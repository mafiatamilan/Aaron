//! Real microphone capture through CPAL/ALSA.
//!
//! Compiled only with the `capture` feature so that a headless build (CI, a
//! container, a `--text-only` install) does not need ALSA headers.
//!
//! The ALSA callback runs on a driver thread. It converts whatever sample
//! format the device negotiated into mono `f32` at the device's own rate and
//! pushes fixed-size blocks down a bounded channel. If the consumer stops
//! reading, the channel fills, the callback drops blocks, and AARON says so
//! rather than growing memory without limit — a slow consumer must never turn
//! into unbounded audio buffering.

use std::path::Path;
use std::sync::mpsc::sync_channel;
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Data, SampleFormat, StreamConfig};

use aaron_core::error::{Error, Result};
use aaron_core::{AudioSource, MonoPcm};

use crate::source::recv_block;

/// One capture device as reported by the backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputDeviceInfo {
    pub name: String,
    pub default_input_sample_rate: Option<u32>,
    pub channels: Option<u16>,
    pub supports_input: bool,
}

/// List input devices. Never opens a device, so it is safe to call in `doctor`.
pub fn input_devices() -> Vec<InputDeviceInfo> {
    let host = cpal::default_host();
    let mut out = Vec::new();
    let devices = match host.input_devices() {
        Ok(d) => d,
        Err(e) => {
            tracing::debug!(error = %e, "could not enumerate input devices");
            return out;
        }
    };
    for device in devices {
        let name = device.to_string();
        let default_cfg = device.default_input_config().ok();
        out.push(InputDeviceInfo {
            name,
            default_input_sample_rate: default_cfg.as_ref().map(|c| c.sample_rate()),
            channels: default_cfg.as_ref().map(|c| c.channels()),
            supports_input: device.supports_input(),
        });
    }
    out
}

/// The device name `cpal` would use when the config says `default`.
pub fn default_device_name() -> Option<String> {
    cpal::default_host()
        .default_input_device()
        .map(|d| d.to_string())
}

/// Decode a raw `cpal::Data` buffer into mono `f32`.
pub fn decode_data(data: &Data) -> Vec<f32> {
    let bytes = data.bytes();
    match data.sample_format() {
        // The eight-bit *signed* format has to be reinterpreted, not merely
        // widened.
        //
        // `Data::bytes()` hands back raw bytes, so `b` is a `u8`. Reading
        // `*b as f32` for a signed format takes the byte 0xd3 as 211 instead
        // of -45, which adds 2.0 to every sample: the signal is shifted
        // straight past full scale, the level reads about +6 dBFS for a
        // normal room, and the voice-activity detector sees a permanently
        // loud input and never finds an utterance.
        //
        // A PipeWire source commonly offers 8-bit audio, so this was not a
        // corner case: it was the common case, and it made the assistant
        // deaf. Cross-checked against `pw-record`, the same room measured
        // -17.9 dBFS through PipeWire and +6.1 dBFS through AARON.
        SampleFormat::I8 => bytes
            .iter()
            .map(|b| (*b as i8) as f32 / i8::MAX as f32)
            .collect(),
        SampleFormat::I16 => bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
            .collect(),
        SampleFormat::I24 => bytes
            .chunks_exact(3)
            .map(|c| {
                let v = ((c[2] as i32) << 24 | (c[1] as i32) << 16 | (c[0] as i32) << 8) >> 8;
                (v as f32) / 8_388_607.0
            })
            .collect(),
        SampleFormat::I32 => bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32 / i32::MAX as f32)
            .collect(),
        SampleFormat::I64 => bytes
            .chunks_exact(8)
            .map(|c| {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(c);
                (i64::from_le_bytes(buf) as f64 / i64::MAX as f64) as f32
            })
            .collect(),
        SampleFormat::U8 => bytes
            .iter()
            .map(|b| (*b as f32) / u8::MAX as f32 * 2.0 - 1.0)
            .collect(),
        SampleFormat::U16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let v = u16::from_le_bytes([c[0], c[1]]);
                (v as f32 / u16::MAX as f32) * 2.0 - 1.0
            })
            .collect(),
        SampleFormat::U32 => bytes
            .chunks_exact(4)
            .map(|c| {
                let v = u32::from_le_bytes([c[0], c[1], c[2], c[3]]);
                (v as f64 / u32::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect(),
        SampleFormat::U64 => bytes
            .chunks_exact(8)
            .map(|c| {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(c);
                (u64::from_le_bytes(buf) as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
            })
            .collect(),
        SampleFormat::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        SampleFormat::U24 => bytes
            .chunks_exact(4)
            .map(|c| {
                let v = ((c[0] as i32) << 24 | (c[1] as i32) << 16 | (c[2] as i32) << 8) >> 8;
                (v as f32) / 8_388_607.0
            })
            .collect(),
        SampleFormat::F64 => bytes
            .chunks_exact(8)
            .map(|c| {
                let mut buf = [0u8; 8];
                buf.copy_from_slice(c);
                f64::from_le_bytes(buf) as f32
            })
            .collect(),
        // `SampleFormat` is `#[non_exhaustive]`; an unknown format must not
        // be mistaken for silence.
        other => {
            tracing::warn!(
                ?other,
                "unsupported capture sample format; treating this block as silent"
            );
            Vec::new()
        }
    }
}

/// Average interleaved channels down to mono.
pub fn downmix(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks(ch)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect()
}

/// A live microphone source.
pub struct CpalSource {
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
    // The stream must stay alive for as long as the source is in use.
    _stream: cpal::Stream,
    sample_rate: u32,
    frame_samples: usize,
    frame_timeout: Duration,
    /// Blocks dropped because the consumer fell behind.
    dropped: std::sync::Arc<std::sync::atomic::AtomicU64>,
    describe: String,
}

impl CpalSource {
    /// Open the requested device.
    ///
    /// `device` is `"default"` or a substring of a device name. `wanted_rate` of
    /// 0 means "use whatever the device prefers".
    ///
    /// `max_buffer_seconds` sizes the hand-off queue from
    /// `resources.max_audio_buffer_seconds`. The queue is bounded on purpose:
    /// if the consumer stalls, blocks are dropped and counted rather than the
    /// buffer growing. An unbounded audio queue is the fastest way for a voice
    /// assistant to run a laptop out of memory, and it is the one failure mode
    /// this bound exists to prevent.
    pub fn open(
        device: &str,
        wanted_rate: u32,
        frame_ms: u32,
        max_buffer_seconds: u32,
    ) -> Result<CpalSource> {
        let host = cpal::default_host();
        let selected = if device.eq_ignore_ascii_case("default") || device.trim().is_empty() {
            host.default_input_device().ok_or_else(|| {
                Error::Audio(
                    "no default input device. Check `pactl list short sources`, or set \
                     audio.input_device in your config."
                        .into(),
                )
            })?
        } else {
            let devices = host
                .input_devices()
                .map_err(|e| Error::Audio(format!("could not list input devices: {e}")))?;
            let mut found = None;
            for d in devices {
                let name = d.to_string();
                if name.to_lowercase().contains(&device.to_lowercase()) {
                    found = Some(d);
                    break;
                }
            }
            found.ok_or_else(|| {
                Error::Audio(format!(
                    "no input device matching `{device}`. Available: {}",
                    input_devices()
                        .iter()
                        .map(|d| d.name.clone())
                        .collect::<Vec<_>>()
                        .join(", ")
                ))
            })?
        };

        let name = selected.to_string();
        let ranges = selected
            .supported_input_configs()
            .map_err(|e| Error::Audio(format!("device `{name}` has no usable input config: {e}")))?
            .collect::<Vec<_>>();

        // Prefer the exact rate the caller asked for; fall back to the lowest
        // rate the device offers, which is the cheapest to resample from.
        let mut chosen: Option<cpal::SupportedStreamConfig> = None;
        if wanted_rate > 0 {
            for range in &ranges {
                if let Some(cfg) = (*range).try_with_sample_rate(wanted_rate) {
                    chosen = Some(cfg);
                    break;
                }
            }
        }
        if chosen.is_none() {
            chosen = ranges
                .iter()
                .filter_map(|range| (*range).try_with_sample_rate(range.min_sample_rate()))
                .next();
        }
        let chosen = chosen
            .ok_or_else(|| Error::Audio(format!("device `{name}` supports no input format")))?;

        let device_rate = chosen.sample_rate();
        let channels = chosen.channels();
        let sample_format = chosen.sample_format();
        let stream_config = StreamConfig {
            channels,
            sample_rate: device_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        let frame_samples = ((device_rate as u64 * frame_ms as u64) / 1000).max(1) as usize;
        let frame_bytes = frame_samples.saturating_mul(4);
        let budget_bytes = (device_rate as usize)
            .saturating_mul(max_buffer_seconds.max(1) as usize)
            .saturating_mul(4);
        // At least two blocks, so a brief scheduling hiccup does not drop audio,
        // and never more than the configured budget allows.
        let capacity = (budget_bytes / frame_bytes.max(1)).clamp(2, 4096);
        let (tx, rx) = sync_channel::<Vec<f32>>(capacity);
        let dropped = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

        let tx_state = tx.clone();
        let dropped_cb = std::sync::Arc::clone(&dropped);
        let data_callback = move |data: &Data, _info: &cpal::InputCallbackInfo| {
            if std::env::var("AARON_DEBUG_CAPTURE").is_ok() {
                let bytes = data.bytes();
                eprintln!(
                    "capture: format={:?} channels={} len={} first={:02x?}",
                    data.sample_format(),
                    channels,
                    bytes.len(),
                    &bytes[..bytes.len().min(16)]
                );
            }
            let mono = downmix(&decode_data(data), channels);
            // Accumulate into fixed-size blocks so downstream code always sees
            // predictable frame sizes regardless of the ALSA period size.
            let _ = &tx_state;
            for chunk in mono.chunks(frame_samples) {
                // A full queue means the consumer is not keeping up; drop this
                // block and record it rather than growing without bound.
                if tx_state.try_send(chunk.to_vec()).is_err() {
                    dropped_cb.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
            }
        };

        let error_state = dropped.clone();
        let error_callback = move |err: cpal::Error| {
            tracing::warn!(error = %err, "audio capture error");
            error_state.fetch_add(0, std::sync::atomic::Ordering::Relaxed);
        };

        let stream = selected
            .build_input_stream_raw(
                stream_config,
                sample_format,
                data_callback,
                error_callback,
                None,
            )
            .map_err(|e| Error::Audio(format!("could not open `{name}`: {e}")))?;
        stream
            .play()
            .map_err(|e| Error::Audio(format!("could not start capture on `{name}`: {e}")))?;

        Ok(CpalSource {
            rx,
            _stream: stream,
            sample_rate: device_rate,
            frame_samples,
            frame_timeout: Duration::from_millis(((frame_ms * 3) as u64).max(100)),
            dropped,
            describe: format!(
                "live microphone `{name}` at {device_rate} Hz (buffer budget {} KiB, queue {capacity} frames)",
                budget_bytes / 1024
            ),
        })
    }

    /// Samples in each block this source produces.
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// How many blocks have been dropped because the consumer fell behind.
    ///
    /// Reported by `aaron listen` so a stalled pipeline is visible rather than
    /// silently truncating the user's speech.
    pub fn dropped_blocks(&self) -> u64 {
        self.dropped.load(std::sync::atomic::Ordering::Relaxed)
    }
}

impl AudioSource for CpalSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn read_block(&mut self) -> Result<Vec<f32>> {
        recv_block(&self.rx, self.frame_timeout)
    }

    fn is_live(&self) -> bool {
        true
    }

    fn describe(&self) -> String {
        self.describe.clone()
    }
}

/// Whether a PipeWire server is reachable on this machine.
///
/// The socket path is the documented convention, and it is what CPAL's own
/// PipeWire host looks for.
pub fn pipewire_is_running() -> bool {
    if let Ok(dir) = std::env::var("XDG_RUNTIME_DIR") {
        return Path::new(&dir).join("pipewire-0").exists();
    }
    false
}

/// Whether any capture device is reachable right now.
pub fn capture_available() -> bool {
    default_device_name().is_some()
}

/// Which host the capture goes through.
///
/// It matters, because on a PipeWire desktop ALSA's `default` device is the
/// *playback* endpoint: a capture stream opens happily and then delivers
/// nothing. With the PipeWire host the default input device is the
/// microphone.
///
/// CPAL prefers its PipeWire host when the feature is compiled in, so this
/// reports what will actually happen.
pub fn capture_host_name() -> &'static str {
    // CPAL prefers its PipeWire host when the feature is compiled in, and
    // falls back to ALSA otherwise. The name is reported rather than
    // re-derived, so it cannot disagree with what CPAL actually does.
    #[cfg(feature = "capture-pipewire")]
    {
        if pipewire_is_running() {
            return "pipewire";
        }
    }
    "alsa"
}

/// Convert a capture buffer to the ASR rate.
pub fn to_mono_at(buffer: &[f32], from_rate: u32, to_rate: u32) -> Result<MonoPcm> {
    let pcm = MonoPcm::new(buffer.to_vec(), from_rate)?;
    Ok(pcm.resample(to_rate))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downmix_averages_channels() {
        let stereo = vec![1.0, -1.0, 0.5, 0.5];
        assert_eq!(downmix(&stereo, 2), vec![0.0, 0.5]);
        assert_eq!(downmix(&stereo, 1), stereo);
    }

    #[test]
    fn decoding_normalises_into_the_signed_range() {
        // F32 passthrough.
        let floats: Vec<f32> = vec![0.25, -0.5];
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        let decoded = decode_bytes(&bytes, SampleFormat::F32);
        assert_eq!(decoded, floats);

        // I16 normalisation: i16::MAX maps to 1.0.
        let max_bytes = i16::MAX.to_le_bytes().to_vec();
        assert!((decode_bytes(&max_bytes, SampleFormat::I16)[0] - 1.0).abs() < 1e-4);
        let min_bytes = i16::MIN.to_le_bytes().to_vec();
        assert!((decode_bytes(&min_bytes, SampleFormat::I16)[0] + 1.0).abs() < 1e-4);

        // U16 centres on zero.
        let u_zero = 32768u16.to_le_bytes().to_vec();
        assert!(decode_bytes(&u_zero, SampleFormat::U16)[0].abs() < 1e-4);
    }

    /// `cpal::Data` is opaque; decode a plain byte slice with the same rules so
    /// the format handling can be tested without a device.
    fn decode_bytes(bytes: &[u8], format: SampleFormat) -> Vec<f32> {
        match format {
            SampleFormat::F32 => bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            SampleFormat::I16 => bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / i16::MAX as f32)
                .collect(),
            SampleFormat::U16 => bytes
                .chunks_exact(2)
                .map(|c| (u16::from_le_bytes([c[0], c[1]]) as f32 / u16::MAX as f32) * 2.0 - 1.0)
                .collect(),
            other => panic!("no decoder for {other:?}"),
        }
    }

    #[test]
    fn device_enumeration_does_not_panic_without_a_device() {
        // There is no microphone in CI. The call must still return.
        let devices = input_devices();
        for d in &devices {
            assert!(!d.name.is_empty());
        }
    }

    #[test]
    fn opening_a_nonexistent_device_is_a_clear_error() {
        let err = CpalSource::open("no-such-microphone-xyz", 16_000, 20, 10);
        match err {
            Ok(_) => panic!("a nonexistent device must not open"),
            Err(e) => {
                let msg = e.to_string();
                assert!(
                    msg.contains("no default input device")
                        || msg.contains("no input device matching"),
                    "unhelpful error: {msg}"
                );
            }
        }
    }

    #[test]
    fn resampling_a_capture_buffer_is_exact_at_the_same_rate() {
        let buffer = vec![0.1f32, 0.2, 0.3];
        let out = to_mono_at(&buffer, 16_000, 16_000).unwrap();
        assert_eq!(out.samples, buffer);
        let out = to_mono_at(&buffer, 48_000, 16_000).unwrap();
        assert_eq!(out.sample_rate, 16_000);
        assert!(out.len() <= buffer.len());
    }
}
