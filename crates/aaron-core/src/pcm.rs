//! Monophonic PCM buffers and a dependency-free WAV codec.
//!
//! Audio crosses module boundaries as [`MonoPcm`]. WAV encoding lives here
//! rather than in `aaron-audio` because both the ASR backends (which need to
//! hand a file to an external binary) and the user-facing "save this clip"
//! command need it, and neither should depend on the capture layer.

use std::path::Path;

use crate::error::{Error, Result};

/// Deinterleaved mono samples plus the sample rate they were captured at.
#[derive(Debug, Clone, PartialEq)]
pub struct MonoPcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

impl MonoPcm {
    /// Construct a buffer, asserting the sample rate is sane.
    pub fn new(samples: Vec<f32>, sample_rate: u32) -> Result<Self> {
        if sample_rate == 0 {
            return Err(Error::Audio("sample rate must be non-zero".into()));
        }
        if samples.iter().any(|s| !s.is_finite()) {
            return Err(Error::Audio("buffer contains NaN or infinity".into()));
        }
        Ok(MonoPcm {
            samples,
            sample_rate,
        })
    }

    /// Duration in milliseconds.
    pub fn duration_ms(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.samples.len() as u64 * 1000) / self.sample_rate as u64
    }

    /// Number of samples.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Root mean square level, linear.
    pub fn rms(&self) -> f32 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        ((sum / self.samples.len() as f64).sqrt()) as f32
    }

    /// RMS level in dBFS. Silence maps to `-120.0` rather than `-inf` so that
    /// comparisons and JSON serialisation stay finite.
    pub fn rms_dbfs(&self) -> f32 {
        let rms = self.rms();
        if rms <= 1e-6 {
            -120.0
        } else {
            20.0 * rms.log10()
        }
    }

    /// Peak absolute sample value in `[0, 1]`.
    pub fn peak(&self) -> f32 {
        self.samples.iter().fold(0.0f32, |acc, s| acc.max(s.abs()))
    }

    /// Resample to a target rate with linear interpolation.
    ///
    /// Whisper models expect 16 kHz; ALSA often delivers 44.1 or 48 kHz. Linear
    /// interpolation is adequate because a preceding band-limited filter is
    /// applied by the capture layer when the device rate differs.
    pub fn resample(&self, target: u32) -> MonoPcm {
        if target == self.sample_rate || self.samples.is_empty() {
            return self.clone();
        }
        let ratio = self.sample_rate as f64 / target as f64;
        let out_len = ((self.samples.len() as f64) / ratio).round() as usize;
        let mut out = Vec::with_capacity(out_len);
        for i in 0..out_len {
            let src = i as f64 * ratio;
            let idx = src.floor() as usize;
            let frac = (src - idx as f64) as f32;
            let a = self.samples.get(idx).copied().unwrap_or(0.0);
            let b = self.samples.get(idx + 1).copied().unwrap_or(a);
            out.push(a + (b - a) * frac);
        }
        MonoPcm {
            samples: out,
            sample_rate: target,
        }
    }

    /// Mix any channel count down to mono by averaging.
    pub fn from_interleaved(frames: &[f32], channels: u16) -> Result<MonoPcm> {
        if channels == 0 {
            return Err(Error::Audio("channel count must be non-zero".into()));
        }
        let ch = channels as usize;
        let mut out = Vec::with_capacity(frames.len() / ch);
        for frame in frames.chunks(ch) {
            let sum: f32 = frame.iter().sum();
            out.push(sum / frame.len() as f32);
        }
        Ok(MonoPcm {
            samples: out,
            sample_rate: 16_000,
        })
    }

    /// Encode to 16-bit little-endian PCM WAV bytes.
    pub fn to_wav_bytes(&self) -> Vec<u8> {
        let data_len = self.samples.len() * 2;
        let mut out = Vec::with_capacity(44 + data_len);
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36u32 + data_len as u32).to_le_bytes());
        out.extend_from_slice(b"WAVE");
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
        out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
        out.extend_from_slice(&1u16.to_le_bytes()); // mono
        out.extend_from_slice(&self.sample_rate.to_le_bytes());
        out.extend_from_slice(&(self.sample_rate * 2).to_le_bytes()); // byte rate
        out.extend_from_slice(&2u16.to_le_bytes()); // block align
        out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_len as u32).to_le_bytes());
        for s in &self.samples {
            let clamped = s.clamp(-1.0, 1.0);
            out.extend_from_slice(&((clamped * 32767.0) as i16).to_le_bytes());
        }
        out
    }

    /// Decode 16-bit PCM WAV bytes.
    ///
    /// Deliberately narrow: it accepts mono and stereo PCM16 in RIFF form and
    /// rejects anything else with a clear message rather than guessing.
    pub fn from_wav_bytes(bytes: &[u8]) -> Result<MonoPcm> {
        if bytes.len() < 44 {
            return Err(Error::Audio("WAV file shorter than 44 bytes".into()));
        }
        if &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
            return Err(Error::Audio("not a RIFF/WAVE file".into()));
        }
        let mut pos = 12usize;
        let mut sample_rate = 0u32;
        let mut channels = 1u16;
        let mut bits = 0u16;
        let mut data: Option<(usize, usize)> = None;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
            let body = pos + 8;
            if body + size > bytes.len() {
                return Err(Error::Audio("WAV chunk overruns file".into()));
            }
            if id == b"fmt " {
                if size < 16 {
                    return Err(Error::Audio("short fmt chunk".into()));
                }
                let format = u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap());
                if format != 1 {
                    return Err(Error::Audio(format!(
                        "unsupported WAV format {format} (only PCM=1 is supported)"
                    )));
                }
                channels = u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap());
                sample_rate = u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap());
                bits = u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap());
            } else if id == b"data" {
                data = Some((body, size));
            }
            // Chunks are word aligned.
            pos = body + size + (size % 2);
        }
        let (body, size) = data.ok_or_else(|| Error::Audio("WAV file has no data chunk".into()))?;
        if bits != 16 {
            return Err(Error::Audio(format!(
                "unsupported bit depth {bits} (expected 16)"
            )));
        }
        if channels == 0 {
            return Err(Error::Audio("WAV declares zero channels".into()));
        }
        if sample_rate == 0 {
            return Err(Error::Audio("WAV declares zero sample rate".into()));
        }
        let count = size / 2;
        let raw: Vec<f32> = (0..count)
            .map(|i| {
                let b = body + i * 2;

                i16::from_le_bytes([bytes[b], bytes[b + 1]]) as f32 / 32767.0
            })
            .collect();
        MonoPcm::from_interleaved_with_rate(&raw, channels, sample_rate)
    }

    /// Read and decode a WAV file from disk.
    pub fn read_wav(path: impl AsRef<Path>) -> Result<MonoPcm> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
        MonoPcm::from_wav_bytes(&bytes)
    }

    /// Write this buffer as a 16-bit PCM WAV file.
    pub fn write_wav(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        std::fs::write(path, self.to_wav_bytes()).map_err(|e| Error::io(path, e))
    }

    /// Downmix helper that preserves the declared sample rate.
    fn from_interleaved_with_rate(
        samples: &[f32],
        channels: u16,
        sample_rate: u32,
    ) -> Result<MonoPcm> {
        let ch = channels as usize;
        let mut out = Vec::with_capacity(samples.len() / ch);
        for frame in samples.chunks(ch) {
            let sum: f32 = frame.iter().sum();
            out.push(sum / frame.len() as f32);
        }
        MonoPcm::new(out, sample_rate)
    }

    /// A pure tone, used by `--demo` capture and by audio tests.
    pub fn sine(duration_ms: u64, sample_rate: u32, freq_hz: f32) -> Result<MonoPcm> {
        let n = (duration_ms * sample_rate as u64 / 1000) as usize;
        let samples = (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq_hz * t).sin() * 0.3
            })
            .collect();
        MonoPcm::new(samples, sample_rate)
    }

    /// Silence of the given duration.
    pub fn silence(duration_ms: u64, sample_rate: u32) -> Result<MonoPcm> {
        let n = (duration_ms * sample_rate as u64 / 1000) as usize;
        MonoPcm::new(vec![0.0; n], sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_round_trip() {
        let pcm = MonoPcm::sine(50, 16_000, 440.0).unwrap();
        let bytes = pcm.to_wav_bytes();
        assert_eq!(&bytes[0..4], b"RIFF");
        let back = MonoPcm::from_wav_bytes(&bytes).unwrap();
        assert_eq!(back.sample_rate, 16_000);
        assert_eq!(back.len(), pcm.len());
        for (a, b) in pcm.samples.iter().zip(back.samples.iter()) {
            assert!((a - b).abs() < 1e-4, "sample drift {a} vs {b}");
        }
    }

    #[test]
    fn stereo_is_downmixed() {
        let pcm = MonoPcm::sine(20, 16_000, 440.0).unwrap();
        // Duplicate every sample to build a fake stereo interleaved stream.
        let mut inter = Vec::new();
        for s in &pcm.samples {
            inter.push(*s);
            inter.push(*s);
        }
        let mixed = MonoPcm::from_interleaved_with_rate(&inter, 2, 16_000).unwrap();
        assert_eq!(mixed.len(), pcm.len());
        for (a, b) in pcm.samples.iter().zip(mixed.samples.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn rejects_garbage_wav() {
        assert!(MonoPcm::from_wav_bytes(&[0u8; 10]).is_err());
        assert!(MonoPcm::from_wav_bytes(b"NOTAWAVFILE_______________").is_err());
    }

    #[test]
    fn resample_changes_length_proportionally() {
        let pcm = MonoPcm::sine(100, 48_000, 440.0).unwrap();
        let out = pcm.resample(16_000);
        assert_eq!(out.sample_rate, 16_000);
        let ratio = out.len() as f32 / pcm.len() as f32;
        assert!((ratio - 1.0 / 3.0).abs() < 0.02, "ratio {ratio}");
    }

    #[test]
    fn dbfs_of_silence_is_finite() {
        let s = MonoPcm::silence(10, 16_000).unwrap();
        assert!(s.rms_dbfs().is_finite());
        assert!(s.rms_dbfs() < -100.0);
    }

    #[test]
    fn rejects_invalid_construction() {
        assert!(MonoPcm::new(vec![0.0], 0).is_err());
        assert!(MonoPcm::new(vec![f32::NAN], 16_000).is_err());
    }
}
