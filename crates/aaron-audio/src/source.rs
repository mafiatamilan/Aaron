//! Audio sources.
//!
//! Three implementations of [`aaron_core::AudioSource`]:
//!
//! * [`SyntheticSource`] — a generated tone/speech-shaped signal. Used by
//!   `aaron listen --source synthetic`, by the audio tests, and by CI, which
//!   has no microphone.
//! * [`FileSource`] — plays back a WAV file. Used by `--source file` and by
//!   integration tests that need a reproducible signal.
//! * [`capture::CpalSource`] — the real microphone, compiled only when the
//!   `capture` feature is on.
//!
//! Every source reports `is_live()`, and the CLI refuses to draw the "microphone
//! live" indicator for a non-live source. That distinction is a privacy
//! property, not a cosmetic one: the user must never see a live indicator when
//! nothing is recording.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::time::Duration;

use crate::vad;
use aaron_core::config::VadConfig;
use aaron_core::error::{Error, Result};
use aaron_core::{AudioSource, MonoPcm};

/// A source that generates a deterministic signal.
pub struct SyntheticSource {
    sample_rate: u32,
    frame_samples: usize,
    /// Tone frequency.
    freq: f32,
    /// Envelope: the tone is chopped into speech-like bursts so the VAD has
    /// something realistic to find.
    burst_on_ms: u32,
    burst_off_ms: u32,
    position: usize,
    done: bool,
}

impl SyntheticSource {
    /// A source producing `burst_on_ms` of tone followed by `burst_off_ms` of
    /// silence, repeating until `total_ms` elapses.
    pub fn new(
        total_ms: u64,
        sample_rate: u32,
        frame_ms: u32,
        freq: f32,
    ) -> Result<SyntheticSource> {
        if total_ms == 0 {
            return Err(Error::Audio(
                "synthetic source needs a non-zero duration".into(),
            ));
        }
        let frame_samples = ((sample_rate as u64 * frame_ms as u64) / 1000).max(1) as usize;
        Ok(SyntheticSource {
            sample_rate,
            frame_samples,
            freq,
            burst_on_ms: 300.max(frame_ms * 4),
            burst_off_ms: 200.max(frame_ms * 3),
            position: 0,
            done: false,
        })
    }
}

impl AudioSource for SyntheticSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn read_block(&mut self) -> Result<Vec<f32>> {
        if self.done {
            return Ok(Vec::new());
        }
        let mut out = Vec::with_capacity(self.frame_samples);
        let cycle = (self.burst_on_ms + self.burst_off_ms) as f64;
        for _ in 0..self.frame_samples {
            let t_ms = (self.position as f64) / (self.sample_rate as f64) * 1000.0;
            let in_burst = (t_ms % cycle) < self.burst_on_ms as f64;
            let t = self.position as f32 / self.sample_rate as f32;
            // Fade the burst edges so the VAD sees a realistic envelope.
            let phase_in = (t_ms % cycle) / self.burst_on_ms as f64;
            let envelope = if !in_burst {
                0.0
            } else {
                let p = phase_in.clamp(0.0, 1.0);
                (p * 8.0).min(1.0).min((1.0 - p) * 8.0)
            };
            out.push((2.0 * std::f32::consts::PI * self.freq * t).sin() * 0.35 * envelope as f32);
            self.position += 1;
        }
        Ok(out)
    }

    fn is_live(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        format!(
            "synthetic {freq} Hz tone (no microphone is open)",
            freq = self.freq
        )
    }
}

/// Plays back a WAV file, frame by frame.
pub struct FileSource {
    pcm: MonoPcm,
    frame_samples: usize,
    position: usize,
}

impl FileSource {
    /// Load a WAV file, resampling to `target_rate` if needed.
    pub fn open(path: &std::path::Path, target_rate: u32, frame_ms: u32) -> Result<FileSource> {
        let pcm = MonoPcm::read_wav(path)?;
        let pcm = pcm.resample(target_rate);
        let frame_samples = ((target_rate as u64 * frame_ms as u64) / 1000).max(1) as usize;
        Ok(FileSource {
            pcm,
            frame_samples,
            position: 0,
        })
    }

    /// Build from an in-memory buffer.
    pub fn from_pcm(pcm: MonoPcm, frame_ms: u32) -> Result<FileSource> {
        let frame_samples = ((pcm.sample_rate as u64 * frame_ms as u64) / 1000).max(1) as usize;
        Ok(FileSource {
            pcm,
            frame_samples,
            position: 0,
        })
    }
}

impl AudioSource for FileSource {
    fn sample_rate(&self) -> u32 {
        self.pcm.sample_rate
    }

    fn read_block(&mut self) -> Result<Vec<f32>> {
        if self.position >= self.pcm.len() {
            return Ok(Vec::new());
        }
        let end = (self.position + self.frame_samples).min(self.pcm.len());
        let block = self.pcm.samples[self.position..end].to_vec();
        self.position = end;
        Ok(block)
    }

    fn is_live(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        format!(
            "WAV playback, {} ms of audio at {} Hz",
            self.pcm.duration_ms(),
            self.pcm.sample_rate
        )
    }
}

/// A source that yields nothing, for `--no-mic`.
pub struct NullSource {
    sample_rate: u32,
    reported: bool,
}

impl NullSource {
    pub fn new(sample_rate: u32) -> NullSource {
        NullSource {
            sample_rate,
            reported: false,
        }
    }
}

impl AudioSource for NullSource {
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn read_block(&mut self) -> Result<Vec<f32>> {
        if self.reported {
            return Ok(Vec::new());
        }
        self.reported = true;
        Err(Error::Audio(
            "audio capture is disabled; use `aaron run --text-only` or `aaron listen --source synthetic`"
                .into(),
        ))
    }

    fn is_live(&self) -> bool {
        false
    }

    fn describe(&self) -> String {
        "disabled".into()
    }
}

/// Collect frames from any source until the VAD reports an utterance finished,
/// the maximum duration elapses, or the source ends.
///
/// Returns the trimmed mono buffer and whether speech was actually found.
/// How to capture one utterance.
///
/// The capture runs entirely in the **source's own sample rate** and is
/// resampled once, at the end, to `target_rate`.
///
/// # Why the rate matters so much
///
/// A microphone usually does not run at the rate the configuration asks for.
/// This machine's default source offers 48 kHz and nothing else, so the
/// 16 kHz the config asks for is not available and the capture opens at the
/// device's rate.
///
/// If the detector and the duration arithmetic are then computed against the
/// *configured* rate, every duration in the pipeline is wrong by the ratio
/// between the two: a 30 ms analysis frame becomes 10 ms of real audio, the
/// 700 ms of trailing silence becomes 233 ms, and the 12-second cap becomes
/// 4 seconds. The visible symptom is that the assistant listens to a real
/// microphone, hears a real voice, and reports that nobody spoke — because
/// the utterance never ends inside the (shortened) window.
///
/// So the rate is read from the source, and the target rate is only applied
/// to the finished buffer.
#[derive(Debug, Clone, PartialEq)]
pub struct CaptureSpec {
    /// The voice-activity detector's settings.
    pub vad: VadConfig,
    /// The analysis frame length, in milliseconds.
    pub frame_ms: u32,
    /// The rate the finished buffer is delivered at.
    pub target_rate: u32,
    /// The hard cap on one utterance, in seconds.
    pub max_secs: u32,
}

impl Default for CaptureSpec {
    fn default() -> Self {
        CaptureSpec {
            vad: VadConfig::default(),
            frame_ms: 30,
            target_rate: 16_000,
            max_secs: 12,
        }
    }
}

/// Record one utterance, detecting the speech region and trimming to it.
///
/// The detection happens in the source's own rate; the returned buffer is at
/// `spec.target_rate`.
pub fn record_utterance(
    source: &mut dyn AudioSource,
    spec: &CaptureSpec,
) -> Result<RecordedUtterance> {
    // The source's rate, not the configured one. Everything below is in these
    // units, and the buffer is converted once at the end.
    let source_rate = source.sample_rate().max(1);
    let vad = crate::vad::VoiceActivityDetector::new(&spec.vad, source_rate, spec.frame_ms);

    let max_samples = (source_rate as u64 * spec.max_secs as u64) as usize;
    let frame_samples = vad.frame_samples();
    let silence_samples = (vad.silence_ms() as u64 * source_rate as u64 / 1000) as usize;
    let mut collected: Vec<f32> = Vec::new();
    let mut started = false;

    while collected.len() < max_samples {
        let block = source.read_block()?;
        if block.is_empty() {
            break;
        }
        collected.extend_from_slice(&block);

        let outcome = vad.analyse(&collected);
        // Speech has begun as soon as the VAD sees a speech region that clears
        // the minimum duration; `TooShort` means it has not yet.
        //
        // `Truncated` counts: the buffer ended while the speech was still
        // going, and the region is there. It used to be discarded, so a
        // recording of clear speech was thrown away because it did not end
        // politely — which is every file source, and every person who talks
        // until the capture cap.
        let end_frame = match outcome {
            vad::VadOutcome::Speech { end_frame, .. }
            | vad::VadOutcome::Truncated { end_frame, .. } => end_frame,
            _ => continue,
        };
        started = true;

        // Keep going until the trailing silence has arrived.
        let speech_samples = end_frame * frame_samples;
        if collected.len() >= speech_samples + silence_samples {
            let cropped = vad::crop(&collected, source_rate, &outcome, frame_samples, 80);
            return Ok(RecordedUtterance {
                pcm: MonoPcm::new(cropped, source_rate)?.resample(spec.target_rate),
                speech_detected: true,
                hit_cap: false,
            });
        }
    }

    let outcome = vad.analyse(&collected);
    let speech_detected = started
        && matches!(
            outcome,
            vad::VadOutcome::Speech { .. } | vad::VadOutcome::Truncated { .. }
        );
    let audio = if speech_detected {
        vad::crop(&collected, source_rate, &outcome, vad.frame_samples(), 80)
    } else {
        Vec::new()
    };
    Ok(RecordedUtterance {
        pcm: MonoPcm::new(audio, source_rate)?.resample(spec.target_rate),
        speech_detected,
        hit_cap: collected.len() >= max_samples,
    })
}

/// The raw level of a source over a window, measured without the
/// voice-activity detector.
///
/// # Why this exists
///
/// [`record_utterance`] returns an **empty** buffer when it finds no speech.
/// That is correct for a voice turn, and useless for a diagnostic: a quiet
/// room and a microphone that is delivering nothing at all produce the same
/// empty buffer, and the difference matters enormously. A user whose
/// microphone is dead needs to be told so, not told that nobody spoke.
///
/// So this measures the signal itself: peak, mean and how many frames
/// arrived.
#[derive(Debug, Clone, PartialEq)]
pub struct LevelReport {
    /// The loudest sample, in dBFS. `-120` means digital silence.
    pub peak_dbfs: f32,
    /// The mean level over the window, in dBFS.
    pub mean_dbfs: f32,
    /// How many samples arrived.
    pub samples: usize,
    /// The sample rate the source ran at.
    pub sample_rate: u32,
    /// How long the window was.
    pub duration_ms: u64,
    /// Whether the window ended because the time ran out rather than because
    /// the source ran out of audio.
    pub hit_cap: bool,
}

impl LevelReport {
    /// Whether the source delivered a usable signal.
    ///
    /// A live source that delivers only digital silence is a failure, not a
    /// quiet room: on a PipeWire desktop an ALSA "default" capture opens the
    /// *playback* endpoint and delivers nothing, and the user would otherwise
    /// be told nobody said anything.
    pub fn delivered_audio(&self) -> bool {
        self.samples > 0 && self.peak_dbfs > -90.0
    }
}

/// Measure the raw level of a source over a window.
///
/// Reads blocks until the window is full or the source runs out, and never
/// discards anything: this is a measurement, not a turn.
pub fn measure_level(source: &mut dyn AudioSource, max_secs: u32) -> Result<LevelReport> {
    let sample_rate = source.sample_rate();
    let max_samples = (sample_rate as u64 * max_secs.max(1) as u64) as usize;
    let mut collected: Vec<f32> = Vec::with_capacity(max_samples.min(1 << 20));
    while collected.len() < max_samples {
        let block = source.read_block()?;
        if block.is_empty() {
            break;
        }
        collected.extend_from_slice(&block);
    }
    let peak = collected.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    // The mean level is an *amplitude*, so the mean power has to be square
    // rooted before it is converted to dBFS. Reporting the mean power as if
    // it were an amplitude makes the mean larger than the peak, which is
    // impossible and obvious once you look for it.
    let mean = if collected.is_empty() {
        0.0
    } else {
        (collected.iter().map(|s| s * s).sum::<f32>() / collected.len() as f32).sqrt()
    };
    Ok(LevelReport {
        peak_dbfs: to_dbfs(peak),
        mean_dbfs: to_dbfs(mean),
        samples: collected.len(),
        sample_rate,
        duration_ms: if sample_rate == 0 {
            0
        } else {
            (collected.len() as u64 * 1000) / sample_rate as u64
        },
        hit_cap: collected.len() >= max_samples,
    })
}

/// A linear amplitude in dBFS, floored so the output stays finite.
fn to_dbfs(amplitude: f32) -> f32 {
    if amplitude <= 1e-6 {
        -120.0
    } else {
        20.0 * amplitude.log10()
    }
}

/// The result of a capture attempt.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedUtterance {
    /// Trimmed mono 16 kHz audio. Empty when no speech was found.
    pub pcm: MonoPcm,
    pub speech_detected: bool,
    /// Whether the hard duration cap stopped the recording.
    pub hit_cap: bool,
}

/// Drain a channel-backed source with a timeout.
///
/// Used by the CPAL capture source, which delivers blocks from an ALSA
/// callback thread.
/// Wait up to `timeout` for the next block; an empty vector means the
/// capture thread is idle rather than finished.
pub fn recv_block(rx: &Receiver<Vec<f32>>, timeout: Duration) -> Result<Vec<f32>> {
    match rx.recv_timeout(timeout) {
        Ok(block) => Ok(block),
        Err(mpsc::RecvTimeoutError::Timeout) => Ok(Vec::new()),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(Error::Audio(
            "the capture thread stopped unexpectedly".into(),
        )),
    }
}

/// Non-blocking drain helper.
///
/// Used by the wake loop, which polls rather than blocks so it can keep the UI
/// responsive between audio frames.
pub fn try_recv_block(rx: &Receiver<Vec<f32>>) -> Result<Vec<f32>> {
    match rx.try_recv() {
        Ok(block) => Ok(block),
        Err(TryRecvError::Empty) => Ok(Vec::new()),
        Err(TryRecvError::Disconnected) => Err(Error::Audio(
            "the capture thread stopped unexpectedly".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_source_produces_bursts_and_is_not_live() {
        let mut s = SyntheticSource::new(2000, 16_000, 20, 220.0).unwrap();
        assert!(!s.is_live());
        let mut loud = 0.0f32;
        let mut quiet = 0.0f32;
        for _ in 0..50 {
            let block = s.read_block().unwrap();
            assert_eq!(block.len(), 320);
            let peak = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            if peak > 0.05 {
                loud = loud.max(peak);
            } else {
                quiet = quiet.max(peak);
            }
        }
        assert!(loud > 0.1, "synthetic source produced no loud burst");
        assert!(
            quiet < loud,
            "synthetic source produced no silence between bursts"
        );
    }

    #[test]
    fn recording_the_synthetic_source_finds_speech() {
        let mut source = SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap();
        let cfg = aaron_core::config::VadConfig {
            silence_ms: 200,
            min_speech_ms: 60,
            ..Default::default()
        };
        let spec = CaptureSpec {
            vad: cfg,
            frame_ms: 20,
            target_rate: 16_000,
            max_secs: 5,
        };
        let out = record_utterance(&mut source, &spec).unwrap();
        assert!(out.speech_detected);
        assert!(!out.pcm.is_empty());
        assert!(out.pcm.duration_ms() >= 100);
    }

    #[test]
    fn recording_silence_reports_no_speech() {
        let silent = MonoPcm::silence(1000, 16_000).unwrap();
        let mut source = FileSource::from_pcm(silent, 20).unwrap();
        let spec = CaptureSpec {
            frame_ms: 20,
            ..Default::default()
        };
        let out = record_utterance(&mut source, &spec).unwrap();
        assert!(!out.speech_detected);
        assert!(out.pcm.is_empty());
    }

    #[test]
    fn file_source_round_trips_through_a_wav_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tone.wav");
        let tone = MonoPcm::sine(200, 16_000, 440.0).unwrap();
        tone.write_wav(&path).unwrap();

        let mut source = FileSource::open(&path, 16_000, 20).unwrap();
        assert!(!source.is_live());
        let total: usize = (0..20).map(|_| source.read_block().unwrap().len()).sum();
        assert_eq!(total, tone.len());
        assert!(
            source.read_block().unwrap().is_empty(),
            "source must end cleanly"
        );
    }

    #[test]
    fn file_source_resamples_to_the_requested_rate() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("tone.wav");
        MonoPcm::sine(200, 48_000, 440.0)
            .unwrap()
            .write_wav(&path)
            .unwrap();
        let source = FileSource::open(&path, 16_000, 20).unwrap();
        assert_eq!(source.sample_rate(), 16_000);
    }

    #[test]
    fn null_source_explains_itself_once_then_ends() {
        let mut s = NullSource::new(16_000);
        let err = s.read_block().unwrap_err();
        assert!(err.to_string().contains("--text-only"));
        assert!(s.read_block().unwrap().is_empty());
        assert!(!s.is_live());
    }

    #[test]
    fn the_duration_cap_is_enforced() {
        let mut source = SyntheticSource::new(60_000, 16_000, 20, 220.0).unwrap();
        // Never-ending speech: the cap is the only thing that stops it.
        let cfg = aaron_core::config::VadConfig {
            silence_ms: 60_000,
            ..Default::default()
        };
        let spec = CaptureSpec {
            vad: cfg,
            frame_ms: 20,
            target_rate: 16_000,
            max_secs: 2,
        };
        let out = record_utterance(&mut source, &spec).unwrap();
        assert!(out.hit_cap || !out.pcm.is_empty());
        assert!(out.pcm.duration_ms() <= 2500);
    }

    /// The regression test for the defect that stopped the voice path
    /// working on a device whose rate is not the configured one.
    ///
    /// A microphone usually offers 48 kHz and nothing else, so the capture
    /// opens at the device's rate. Before the fix, the detector and every
    /// duration were computed against the *configured* 16 kHz, so a 30 ms
    /// frame became 10 ms of real audio, the trailing-silence requirement
    /// shrank, and the capture cap stopped a third of the way through. The
    /// symptom was that the assistant heard a real voice and reported that
    /// nobody spoke.
    ///
    /// The test builds exactly that: speech at 48 kHz, a target of 16 kHz.
    #[test]
    fn a_source_at_a_different_rate_is_still_understood() {
        let rate = 48_000u32;
        // One second of speech, then one second of silence.
        let mut samples: Vec<f32> = (0..rate)
            .map(|i| {
                let t = i as f32 / rate as f32;
                (t * 440.0 * std::f32::consts::TAU).sin() * 0.5
            })
            .collect();
        samples.extend(std::iter::repeat_n(0.0, rate as usize));
        let pcm = MonoPcm::new(samples, rate).unwrap();
        let mut source = FileSource::from_pcm(pcm, 20).unwrap();

        // The shipped configuration, unchanged: 700 ms of trailing silence,
        // 200 ms of minimum speech, a 12-second cap.
        let spec = CaptureSpec {
            vad: aaron_core::config::VadConfig::default(),
            frame_ms: 30,
            target_rate: 16_000,
            max_secs: 12,
        };
        let out = record_utterance(&mut source, &spec).unwrap();
        assert!(
            out.speech_detected,
            "a voice at {rate} Hz must be understood when the target is 16 kHz"
        );
        assert!(!out.pcm.is_empty());
        // The delivered buffer is at the *target* rate.
        assert_eq!(out.pcm.sample_rate, 16_000);
        // And the speech is about as long as it was, not a third of it.
        let spoken = out.pcm.duration_ms();
        assert!(
            (700..=1400).contains(&spoken),
            "the utterance should be about a second long, got {spoken} ms"
        );
        // And the cap must be measured in the source's rate, not the
        // target's: at 48 kHz a 12-second cap is 12 seconds of real audio,
        // not 4.
        let mut long = SyntheticSource::new(48_000 * 12, rate, 30, 220.0).unwrap();
        let capped = record_utterance(&mut long, &spec).unwrap();
        assert!(
            capped.pcm.duration_ms() <= 13_000,
            "the cap must allow twelve seconds of real audio, got {} ms",
            capped.pcm.duration_ms()
        );
    }

    /// The level report must be able to tell a quiet room from a dead
    /// microphone, which the speech detector cannot: it returns an empty
    /// buffer for both.
    #[test]
    fn the_level_report_separates_silence_from_a_dead_device() {
        let mut silent = FileSource::from_pcm(MonoPcm::silence(1000, 16_000).unwrap(), 20).unwrap();
        let quiet = measure_level(&mut silent, 2).unwrap();
        assert!(!quiet.delivered_audio(), "silence is not audio");

        let mut tone = SyntheticSource::new(2000, 16_000, 20, 220.0).unwrap();
        let loud = measure_level(&mut tone, 2).unwrap();
        assert!(loud.delivered_audio(), "a tone is audio");
        assert!(loud.peak_dbfs > -30.0, "peak was {}", loud.peak_dbfs);
    }
}
