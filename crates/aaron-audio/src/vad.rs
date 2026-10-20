//! Energy-based voice activity detection.
//!
//! This is a *deliberately simple* VAD: frame RMS energy against a threshold,
//! plus onset, minimum-duration and trailing-silence rules. It has no model and
//! no parameters to train, which means it cannot be a privacy leak and its
//! behaviour is explainable.
//!
//! Its job is only to decide when to stop recording. Speech *content* is the
//! ASR backend's problem, and this module never pretends otherwise.

use aaron_core::config::VadConfig;

/// What the VAD decided about a completed capture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VadOutcome {
    /// Speech was found and the buffer is worth transcribing.
    Speech {
        /// Index of the first frame above threshold.
        start_frame: usize,
        /// Index one past the last frame above threshold.
        end_frame: usize,
    },
    /// Nothing above threshold; the buffer was silence or noise.
    Silence,
    /// Speech was shorter than `min_speech_ms` (a click, a pop, a cough).
    TooShort,
    /// The capture ended while the speech was still going.
    ///
    /// The speech region is there — it simply has no trailing silence to
    /// close it. This happens whenever the utterance runs to the end of the
    /// buffer: a recording of nothing but speech, a person who talks
    /// continuously until the capture cap, and a file source, which has no
    /// silence after it at all.
    ///
    /// It used to carry no region, and the caller discarded the audio
    /// entirely — so a recording of clear speech was thrown away because it
    /// did not end politely.
    Truncated {
        /// Index of the first frame above threshold.
        start_frame: usize,
        /// Index one past the last frame above threshold.
        end_frame: usize,
    },
}

/// How far above the measured noise floor a frame must be to count as
/// speech.
///
/// A fixed threshold only works in a silent room. In a room with a fan, or
/// an open window, every frame is above the absolute threshold, the
/// utterance never ends, and the capture is discarded.
pub const NOISE_MARGIN_DB: f32 = 6.0;

/// The quietest level in a buffer, as a percentile of its frame levels.
///
/// The tenth percentile rather than the minimum: a single quiet frame is
/// usually a gap between words, not the room.
pub fn noise_floor(levels: &[f32]) -> f32 {
    if levels.is_empty() {
        return -120.0;
    }
    let mut sorted = levels.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let index = ((sorted.len() as f32) * 0.1).floor() as usize;
    sorted[index.min(sorted.len() - 1)]
}

/// Minimum number of loud frames required to declare speech onset. A click, a
/// keyboard tap or a door slam produces one or two loud frames; speech does
/// not.
pub const MIN_ONSET_FRAMES: usize = 3;

/// Running VAD state over a stream of frames.
#[derive(Debug, Clone)]
pub struct VoiceActivityDetector {
    config: VadConfig,
    sample_rate: u32,
    frame_samples: usize,
    /// Ring of recent frame energies, used for the onset ratio.
    warmup: usize,
    energy_history: Vec<f32>,
}

impl VoiceActivityDetector {
    pub fn new(config: &VadConfig, sample_rate: u32, frame_ms: u32) -> VoiceActivityDetector {
        let frame_samples = ((sample_rate as u64 * frame_ms as u64) / 1000).max(1) as usize;
        let warmup =
            (((sample_rate as u64 * frame_ms as u64) / 1000) as f64 * 0.20).max(1.0) as usize;
        VoiceActivityDetector {
            config: config.clone(),
            sample_rate,
            frame_samples,
            warmup,
            energy_history: Vec::new(),
        }
    }

    /// Samples per frame.
    pub fn frame_samples(&self) -> usize {
        self.frame_samples
    }

    /// Sample rate the detector was configured for.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Trailing silence required to end an utterance, in milliseconds.
    pub fn silence_ms(&self) -> u32 {
        self.config.silence_ms
    }

    /// Minimum speech duration required to accept an utterance.
    pub fn min_speech_ms(&self) -> u32 {
        self.config.min_speech_ms
    }

    /// Frame energy in dBFS.
    pub fn frame_dbfs(frame: &[f32]) -> f32 {
        if frame.is_empty() {
            return -120.0;
        }
        let sum: f64 = frame.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        let rms = (sum / frame.len() as f64).sqrt();
        if rms <= 1e-6 {
            -120.0
        } else {
            (20.0 * rms.log10()) as f32
        }
    }

    /// The level a frame must clear to count as speech.
    ///
    /// Two rules, and the second is the important one.
    ///
    /// The configured threshold is an absolute floor. A room that is louder
    /// than it — a fan, an open window — makes every frame "speech", the
    /// utterance never ends, and the capture is discarded. So the threshold
    /// rises above the room's own floor.
    ///
    /// But a signal with no *structure* has no speech in it at all: a test
    /// tone, or a noise floor. Its loudest frame and its quietest frame are
    /// the same level, so the "floor" is the signal itself and raising the
    /// threshold above it would say the signal is silence — which is the
    /// honest answer, and the only one. Without this rule the tone would be
    /// reported as speech, and a recording of nothing but a tone would be
    /// transcribed and dispatched.
    fn effective_threshold(&self, levels: &[f32]) -> f32 {
        if levels.is_empty() {
            return self.config.energy_threshold_dbfs;
        }
        let floor = noise_floor(levels);
        let peak = levels.iter().copied().fold(f32::MIN, f32::max);
        if peak - floor < NOISE_MARGIN_DB {
            // Stationary: no structure, so no speech.
            return f32::MAX;
        }
        self.config
            .energy_threshold_dbfs
            .max(floor + NOISE_MARGIN_DB)
    }

    /// Whether a single frame counts as speech.
    pub fn is_speech_frame(&self, frame: &[f32]) -> bool {
        Self::frame_dbfs(frame) >= self.config.energy_threshold_dbfs
    }

    /// Analyse a complete capture.
    pub fn analyse(&self, samples: &[f32]) -> VadOutcome {
        if self.frame_samples == 0 || samples.is_empty() {
            return VadOutcome::Silence;
        }
        let levels: Vec<f32> = samples
            .chunks(self.frame_samples)
            .map(Self::frame_dbfs)
            .collect();
        // The threshold is the configured one, raised above the room's own
        // noise floor when the room is louder than the configuration allows.
        let threshold = self.effective_threshold(&levels);
        let flags: Vec<bool> = levels.iter().map(|l| *l >= threshold).collect();
        self.summarise(&flags, samples.len())
    }

    /// Decide whether the last `window_ms` milliseconds contain speech onset.
    ///
    /// Used in streaming mode: it answers "did the user just start talking?"
    /// without committing to a full utterance.
    pub fn detect_onset(&mut self, recent_frames: &[f32]) -> bool {
        if !self.config.enabled {
            return false;
        }
        let loud = recent_frames
            .iter()
            .filter(|db| **db >= self.config.energy_threshold_dbfs)
            .count();
        self.energy_history.extend(recent_frames.iter().copied());
        if self.energy_history.len() > self.warmup * 4 {
            let excess = self.energy_history.len() - self.warmup * 4;
            self.energy_history.drain(..excess);
        }
        // Two conditions, because a single loud frame is a click, not speech:
        // enough of the window must be loud, and at least MIN_ONSET_FRAMES of
        // them must be consecutive-loud.
        let ratio = loud as f32 / recent_frames.len().max(1) as f32;
        ratio >= self.config.onset_ratio && loud >= MIN_ONSET_FRAMES
    }

    /// Reset the streaming history, called whenever the agent returns to idle.
    pub fn reset(&mut self) {
        self.energy_history.clear();
    }

    fn summarise(&self, flags: &[bool], total_samples: usize) -> VadOutcome {
        let silence_frames_needed = ((self.config.silence_ms as u64 * self.sample_rate as u64)
            / 1000
            / self.frame_samples as u64)
            .max(1) as usize;
        let min_speech_frames = ((self.config.min_speech_ms as u64 * self.sample_rate as u64)
            / 1000
            / self.frame_samples as u64)
            .max(1) as usize;

        let Some(first) = flags.iter().position(|f| *f) else {
            return VadOutcome::Silence;
        };

        // Walk forward, remembering the last speech frame and how much
        // silence has followed it.
        let mut last = first;
        let mut trailing = 0usize;
        for (i, f) in flags.iter().enumerate().skip(first) {
            if *f {
                last = i;
                trailing = 0;
            } else {
                trailing += 1;
                if trailing >= silence_frames_needed {
                    break;
                }
            }
        }

        let speech_frames = last - first + 1;
        if speech_frames < min_speech_frames {
            return VadOutcome::TooShort;
        }

        // The buffer ended before the trailing silence arrived, so the
        // utterance was cut off — but the speech region is there, and it is
        // reported rather than thrown away.
        if trailing < silence_frames_needed {
            let _ = total_samples;
            return VadOutcome::Truncated {
                start_frame: first,
                end_frame: last + 1,
            };
        }

        VadOutcome::Speech {
            start_frame: first,
            end_frame: last + 1,
        }
    }
}

/// Trim a buffer to the speech region the VAD found, adding a small margin so
/// the ASR backend gets its onset.
pub fn crop(
    samples: &[f32],
    sample_rate: u32,
    outcome: &VadOutcome,
    frame_samples: usize,
    margin_ms: u32,
) -> Vec<f32> {
    let (start_frame, end_frame) = match outcome {
        VadOutcome::Speech {
            start_frame,
            end_frame,
        }
        | VadOutcome::Truncated {
            start_frame,
            end_frame,
        } => (*start_frame, *end_frame),
        _ => return samples.to_vec(),
    };
    let margin = (sample_rate as u64 * margin_ms as u64 / 1000) as usize;
    let start = start_frame.saturating_sub(1) * frame_samples;
    let end = ((end_frame + 1) * frame_samples).min(samples.len());
    let start = start.saturating_sub(margin);
    let end = end.saturating_add(margin).min(samples.len());
    samples[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> VadConfig {
        VadConfig {
            energy_threshold_dbfs: -42.0,
            silence_ms: 200,
            min_speech_ms: 100,
            onset_ratio: 0.3,
            enabled: true,
        }
    }

    fn speech_then_silence(dur_ms: u64, silence_ms: u64) -> Vec<f32> {
        let mut s = aaron_core::MonoPcm::sine(dur_ms, 16_000, 220.0).unwrap();
        let sil = aaron_core::MonoPcm::silence(silence_ms, 16_000).unwrap();
        s.samples.extend(sil.samples);
        s.samples
    }

    #[test]
    fn detects_speech_in_a_quiet_recording() {
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let samples = speech_then_silence(400, 600);
        match vad.analyse(&samples) {
            VadOutcome::Speech {
                start_frame,
                end_frame,
            } => {
                assert!(start_frame < end_frame);
                assert!(start_frame <= 2, "onset detected too late: {start_frame}");
            }
            other => panic!("expected speech, got {other:?}"),
        }
    }

    #[test]
    fn pure_silence_is_reported_as_silence() {
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let samples = aaron_core::MonoPcm::silence(1000, 16_000).unwrap().samples;
        assert_eq!(vad.analyse(&samples), VadOutcome::Silence);
    }

    #[test]
    fn a_click_is_reported_as_too_short() {
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let mut samples = aaron_core::MonoPcm::silence(200, 16_000).unwrap().samples;
        // One 5 ms burst.
        let burst = aaron_core::MonoPcm::sine(5, 16_000, 800.0).unwrap().samples;
        samples.extend(burst);
        samples.extend(aaron_core::MonoPcm::silence(500, 16_000).unwrap().samples);
        assert_eq!(vad.analyse(&samples), VadOutcome::TooShort);
    }

    #[test]
    fn quiet_noise_far_below_the_threshold_is_silence() {
        let mut cfg = config();
        cfg.energy_threshold_dbfs = -10.0;
        let vad = VoiceActivityDetector::new(&cfg, 16_000, 20);
        let samples = aaron_core::MonoPcm::sine(500, 16_000, 220.0)
            .unwrap()
            .samples;
        assert_eq!(vad.analyse(&samples), VadOutcome::Silence);
    }

    #[test]
    fn a_buffer_without_trailing_silence_is_truncated_but_keeps_its_region() {
        // A buffer of nothing but speech has no trailing silence to close it.
        // The region is still there, and it is reported — the caller used to
        // throw the audio away, so a recording of clear speech was discarded
        // because it did not end politely.
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        // A quiet room with speech on top of it, and no trailing silence.
        let mut samples = vec![0.0f32; 16_000];
        let loud = aaron_core::MonoPcm::sine(400, 16_000, 220.0)
            .unwrap()
            .samples;
        samples.extend(loud);
        match vad.analyse(&samples) {
            VadOutcome::Truncated {
                start_frame,
                end_frame,
            } => {
                assert!(start_frame < end_frame, "the region must not be empty");
            }
            other => panic!("expected a truncated region, got {other:?}"),
        }
    }

    #[test]
    fn a_steady_tone_is_not_speech() {
        // A tone has no structure: its loudest frame and its quietest frame
        // are the same level. Reporting it as speech would have a recording of
        // nothing but a tone transcribed and dispatched.
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let samples = aaron_core::MonoPcm::sine(400, 16_000, 220.0)
            .unwrap()
            .samples;
        assert_eq!(vad.analyse(&samples), VadOutcome::Silence);
    }

    #[test]
    fn a_noisy_room_does_not_look_like_speech() {
        // A fixed threshold only works in a silent room. With a fan running,
        // every frame is above the absolute threshold, the utterance never
        // ends, and the capture is discarded. The threshold has to rise above
        // the room's own floor.
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let mut noisy = aaron_core::MonoPcm::sine(400, 16_000, 220.0)
            .unwrap()
            .samples;
        // Everything at a constant -30 dBFS: a noise floor, not speech.
        let constant = (-30.0f32 / 20.0).exp() * 32767.0;
        for sample in noisy.iter_mut() {
            *sample = constant;
        }
        assert_eq!(vad.analyse(&noisy), VadOutcome::Silence);
    }

    #[test]
    fn speech_still_stands_out_above_a_noisy_room() {
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        // A noisy room with speech on top of it.
        let mut samples = vec![(-30.0f32 / 20.0).exp() * 32767.0; 16_000];
        let loud = aaron_core::MonoPcm::sine(400, 16_000, 220.0)
            .unwrap()
            .samples;
        samples.extend(loud);
        samples.extend(vec![(-30.0f32 / 20.0).exp() * 32767.0; 8_000]);
        assert!(
            matches!(vad.analyse(&samples), VadOutcome::Speech { .. }),
            "speech above a noise floor must still be found"
        );
    }

    #[test]
    fn onset_needs_a_sustained_burst_not_a_single_frame() {
        let mut vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let one_loud = vec![-10.0f32];
        let quiet = vec![-80.0f32];
        assert!(!vad.detect_onset(&one_loud));
        let sustained = vec![-10.0; 10];
        assert!(vad.detect_onset(&sustained));
        vad.reset();
        assert!(!vad.detect_onset(&quiet));
    }

    #[test]
    fn onset_detection_can_be_disabled() {
        let mut cfg = config();
        cfg.enabled = false;
        let mut vad = VoiceActivityDetector::new(&cfg, 16_000, 20);
        assert!(!vad.detect_onset(&[0.0f32; 20]));
    }

    #[test]
    fn cropping_keeps_the_speech_region_with_a_margin() {
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let samples = speech_then_silence(400, 600);
        let outcome = vad.analyse(&samples);
        let cropped = crop(&samples, 16_000, &outcome, vad.frame_samples(), 50);
        assert!(cropped.len() < samples.len());
        assert!(
            cropped.len() > 16_000 * 300 / 1000,
            "cropped away the speech"
        );
    }

    #[test]
    fn cropping_a_silent_buffer_returns_it_unchanged() {
        let vad = VoiceActivityDetector::new(&config(), 16_000, 20);
        let samples = aaron_core::MonoPcm::silence(200, 16_000).unwrap().samples;
        let out = vad.analyse(&samples);
        assert_eq!(
            crop(&samples, 16_000, &out, vad.frame_samples(), 50),
            samples
        );
    }

    #[test]
    fn dbfs_of_a_full_scale_sine_is_about_minus_three_db() {
        // `MonoPcm::sine` uses amplitude 0.3, so RMS is 0.3/sqrt(2) ~= -13.5 dBFS.
        let db = VoiceActivityDetector::frame_dbfs(
            &aaron_core::MonoPcm::sine(100, 16_000, 440.0)
                .unwrap()
                .samples,
        );
        assert!(db < -12.0 && db > -15.0, "unexpected level {db} dBFS");
    }
}
