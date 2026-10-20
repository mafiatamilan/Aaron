//! The constrained recogniser and the disabled backend.
//!
//! # The constrained recogniser is not speech recognition
//!
//! It exists so that `aaron listen` can be exercised end to end without any
//! model installed: given the same synthetic audio, it deterministically
//! produces one utterance from a fixed list. It performs no acoustic analysis
//! whatsoever. It is documented as such in `aaron doctor`, it is never the
//! default when a real backend is available, and AARON does not claim it can
//! hear anything.
//!
//! Its one legitimate use is as a **test double** for the audio pipeline, and
//! as an explicit opt-in for users who want a zero-dependency demonstration.

use aaron_core::error::{Error, Result};
use aaron_core::{AsrInfo, MonoPcm, SpeechRecognizer, Transcript};

/// Utterances the constrained recogniser can emit.
pub const PHRASES: &[&str] = &[
    "hey aaron check system status",
    "hey aaron what can you do",
    "hey aaron remember that i drink tea",
    "hey aaron what do you remember about coffee",
    "hey aaron run the tests",
    "hey aaron show git status",
    "hey aaron cancel",
];

/// Deterministic, model-free recogniser.
pub struct ConstrainedAsr;

impl ConstrainedAsr {
    /// Which phrase the recogniser emits for a buffer.
    ///
    /// The choice is a pure function of the buffer's duration and energy, so a
    /// test can predict it, and the same audio always yields the same text.
    pub fn choose(audio: &MonoPcm) -> String {
        if audio.is_empty() {
            return String::new();
        }
        let duration_bucket = (audio.duration_ms() / 250) as usize;
        let energy_bucket = (audio.peak() * 10.0).round().max(0.0) as usize;
        let idx = (duration_bucket + energy_bucket) % PHRASES.len();
        PHRASES[idx].to_string()
    }
}

impl SpeechRecognizer for ConstrainedAsr {
    fn backend(&self) -> String {
        "constrained-grammar".into()
    }

    fn transcribe(&self, audio: &MonoPcm) -> Result<Transcript> {
        let started = std::time::Instant::now();
        if audio.is_empty() {
            return Ok(Transcript::empty("constrained-grammar"));
        }
        Ok(Transcript {
            text: Self::choose(audio),
            backend: "constrained-grammar".into(),
            latency_ms: started.elapsed().as_micros() as u64,
        })
    }

    fn info(&self) -> AsrInfo {
        AsrInfo {
            backend: "constrained-grammar".into(),
            // Not ready, and not production. It was reported as ready, which
            // meant `aaron doctor` did not list it as a problem and
            // `aaron listen` did not warn: any loud sound could become one
            // of seven canned commands and be dispatched.
            ready: false,
            model_path: None,
            detail: "NOT speech recognition: emits one of a fixed list of phrases \
                     deterministically, for testing and demonstration only"
                .into(),
            production: false,
        }
    }
}

/// A backend that is explicitly switched off.
pub struct DisabledAsr;

impl SpeechRecognizer for DisabledAsr {
    fn backend(&self) -> String {
        "disabled".into()
    }

    fn transcribe(&self, _audio: &MonoPcm) -> Result<Transcript> {
        Err(Error::Asr(
            "speech recognition is disabled. Run `aaron listen --text-only`, or install \
             whisper.cpp and point asr.model_path at a ggml model."
                .into(),
        ))
    }

    fn info(&self) -> AsrInfo {
        AsrInfo {
            backend: "disabled".into(),
            ready: false,
            model_path: None,
            detail: "disabled by configuration (asr.backend = \"disabled\")".into(),
            production: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_is_deterministic_for_the_same_audio() {
        let audio = MonoPcm::sine(500, 16_000, 220.0).unwrap();
        let a = ConstrainedAsr::choose(&audio);
        let b = ConstrainedAsr::choose(&audio);
        assert_eq!(a, b);
        assert!(PHRASES.contains(&a.as_str()));
    }

    #[test]
    fn empty_audio_produces_no_text() {
        let empty = MonoPcm::new(Vec::new(), 16_000).unwrap();
        assert_eq!(ConstrainedAsr::choose(&empty), "");
        let t = ConstrainedAsr.transcribe(&empty).unwrap();
        assert!(t.text.is_empty());
    }

    #[test]
    fn the_backend_advertises_that_it_is_not_asr() {
        let info = ConstrainedAsr.info();
        assert!(info.detail.contains("NOT speech recognition"));
    }

    #[test]
    fn disabled_backend_fails_with_instructions() {
        let err = DisabledAsr
            .transcribe(&MonoPcm::silence(100, 16_000).unwrap())
            .unwrap_err();
        assert!(matches!(err, Error::Asr(_)));
        assert!(err.to_string().contains("--text-only"));
        assert!(!DisabledAsr.info().ready);
    }
}
