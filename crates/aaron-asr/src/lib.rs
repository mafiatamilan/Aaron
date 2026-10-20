//! Pluggable offline speech recognition.
//!
//! Every backend in this crate runs on the local machine. There is no online
//! ASR, no API key, and no code path that opens a socket: AARON's offline
//! guarantee is enforced by the *absence* of a network backend, not by a flag
//! that might be set wrongly.

pub mod constrained;
pub mod phonon;
pub mod whisper;

use std::sync::Arc;

use aaron_core::config::{AsrBackend, AsrConfig};
use aaron_core::error::Result;
use aaron_core::SpeechRecognizer;

pub use constrained::{ConstrainedAsr, DisabledAsr};
pub use phonon::PhononAsr;
pub use whisper::WhisperAsr;

/// Build the backend the configuration asks for.
///
/// `auto` prefers a backend that performs **real** recognition: the Phonon
/// engine first, then whisper.cpp, and only then the fixed-phrase test
/// double. The test double is not speech recognition — it emits one of seven
/// canned commands — so `aaron listen` refuses to dispatch what it hears
/// unless the user asks for the demonstration explicitly.
pub fn build(config: &AsrConfig) -> Result<Arc<dyn SpeechRecognizer>> {
    match config.backend {
        AsrBackend::Disabled => Ok(Arc::new(DisabledAsr)),
        AsrBackend::Constrained => Ok(Arc::new(ConstrainedAsr)),
        AsrBackend::WhisperCpp => {
            let asr = WhisperAsr::new(config)?;
            if !asr.available() {
                tracing::info!("whisper.cpp is not installed; reporting it rather than pretending");
            }
            Ok(Arc::new(asr))
        }
        AsrBackend::Phonon => {
            let asr = PhononAsr::new(config)?;
            if !asr.available() {
                tracing::info!(
                    "the Phonon engine is not installed; reporting it rather than pretending"
                );
            }
            Ok(Arc::new(asr))
        }
        AsrBackend::Auto => {
            let phonon = PhononAsr::new(config)?;
            if phonon.available() {
                return Ok(Arc::new(phonon));
            }
            match WhisperAsr::new(config) {
                Ok(asr) if asr.available() => Ok(Arc::new(asr)),
                Ok(asr) => {
                    tracing::info!(detail = %asr.info().detail, "falling back to the constrained recogniser");
                    Ok(Arc::new(ConstrainedAsr))
                }
                Err(e) => {
                    tracing::info!(error = %e, "no real recogniser is installed; using the constrained one");
                    Ok(Arc::new(ConstrainedAsr))
                }
            }
        }
    }
}

/// Sample rate the ASR backends expect.
pub const EXPECTED_SAMPLE_RATE: u32 = 16_000;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_backend_is_selected_directly() {
        let cfg = AsrConfig {
            backend: AsrBackend::Disabled,
            ..Default::default()
        };
        assert_eq!(build(&cfg).unwrap().backend(), "disabled");
    }

    #[test]
    fn constrained_backend_is_selected_directly() {
        let cfg = AsrConfig {
            backend: AsrBackend::Constrained,
            ..Default::default()
        };
        assert_eq!(build(&cfg).unwrap().backend(), "constrained-grammar");
    }

    #[test]
    fn auto_falls_back_when_no_model_is_installed() {
        let cfg = AsrConfig {
            backend: AsrBackend::Auto,
            model_path: String::new(),
            ..Default::default()
        };
        // A fresh install has no model; AARON must still be able to run.
        assert_eq!(build(&cfg).unwrap().backend(), "constrained-grammar");
    }

    #[test]
    fn whisper_backend_selection_does_not_panic_without_a_binary() {
        let cfg = AsrConfig {
            backend: AsrBackend::WhisperCpp,
            model_path: "/no/such/model.bin".into(),
            ..Default::default()
        };
        let asr = build(&cfg).unwrap();
        assert_eq!(asr.backend(), "whisper.cpp");
        assert!(!asr.info().ready);
    }

    #[test]
    fn every_backend_is_offline_only() {
        // The offline guarantee is structural: AARON's ASR crate must not be
        // able to reach the network at all. Assert it on the dependency list,
        // which is the only thing that could give it that ability.
        let manifest = include_str!("../Cargo.toml");
        let needle = |a: &str, b: &str| format!("{a}{b}");
        for forbidden in [
            needle("req", "west"),
            needle("hy", "per"),
            needle("ur", "eq"),
            needle("tok", "io"),
        ] {
            assert!(
                !manifest.contains(&forbidden),
                "aaron-asr must not depend on `{forbidden}`: ASR is offline only"
            );
        }
        // A second, independent check: the compiled crate must not pull in a
        // socket type. `std::net` is reachable without any dependency, so this
        // is asserted against the source with the needle built at runtime.
        let src = [
            include_str!("lib.rs"),
            include_str!("whisper.rs"),
            include_str!("constrained.rs"),
        ]
        .concat();
        let needle = format!("std::net::{0}", "Tcp");
        assert!(
            !src.contains(&needle),
            "aaron-asr must not open sockets: ASR is offline only"
        );
    }
}
