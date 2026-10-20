//! Shared foundations for the AARON assistant.
//!
//! This crate deliberately contains no neural network, audio, database or
//! process-management code. It defines the vocabulary every other crate agrees
//! on: the intent schema, the configuration surface, the error taxonomy, the
//! pluggable traits, and the deterministic text pipeline that the brain and the
//! heuristic router both depend on.

pub mod config;
pub mod error;
pub mod hardware;
pub mod intent;
pub mod logging;
pub mod paths;
pub mod pcm;
pub mod readiness;
pub mod text;
pub mod traits;

pub use config::Config;
pub use error::{Error, ErrorKind, Result};
pub use hardware::HardwareReport;
pub use intent::{
    Calibration, ClassifySource, Command, Entities, Intent, IntentClassifier, Readiness,
    INTENT_COUNT,
};
pub use pcm::MonoPcm;
pub use traits::{
    AsrInfo, AudioSource, InputSource, SharedSource, SpeechRecognizer, SpeechSynthesizer,
    Transcript, TtsInfo, WakeDetector,
};

/// Crate version, surfaced by `aaron --version --verbose`.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
