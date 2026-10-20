//! Pluggable interfaces for the voice pipeline.
//!
//! Every optional external engine is reached through one of these traits, so a
//! missing binary becomes a configuration state rather than a hard dependency,
//! and the text-only path never touches any of it.

use std::fmt;

use crate::config::TtsBackend;
use crate::error::Result;
use crate::pcm::MonoPcm;

/// Result of transcribing one utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Recognised text, trimmed, original casing preserved.
    pub text: String,
    /// Backend that produced this text.
    pub backend: String,
    /// Wall-clock time spent inside the recogniser.
    pub latency_ms: u64,
}

impl Transcript {
    /// A transcript with no recognised speech.
    pub fn empty(backend: impl Into<String>) -> Self {
        Transcript {
            text: String::new(),
            backend: backend.into(),
            latency_ms: 0,
        }
    }
}

/// Description of an ASR backend's installation state, for `aaron doctor`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AsrInfo {
    /// Configured backend name.
    pub backend: String,
    /// Whether the backend is fully usable right now.
    pub ready: bool,
    /// Path to the model file, if any.
    pub model_path: Option<String>,
    /// Human readable note about what is missing or substituted.
    pub detail: String,
    /// Whether this backend performs real speech recognition.
    ///
    /// A backend that does not — a test double that emits a fixed list of
    /// phrases — must say so here. It is not merely "not ready": it is
    /// *dangerous*, because a fabricated transcript is an instruction. AARON
    /// must not act on one.
    pub production: bool,
}

/// Offline speech recognition.
pub trait SpeechRecognizer: Send + Sync {
    /// Backend identifier, e.g. `whisper.cpp` or `constrained`.
    fn backend(&self) -> String;

    /// Transcribe a mono 16 kHz buffer.
    fn transcribe(&self, audio: &MonoPcm) -> Result<Transcript>;

    /// Whether the backend can run right now, and why not if it cannot.
    fn info(&self) -> AsrInfo;
}

/// Description of a TTS backend's installation state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TtsInfo {
    pub backend: String,
    pub ready: bool,
    pub detail: String,
}

/// Offline speech synthesis.
pub trait SpeechSynthesizer: Send + Sync {
    /// Backend identifier, e.g. `espeak-ng`.
    fn backend(&self) -> String;

    /// Speak `text`. Implementations must never raise privileges or shell out
    /// through a shell; they exec a fixed binary with an argument vector.
    fn speak(&self, text: &str) -> Result<()>;

    /// Write synthesised audio to `wav_path`. Backends that cannot render to a
    /// file return `None`.
    fn render_wav(&self, text: &str, wav_path: &std::path::Path) -> Result<bool>;

    fn info(&self) -> TtsInfo;
}

/// Audio capture source.
pub trait AudioSource: Send {
    /// Sample rate the source will deliver.
    fn sample_rate(&self) -> u32;

    /// Pull the next block of samples. Returns an empty vector at end of stream.
    fn read_block(&mut self) -> Result<Vec<f32>>;

    /// Whether a real microphone is behind this source. Used by the UI so the
    /// user can never mistake a synthetic source for live capture.
    fn is_live(&self) -> bool {
        true
    }

    /// Name for `aaron doctor`.
    fn describe(&self) -> String {
        "audio source".to_string()
    }
}

/// Blanket delegation so a boxed source can itself be used as a source; the
/// voice loop holds `Box<dyn AudioSource>` and should not need to unwrap it.
impl<T: AudioSource + ?Sized> AudioSource for Box<T> {
    fn sample_rate(&self) -> u32 {
        (**self).sample_rate()
    }

    fn read_block(&mut self) -> crate::error::Result<Vec<f32>> {
        (**self).read_block()
    }

    fn is_live(&self) -> bool {
        (**self).is_live()
    }

    fn describe(&self) -> String {
        (**self).describe()
    }
}

/// A microphone that more than one component can take turns with.
///
/// The voice loop and the spoken-confirmation handler both need the audio
/// source, and only one of them can hold it at a time. The handle is what
/// lets a session capture an utterance, **release** the microphone, and then
/// let the confirmer capture the answer to a question it asked out loud.
///
/// The lock is deliberately held for as short a time as possible: a capture
/// holds it, and nothing else does. A held-across-the-turn lock would make
/// the spoken confirmation deadlock against itself.
pub type SharedSource = std::sync::Arc<std::sync::Mutex<Box<dyn AudioSource>>>;

/// Wake-phrase detector.
pub trait WakeDetector: Send + Sync {
    /// Current detector name, e.g. `transcript-keyword`.
    fn name(&self) -> String;

    /// Whether this detector is usable in the current environment.
    fn ready(&self) -> Result<()>;

    /// Reset internal state (called on every wake and on cancel).
    fn reset(&self);
}

/// How the agent obtained an utterance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputSource {
    /// Typed at the CLI or read from a file.
    Text(String),
    /// Transcribed from microphone audio.
    Voice(Transcript),
}

impl InputSource {
    /// The text to classify.
    pub fn text(&self) -> &str {
        match self {
            InputSource::Text(t) => t,
            InputSource::Voice(v) => &v.text,
        }
    }
}

impl fmt::Display for InputSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InputSource::Text(_) => f.write_str("text"),
            InputSource::Voice(_) => f.write_str("voice"),
        }
    }
}

/// Convenience: the TTS backend enum as it appears in diagnostics.
pub fn tts_backend_name(b: TtsBackend) -> &'static str {
    match b {
        TtsBackend::Auto => "auto",
        TtsBackend::Espeak => "espeak-ng",
        TtsBackend::Piper => "piper",
        TtsBackend::SpdSay => "spd-say",
        TtsBackend::Disabled => "disabled",
    }
}
