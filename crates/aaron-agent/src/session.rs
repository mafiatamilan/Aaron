//! The voice session: capture, wake, transcribe, dispatch.
//!
//! This is the piece that wires [`aaron_core::AudioSource`],
//! [`aaron_core::WakeDetector`], [`aaron_core::SpeechRecognizer`] and [`Agent`]
//! together, and it is written so that **every** part is injectable. The CLI
//! passes real devices; the tests pass a synthetic source and a stub
//! recogniser, so the entire loop is covered without a microphone and without a
//! model.

use std::sync::Arc;
use std::time::{Duration, Instant};

use aaron_audio::{record_utterance, CaptureSpec};
use aaron_core::config::Config;
use aaron_core::error::{Error, Result};
use aaron_core::{AudioSource, SharedSource, SpeechRecognizer, WakeDetector};

use crate::agent::Agent;

/// What one listening session produced.
#[derive(Debug, Clone, PartialEq)]
pub enum SessionOutcome {
    /// A wake phrase was heard and a command was handled.
    Command {
        transcript: String,
        reply: String,
        elapsed_ms: u64,
    },
    /// Speech was heard but it was not the wake phrase.
    NotWoken,
    /// Nothing was heard in this window.
    Silence,
    /// The capture hit its duration cap without usable speech.
    TimedOut,
    /// The caller asked to stop.
    Stopped,
}

/// Drives the voice loop.
pub struct VoiceSession<'a> {
    agent: &'a mut Agent,
    /// Shared rather than borrowed, so the microphone can be **released**
    /// between the command and its spoken confirmation.
    ///
    /// A borrowed device would stay locked for the whole turn, and the
    /// confirmer — which has to capture the user's answer to a question it
    /// asked out loud — would deadlock against the session that asked it.
    ///
    /// One capture device stays open for the whole listening session either
    /// way. A reopened device is both slower and a source of transient
    /// errors.
    source: SharedSource,
    asr: Arc<dyn SpeechRecognizer>,
    wake: Arc<dyn WakeDetector>,
    config: Config,
    /// Text of the wake phrase, used by the transcript-based detector.
    wake_phrase: String,
}

impl<'a> VoiceSession<'a> {
    /// Build a session. `wake_phrase` is only used for the transcript strategy.
    pub fn new(
        agent: &'a mut Agent,
        source: SharedSource,
        asr: Arc<dyn SpeechRecognizer>,
        wake: Arc<dyn WakeDetector>,
        config: Config,
        wake_phrase: String,
    ) -> Result<VoiceSession<'a>> {
        wake.ready()?;
        Ok(VoiceSession {
            agent,
            source,
            asr,
            wake,
            config,
            wake_phrase,
        })
    }

    /// Listen for one utterance and dispatch it.
    ///
    /// `wake_check` decides whether a transcript should be tested against the
    /// wake phrase. The default closure strips a leading "hey aaron" and
    /// requires one; passing `|_| false` dispatches whatever was heard, which
    /// is what `aaron listen --no-wake` does.
    pub fn listen_once<F>(&mut self, wake_check: F) -> Result<SessionOutcome>
    where
        F: Fn(&str, &str) -> bool,
    {
        let started = Instant::now();
        let spec = self.capture_spec();
        // The microphone is held for the capture and released before the
        // command is dispatched, so the confirmer can take it for the answer.
        let recorded = {
            let mut guard = self.lock_source()?;
            record_utterance(&mut **guard, &spec)?
        };

        if !recorded.speech_detected {
            return Ok(if recorded.hit_cap {
                SessionOutcome::TimedOut
            } else {
                SessionOutcome::Silence
            });
        }
        if recorded.pcm.is_empty() {
            return Ok(SessionOutcome::Silence);
        }

        let transcript = self.asr.transcribe(&recorded.pcm)?;
        if std::env::var("AARON_DEBUG_ASR").is_ok() {
            eprintln!("transcript: {:?}", transcript.text);
        }
        if transcript.text.trim().is_empty() {
            return Ok(SessionOutcome::Silence);
        }

        // The wake check inspects a *string*, and never executes anything.
        let command_text = if wake_check(&transcript.text, &self.wake_phrase) {
            strip_wake_phrase(&transcript.text, &self.wake_phrase)
        } else {
            return Ok(SessionOutcome::NotWoken);
        };

        if command_text.trim().is_empty() {
            // The user only said the wake phrase: acknowledge and stay idle
            // rather than guessing a command.
            return Ok(SessionOutcome::NotWoken);
        }

        let response = self.agent.handle_text(&command_text)?;
        self.wake.reset();
        Ok(SessionOutcome::Command {
            transcript: command_text,
            reply: response.reply,
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    /// How to capture one utterance.
    ///
    /// The capture runs in the *source's* sample rate and is resampled once,
    /// at the end, to the configured rate. A microphone that offers only
    /// 48 kHz is normal, and computing the detector's frames and the
    /// duration cap against the configured 16 kHz makes every duration in the
    /// pipeline wrong by the ratio between the two.
    fn capture_spec(&self) -> CaptureSpec {
        CaptureSpec {
            vad: self.config.audio.vad.clone(),
            frame_ms: self.config.audio.frame_ms,
            target_rate: self.config.audio.sample_rate,
            max_secs: self.config.audio.max_capture_secs,
        }
    }

    /// Whether a microphone is genuinely open behind this session.
    pub fn is_live(&self) -> bool {
        match self.source.lock() {
            Ok(guard) => guard.is_live(),
            Err(_) => false,
        }
    }

    /// Description of the active source, for `aaron listen`.
    pub fn describe_source(&self) -> String {
        match self.source.lock() {
            Ok(guard) => guard.describe(),
            Err(_) => "audio source (unavailable)".into(),
        }
    }

    /// Take the microphone for the length of one capture.
    ///
    /// The lock is poisoned only if a capture panicked while holding it, and
    /// a poisoned lock is reported rather than silently ignored: a source
    /// that panicked is not a source that should keep running.
    fn lock_source(&self) -> Result<std::sync::MutexGuard<'_, Box<dyn AudioSource>>> {
        self.source
            .lock()
            .map_err(|_| Error::Audio("the audio source lock was poisoned by a panic".into()))
    }

    /// Total time the caller may spend listening.
    pub fn max_duration(&self) -> Duration {
        Duration::from_secs(self.config.audio.max_capture_secs as u64)
    }
}

/// Remove a leading wake phrase from a transcript.
pub fn strip_wake_phrase(transcript: &str, phrase: &str) -> String {
    let t = aaron_core::text::normalize_entities(transcript);
    let p = aaron_core::text::normalize_entities(phrase);
    if p.is_empty() {
        return transcript.to_string();
    }
    if let Some(rest) = t.strip_prefix(&p) {
        return rest.trim().to_string();
    }
    // The recogniser may have inserted a filler word before the phrase. Only
    // strip when the phrase is actually present, so an utterance that simply
    // does not contain it is returned unchanged.
    let phrase_tokens: Vec<&str> = p.split(' ').collect();
    match t.find(&p) {
        Some(idx) => t[idx + p.len()..].trim().to_string(),
        None => {
            let Some(first) = phrase_tokens.first() else {
                return t;
            };
            match t.find(first) {
                Some(idx) => t[idx..].trim().to_string(),
                None => t,
            }
        }
    }
}

/// Default wake check: the transcript must contain the configured phrase.
pub fn default_wake_check(transcript: &str, phrase: &str) -> bool {
    aaron_wake::phrase_similarity(phrase, transcript) >= 0.72
}

/// Build the capture settings for the configured sample rate.
///
/// The detector itself is built inside the capture, from the *source's* rate,
/// so that a device that does not offer the configured rate cannot make the
/// durations wrong.
pub fn build_capture_spec(config: &Config) -> CaptureSpec {
    CaptureSpec {
        vad: config.audio.vad.clone(),
        frame_ms: config.audio.frame_ms,
        target_rate: config.audio.sample_rate,
        max_secs: config.audio.max_capture_secs,
    }
}

/// Reject a configuration whose sample rate the pipeline cannot use.
pub fn validate_audio(config: &Config) -> Result<()> {
    if config.audio.sample_rate < 8_000 {
        return Err(Error::Config(
            "audio.sample_rate must be at least 8000".into(),
        ));
    }
    if !matches!(
        config.audio.backend,
        aaron_core::config::AudioBackend::Disabled
    ) && config.audio.max_capture_secs == 0
    {
        return Err(Error::Config("audio.max_capture_secs must be > 0".into()));
    }
    Ok(())
}

/// Outcome of a VAD pass, re-exported for convenience.
pub use aaron_audio::VadOutcome as VADOutcome;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{AgentDeps, AutoConfirm};
    use aaron_audio::{FileSource, SyntheticSource};
    use aaron_core::{AsrInfo, MonoPcm, Transcript};
    use aaron_memory::MemoryStore;
    use aaron_security::{AuditLog, PathSandbox, PolicyEngine, PolicyFile};
    use aaron_tools::{ToolContext, ToolRegistry};
    use std::sync::Mutex;

    const ALLOW: &str = r#"
version = 1
[tools.system_status]
effect = "allow"
[tools.help_me]
effect = "allow"
"#;

    struct StubAsr {
        text: String,
    }

    impl SpeechRecognizer for StubAsr {
        fn backend(&self) -> String {
            "stub".into()
        }
        fn transcribe(&self, _audio: &MonoPcm) -> Result<Transcript> {
            Ok(Transcript {
                text: self.text.clone(),
                backend: "stub".into(),
                latency_ms: 1,
            })
        }
        fn info(&self) -> AsrInfo {
            AsrInfo {
                backend: "stub".into(),
                ready: true,
                model_path: None,
                detail: "test double".into(),
                production: false,
            }
        }
    }

    fn build_agent() -> (tempfile::TempDir, Agent) {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = Config::default();
        config.tts.echo_to_terminal = false;
        config.security.always_confirm = Vec::new();
        let memory = Arc::new(Mutex::new(
            MemoryStore::open_in_memory(Default::default()).unwrap(),
        ));
        let ctx = ToolContext::new(
            Arc::new(PathSandbox::deny_all()),
            Arc::clone(&memory),
            Arc::new(config.clone()),
            tmp.path().to_path_buf(),
        );
        let agent = Agent::new(AgentDeps {
            classifier: Arc::new(aaron_brain::HeuristicClassifier::new(0.5)),
            registry: ToolRegistry::builtin(),
            policy: PolicyEngine::new(PolicyFile::from_toml(ALLOW).unwrap(), Vec::new()),
            audit: AuditLog::disabled(),
            context: ctx,
            memory,
            tts: Arc::new(aaron_tts::TextOnlyTts),
            confirmer: Arc::new(AutoConfirm),
            config,
        });
        (tmp, agent)
    }

    /// Wrap a source so the session and a confirmer can take turns with it.
    fn shared(source: impl AudioSource + 'static) -> SharedSource {
        Arc::new(std::sync::Mutex::new(
            Box::new(source) as Box<dyn AudioSource>
        ))
    }

    fn config() -> Config {
        let mut c = Config::default();
        c.tts.echo_to_terminal = false;
        c.audio.max_capture_secs = 4;
        // The synthetic source alternates 300 ms of tone with 200 ms of
        // silence, so the trailing-silence threshold must fit inside the gap.
        c.audio.vad.silence_ms = 150;
        c.audio.vad.min_speech_ms = 60;
        c
    }

    #[test]
    fn a_wake_phrase_plus_command_is_dispatched() {
        let (_tmp, mut agent) = build_agent();
        let source = SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap();
        let asr = Arc::new(StubAsr {
            text: "hey aaron check system status".into(),
        });
        let mut session = VoiceSession::new(
            &mut agent,
            shared(source),
            asr,
            aaron_wake::build(&config().wake),
            config(),
            "hey aaron".into(),
        )
        .unwrap();

        match session.listen_once(default_wake_check).unwrap() {
            SessionOutcome::Command {
                transcript, reply, ..
            } => {
                assert_eq!(transcript, "check system status");
                assert!(reply.contains("CPU"));
            }
            other => panic!("expected a command, got {other:?}"),
        }
    }

    #[test]
    fn speech_without_the_wake_phrase_is_ignored() {
        let (_tmp, mut agent) = build_agent();
        let source = SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap();
        let asr = Arc::new(StubAsr {
            text: "tell me a joke".into(),
        });
        let mut session = VoiceSession::new(
            &mut agent,
            shared(source),
            asr,
            aaron_wake::build(&config().wake),
            config(),
            "hey aaron".into(),
        )
        .unwrap();
        assert_eq!(
            session.listen_once(default_wake_check).unwrap(),
            SessionOutcome::NotWoken
        );
    }

    #[test]
    fn the_wake_phrase_alone_does_not_invent_a_command() {
        let (_tmp, mut agent) = build_agent();
        let source = SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap();
        let asr = Arc::new(StubAsr {
            text: "hey aaron".into(),
        });
        let mut session = VoiceSession::new(
            &mut agent,
            shared(source),
            asr,
            aaron_wake::build(&config().wake),
            config(),
            "hey aaron".into(),
        )
        .unwrap();
        assert_eq!(
            session.listen_once(default_wake_check).unwrap(),
            SessionOutcome::NotWoken
        );
    }

    #[test]
    fn no_wake_mode_dispatches_everything() {
        let (_tmp, mut agent) = build_agent();
        let source = SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap();
        let asr = Arc::new(StubAsr {
            text: "check system status".into(),
        });
        let mut session = VoiceSession::new(
            &mut agent,
            shared(source),
            asr,
            aaron_wake::build(&config().wake),
            config(),
            "hey aaron".into(),
        )
        .unwrap();
        // `--no-wake` passes an always-true check, so everything heard is a command.
        match session.listen_once(|_, _| true).unwrap() {
            SessionOutcome::Command { transcript, .. } => {
                assert_eq!(transcript, "check system status")
            }
            other => panic!("expected a command, got {other:?}"),
        }
    }

    #[test]
    fn silence_reports_silence_not_an_error() {
        let (_tmp, mut agent) = build_agent();
        let silent = MonoPcm::silence(1500, 16_000).unwrap();
        let source = FileSource::from_pcm(silent, 20).unwrap();
        let asr = Arc::new(StubAsr {
            text: String::new(),
        });
        let mut session = VoiceSession::new(
            &mut agent,
            shared(source),
            asr,
            aaron_wake::build(&config().wake),
            config(),
            "hey aaron".into(),
        )
        .unwrap();
        assert_eq!(
            session.listen_once(default_wake_check).unwrap(),
            SessionOutcome::Silence
        );
    }

    #[test]
    fn a_synthetic_source_is_never_reported_as_live() {
        let (_tmp, mut agent) = build_agent();
        let source = SyntheticSource::new(1000, 16_000, 20, 220.0).unwrap();
        let asr = Arc::new(StubAsr {
            text: String::new(),
        });
        let session = VoiceSession::new(
            &mut agent,
            shared(source),
            asr,
            aaron_wake::build(&config().wake),
            config(),
            "hey aaron".into(),
        )
        .unwrap();
        assert!(
            !session.is_live(),
            "a synthetic source must not claim a live microphone"
        );
        assert!(session.describe_source().contains("no microphone"));
    }

    #[test]
    fn stripping_the_wake_phrase_handles_punctuation_and_fillers() {
        assert_eq!(
            strip_wake_phrase("Hey, Aaron! check status", "hey aaron"),
            "check status"
        );
        assert_eq!(strip_wake_phrase("hey aaron", "hey aaron"), "");
        assert_eq!(
            strip_wake_phrase("check status", "hey aaron"),
            "check status"
        );
    }

    #[test]
    fn audio_config_validation_rejects_impossible_rates() {
        let mut cfg = Config::default();
        cfg.audio.sample_rate = 100;
        assert!(validate_audio(&cfg).is_err());
        assert!(validate_audio(&Config::default()).is_ok());
    }
}
