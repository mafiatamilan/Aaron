//! Spoken confirmation: asking a question out loud and hearing the answer.
//!
//! The terminal confirmer is the right one when a terminal is attached: it
//! shows the exact command and target, and the user can read them. But a
//! background assistant has no terminal, and the alternative — declining
//! every confirmation-gated action — quietly makes the assistant useless for
//! exactly the actions people use it for.
//!
//! This confirmer asks the question **out loud**, captures one short
//! utterance, and accepts a yes or a no. It is deliberately narrow:
//!
//! * It only ever runs when there is no terminal, and only for an action the
//!   policy engine has already marked as needing confirmation. The policy
//!   engine is still the boundary; this only replaces the *prompt*.
//! * It answers "no" by default. Silence, an unclear word, or anything that
//!   is not recognisably a yes is a refusal.
//! * It never acts on the answer. The answer unblocks an action the policy
//!   engine has already authorised for asking.

use std::sync::Arc;
use std::time::Duration;

use aaron_audio::record_utterance;
use aaron_core::config::Config;
use aaron_core::error::Result;
use aaron_core::{SharedSource, SpeechRecognizer, SpeechSynthesizer};

use crate::agent::Confirmer;

/// Asks a confirmation question out loud and listens for the answer.
pub struct VoiceConfirmer {
    /// The microphone, shared with the voice loop.
    source: SharedSource,
    asr: Arc<dyn SpeechRecognizer>,
    tts: Arc<dyn SpeechSynthesizer>,
    config: Config,
    /// How long to wait for the answer.
    timeout: Duration,
}

impl VoiceConfirmer {
    /// Build a confirmer that speaks the question and listens for the answer.
    ///
    /// `timeout` is the longest it will wait for an answer; the capture
    /// itself is bounded by the same number of seconds, rounded down to at
    /// least one.
    pub fn new(
        source: SharedSource,
        asr: Arc<dyn SpeechRecognizer>,
        tts: Arc<dyn SpeechSynthesizer>,
        config: Config,
        timeout: Duration,
    ) -> Self {
        Self {
            source,
            asr,
            tts,
            config,
            timeout,
        }
    }

    /// The question as it should be *spoken*.
    ///
    /// The policy engine writes a prompt for a terminal: a line of
    /// explanation, then an indented detail line. Indentation and newlines
    /// are meaningless out loud, and the detail is worth hearing, so the
    /// detail is kept and the layout is flattened.
    ///
    /// The tool name is unquoted and the risk level is dropped. "Run
    /// `run_tests` (medium)?" is a line for a screen; read aloud, the
    /// backticks are noise and the risk level means nothing without the
    /// terminal's context to weigh it against. The risk level is still
    /// shown, and still audited, on the terminal path.
    fn spoken_question(prompt: &str) -> String {
        let mut parts: Vec<String> = Vec::new();
        for line in prompt.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let spoken = Self::speakable(line);
            // The detail line is the tool's name and its arguments. When it
            // is *only* the name, it says nothing the prompt has not already
            // said, and hearing it twice is just noise.
            let is_bare_name =
                !spoken.contains('=') && !spoken.contains(char::is_whitespace) && !parts.is_empty();
            if is_bare_name {
                continue;
            }
            parts.push(spoken);
        }
        if parts.is_empty() {
            "Shall I?".to_string()
        } else {
            let mut spoken = String::new();
            for part in parts {
                if spoken.is_empty() {
                    spoken = part;
                } else if spoken.ends_with('?') {
                    spoken.push(' ');
                    spoken.push_str(&part);
                } else {
                    spoken.push_str(". ");
                    spoken.push_str(&part);
                }
            }
            format!("{spoken} Shall I?")
        }
    }

    /// Strip a trailing risk parenthetical from one line.
    fn speakable(line: &str) -> String {
        let text = line.replace('`', "");
        for risk in ["critical", "high", "medium", "low"] {
            // The parenthetical sits at the end of the prompt, sometimes
            // followed by a question mark.
            let stripped = match text.strip_suffix(&format!(" ({risk})?")) {
                Some(rest) => Some(format!("{rest}?")),
                None => text
                    .strip_suffix(&format!(" ({risk})"))
                    .map(|rest| rest.to_string()),
            };
            if let Some(rest) = stripped {
                return rest;
            }
        }
        text
    }

    /// Whether an answer means yes.
    ///
    /// The list is short on purpose. A long list of synonyms is a long list
    /// of ways to be wrong: "no" contains no substring of "yes", but
    /// "yesterday" contains "yes", and a fuzzy matcher that accepts
    /// "yesterday" as consent is worse than one that asks again.
    fn is_yes(answer: &str) -> bool {
        let normalised = aaron_core::text::normalize_entities(answer);
        normalised
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| matches!(word, "yes" | "yeah" | "yep" | "ok" | "okay" | "do" | "go"))
    }
}

impl Confirmer for VoiceConfirmer {
    fn confirm(&self, prompt: &str, _timeout: Duration) -> Result<bool> {
        let question = Self::spoken_question(prompt);

        // Speak the question. The synthesiser plays it synchronously, so by
        // the time it returns the speaker has finished — but the tail of it
        // may still be in the room, and the microphone is open. A short
        // settle lets the room go quiet before the answer is captured, so
        // the assistant does not transcribe its own question.
        if let Err(e) = self.tts.speak(&question) {
            // Without a working speaker there is no way to ask, so there is
            // no way to answer, and the answer is no.
            tracing::warn!(error = %e, "cannot speak the confirmation question; declining");
            return Ok(false);
        }
        std::thread::sleep(Duration::from_millis(350));

        let spec = aaron_audio::CaptureSpec {
            vad: self.config.audio.vad.clone(),
            frame_ms: self.config.audio.frame_ms,
            target_rate: self.config.audio.sample_rate,
            // The answer is short. A cap of two seconds past the configured
            // limit keeps a silent room from holding the microphone open for
            // the whole session.
            max_secs: self.timeout.as_secs().clamp(1, 10) as u32,
        };

        let recorded = {
            let mut guard = match self.source.lock() {
                Ok(g) => g,
                Err(_) => {
                    tracing::warn!("the audio source lock was poisoned; declining");
                    return Ok(false);
                }
            };
            record_utterance(&mut **guard, &spec)?
        };

        if !recorded.speech_detected || recorded.pcm.is_empty() {
            tracing::info!("no spoken answer to the confirmation question; declining");
            return Ok(false);
        }

        let transcript = self.asr.transcribe(&recorded.pcm)?;
        let answer = transcript.text.trim();
        if answer.is_empty() {
            return Ok(false);
        }
        if std::env::var("AARON_DEBUG_ASR").is_ok() {
            eprintln!("confirmation answer: {answer:?}");
        }
        let approved = Self::is_yes(answer);
        tracing::info!(answer = %answer, approved, "spoken confirmation answered");
        Ok(approved)
    }

    /// The user is answering by voice, so the session is interactive.
    ///
    /// This matters: the agent refuses a confirmation-gated action outright
    /// when the confirmer says it is not interactive, so claiming otherwise
    /// without asking would run the action unconfirmed.
    fn is_interactive(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aaron_audio::{FileSource, SyntheticSource};
    use aaron_core::{AsrInfo, MonoPcm, Transcript};

    /// A recogniser that always says the same thing.
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
                production: true,
            }
        }
    }

    /// A synthesiser that records what it was asked to say.
    struct StubTts {
        spoken: std::sync::Mutex<Vec<String>>,
    }

    impl SpeechSynthesizer for StubTts {
        fn backend(&self) -> String {
            "stub".into()
        }
        fn speak(&self, text: &str) -> Result<()> {
            self.spoken.lock().unwrap().push(text.to_string());
            Ok(())
        }
        fn render_wav(&self, _text: &str, _path: &std::path::Path) -> Result<bool> {
            Ok(false)
        }
        fn info(&self) -> aaron_core::TtsInfo {
            aaron_core::TtsInfo {
                backend: "stub".into(),
                ready: true,
                detail: "test double".into(),
            }
        }
    }

    fn config() -> Config {
        let mut c = Config::default();
        c.audio.max_capture_secs = 4;
        c.audio.vad.silence_ms = 150;
        c.audio.vad.min_speech_ms = 60;
        c
    }

    fn shared(source: impl aaron_core::AudioSource + 'static) -> SharedSource {
        Arc::new(std::sync::Mutex::new(
            Box::new(source) as Box<dyn aaron_core::AudioSource>
        ))
    }

    fn confirmer(source: SharedSource, answer: &str, tts: &Arc<StubTts>) -> VoiceConfirmer {
        VoiceConfirmer::new(
            source,
            Arc::new(StubAsr {
                text: answer.to_string(),
            }),
            Arc::clone(tts) as Arc<dyn SpeechSynthesizer>,
            config(),
            Duration::from_secs(5),
        )
    }

    #[test]
    fn a_yes_approves() {
        let tts = Arc::new(StubTts {
            spoken: std::sync::Mutex::new(Vec::new()),
        });
        let source = shared(SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap());
        let c = confirmer(source, "yes", &tts);
        assert!(c
            .confirm("Shall I run the tests?", Duration::from_secs(5))
            .unwrap());
        assert_eq!(tts.spoken.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_no_refuses() {
        let tts = Arc::new(StubTts {
            spoken: std::sync::Mutex::new(Vec::new()),
        });
        let source = shared(SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap());
        let c = confirmer(source, "no", &tts);
        assert!(!c
            .confirm("Shall I run the tests?", Duration::from_secs(5))
            .unwrap());
    }

    #[test]
    fn an_unclear_answer_refuses() {
        let tts = Arc::new(StubTts {
            spoken: std::sync::Mutex::new(Vec::new()),
        });
        let source = shared(SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap());
        let c = confirmer(source, "maybe later", &tts);
        assert!(!c
            .confirm("Shall I run the tests?", Duration::from_secs(5))
            .unwrap());
    }

    #[test]
    fn silence_refuses() {
        let tts = Arc::new(StubTts {
            spoken: std::sync::Mutex::new(Vec::new()),
        });
        let silent = MonoPcm::silence(1500, 16_000).unwrap();
        let source = shared(FileSource::from_pcm(silent, 20).unwrap());
        let c = confirmer(source, "yes", &tts);
        assert!(!c
            .confirm("Shall I run the tests?", Duration::from_secs(5))
            .unwrap());
    }

    #[test]
    fn yesterday_is_not_a_yes() {
        // "yesterday" contains "yes". A substring matcher would approve an
        // action on the word "yesterday", which is the kind of mistake that
        // matters most here.
        assert!(!VoiceConfirmer::is_yes("yesterday"));
        assert!(VoiceConfirmer::is_yes("yes"));
        assert!(VoiceConfirmer::is_yes("Yes, do it"));
        assert!(VoiceConfirmer::is_yes("okay"));
        assert!(!VoiceConfirmer::is_yes("no"));
        assert!(!VoiceConfirmer::is_yes("not yet"));
        assert!(!VoiceConfirmer::is_yes(""));
    }

    #[test]
    fn the_question_is_spoken_without_the_terminal_layout() {
        let q =
            VoiceConfirmer::spoken_question("May I read the file main.rs?\n  read_file main.rs");
        assert!(q.contains("May I read the file main.rs?"));
        assert!(q.contains("read_file main.rs"));
        assert!(q.ends_with("Shall I?"));
        assert!(!q.contains('\n'));
    }

    #[test]
    fn the_terminal_layout_is_stripped_for_speaking() {
        // The real prompt the policy engine writes for a gated tool.
        let q = VoiceConfirmer::spoken_question("Run `run_tests` (medium)?\n  run_tests");
        assert_eq!(q, "Run run_tests? Shall I?");
        assert!(!q.contains('`'));
        assert!(!q.contains("medium"));
    }

    #[test]
    fn a_parenthetical_that_is_not_a_risk_is_kept() {
        let q = VoiceConfirmer::spoken_question("Run `open_url` (example.com)?");
        assert!(q.contains("(example.com)"), "{q}");
    }
}

#[cfg(test)]
mod end_to_end {
    //! The whole path: a spoken command, a spoken question, a spoken answer.

    use super::*;
    use crate::agent::{Agent, AgentDeps};
    use crate::session::{default_wake_check, SessionOutcome, VoiceSession};
    use aaron_core::config::Config;
    use aaron_core::{AsrInfo, IntentClassifier, MonoPcm, Transcript, WakeDetector};
    use aaron_memory::MemoryStore;
    use aaron_security::{AuditLog, PathSandbox, PolicyEngine, PolicyFile};
    use aaron_tools::{ToolContext, ToolRegistry};

    const ASK: &str = r#"
version = 1
[tools.system_status]
effect = "ask"
"#;

    /// A recogniser that answers with a different phrase each time.
    ///
    /// The first call is the command; the second is the answer to the
    /// confirmation question.
    struct QueueAsr {
        texts: std::sync::Mutex<Vec<String>>,
    }

    impl SpeechRecognizer for QueueAsr {
        fn backend(&self) -> String {
            "queue".into()
        }
        fn transcribe(&self, _audio: &MonoPcm) -> Result<Transcript> {
            let mut texts = self.texts.lock().unwrap();
            let text = if texts.is_empty() {
                String::new()
            } else {
                texts.remove(0)
            };
            Ok(Transcript {
                text,
                backend: "queue".into(),
                latency_ms: 1,
            })
        }
        fn info(&self) -> AsrInfo {
            AsrInfo {
                backend: "queue".into(),
                ready: true,
                model_path: None,
                detail: "test double".into(),
                production: true,
            }
        }
    }

    struct StubTts {
        spoken: std::sync::Mutex<Vec<String>>,
    }

    impl SpeechSynthesizer for StubTts {
        fn backend(&self) -> String {
            "stub".into()
        }
        fn speak(&self, text: &str) -> Result<()> {
            self.spoken.lock().unwrap().push(text.to_string());
            Ok(())
        }
        fn render_wav(&self, _text: &str, _path: &std::path::Path) -> Result<bool> {
            Ok(false)
        }
        fn info(&self) -> aaron_core::TtsInfo {
            aaron_core::TtsInfo {
                backend: "stub".into(),
                ready: true,
                detail: "test double".into(),
            }
        }
    }

    struct StubWake;

    impl WakeDetector for StubWake {
        fn name(&self) -> String {
            "stub".into()
        }
        fn ready(&self) -> Result<()> {
            Ok(())
        }
        fn reset(&self) {}
    }

    fn config() -> Config {
        let mut c = Config::default();
        c.tts.echo_to_terminal = false;
        c.audio.max_capture_secs = 4;
        c.audio.vad.silence_ms = 150;
        c.audio.vad.min_speech_ms = 60;
        c
    }

    /// Run one turn: a spoken command, then a spoken answer to the question
    /// the assistant asked.
    fn one_turn(answer: &str) -> (SessionOutcome, Vec<String>) {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = config();
        let memory = Arc::new(std::sync::Mutex::new(
            MemoryStore::open_in_memory(Default::default()).unwrap(),
        ));
        let ctx = ToolContext::new(
            Arc::new(PathSandbox::deny_all()),
            Arc::clone(&memory),
            Arc::new(cfg.clone()),
            tmp.path().to_path_buf(),
        );

        let source: SharedSource = Arc::new(std::sync::Mutex::new(Box::new(
            aaron_audio::SyntheticSource::new(4000, 16_000, 20, 220.0).unwrap(),
        )
            as Box<dyn aaron_core::AudioSource>));
        let tts = Arc::new(StubTts {
            spoken: std::sync::Mutex::new(Vec::new()),
        });
        let asr = Arc::new(QueueAsr {
            texts: std::sync::Mutex::new(vec![
                "aaron check system status".to_string(),
                answer.to_string(),
            ]),
        });

        let confirmer = Arc::new(VoiceConfirmer::new(
            Arc::clone(&source),
            asr.clone() as Arc<dyn SpeechRecognizer>,
            Arc::clone(&tts) as Arc<dyn SpeechSynthesizer>,
            cfg.clone(),
            Duration::from_secs(5),
        ));

        let mut agent = Agent::new(AgentDeps {
            classifier: Arc::new(aaron_brain::HeuristicClassifier::new(0.5))
                as Arc<dyn IntentClassifier>,
            registry: ToolRegistry::builtin(),
            policy: PolicyEngine::new(PolicyFile::from_toml(ASK).unwrap(), Vec::new()),
            audit: AuditLog::disabled(),
            context: ctx,
            memory,
            tts: Arc::clone(&tts) as Arc<dyn SpeechSynthesizer>,
            confirmer: confirmer as Arc<dyn crate::agent::Confirmer>,
            config: cfg.clone(),
        });

        let mut session = VoiceSession::new(
            &mut agent,
            Arc::clone(&source),
            asr as Arc<dyn SpeechRecognizer>,
            Arc::new(StubWake),
            cfg,
            "aaron".into(),
        )
        .unwrap();

        let outcome = session.listen_once(default_wake_check).unwrap();
        let spoken = tts.spoken.lock().unwrap().clone();
        (outcome, spoken)
    }

    #[test]
    fn a_spoken_yes_runs_the_gated_action() {
        let (outcome, spoken) = one_turn("yes");
        match outcome {
            SessionOutcome::Command { reply, .. } => {
                assert!(
                    reply.contains("CPU") || reply.contains("System"),
                    "the tool should have run, got: {reply}"
                );
            }
            other => panic!("expected the action to run, got {other:?}"),
        }
        // The question was spoken before the answer was accepted.
        assert!(
            !spoken.is_empty(),
            "the confirmation question must be spoken"
        );
        assert!(
            spoken[0].contains("Shall I?"),
            "the spoken question was {:?}",
            spoken[0]
        );
    }

    #[test]
    fn a_spoken_no_refuses_the_gated_action() {
        let (outcome, _) = one_turn("no");
        match outcome {
            SessionOutcome::Command { reply, .. } => {
                assert!(
                    reply.contains("Cancelled"),
                    "the action should have been refused, got: {reply}"
                );
                assert!(
                    !reply.contains("CPU"),
                    "the tool must not have run, got: {reply}"
                );
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
    }
}
