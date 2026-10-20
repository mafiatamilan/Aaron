//! Offline speech synthesis backends.
//!
//! Three local engines are supported, in the order `auto` prefers them:
//!
//! | backend    | binary                | model file | notes                                   |
//! |------------|-----------------------|------------|-----------------------------------------|
//! | espeak-ng  | `espeak-ng`           | none       | robotic but tiny; no download needed    |
//! | piper      | `piper`               | ONNX voice | neural, better sounding, user-installed |
//! | spd-say    | `spd-say`             | none       | routes through speech-dispatcher        |
//!
//! None of them contacts a network service. Each is invoked as a fixed program
//! with an argument vector; the text is passed as one argument and never
//! through a shell, so a response containing `;` or `$(...)` is spoken rather
//! than executed.
//!
//! AARON will not download a voice model. If `piper` is selected and its model
//! file is missing, the backend says so and `auto` falls back to espeak-ng.

use std::path::{Path, PathBuf};

use aaron_core::config::{TtsBackend, TtsConfig};
use aaron_core::error::{Error, Result};
use aaron_core::{SpeechSynthesizer, TtsInfo};

/// The local players AARON can hand a rendered WAV to, in order of
/// preference.
///
/// `espeak-ng` and `piper` render; they do not play. Something has to open
/// the file, and the honest options are the three that ship with a normal
/// desktop.
const PLAYERS: &[&str] = &["paplay", "aplay", "pw-play"];

/// The first available local player, if any.
fn available_player() -> Option<&'static str> {
    PLAYERS
        .iter()
        .copied()
        .find(|p| aaron_core::hardware::which(p).is_some())
}

/// Play a rendered WAV through a local player.
///
/// Returns the player that was used. A failure is an error, never a silent
/// success: an assistant that says it spoke when nothing came out of the
/// speakers is worse than one that admits the problem.
fn play_wav(path: &Path) -> Result<&'static str> {
    let player = available_player().ok_or_else(|| {
        Error::missing_with_hint(
            "a playback program",
            "install `pulseaudio-utils` (paplay) or `alsa-utils` (aplay); AARON renders the \
             audio and hands it to a player",
        )
    })?;
    let status = std::process::Command::new(player)
        .arg(path)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map_err(|e| Error::Tts(format!("could not run {player}: {e}")))?;
    if !status.success() {
        return Err(Error::Tts(format!("{player} exited with {status}")));
    }
    Ok(player)
}

/// A unique path in the temporary directory for a rendered WAV.
fn scratch_wav(tag: &str) -> std::path::PathBuf {
    use std::sync::atomic::{AtomicU32, Ordering};
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("aaron-{tag}-{}-{n}.wav", std::process::id()))
}

/// espeak-ng. Small, always installable from the distro, no model file.
pub struct EspeakTts {
    binary: PathBuf,
    voice: String,
    rate_wpm: u32,
}

impl EspeakTts {
    pub fn new(config: &TtsConfig) -> EspeakTts {
        let binary = if config.binary_path.trim().is_empty() {
            PathBuf::from("espeak-ng")
        } else {
            PathBuf::from(&config.binary_path)
        };
        EspeakTts {
            binary,
            voice: config.voice.clone(),
            rate_wpm: config.rate_wpm,
        }
    }

    /// The fixed argument vector, exposed for tests.
    pub fn argv(&self, text: &str, wav: Option<&Path>) -> Vec<String> {
        let mut args = vec![
            "-v".to_string(),
            self.voice.clone(),
            "-s".to_string(),
            self.rate_wpm.to_string(),
        ];
        match wav {
            Some(path) => {
                args.push("-w".into());
                args.push(path.display().to_string());
            }
            // Without `-w` espeak-ng writes audio to stdout, which we discard.
            None => args.push("-q".into()),
        }
        // The text is a single argv entry: no shell, ever.
        args.push(text.to_string());
        args
    }

    /// Whether the binary can be found.
    pub fn available(&self) -> bool {
        self.binary.exists()
            || aaron_core::hardware::which(&self.binary.display().to_string()).is_some()
    }
}

impl SpeechSynthesizer for EspeakTts {
    fn backend(&self) -> String {
        "espeak-ng".into()
    }

    fn speak(&self, text: &str) -> Result<()> {
        // It used to render to `/dev/null` and discard the boolean that says
        // nothing was produced, so `speak` returned success and the assistant
        // believed it had spoken while the room stayed silent.
        //
        // Now the audio goes to a real file and a local player opens it. If
        // either step fails, that is an error the caller can report.
        let scratch = scratch_wav("espeak");
        let rendered = self.render_wav(text, &scratch);
        let played = match rendered {
            Ok(true) => play_wav(&scratch).map(|_| ()),
            Ok(false) => Err(Error::Tts("espeak-ng produced no audio file".into())),
            Err(e) => Err(e),
        };
        let _ = std::fs::remove_file(&scratch);
        played
    }

    fn render_wav(&self, text: &str, wav_path: &Path) -> Result<bool> {
        if !self.available() {
            return Err(Error::missing_with_hint(
                "espeak-ng",
                "install the `espeak-ng` package, or set tts.backend = \"disabled\" for text-only replies",
            ));
        }
        let program = if self.binary.is_file() {
            self.binary.clone()
        } else {
            aaron_core::hardware::which(&self.binary.display().to_string()).ok_or_else(|| {
                Error::missing_with_hint("espeak-ng", "install the `espeak-ng` package")
            })?
        };
        let status = std::process::Command::new(program)
            .args(self.argv(text, Some(wav_path)))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map_err(|e| Error::Tts(format!("could not run espeak-ng: {e}")))?;
        if !status.success() {
            return Err(Error::Tts(format!("espeak-ng exited with {status}")));
        }
        Ok(wav_path.is_file())
    }

    fn info(&self) -> TtsInfo {
        TtsInfo {
            backend: "espeak-ng".into(),
            ready: self.available(),
            detail: if self.available() {
                format!("voice {}, {} wpm", self.voice, self.rate_wpm)
            } else {
                "espeak-ng is not installed".into()
            },
        }
    }
}

/// Piper: a small local neural TTS.
pub struct PiperTts {
    binary: PathBuf,
    model: PathBuf,
    length_scale: f32,
    /// Optional voice override; empty means "use the model's default voice".
    voice: String,
}

impl PiperTts {
    pub fn new(config: &TtsConfig) -> PiperTts {
        let binary = if config.binary_path.trim().is_empty() {
            PathBuf::from("piper")
        } else {
            PathBuf::from(&config.binary_path)
        };
        PiperTts {
            binary,
            model: PathBuf::from(&config.model_path),
            // 1.0 is the model's natural pace; below 1.0 is faster.
            length_scale: 1.0,
            voice: config.voice.clone(),
        }
    }

    /// The exact argument vector AARON passes.
    ///
    /// The text is **not** in it: piper reads the words on stdin, so a reply
    /// containing a newline, a quote or a shell metacharacter is spoken
    /// rather than interpreted.
    ///
    /// The long flag names moved between piper generations — `--length_scale`
    /// became `--length-scale` — so the spelling is probed from the binary
    /// rather than guessed. A wrong guess is a usage error and a silent
    /// reply.
    pub fn argv(&self, wav_path: &Path) -> Vec<String> {
        let mut args = vec![
            "--model".to_string(),
            self.model.display().to_string(),
            self.length_scale_flag().to_string(),
            self.length_scale.to_string(),
            "--output_file".to_string(),
            wav_path.display().to_string(),
        ];
        let config = self.model.with_extension("onnx.json");
        if config.is_file() {
            args.push("--config".to_string());
            args.push(config.display().to_string());
        }
        // Piper's `--speaker` is a *numeric* speaker id for a multi-speaker
        // model, not a voice name. The configuration's `voice` is an espeak
        // voice ("en-us"), and passing it here made piper exit with a usage
        // error — and AARON reported "speech synthesis unavailable" rather
        // than the real cause. So the flag is only passed when the value
        // actually looks like a speaker id, and piper uses the model's own
        // default otherwise.
        if self.voice.trim().parse::<u32>().is_ok() {
            args.push("--speaker".to_string());
            args.push(self.voice.trim().to_string());
        }
        args
    }

    /// Which spelling of the length-scale flag this piper understands.
    fn length_scale_flag(&self) -> &'static str {
        match self.help_text() {
            Some(help) if help.contains("--length-scale") => "--length-scale",
            _ => "--length_scale",
        }
    }

    /// The binary's own help text, if it can be produced.
    fn help_text(&self) -> Option<String> {
        let output = std::process::Command::new(&self.binary)
            .arg("--help")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .output()
            .ok()?;
        Some(format!(
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }

    pub fn available(&self) -> bool {
        let bin = if self.binary.is_file() {
            true
        } else {
            aaron_core::hardware::which(&self.binary.display().to_string()).is_some()
        };
        bin && self.model.is_file()
    }
}

impl SpeechSynthesizer for PiperTts {
    fn backend(&self) -> String {
        "piper".into()
    }

    fn speak(&self, text: &str) -> Result<()> {
        // Render to a real file and hand it to a local player.
        //
        // It used to render to `/dev/null` and discard the boolean that says
        // nothing was produced, so the assistant believed it had spoken while
        // the room stayed silent.
        let scratch = scratch_wav("piper");
        let rendered = self.render_wav(text, &scratch);
        let played = match rendered {
            Ok(true) => play_wav(&scratch).map(|_| ()),
            Ok(false) => Err(Error::Tts("piper produced no audio file".into())),
            Err(e) => Err(e),
        };
        let _ = std::fs::remove_file(&scratch);
        played
    }

    fn render_wav(&self, text: &str, wav_path: &Path) -> Result<bool> {
        if !self.available() {
            return Err(Error::missing_with_hint(
                "a piper voice model",
                "download a voice yourself and set tts.model_path; AARON never downloads models",
            ));
        }
        // The text goes in on stdin: piper's command line takes the model,
        // the config and the output path, and the words separately.
        if std::env::var("AARON_DEBUG_TTS").is_ok() {
            eprintln!("piper argv: {:?}", self.argv(wav_path));
        }
        let mut child = std::process::Command::new(&self.binary)
            .args(self.argv(wav_path))
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| Error::Tts(format!("could not run piper: {e}")))?;
        if let Some(stdin) = child.stdin.as_mut() {
            use std::io::Write;
            stdin
                .write_all(text.as_bytes())
                .map_err(|e| Error::Tts(format!("could not write to piper: {e}")))?;
        }
        let out = child
            .wait_with_output()
            .map_err(|e| Error::Tts(format!("piper did not finish: {e}")))?;
        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(Error::Tts(format!(
                "piper exited with {}: {}",
                out.status,
                aaron_core::text::truncate_chars(stderr.trim(), 400)
            )));
        }
        Ok(wav_path.is_file())
    }

    fn info(&self) -> TtsInfo {
        TtsInfo {
            backend: "piper".into(),
            ready: self.available(),
            detail: if self.available() {
                format!("model {}", self.model.display())
            } else if !self.model.is_file() {
                format!("no voice model at {}", self.model.display())
            } else {
                "the `piper` binary is not installed".into()
            },
        }
    }
}

/// speech-dispatcher.
pub struct SpdSayTts {
    binary: PathBuf,
    voice: String,
    rate: i32,
}

impl SpdSayTts {
    pub fn new(config: &TtsConfig) -> SpdSayTts {
        let binary = if config.binary_path.trim().is_empty() {
            PathBuf::from("spd-say")
        } else {
            PathBuf::from(&config.binary_path)
        };
        // speech-dispatcher takes a signed percentage, not words per minute.
        let rate = (((config.rate_wpm as f32 / 175.0) - 1.0) * 100.0).round() as i32;
        SpdSayTts {
            binary,
            voice: config.voice.clone(),
            rate,
        }
    }

    pub fn argv(&self, text: &str) -> Vec<String> {
        let mut args = vec!["-w".to_string()];
        if !self.voice.is_empty() {
            args.push(self.voice.clone());
        }
        args.push("-r".into());
        args.push(self.rate.to_string());
        args.push(text.to_string());
        args
    }

    pub fn available(&self) -> bool {
        self.binary.is_file()
            || aaron_core::hardware::which(&self.binary.display().to_string()).is_some()
    }
}

impl SpeechSynthesizer for SpdSayTts {
    fn backend(&self) -> String {
        "spd-say".into()
    }

    fn speak(&self, text: &str) -> Result<()> {
        if !self.available() {
            return Err(Error::missing_with_hint(
                "spd-say",
                "install `speech-dispatcher` and `speech-dispatcher-espeak-ng`",
            ));
        }
        let program = if self.binary.is_file() {
            self.binary.clone()
        } else {
            aaron_core::hardware::which(&self.binary.display().to_string()).ok_or_else(|| {
                Error::missing_with_hint("spd-say", "install the `speech-dispatcher` package")
            })?
        };
        let status = std::process::Command::new(program)
            .args(self.argv(text))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .status()
            .map_err(|e| Error::Tts(format!("could not run spd-say: {e}")))?;
        if !status.success() {
            return Err(Error::Tts(format!("spd-say exited with {status}")));
        }
        Ok(())
    }

    fn render_wav(&self, _text: &str, _wav_path: &Path) -> Result<bool> {
        Ok(false)
    }

    fn info(&self) -> TtsInfo {
        TtsInfo {
            backend: "spd-say".into(),
            ready: self.available(),
            detail: if self.available() {
                format!("rate {}", self.rate)
            } else {
                "spd-say is not installed".into()
            },
        }
    }
}

/// Writes to standard output only. Always available.
pub struct TextOnlyTts;

impl SpeechSynthesizer for TextOnlyTts {
    fn backend(&self) -> String {
        "text-only".into()
    }

    fn speak(&self, _text: &str) -> Result<()> {
        Ok(())
    }

    fn render_wav(&self, _text: &str, _wav_path: &Path) -> Result<bool> {
        Ok(false)
    }

    fn info(&self) -> TtsInfo {
        TtsInfo {
            backend: "text-only".into(),
            ready: true,
            detail: "replies are printed, never spoken".into(),
        }
    }
}

/// Speak `text`, falling back to a terminal echo when no engine is installed.
///
/// AARON never fails an assistant turn because a speech engine is missing: the
/// user still gets the answer, on screen.
pub fn speak_or_echo(synth: &dyn SpeechSynthesizer, text: &str, echo: bool) {
    match synth.speak(text) {
        Ok(()) => {
            if echo {
                println!("{text}");
            }
        }
        Err(e) => {
            tracing::debug!(error = %e, "speech synthesis unavailable; answering in text");
            if echo {
                println!("{text}");
            }
        }
    }
}

/// Build the configured backend.
pub fn build(config: &TtsConfig) -> std::sync::Arc<dyn SpeechSynthesizer> {
    match config.backend {
        TtsBackend::Disabled => std::sync::Arc::new(TextOnlyTts),
        TtsBackend::Espeak => std::sync::Arc::new(EspeakTts::new(config)),
        TtsBackend::Piper => std::sync::Arc::new(PiperTts::new(config)),
        TtsBackend::SpdSay => std::sync::Arc::new(SpdSayTts::new(config)),
        TtsBackend::Auto => {
            // In order of "will this actually be heard".
            //
            // `espeak-ng` renders and a local player opens the file, so it
            // needs both. `spd-say` plays by itself. `piper` renders but
            // AARON does not pipe it to a device, so it is never preferred
            // for a spoken reply — it used to be preferred over a backend
            // that works, and every turn then failed.
            let espeak = EspeakTts::new(config);
            if espeak.available() && available_player().is_some() {
                return std::sync::Arc::new(espeak);
            }
            let spd = SpdSayTts::new(config);
            if spd.available() {
                return std::sync::Arc::new(spd);
            }
            let piper = PiperTts::new(config);
            if piper.available() {
                return std::sync::Arc::new(piper);
            }
            std::sync::Arc::new(TextOnlyTts)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aaron_core::config::TtsConfig;

    #[test]
    fn text_only_backend_always_works() {
        let t = TextOnlyTts;
        assert!(t.speak("hello").is_ok());
        assert!(!t.render_wav("hello", Path::new("/tmp/x.wav")).unwrap());
        assert!(t.info().ready);
    }

    #[test]
    fn espeak_argv_passes_text_as_one_argument() {
        let cfg = TtsConfig {
            voice: "en-us".into(),
            rate_wpm: 175,
            ..Default::default()
        };
        let t = EspeakTts::new(&cfg);
        let args = t.argv("hello; rm -rf /", Some(Path::new("/tmp/o.wav")));
        assert_eq!(args.last().unwrap(), "hello; rm -rf /");
        // The dangerous text is one argv entry, not a shell fragment.
        assert!(!args.iter().any(|a| a == "; rm -rf /"));
    }

    #[test]
    fn espeak_reports_a_missing_binary_with_a_hint() {
        let cfg = TtsConfig {
            binary_path: "/usr/bin/definitely-not-espeak".into(),
            ..Default::default()
        };
        let t = EspeakTts::new(&cfg);
        assert!(!t.available());
        let err = t.speak("hi").unwrap_err();
        assert!(matches!(err, Error::MissingDependency(_)));
        assert!(err.to_string().contains("espeak-ng"));
    }

    #[test]
    fn an_out_of_range_rate_is_rejected_at_config_load_time() {
        let mut cfg = aaron_core::Config::default();
        cfg.tts.rate_wpm = 10;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("tts.rate_wpm"));
    }

    #[test]
    fn piper_requires_both_binary_and_model() {
        let cfg = TtsConfig {
            model_path: "/no/such/voice.onnx".into(),
            ..Default::default()
        };
        let t = PiperTts::new(&cfg);
        assert!(!t.available());
        assert!(t.info().detail.contains("no voice model"));
    }

    #[test]
    fn piper_never_claims_readiness_without_a_model() {
        let cfg = TtsConfig::default();
        let t = PiperTts::new(&cfg);
        assert!(!t.available());
        assert!(!t.info().ready);
    }

    #[test]
    fn spd_rate_is_converted_from_wpm_to_a_percentage() {
        let slow = SpdSayTts::new(&TtsConfig {
            rate_wpm: 100,
            ..Default::default()
        });
        let fast = SpdSayTts::new(&TtsConfig {
            rate_wpm: 350,
            ..Default::default()
        });
        assert!(slow.rate < 0);
        assert!(fast.rate > 0);
    }

    #[test]
    fn auto_selection_falls_back_to_text_only_with_nothing_installed() {
        let cfg = TtsConfig {
            binary_path: "/usr/bin/definitely-not-a-voice".into(),
            model_path: "/no/such/model.onnx".into(),
            ..Default::default()
        };
        let synth = build(&cfg);
        // Whatever it picked must be usable and honest about itself.
        assert!(synth.info().ready || synth.backend() == "text-only");
    }

    #[test]
    fn disabled_backend_forces_text_only() {
        let cfg = TtsConfig {
            backend: TtsBackend::Disabled,
            ..Default::default()
        };
        assert_eq!(build(&cfg).backend(), "text-only");
    }

    #[test]
    fn speak_or_echo_never_fails_the_turn() {
        let broken = EspeakTts::new(&TtsConfig {
            binary_path: "/usr/bin/nope".into(),
            ..Default::default()
        });
        // The broken backend must not panic or propagate; the caller keeps
        // running and the text still reaches the user.
        speak_or_echo(&broken, "the answer is 42", false);
    }
}
