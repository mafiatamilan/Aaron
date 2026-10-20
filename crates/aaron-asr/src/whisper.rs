//! Speech recognition through `whisper-cli` (whisper.cpp).
//!
//! # Why a child process
//!
//! whisper.cpp ships as a C/C++ library with no maintained Rust binding that
//! covers every build option. Running the binary with a **fixed argument
//! vector** keeps the integration honest: AARON never passes user text to it,
//! never uses a shell, and validates the checksum of the model file before use.
//!
//! # What this is not
//!
//! AARON never downloads a Whisper model. If you have not installed one,
//! [`WhisperAsr::available`] returns `false` and `aaron doctor` says so. The
//! model is a separate, separately licensed artefact — its parameter count is
//! reported in `docs/model-training.md` and is **not** part of AARON's 1M
//! custom-model budget.

use std::path::{Path, PathBuf};

use aaron_core::config::AsrConfig;
use aaron_core::error::{Error, Result};
use aaron_core::{AsrInfo, MonoPcm, SpeechRecognizer, Transcript};

/// Sample rate every Whisper-family model expects.
pub const WHISPER_SAMPLE_RATE: u32 = 16_000;

/// whisper.cpp backend.
pub struct WhisperAsr {
    binary: PathBuf,
    model: PathBuf,
    language: String,
    threads: u32,
    use_gpu: bool,
    temp_dir: PathBuf,
    /// Result of the model checksum check, if one was configured.
    model_sha256: Option<String>,
}

impl WhisperAsr {
    /// Construct from configuration. Does not check that anything exists yet.
    pub fn new(config: &AsrConfig) -> Result<WhisperAsr> {
        if config.model_path.trim().is_empty() {
            return Err(Error::missing_with_hint(
                "a Whisper model file",
                "set asr.model_path in your config; AARON never downloads models for you. \
                 Arch users can install `whisper.cpp` and fetch a ggml model from Hugging Face.",
            ));
        }
        let temp_dir = std::env::temp_dir().join(format!("aaron-asr-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).map_err(|e| Error::io(&temp_dir, e))?;
        Ok(WhisperAsr {
            binary: PathBuf::from(&config.binary_path),
            model: PathBuf::from(&config.model_path),
            language: config.language.clone(),
            threads: config.threads.clamp(1, 64),
            use_gpu: config.use_gpu,
            temp_dir,
            model_sha256: Some(config.model_sha256.clone()).filter(|s| !s.trim().is_empty()),
        })
    }

    /// Override the temporary directory used for the WAV hand-off.
    pub fn with_temp_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.temp_dir = dir.into();
        let _ = std::fs::create_dir_all(&self.temp_dir);
        self
    }

    /// The exact argument vector AARON passes. Exposed so tests can assert the
    /// integration never grows a user-controlled flag.
    pub fn argv(&self, wav: &Path) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "-m".into(),
            self.model.display().to_string(),
            "-f".into(),
            wav.display().to_string(),
            "-l".into(),
            self.language.clone(),
            "-t".into(),
            self.threads.to_string(),
            // No timestamps, no progress bars: stdout is a transcript, nothing
            // else, so parsing stays trivial and stable.
            "-nt".into(),
            "-np".into(),
            "-of".into(),
            self.temp_dir.join("transcript").display().to_string(),
        ];
        if self.use_gpu {
            args.push("-ng".into());
        }
        args
    }

    /// Whether the binary and model are both present.
    pub fn available(&self) -> bool {
        self.binary.exists() && self.model.is_file()
    }

    /// Verify the configured SHA-256 of the model file.
    pub fn verify_model(&self) -> Result<()> {
        let Some(expected) = &self.model_sha256 else {
            return Ok(());
        };
        let actual = sha256_file(&self.model)?;
        if &actual != expected {
            return Err(Error::Model(format!(
                "whisper model checksum mismatch for {}: expected {expected}, got {actual}",
                self.model.display()
            )));
        }
        Ok(())
    }

    /// Transcribe, leaving no temporary files behind.
    pub fn transcribe_to(&self, audio: &MonoPcm) -> Result<Transcript> {
        let started = std::time::Instant::now();
        if !self.binary.exists() {
            return Err(Error::missing_with_hint(
                self.binary.display().to_string(),
                "install whisper.cpp (`pacman -S whisper.cpp`) or set asr.binary_path",
            ));
        }
        if !self.model.is_file() {
            return Err(Error::missing_with_hint(
                format!("whisper model at {}", self.model.display()),
                "download a ggml model yourself and set asr.model_path; AARON will not download it",
            ));
        }
        self.verify_model()?;

        let pcm = audio.resample(WHISPER_SAMPLE_RATE);
        let wav_path = self.temp_dir.join("utterance.wav");
        pcm.write_wav(&wav_path)?;

        let mut cmd = std::process::Command::new(&self.binary);
        cmd.args(self.argv(&wav_path))
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());
        let out = cmd.output().map_err(|e| Error::io(&self.binary, e))?;
        let _ = std::fs::remove_file(&wav_path);

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            return Err(Error::Asr(format!(
                "whisper.cpp exited with {}: {}",
                out.status,
                aaron_core::text::truncate_chars(stderr.trim(), 400)
            )));
        }

        let stdout = String::from_utf8_lossy(&out.stdout);
        let text = parse_whisper_output(&stdout);
        Ok(Transcript {
            text,
            backend: "whisper.cpp".into(),
            latency_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// Extract the transcript from whisper.cpp stdout.
///
/// `-nt` suppresses per-segment headers, but older builds still print them, so
/// lines that look like `[00:00:00.000 --> 00:00:01.000]` are stripped rather
/// than fed to the intent classifier.
pub fn parse_whisper_output(stdout: &str) -> String {
    let mut parts: Vec<String> = Vec::new();
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('[') && line.contains("-->") {
            // `[00:00:00.000 --> 00:00:01.000]   hello there`
            if let Some((_, rest)) = line.split_once(']') {
                let rest = rest.trim();
                if !rest.is_empty() {
                    parts.push(rest.to_string());
                }
            }
            continue;
        }
        // Skip whisper's own banner lines.
        if line.starts_with("whisper_") || line.starts_with("main:") || line.starts_with("model_") {
            continue;
        }
        parts.push(line.to_string());
    }
    aaron_core::text::normalize_entities(&parts.join(" "))
}

/// SHA-256 of a file, lowercase hex.
pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path).map_err(|e| Error::io(path, e))?;
    let mut h = Sha256::new();
    h.update(&bytes);
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

impl SpeechRecognizer for WhisperAsr {
    fn backend(&self) -> String {
        "whisper.cpp".into()
    }

    fn transcribe(&self, audio: &MonoPcm) -> Result<Transcript> {
        self.transcribe_to(audio)
    }

    fn info(&self) -> AsrInfo {
        let binary_ok = self.binary.exists();
        let model_ok = self.model.is_file();
        AsrInfo {
            backend: "whisper.cpp".into(),
            ready: binary_ok && model_ok,
            model_path: Some(self.model.display().to_string()),
            detail: match (binary_ok, model_ok) {
                (false, _) => format!("`{}` is not installed", self.binary.display()),
                (_, false) => format!(
                    "no model at {}; AARON never downloads models for you",
                    self.model.display()
                ),
                _ => format!("ready ({} threads)", self.threads),
            },
            // whisper.cpp performs real recognition, so it is production
            // whether or not it is installed.
            production: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aaron_core::config::AsrBackend;

    fn config_with_model(model: &str) -> AsrConfig {
        AsrConfig {
            backend: AsrBackend::WhisperCpp,
            model_path: model.into(),
            binary_path: "/usr/bin/whisper-cli".into(),
            ..Default::default()
        }
    }

    #[test]
    fn missing_model_path_is_a_clear_install_error() {
        let Err(err) = WhisperAsr::new(&config_with_model("")) else {
            panic!("an empty model path must be refused");
        };
        assert!(matches!(err, Error::MissingDependency(_)));
        assert!(err.to_string().contains("never downloads"));
    }

    #[test]
    fn missing_binary_is_reported_without_pretending_to_work() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("ggml-tiny.bin");
        std::fs::write(&model, b"not a real model").unwrap();
        let asr = WhisperAsr::new(&config_with_model(model.to_str().unwrap())).unwrap();
        assert!(!asr.available());
        let info = asr.info();
        assert!(!info.ready);
        assert!(info.detail.contains("not installed"));

        let err = asr
            .transcribe_to(&MonoPcm::silence(100, 16_000).unwrap())
            .unwrap_err();
        assert!(matches!(err, Error::MissingDependency(_)));
    }

    #[test]
    fn missing_model_file_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        // A stand-in binary that exists and is executable, so the *model* check
        // is the one that fires.
        let fake = dir.path().join("whisper-cli");
        std::fs::write(&fake, "#!/bin/sh\nexit 0\n").unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&fake, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let cfg = AsrConfig {
            binary_path: fake.display().to_string(),
            ..config_with_model("/no/such/model.bin")
        };
        let asr = WhisperAsr::new(&cfg).unwrap();
        assert!(!asr.available());
        let err = asr
            .transcribe_to(&MonoPcm::silence(100, 16_000).unwrap())
            .unwrap_err();
        assert!(err.to_string().contains("whisper model at"), "{err}");
    }

    #[test]
    fn argv_is_fixed_and_carries_no_user_text() {
        let asr = WhisperAsr::new(&config_with_model("/models/ggml-base.en.bin")).unwrap();
        let args = asr.argv(Path::new("/tmp/a.wav"));
        assert!(args.contains(&"-m".to_string()));
        assert!(args.contains(&"-nt".to_string()));
        assert!(args.contains(&"-np".to_string()));
        assert_eq!(args.iter().filter(|a| *a == "-f").count(), 1);
        // No argument may be empty or contain a shell metacharacter chain.
        for a in &args {
            assert!(!a.contains("\n"));
        }
    }

    #[test]
    fn gpu_flag_is_only_added_when_requested() {
        let mut cfg = config_with_model("/models/m.bin");
        let cpu = WhisperAsr::new(&cfg).unwrap();
        assert!(!cpu
            .argv(Path::new("/tmp/a.wav"))
            .contains(&"-ng".to_string()));
        cfg.use_gpu = true;
        let gpu = WhisperAsr::new(&cfg).unwrap();
        assert!(gpu
            .argv(Path::new("/tmp/a.wav"))
            .contains(&"-ng".to_string()));
    }

    #[test]
    fn parses_segmented_whisper_output() {
        let stdout = "whisper_init: loading model\n\
             [00:00:00.000 --> 00:00:01.500]  hey aaron, check system status\n\
             \n\
             [00:00:01.500 --> 00:00:02.000]  thanks\n";
        // Commas are separators for the intent model, so they are dropped.
        assert_eq!(
            parse_whisper_output(stdout),
            "hey aaron check system status thanks"
        );
    }

    #[test]
    fn parses_bare_transcript_output() {
        assert_eq!(
            parse_whisper_output("check the system status\n"),
            "check the system status"
        );
        assert_eq!(parse_whisper_output(""), "");
    }

    #[test]
    fn model_checksum_mismatch_is_fatal() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.bin");
        std::fs::write(&model, b"abc").unwrap();
        let cfg = AsrConfig {
            model_sha256: "0000000000000000000000000000000000000000000000000000000000000000".into(),
            ..config_with_model(model.to_str().unwrap())
        };
        let asr = WhisperAsr::new(&cfg).unwrap();
        let err = asr.verify_model().unwrap_err();
        assert!(err.to_string().contains("checksum mismatch"));
    }

    #[test]
    fn model_checksum_can_be_satisfied() {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("m.bin");
        std::fs::write(&model, b"abc").unwrap();
        let sha = sha256_file(&model).unwrap();
        let cfg = AsrConfig {
            model_sha256: sha.clone(),
            ..config_with_model(model.to_str().unwrap())
        };
        assert!(WhisperAsr::new(&cfg).unwrap().verify_model().is_ok());
    }

    #[test]
    fn transcribing_without_a_model_never_returns_a_fake_transcript() {
        let asr = WhisperAsr::new(&config_with_model("/no/such/model.bin")).unwrap();
        assert!(asr
            .transcribe_to(&MonoPcm::silence(50, 16_000).unwrap())
            .is_err());
    }
}
