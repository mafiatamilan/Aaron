//! Speech recognition through the Phonon engine (Fermion Research).
//!
//! # What Phonon-2 is
//!
//! Phonon-2 is an open speech recognition model: a 164 MB download, 5.21 %
//! average word error rate across seven English benchmarks, and it runs on an
//! ordinary x86-64 CPU through the `avx2` kernel tier. The weights are
//! CC-BY-4.0; the command line is Apache-2.0.
//!
//! It is **not** a text-to-speech model. The name is shared with a separate
//! on-device TTS model that is not publicly available, and the Phonon CLI
//! has no speak command. AARON's TTS path is espeak-ng, piper or
//! speech-dispatcher, and the Phonon engine is used for recognition only.
//!
//! # Why a child process
//!
//! The engine ships as a Python package with its own kernels. AARON runs the
//! `fermion` command with a **fixed argument vector** and never a shell, and
//! it goes through the same hardened process harness the tool layer uses: a
//! timeout, a process group, a cleared environment and a capped output. The
//! whisper.cpp backend grew up without those, and a hung recogniser used to
//! block `aaron listen` indefinitely.
//!
//! # What this is not
//!
//! AARON never downloads the model. The `fermion` CLI fetches it on first
//! use and caches it under `~/.cache/fermion`; that is the engine's own
//! behaviour, not AARON's, and AARON's own rule is that it never fetches a
//! model on the user's behalf. If the engine is not installed, the backend
//! reports that it is not ready and `aaron doctor` says so.

use std::path::{Path, PathBuf};

use aaron_core::config::AsrConfig;
use aaron_core::error::{Error, Result};
use aaron_core::{AsrInfo, MonoPcm, SpeechRecognizer, Transcript};

/// Sample rate the engine expects; it resamples internally, but the WAV
/// hand-off is cheaper at the native rate.
pub const PHONON_SAMPLE_RATE: u32 = 16_000;

/// The default model name.
pub const DEFAULT_MODEL: &str = "phonon-2";

/// The Phonon backend.
pub struct PhononAsr {
    binary: PathBuf,
    /// The model name: `phonon-2`, `phonon-1`, `phonon-1-micro`, …
    model: String,
    threads: u32,
    temp_dir: PathBuf,
    /// Terms to favour, comma separated. Empty disables them.
    hotwords: String,
}

impl PhononAsr {
    /// Construct from configuration. Does not check that anything exists yet.
    pub fn new(config: &AsrConfig) -> Result<PhononAsr> {
        let temp_dir = std::env::temp_dir().join(format!("aaron-phonon-{}", std::process::id()));
        std::fs::create_dir_all(&temp_dir).map_err(|e| Error::io(&temp_dir, e))?;
        Ok(PhononAsr {
            binary: PathBuf::from(&config.phonon_binary_path),
            model: if config.model.trim().is_empty() {
                DEFAULT_MODEL.to_string()
            } else {
                config.model.trim().to_string()
            },
            threads: config.threads.clamp(1, 64),
            temp_dir,
            hotwords: config.hotwords.clone(),
        })
    }

    /// Override the temporary directory used for the WAV hand-off.
    pub fn with_temp_dir(mut self, dir: impl Into<PathBuf>) -> PhononAsr {
        self.temp_dir = dir.into();
        let _ = std::fs::create_dir_all(&self.temp_dir);
        self
    }

    /// Use a different model.
    pub fn with_model(mut self, model: impl Into<String>) -> PhononAsr {
        self.model = model.into();
        self
    }

    /// Favour these names and terms.
    pub fn with_hotwords(mut self, hotwords: impl Into<String>) -> PhononAsr {
        self.hotwords = hotwords.into();
        self
    }

    /// The exact argument vector AARON passes. Exposed so tests can assert
    /// the integration never grows a user-controlled flag.
    pub fn argv(&self, wav: &Path) -> Vec<String> {
        let mut args: Vec<String> = vec![
            "transcribe".into(),
            self.model.clone(),
            wav.display().to_string(),
            // JSON, so the transcript is parsed from a field rather than
            // scraped off whatever the engine printed.
            "--json".into(),
            "--threads".into(),
            self.threads.to_string(),
        ];
        if !self.hotwords.trim().is_empty() {
            args.push("--hotwords".into());
            args.push(self.hotwords.trim().to_string());
        }
        args
    }

    /// Whether the engine is installed.
    pub fn available(&self) -> bool {
        self.binary.exists()
            || aaron_core::hardware::which(&self.binary.display().to_string()).is_some()
    }

    /// Transcribe, leaving no temporary files behind.
    pub fn transcribe_to(&self, audio: &MonoPcm) -> Result<Transcript> {
        let started = std::time::Instant::now();
        if !self.available() {
            return Err(Error::missing_with_hint(
                self.binary.display().to_string(),
                "install the `fermion-research` package (`pip install fermion-research`), \
                 or set asr.binary_path to the `fermion` command",
            ));
        }

        let pcm = audio.resample(PHONON_SAMPLE_RATE);
        let wav_path = self.temp_dir.join("utterance.wav");
        pcm.write_wav(&wav_path)?;

        // The hardened harness: a timeout, a process group, a cleared
        // environment and a capped output. A hung recogniser must not block
        // the assistant, and a noisy one must not exhaust its memory.
        if std::env::var("AARON_DEBUG_ASR").is_ok() {
            eprintln!("phonon argv: {:?}", self.argv(&wav_path));
        }
        let options = aaron_tools::proc::ExecOptions::new(
            self.binary.display().to_string(),
            self.argv(&wav_path),
        )
        .timeout(120)
        .max_output(1 << 20)
        // The engine reserves a large virtual address space for its model
        // runtimes, so the default four-gibibyte address-space cap makes it
        // fail to allocate a single megabyte. The timeout is what bounds it.
        .without_address_space_limit();
        let outcome = aaron_tools::proc::run(&options);
        let _ = std::fs::remove_file(&wav_path);
        let output = outcome?;

        if !output.ok() {
            let detail = aaron_core::text::truncate_chars(output.combined().trim(), 400);
            return Err(Error::Asr(format!(
                "the Phonon engine exited with an error: {detail}"
            )));
        }

        let stdout = String::from_utf8_lossy(output.stdout.as_bytes());
        let text = parse_phonon_output(&stdout);
        Ok(Transcript {
            text,
            backend: "phonon".into(),
            latency_ms: started.elapsed().as_millis() as u64,
        })
    }
}

/// Pull the transcript out of the engine's JSON output.
///
/// The engine's contract is that stdout carries only the transcript, or only
/// the JSON object. If the JSON does not parse, the raw text is used: a
/// transcript is better than an error, and the engine's own plain-text mode
/// exists.
pub fn parse_phonon_output(stdout: &str) -> String {
    let trimmed = stdout.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
            return text.trim().to_string();
        }
    }
    // Plain-text mode, or a JSON object without a `text` field.
    trimmed.to_string()
}

impl SpeechRecognizer for PhononAsr {
    fn backend(&self) -> String {
        "phonon".into()
    }

    fn transcribe(&self, audio: &MonoPcm) -> Result<Transcript> {
        self.transcribe_to(audio)
    }

    fn info(&self) -> AsrInfo {
        let installed = self.available();
        AsrInfo {
            backend: "phonon".into(),
            ready: installed,
            model_path: None,
            detail: if installed {
                format!(
                    "model {}; the engine fetches it on first use and caches it under \
                     ~/.cache/fermion",
                    self.model
                )
            } else {
                format!(
                    "`{}` is not installed; install the `fermion-research` package",
                    self.binary.display()
                )
            },
            // The engine performs real recognition.
            production: true,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_argument_vector_is_fixed() {
        let asr = PhononAsr {
            binary: PathBuf::from("fermion"),
            model: DEFAULT_MODEL.to_string(),
            threads: 4,
            temp_dir: PathBuf::from("/tmp"),
            hotwords: String::new(),
        };
        let argv = asr.argv(Path::new("/tmp/x.wav"));
        assert_eq!(
            argv,
            vec![
                "transcribe",
                "phonon-2",
                "/tmp/x.wav",
                "--json",
                "--threads",
                "4"
            ]
        );
    }

    #[test]
    fn hotwords_are_added_only_when_there_are_some() {
        let asr = PhononAsr {
            binary: PathBuf::from("fermion"),
            model: DEFAULT_MODEL.to_string(),
            threads: 2,
            temp_dir: PathBuf::from("/tmp"),
            hotwords: "Ada, Quillon".into(),
        };
        let argv = asr.argv(Path::new("/tmp/x.wav"));
        assert!(argv.contains(&"--hotwords".to_string()));
        assert!(argv.contains(&"Ada, Quillon".to_string()));
    }

    #[test]
    fn the_transcript_is_read_from_the_json() {
        assert_eq!(
            parse_phonon_output(r#"{"text": "hello there", "model": "x"}"#),
            "hello there"
        );
    }

    #[test]
    fn plain_text_output_is_accepted() {
        assert_eq!(parse_phonon_output("hello there\n"), "hello there");
        assert_eq!(parse_phonon_output("   "), "");
    }
}
