//! Validated TOML configuration.
//!
//! Every section uses `#[serde(default)]` so a partial file is valid, but the
//! whole struct is `deny_unknown_fields` so a typo such as `[asr]` vs `[asr]`
//! misspelled as `[asr_typo]` is a hard error rather than a silently ignored
//! setting.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Root configuration object.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
#[derive(Default)]
pub struct Config {
    /// Assistant identity and global switches.
    #[serde(default)]
    pub assistant: AssistantConfig,
    /// Microphone capture settings.
    #[serde(default)]
    pub audio: AudioConfig,
    /// Wake phrase detection.
    #[serde(default)]
    pub wake: WakeConfig,
    /// Speech recognition.
    #[serde(default)]
    pub asr: AsrConfig,
    /// Speech synthesis.
    #[serde(default)]
    pub tts: TtsConfig,
    /// Intent classification.
    #[serde(default)]
    pub brain: BrainConfig,
    /// Agent loop behaviour and hard runtime limits.
    #[serde(default)]
    pub runtime: RuntimeConfig,
    /// Logging.
    #[serde(default)]
    pub logging: LoggingConfig,
    /// Long-term memory.
    #[serde(default)]
    pub memory: MemoryConfig,
    /// Review-based learning pipeline.
    #[serde(default)]
    pub learning: LearningConfig,
    /// Security policy wiring.
    #[serde(default)]
    pub security: SecurityConfig,
    /// Shared size limits.
    #[serde(default)]
    pub limits: LimitsConfig,
    /// Tool allow-lists: which programs may ever be launched, and with which
    /// fixed argument vectors.
    #[serde(default)]
    pub tools: ToolsConfig,
    /// Resource budgets and the low-memory state machine.
    #[serde(default)]
    pub resources: ResourceConfig,
}

/// Model load/unload policy.
///
/// AARON deliberately does **not** keep a speech model resident by default.
/// On a 4 GB card the display server and everything else on the desktop are
/// already using most of it, and AARON must never assume the whole budget is
/// available to it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ModelResourceConfig {
    /// Load the ASR backend only when something is actually being transcribed.
    pub load_asr_on_demand: bool,
    /// Load the TTS backend only when something is actually being spoken.
    pub load_tts_on_demand: bool,
    /// Release an idle model to free its memory.
    pub unload_idle_models: bool,
    /// How long a model may sit unused before it is eligible for release.
    pub idle_unload_seconds: u32,
}

impl Default for ModelResourceConfig {
    fn default() -> Self {
        ModelResourceConfig {
            load_asr_on_demand: true,
            load_tts_on_demand: true,
            unload_idle_models: true,
            idle_unload_seconds: 120,
        }
    }
}

/// Memory-related knobs for the SQLite store and the in-RAM working set.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryResourceConfig {
    /// SQLite page cache, in KiB, i.e. the `-2048` in `PRAGMA cache_size`.
    pub sqlite_cache_kib: u32,
    /// Hard cap on rows returned by any single retrieval query.
    pub max_retrieved_memories: usize,
    /// Hard cap on a single stored memory's size.
    pub max_memory_content_bytes: usize,
    /// Corrections held in RAM for review before paging to disk.
    pub max_learning_queue_items_in_memory: usize,
}

impl Default for MemoryResourceConfig {
    fn default() -> Self {
        MemoryResourceConfig {
            sqlite_cache_kib: 2048,
            max_retrieved_memories: 10,
            max_memory_content_bytes: 16 * 1024,
            max_learning_queue_items_in_memory: 100,
        }
    }
}

/// `[resources]`
///
/// Every value here is a **budget**, not a measurement. AARON reads the real
/// machine at startup and adapts conservatively inside these limits.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ResourceConfig {
    /// Enable the low-memory state machine at all.
    pub low_memory_mode: bool,
    /// Seconds between resource samples.
    pub monitor_interval_seconds: u32,
    /// System RAM AARON refuses to use, leaving it for the rest of the desktop.
    pub reserve_system_ram_mb: u64,
    /// VRAM AARON refuses to use, leaving it for the compositor and the desktop.
    pub reserve_gpu_vram_mb: u64,
    /// Fraction of *free* memory below which low-memory mode engages.
    ///
    /// Expressed as a percentage, not bytes, so it works on a 8 GB laptop and
    /// a 64 GB workstation without editing the config.
    pub low_memory_free_percent: u8,
    /// Fraction of free memory that must return before low-memory mode exits.
    ///
    /// Strictly greater than `low_memory_free_percent`: that gap is the
    /// hysteresis band that stops AARON flapping between modes near a
    /// threshold.
    pub low_memory_recover_percent: u8,
    /// Seconds of audio retained in RAM at once.
    pub max_audio_buffer_seconds: u32,
    /// Queued requests permitted before new ones are refused.
    pub max_pending_requests: usize,
    /// Conversation turns kept in RAM for context.
    pub max_conversation_turns_in_context: usize,
    /// Ceiling for a single tool's output held in RAM.
    pub max_tool_output_bytes: usize,
    /// Ceiling for a retained transcript.
    pub max_transcript_chars: usize,
    /// Whether idle models may be released at all.
    pub allow_model_unloading: bool,
    #[serde(default)]
    pub models: ModelResourceConfig,
    #[serde(default)]
    pub memory: MemoryResourceConfig,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        ResourceConfig {
            low_memory_mode: true,
            monitor_interval_seconds: 5,
            reserve_system_ram_mb: 1024,
            reserve_gpu_vram_mb: 768,
            low_memory_free_percent: 12,
            low_memory_recover_percent: 20,
            max_audio_buffer_seconds: 10,
            max_pending_requests: 4,
            max_conversation_turns_in_context: 8,
            max_tool_output_bytes: 65_536,
            max_transcript_chars: 12_000,
            allow_model_unloading: true,
            models: ModelResourceConfig::default(),
            memory: MemoryResourceConfig::default(),
        }
    }
}

/// A registered application AARON is allowed to launch or stop.
///
/// Only the name is user-selectable. The program and argument vector are fixed
/// here, so a spoken or typed command can never influence how a process is
/// started.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ApplicationSpec {
    /// Executable name or absolute path.
    pub program: String,
    /// Fixed arguments. Never built from user input.
    #[serde(default)]
    pub args: Vec<String>,
    /// Signal number used to stop the application; `None` means SIGTERM (15).
    pub stop_signal: Option<i32>,
}

/// Signal used to stop an application when the configuration does not name
/// one: SIGTERM.
pub const DEFAULT_STOP_SIGNAL: i32 = 15;

/// Highest signal number AARON is willing to send to a process.
///
/// Linux reserves 1..=31 for the classic signals and uses 32..=64 for the
/// real-time ones, so 64 is the largest value that can exist.
pub const MAX_STOP_SIGNAL: i32 = 64;

/// Validate a configured stop signal.
///
/// Signal 0 is not a signal at all — it is the POSIX "does this process
/// exist" probe — negative numbers are meaningless, and anything above
/// [`MAX_STOP_SIGNAL`] does not exist on Linux.
pub fn validate_stop_signal(signal: i32) -> Result<()> {
    if signal <= 0 || signal > MAX_STOP_SIGNAL {
        return Err(Error::Config(format!(
            "stop_signal {signal} must be in 1..={MAX_STOP_SIGNAL}"
        )));
    }
    Ok(())
}

/// A registered test suite.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TestSuiteSpec {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Repository-relative directory to run in.
    pub cwd: Option<String>,
}

/// `[tools]`
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolsConfig {
    /// Name -> application definition.
    pub applications: std::collections::BTreeMap<String, ApplicationSpec>,
    /// Name -> fixed test command.
    pub test_suites: std::collections::BTreeMap<String, TestSuiteSpec>,
    /// Program used to hand a URL to the desktop.
    pub url_opener: String,
    /// Program used for `git_status`.
    pub git_binary: String,
    /// Schemes accepted by `open_url`. `http`/`https` are always accepted.
    pub allowed_url_schemes: Vec<String>,
}

impl Default for ToolsConfig {
    fn default() -> Self {
        ToolsConfig {
            applications: default_applications(),
            test_suites: default_test_suites(),
            url_opener: "xdg-open".into(),
            git_binary: "git".into(),
            allowed_url_schemes: vec!["http".into(), "https".into()],
        }
    }
}

/// Two harmless, universally available applications so the registry is not
/// empty on a fresh install. Users add their own in `~/.config/aaron/config.toml`.
fn default_applications() -> std::collections::BTreeMap<String, ApplicationSpec> {
    let mut m = std::collections::BTreeMap::new();
    m.insert(
        "browser".to_string(),
        ApplicationSpec {
            program: "xdg-open".into(),
            args: vec!["https://example.com".into()],
            stop_signal: None,
        },
    );
    m.insert(
        "terminal".to_string(),
        ApplicationSpec {
            program: "x-terminal-emulator".into(),
            args: Vec::new(),
            stop_signal: None,
        },
    );
    m
}

/// Cargo and npm, the two test runners an Arch developer actually uses.
fn default_test_suites() -> std::collections::BTreeMap<String, TestSuiteSpec> {
    let mut m = std::collections::BTreeMap::new();
    m.insert(
        "cargo".to_string(),
        TestSuiteSpec {
            program: "cargo".into(),
            args: vec!["test".into()],
            cwd: None,
        },
    );
    m.insert(
        "cargo-unit".to_string(),
        TestSuiteSpec {
            program: "cargo".into(),
            args: vec!["test".into(), "--lib".into()],
            cwd: None,
        },
    );
    m
}

/// `[assistant]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AssistantConfig {
    /// Display name used in prompts and greetings.
    pub name: String,
    /// When true, any component that would require network access refuses to run.
    pub offline_only: bool,
    /// Accent colour is deliberately not configurable; keep output plain.
    pub greeting_enabled: bool,
}

impl Default for AssistantConfig {
    fn default() -> Self {
        Self {
            name: "AARON".into(),
            offline_only: true,
            greeting_enabled: true,
        }
    }
}

/// Which audio backend to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AudioBackend {
    /// Real microphone capture through CPAL/ALSA.
    #[default]
    Cpal,
    /// A synthetic sine source, used by tests and `--demo`.
    Synthetic,
    /// No capture at all; the agent runs text-only.
    Disabled,
}

/// Checks that every budget is present, finite and actually safe.
///
/// These are budgets, so an unsafe value is rejected at load time rather than
/// discovered when the machine is already out of memory.
impl ResourceConfig {
    pub fn validate(&self) -> Result<()> {
        let bad = |m: &str| Err(Error::Config(format!("resources: {m}")));
        if self.monitor_interval_seconds == 0 {
            return bad("monitor_interval_seconds must be > 0");
        }
        if self.monitor_interval_seconds > 3600 {
            return bad("monitor_interval_seconds must be <= 3600 (once an hour)");
        }
        if self.reserve_system_ram_mb < 128 {
            return bad("reserve_system_ram_mb must be at least 128");
        }
        if self.reserve_gpu_vram_mb < 128 {
            return bad("reserve_gpu_vram_mb must be at least 128");
        }
        if self.low_memory_free_percent == 0 || self.low_memory_free_percent > 90 {
            return bad("low_memory_free_percent must be in 1..=90");
        }
        // The hysteresis band. Without a strict gap the mode oscillates.
        if self.low_memory_recover_percent <= self.low_memory_free_percent {
            return bad(
                "low_memory_recover_percent must be strictly greater than low_memory_free_percent \
                 so the low-memory state has hysteresis",
            );
        }
        if self.low_memory_recover_percent > 100 {
            return bad("low_memory_recover_percent must be <= 100");
        }
        if self.max_audio_buffer_seconds == 0 || self.max_audio_buffer_seconds > 300 {
            return bad("max_audio_buffer_seconds must be in 1..=300");
        }
        if self.max_pending_requests == 0 || self.max_pending_requests > 1024 {
            return bad("max_pending_requests must be in 1..=1024");
        }
        if self.max_conversation_turns_in_context == 0
            || self.max_conversation_turns_in_context > 512
        {
            return bad("max_conversation_turns_in_context must be in 1..=512");
        }
        if self.max_tool_output_bytes < 1024 {
            return bad("max_tool_output_bytes must be at least 1024");
        }
        if self.max_tool_output_bytes > 16 * 1024 * 1024 {
            return bad("max_tool_output_bytes must be <= 16 MiB");
        }
        if self.max_transcript_chars < 64 {
            return bad("max_transcript_chars must be at least 64");
        }
        if self.max_transcript_chars > 1_000_000 {
            return bad("max_transcript_chars must be <= 1,000,000");
        }
        if !self.allow_model_unloading && self.models.unload_idle_models {
            return bad(
                "allow_model_unloading = false contradicts resources.models.unload_idle_models \
                 = true",
            );
        }
        if self.models.idle_unload_seconds == 0 || self.models.idle_unload_seconds > 86_400 {
            return bad("resources.models.idle_unload_seconds must be in 1..=86400");
        }
        if self.memory.sqlite_cache_kib < 64 {
            return bad("resources.memory.sqlite_cache_kib must be at least 64");
        }
        // SQLite refuses a negative cache; an enormous one silently pages the
        // whole database into RAM, which is exactly what this budget forbids.
        if self.memory.sqlite_cache_kib > 262_144 {
            return bad("resources.memory.sqlite_cache_kib must be <= 262144 (256 MiB)");
        }
        if self.memory.max_retrieved_memories == 0 || self.memory.max_retrieved_memories > 1000 {
            return bad("resources.memory.max_retrieved_memories must be in 1..=1000");
        }
        if self.memory.max_memory_content_bytes < 64
            || self.memory.max_memory_content_bytes > 16 * 1024 * 1024
        {
            return bad("resources.memory.max_memory_content_bytes must be in 64..=16777216");
        }
        if self.memory.max_learning_queue_items_in_memory == 0
            || self.memory.max_learning_queue_items_in_memory > 100_000
        {
            return bad(
                "resources.memory.max_learning_queue_items_in_memory must be in 1..=100000",
            );
        }
        Ok(())
    }

    /// Bytes of audio retained in RAM, from the configured rate and seconds.
    pub fn max_audio_buffer_bytes(&self, sample_rate: u32) -> usize {
        (sample_rate as usize)
            .saturating_mul(self.max_audio_buffer_seconds as usize)
            .saturating_mul(4)
    }
}

/// `[audio]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AudioConfig {
    /// `default` or a substring of the device name.
    pub input_device: String,
    /// Capture sample rate. Whisper-family models expect 16 kHz.
    pub sample_rate: u32,
    pub channels: u16,
    /// Frame size in milliseconds for streaming capture.
    pub frame_ms: u32,
    /// Hard cap on a single utterance, independent of VAD.
    pub max_capture_secs: u32,
    /// Recording backend.
    pub backend: AudioBackend,
    /// Where the UI writes a green "microphone live" indicator; empty disables.
    pub activity_indicator_path: String,
    #[serde(default)]
    pub vad: VadConfig,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            input_device: "default".into(),
            sample_rate: 16_000,
            channels: 1,
            frame_ms: 30,
            max_capture_secs: 12,
            backend: AudioBackend::Cpal,
            activity_indicator_path: String::new(),
            vad: VadConfig::default(),
        }
    }
}

/// `[audio.vad]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VadConfig {
    pub enabled: bool,
    /// RMS energy threshold in dBFS; below this the frame is silence.
    pub energy_threshold_dbfs: f32,
    /// Consecutive silence that ends an utterance.
    pub silence_ms: u32,
    /// Speech shorter than this is discarded as a click or pop.
    pub min_speech_ms: u32,
    /// Fraction of frames above threshold required to declare speech onset.
    pub onset_ratio: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            energy_threshold_dbfs: -42.0,
            silence_ms: 700,
            min_speech_ms: 200,
            onset_ratio: 0.30,
        }
    }
}

/// Wake detection strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum WakeStrategy {
    /// Match the wake phrase against the ASR transcript of a short utterance.
    #[default]
    Transcript,
    /// Trigger on a sustained energy burst. Prone to false activations; useful
    /// only when no ASR backend is installed.
    Energy,
    /// Delegate to an external wake-word engine configured by the user.
    External,
}

/// `[wake]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WakeConfig {
    pub enabled: bool,
    pub phrase: String,
    pub strategy: WakeStrategy,
    /// How long after wake detection the assistant listens for a command.
    pub command_window_secs: u32,
    /// Similarity (0..1) required between the spoken phrase and the configured
    /// wake phrase for a transcript match.
    pub match_threshold: f32,
}

impl Default for WakeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            phrase: "hey aaron".into(),
            strategy: WakeStrategy::Transcript,
            command_window_secs: 8,
            match_threshold: 0.72,
        }
    }
}

/// ASR backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum AsrBackend {
    /// Use whisper.cpp's `whisper-cli` if present, otherwise fall back.
    #[default]
    Auto,
    /// Require whisper.cpp; fail loudly when absent.
    WhisperCpp,
    /// Require the Phonon engine (Fermion Research). Real speech
    /// recognition, on an ordinary CPU.
    Phonon,
    /// Deterministic constrained recogniser. Recognises only phrases built from
    /// the command grammar. Explicitly *not* general speech recognition.
    Constrained,
    /// No ASR at all.
    Disabled,
}

/// `[asr]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AsrConfig {
    pub backend: AsrBackend,
    /// Path to a ggml Whisper model file. Never downloaded automatically.
    pub model_path: String,
    /// The model a *named* backend should use. Phonon names its models
    /// (`phonon-2`, `phonon-1-micro`, …) and fetches them itself; whisper.cpp
    /// takes a file path instead, and uses `model_path`.
    pub model: String,
    /// Expected SHA-256 of the model file. Empty disables checksum checking.
    pub model_sha256: String,
    /// Absolute path or bare name resolved through `PATH`.
    pub binary_path: String,
    /// Names and terms the recogniser should favour, comma separated.
    ///
    /// Empty means "use the wake phrase". A recogniser that mishears the
    /// wake word does not wake the assistant: "hey aaron" heard as "hey
    /// erin" is a different phrase, and the wake matcher will not fire on
    /// it. Telling the recogniser what the word is supposed to be is the
    /// difference between a working wake word and one that only works when
    /// the room is silent.
    pub hotwords: String,
    /// The Phonon engine's command, resolved through `PATH`.
    ///
    /// Separate from `binary_path` because the two backends ship as
    /// different programs: `fermion` with the `fermion-research` package,
    /// `whisper-cli` with whisper.cpp.
    pub phonon_binary_path: String,
    pub language: String,
    /// Number of inference threads passed to the backend.
    pub threads: u32,
    /// Pass GPU flags to whisper.cpp. Ignored when CUDA is unavailable.
    pub use_gpu: bool,
}

impl Default for AsrConfig {
    fn default() -> Self {
        Self {
            backend: AsrBackend::Auto,
            model_path: String::new(),
            model: "phonon-2".into(),
            model_sha256: String::new(),
            binary_path: "whisper-cli".into(),
            hotwords: String::new(),
            phonon_binary_path: "fermion".into(),
            language: "en".into(),
            threads: 4,
            use_gpu: false,
        }
    }
}

/// TTS backend selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TtsBackend {
    /// Pick the first installed of espeak-ng, piper, spd-say.
    #[default]
    Auto,
    /// `espeak-ng` (Arch package `espeak-ng`). Small, robotic, no model file.
    Espeak,
    /// `piper` (Arch package `piper-tts` or a manual build). Neural, local.
    Piper,
    /// speech-dispatcher `spd-say`.
    SpdSay,
    Disabled,
}

/// `[tts]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TtsConfig {
    pub backend: TtsBackend,
    pub binary_path: String,
    /// espeak-ng voice such as `en-us`; piper voice name or path.
    pub voice: String,
    /// espeak-ng words per minute (roughly 80-450).
    pub rate_wpm: u32,
    /// Piper ONNX model path. Never downloaded automatically.
    pub model_path: String,
    /// Also mirror spoken output to the terminal.
    pub echo_to_terminal: bool,
}

impl Default for TtsConfig {
    fn default() -> Self {
        Self {
            backend: TtsBackend::Auto,
            binary_path: String::new(),
            voice: "en-us".into(),
            rate_wpm: 175,
            model_path: String::new(),
            echo_to_terminal: true,
        }
    }
}

/// Intent classification backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum BrainBackend {
    /// Use the neural checkpoint when present, else the deterministic rules.
    #[default]
    Auto,
    /// Require the 1M-parameter checkpoint.
    Neural,
    /// Deterministic keyword/grammar router only.
    Heuristic,
}

/// `[brain]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct BrainConfig {
    pub backend: BrainBackend,
    /// Path to the trained checkpoint. Empty means "search the model directory".
    pub checkpoint_path: String,
    /// Softmax probabilities below this become `unknown`.
    pub confidence_threshold: f32,
    /// Logit temperature applied before softmax.
    pub temperature: f32,
    /// Maximum number of hashed features per utterance.
    pub max_features: usize,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            backend: BrainBackend::Auto,
            checkpoint_path: String::new(),
            // Selected on the validation split in release 0.3. 0.55 and 0.60
            // are *exactly tied* on validation accuracy, macro-F1 and
            // false-accept rate, so the tie-break is the more conservative
            // threshold: it cannot reduce measured validation performance and
            // it abstains more often on out-of-schema input. The full sweep
            // is in `training/threshold-sweep.jsonl`.
            //
            // This threshold is an *abstention* control, not a safety
            // control: the release-0.3 sweep showed the model's errors on
            // adversarial input are confident, so no threshold reaches them.
            // See docs/release-0.3-report.md §7.
            confidence_threshold: 0.60,
            temperature: 1.0,
            max_features: 256,
        }
    }
}

/// `[runtime]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RuntimeConfig {
    /// How long a confirmation prompt stays open before auto-cancelling.
    pub confirm_timeout_secs: u32,
    /// Ceiling for any single tool invocation.
    pub max_tool_runtime_secs: u32,
    /// Ceiling for captured tool output, in bytes.
    pub max_output_bytes: usize,
    /// The agent refuses to run more than this many tools for one utterance.
    pub max_tool_calls_per_utterance: usize,
    /// Repeating the *same* action more than this many times is blocked.
    pub max_repeat_identical_actions: usize,
    /// Idle shutdown after N consecutive ASR errors (0 disables).
    pub abort_after_consecutive_errors: usize,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            confirm_timeout_secs: 60,
            max_tool_runtime_secs: 30,
            max_output_bytes: 64 * 1024,
            max_tool_calls_per_utterance: 1,
            max_repeat_identical_actions: 2,
            abort_after_consecutive_errors: 5,
        }
    }
}

/// `[logging]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LoggingConfig {
    /// `error`, `warn`, `info`, `debug`, `trace`.
    pub level: String,
    /// `text` or `json`.
    pub format: String,
    /// Optional log file. Empty logs to stderr only.
    pub file: String,
    /// Retain at most this many bytes per log file before rotation.
    pub max_file_bytes: u64,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            level: "info".into(),
            format: "text".into(),
            file: String::new(),
            max_file_bytes: 8 * 1024 * 1024,
        }
    }
}

/// `[memory]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MemoryConfig {
    pub database_path: String,
    /// Delete non-pinned entries older than this. 0 disables automatic cleanup.
    pub retention_days: u32,
    /// Soft cap on stored entries; oldest unpinned entries are pruned first.
    pub max_entries: usize,
    /// Copy the database next to itself at startup.
    pub backup_on_start: bool,
    /// Maximum characters stored in a single memory value.
    pub max_value_chars: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            database_path: "data/memory.db".into(),
            retention_days: 365,
            max_entries: 10_000,
            backup_on_start: true,
            max_value_chars: 4_000,
        }
    }
}

/// `[learning]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LearningConfig {
    pub enabled: bool,
    /// JSONL file holding corrections awaiting review.
    pub queue_path: String,
    /// Dataset currently used for training.
    pub dataset_path: String,
    /// Promote a candidate automatically when thresholds are met. Off by default:
    /// promotion is an explicit user action.
    pub auto_promote: bool,
    pub min_macro_f1: f64,
    pub min_accuracy: f64,
    /// Drop examples whose text matches these substrings during sanitisation.
    pub redact_patterns: Vec<String>,
}

impl Default for LearningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            queue_path: "data/learning/pending.jsonl".into(),
            dataset_path: "training/dataset/current.jsonl".into(),
            auto_promote: false,
            min_macro_f1: 0.92,
            min_accuracy: 0.93,
            redact_patterns: vec![
                "password".into(),
                "api_key".into(),
                "api-key".into(),
                "secret".into(),
                "bearer ".into(),
                "ssn".into(),
            ],
        }
    }
}

/// `[security]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SecurityConfig {
    /// Path to the policy file, resolved relative to the configuration file's
    /// own directory rather than the working directory.
    pub permissions_file: String,
    /// Directories the file tools may touch. `~` is expanded.
    pub allowed_roots: Vec<String>,
    /// Always ask before running a tool in this list, regardless of policy.
    pub always_confirm: Vec<String>,
    /// Master switch for outbound network access.
    pub allow_network: bool,
    /// Refuse to run anything as root. Keep true.
    pub refuse_root: bool,
    /// Append-only audit log path. Empty disables auditing (not recommended).
    pub audit_log_path: String,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            permissions_file: "permissions.toml".into(),
            allowed_roots: vec![],
            always_confirm: vec![
                "open_url".into(),
                "start_application".into(),
                "stop_application".into(),
                "run_tests".into(),
                "read_file".into(),
            ],
            allow_network: false,
            refuse_root: true,
            audit_log_path: "data/audit/audit.jsonl".into(),
        }
    }
}

/// `[limits]`
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimitsConfig {
    pub max_entity_chars: usize,
    /// Maximum bytes returned by `read_file`.
    pub max_read_bytes: usize,
    /// Maximum number of paths returned by `search_files`.
    pub max_search_results: usize,
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            max_entity_chars: 2000,
            max_read_bytes: 64 * 1024,
            max_search_results: 200,
        }
    }
}

impl Config {
    /// Parse a configuration from TOML text.
    pub fn from_toml(text: &str) -> Result<Config> {
        let cfg: Config = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Read and validate a configuration file.
    pub fn load(path: impl AsRef<Path>) -> Result<Config> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let cfg = Config::from_toml(&text).map_err(|e| match e {
            Error::Config(msg) => Error::Config(format!("{}: {}", path.display(), msg)),
            other => other,
        })?;
        Ok(cfg)
    }

    /// Serialise back to TOML.
    pub fn to_toml(&self) -> Result<String> {
        toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))
    }

    /// Cross-field validation. Catches combinations that would otherwise fail
    /// deep inside a tool at 3am.
    pub fn validate(&self) -> Result<()> {
        if self.audio.sample_rate < 8_000 || self.audio.sample_rate > 192_000 {
            return Err(Error::Config(format!(
                "audio.sample_rate {} is outside 8000..=192000",
                self.audio.sample_rate
            )));
        }
        if self.audio.channels == 0 || self.audio.channels > 2 {
            return Err(Error::Config("audio.channels must be 1 or 2".into()));
        }
        if self.audio.frame_ms == 0 {
            return Err(Error::Config("audio.frame_ms must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.brain.confidence_threshold) {
            return Err(Error::Config(
                "brain.confidence_threshold must be in 0..=1".into(),
            ));
        }
        if self.brain.temperature <= 0.0 || !self.brain.temperature.is_finite() {
            return Err(Error::Config(
                "brain.temperature must be a positive finite number".into(),
            ));
        }
        if self.brain.max_features == 0 {
            return Err(Error::Config("brain.max_features must be > 0".into()));
        }
        if !(0.0..=1.0).contains(&self.wake.match_threshold) {
            return Err(Error::Config(
                "wake.match_threshold must be in 0..=1".into(),
            ));
        }
        for (name, app) in &self.tools.applications {
            if app.program.trim().is_empty() {
                return Err(Error::Config(format!(
                    "tools.applications.{name}.program must not be empty"
                )));
            }
            if app.args.iter().any(|a| a.contains('\0')) {
                return Err(Error::Config(format!(
                    "tools.applications.{name}.args contains a NUL byte"
                )));
            }
            // A bad stop signal is caught here, at startup, rather than
            // deep inside a tool when the user finally asks to stop it.
            if let Some(signal) = app.stop_signal {
                validate_stop_signal(signal).map_err(|_| {
                    Error::Config(format!(
                        "tools.applications.{name}.stop_signal {signal} must be in 1..={MAX_STOP_SIGNAL}"
                    ))
                })?;
            }
        }
        for (name, suite) in &self.tools.test_suites {
            if suite.program.trim().is_empty() {
                return Err(Error::Config(format!(
                    "tools.test_suites.{name}.program must not be empty"
                )));
            }
            // A test suite is never allowed to be a shell pipeline.
            if suite
                .args
                .iter()
                .any(|a| a.contains(';') || a.contains('|') || a.contains('$'))
            {
                return Err(Error::Config(format!(
                    "tools.test_suites.{name}.args must be a fixed argument vector, \
                     not a shell fragment"
                )));
            }
        }
        if self.tools.url_opener.trim().is_empty() {
            return Err(Error::Config("tools.url_opener must not be empty".into()));
        }
        if self.wake.phrase.trim().is_empty() {
            return Err(Error::Config("wake.phrase must not be empty".into()));
        }
        if self.runtime.max_tool_runtime_secs == 0 {
            return Err(Error::Config(
                "runtime.max_tool_runtime_secs must be > 0".into(),
            ));
        }
        if self.runtime.max_output_bytes == 0 {
            return Err(Error::Config("runtime.max_output_bytes must be > 0".into()));
        }
        if self.runtime.max_tool_calls_per_utterance == 0 {
            return Err(Error::Config(
                "runtime.max_tool_calls_per_utterance must be > 0 (0 would disable tools)".into(),
            ));
        }
        if self.limits.max_entity_chars == 0 {
            return Err(Error::Config("limits.max_entity_chars must be > 0".into()));
        }
        if self.tts.rate_wpm < 80 || self.tts.rate_wpm > 450 {
            return Err(Error::Config("tts.rate_wpm must be in 80..=450".into()));
        }
        self.resources.validate()?;
        match self.logging.level.as_str() {
            "error" | "warn" | "info" | "debug" | "trace" => {}
            other => return Err(Error::Config(format!("logging.level `{other}` is invalid"))),
        }
        match self.logging.format.as_str() {
            "text" | "json" => {}
            other => {
                return Err(Error::Config(format!(
                    "logging.format `{other}` is invalid"
                )))
            }
        }
        Ok(())
    }

    /// Resolve a possibly relative, possibly `~`-prefixed path against `base`.
    pub fn resolve(base: &Path, raw: &str) -> PathBuf {
        let expanded = if let Some(rest) = raw.strip_prefix("~/") {
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(home).join(rest);
            }
            PathBuf::from(rest)
        } else if raw == "~" {
            std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from(raw))
        } else {
            PathBuf::from(raw)
        };
        if expanded.is_absolute() {
            expanded
        } else {
            base.join(expanded)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_validate() {
        Config::default()
            .validate()
            .expect("defaults must be valid");
    }

    #[test]
    fn rejects_unknown_keys() {
        let err = Config::from_toml("[brian]\nbackend = \"neural\"\n").unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn an_out_of_range_stop_signal_is_a_config_error() {
        let mut cfg = Config::default();
        cfg.tools.applications.insert(
            "custom".to_string(),
            ApplicationSpec {
                program: "xdg-open".into(),
                args: Vec::new(),
                stop_signal: Some(0),
            },
        );
        let err = cfg.validate().unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(
            err.to_string()
                .contains("tools.applications.custom.stop_signal"),
            "unexpected message: {err}"
        );

        for bad in [-1, 65] {
            cfg.tools
                .applications
                .get_mut("custom")
                .expect("the application was just inserted")
                .stop_signal = Some(bad);
            let err = cfg.validate().unwrap_err();
            assert!(matches!(err, Error::Config(_)), "{bad} should be rejected");
        }
    }

    #[test]
    fn a_valid_stop_signal_passes_validation() {
        let mut cfg = Config::default();
        cfg.tools.applications.insert(
            "custom".to_string(),
            ApplicationSpec {
                program: "xdg-open".into(),
                args: Vec::new(),
                stop_signal: Some(9),
            },
        );
        cfg.validate().expect("signal 9 is a valid signal");

        cfg.tools
            .applications
            .get_mut("custom")
            .expect("the application was just inserted")
            .stop_signal = None;
        cfg.validate()
            .expect("no signal means the default, SIGTERM");
    }

    #[test]
    fn rejects_out_of_range_threshold() {
        let cfg = Config::default();
        let text = cfg.to_toml().unwrap();
        // Replace the threshold with an illegal value, whatever it currently
        // is. Hard-coding the default's string form made this test silently
        // pass whenever the default changed — it was checking that the
        // replacement had happened, not that the config was rejected.
        let replaced = text
            .lines()
            .map(|line| {
                if line.trim_start().starts_with("confidence_threshold") {
                    "confidence_threshold = 1.5"
                } else {
                    line
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        assert_ne!(
            replaced, text,
            "the threshold line was not found; the test would prove nothing"
        );
        let err = Config::from_toml(&replaced).unwrap_err();
        assert!(err.to_string().contains("confidence_threshold"));
    }

    #[test]
    fn round_trips_through_toml() {
        let cfg = Config::default();
        let text = cfg.to_toml().unwrap();
        let back = Config::from_toml(&text).unwrap();
        assert_eq!(cfg, back);
    }

    #[test]
    fn partial_config_fills_defaults() {
        let cfg = Config::from_toml("[assistant]\nname = \"Test\"\n").unwrap();
        assert_eq!(cfg.assistant.name, "Test");
        assert_eq!(cfg.audio.sample_rate, 16_000);
    }

    #[test]
    fn resolve_expands_home_and_relative() {
        let base = Path::new("/tmp/aaron-base");
        assert_eq!(
            Config::resolve(base, "/abs/path"),
            PathBuf::from("/abs/path")
        );
        assert_eq!(
            Config::resolve(base, "rel/x"),
            PathBuf::from("/tmp/aaron-base/rel/x")
        );
        let home = Config::resolve(base, "~/x");
        assert!(home.is_absolute());
    }

    // ------------------------------------------------------- resource budgets

    #[test]
    fn resource_defaults_validate() {
        ResourceConfig::default()
            .validate()
            .expect("shipped resource defaults must be valid");
    }

    /// Builds a budget with a given trigger/recovery pair.
    fn hysteresis(trigger: u8, recover: u8) -> ResourceConfig {
        ResourceConfig {
            low_memory_free_percent: trigger,
            low_memory_recover_percent: recover,
            ..Default::default()
        }
    }

    #[test]
    fn resource_hysteresis_is_required() {
        let err = hysteresis(20, 20).validate().unwrap_err();
        assert!(err.to_string().contains("hysteresis"));

        assert!(
            hysteresis(20, 19).validate().is_err(),
            "recovery below the trigger is not hysteresis"
        );
        assert!(hysteresis(20, 21).validate().is_ok());
    }

    #[test]
    fn unsafe_resource_values_are_rejected() {
        /// One mutation that must be rejected, with the name it sets.
        type Case = (&'static str, Box<dyn Fn(&mut ResourceConfig)>);
        let cases: Vec<Case> = vec![
            (
                "monitor_interval_seconds",
                Box::new(|r: &mut ResourceConfig| r.monitor_interval_seconds = 0),
            ),
            (
                "monitor_interval_seconds",
                Box::new(|r: &mut ResourceConfig| r.monitor_interval_seconds = 100_000),
            ),
            (
                "reserve_system_ram_mb",
                Box::new(|r: &mut ResourceConfig| r.reserve_system_ram_mb = 8),
            ),
            (
                "reserve_gpu_vram_mb",
                Box::new(|r: &mut ResourceConfig| r.reserve_gpu_vram_mb = 0),
            ),
            (
                "max_audio_buffer_seconds",
                Box::new(|r: &mut ResourceConfig| r.max_audio_buffer_seconds = 0),
            ),
            (
                "max_audio_buffer_seconds",
                Box::new(|r: &mut ResourceConfig| r.max_audio_buffer_seconds = 10_000),
            ),
            (
                "max_pending_requests",
                Box::new(|r: &mut ResourceConfig| r.max_pending_requests = 0),
            ),
            (
                "max_conversation_turns_in_context",
                Box::new(|r: &mut ResourceConfig| r.max_conversation_turns_in_context = 0),
            ),
            (
                "max_tool_output_bytes",
                Box::new(|r: &mut ResourceConfig| r.max_tool_output_bytes = 10),
            ),
            (
                "max_tool_output_bytes",
                Box::new(|r: &mut ResourceConfig| r.max_tool_output_bytes = 64 * 1024 * 1024),
            ),
            (
                "max_transcript_chars",
                Box::new(|r: &mut ResourceConfig| r.max_transcript_chars = 4),
            ),
            (
                "sqlite_cache_kib",
                Box::new(|r: &mut ResourceConfig| r.memory.sqlite_cache_kib = 1),
            ),
            (
                "sqlite_cache_kib",
                Box::new(|r: &mut ResourceConfig| r.memory.sqlite_cache_kib = 1_000_000),
            ),
            (
                "max_retrieved_memories",
                Box::new(|r: &mut ResourceConfig| r.memory.max_retrieved_memories = 0),
            ),
            (
                "max_memory_content_bytes",
                Box::new(|r: &mut ResourceConfig| r.memory.max_memory_content_bytes = 2),
            ),
            (
                "idle_unload_seconds",
                Box::new(|r: &mut ResourceConfig| r.models.idle_unload_seconds = 0),
            ),
        ];
        for (name, mutate) in cases {
            let mut r = ResourceConfig::default();
            mutate(&mut r);
            let err = r.validate();
            assert!(
                err.is_err(),
                "`{name}` with that value should have been rejected"
            );
            assert!(
                err.unwrap_err().to_string().contains("resources:"),
                "error for `{name}` should name the resources section"
            );
        }
    }

    #[test]
    fn contradictory_unloading_settings_are_rejected() {
        let unloading = |unload_idle_models| ResourceConfig {
            allow_model_unloading: false,
            models: ModelResourceConfig {
                unload_idle_models,
                ..Default::default()
            },
            ..Default::default()
        };
        assert!(unloading(true).validate().is_err());
        assert!(unloading(false).validate().is_ok());
    }

    #[test]
    fn audio_buffer_budget_scales_with_the_sample_rate() {
        let r = ResourceConfig::default();
        assert_eq!(r.max_audio_buffer_seconds, 10);
        assert_eq!(r.max_audio_buffer_bytes(16_000), 10 * 16_000 * 4);
        assert_eq!(r.max_audio_buffer_bytes(48_000), 10 * 48_000 * 4);
    }

    #[test]
    fn resources_survive_a_toml_round_trip() {
        let mut cfg = Config::default();
        cfg.resources.reserve_system_ram_mb = 2048;
        cfg.resources.memory.sqlite_cache_kib = 4096;
        let text = cfg.to_toml().unwrap();
        let back = Config::from_toml(&text).unwrap();
        assert_eq!(cfg, back);
        assert_eq!(back.resources.reserve_system_ram_mb, 2048);
        assert_eq!(back.resources.memory.sqlite_cache_kib, 4096);
    }

    #[test]
    fn an_unsafe_resource_block_in_a_file_is_a_config_error() {
        let cfg = Config::default();
        let mut text = cfg.to_toml().unwrap();
        text = text.replace("max_pending_requests = 4", "max_pending_requests = 0");
        let err = Config::from_toml(&text).unwrap_err();
        assert!(err.to_string().contains("max_pending_requests"));
    }
}
