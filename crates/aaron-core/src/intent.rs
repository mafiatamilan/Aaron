//! The intent schema, entity extraction results and the classifier interface.
//!
//! The intent enum is the single contract between the custom 1M-parameter brain
//! and every downstream component. The set is closed on purpose: adding an
//! intent changes the classification head and therefore the parameter budget,
//! which is enforced by a test in `aaron-brain`.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Number of classes in the intent schema. The classifier output head is sized
/// from this constant, so it is part of the parameter budget.
pub const INTENT_COUNT: usize = 14;

/// A closed set of command intents understood by AARON.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Intent {
    /// Greeting / small talk that requires no tool.
    Greeting,
    /// "what can you do?"
    Help,
    /// Report CPU/RAM/GPU/audio health.
    SystemStatus,
    /// Persist a user-approved fact.
    Remember,
    /// Retrieve previously stored facts.
    Recall,
    /// Find files by name/glob inside allowed roots.
    SearchFiles,
    /// Read the first N bytes of a text file inside allowed roots.
    ReadFile,
    /// Show git working-tree status for a repository.
    GitStatus,
    /// Run a named, pre-registered test command in a project.
    RunTests,
    /// Open a URL in the user's browser.
    OpenUrl,
    /// Launch a registered desktop application.
    StartApplication,
    /// Stop a registered desktop application.
    StopApplication,
    /// Abandon whatever is currently in flight.
    Cancel,
    /// Nothing matched above the confidence threshold.
    Unknown,
}

impl Intent {
    /// All intents in canonical order. Index in this slice is the class index
    /// used by the classifier output head and by the dataset label encoding.
    pub const ALL: [Intent; INTENT_COUNT] = [
        Intent::Greeting,
        Intent::Help,
        Intent::SystemStatus,
        Intent::Remember,
        Intent::Recall,
        Intent::SearchFiles,
        Intent::ReadFile,
        Intent::GitStatus,
        Intent::RunTests,
        Intent::OpenUrl,
        Intent::StartApplication,
        Intent::StopApplication,
        Intent::Cancel,
        Intent::Unknown,
    ];

    /// Stable snake_case identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Intent::Greeting => "greeting",
            Intent::Help => "help",
            Intent::SystemStatus => "system_status",
            Intent::Remember => "remember",
            Intent::Recall => "recall",
            Intent::SearchFiles => "search_files",
            Intent::ReadFile => "read_file",
            Intent::GitStatus => "git_status",
            Intent::RunTests => "run_tests",
            Intent::OpenUrl => "open_url",
            Intent::StartApplication => "start_application",
            Intent::StopApplication => "stop_application",
            Intent::Cancel => "cancel",
            Intent::Unknown => "unknown",
        }
    }

    /// Class index for the model output head.
    pub fn index(self) -> usize {
        match self {
            Intent::Greeting => 0,
            Intent::Help => 1,
            Intent::SystemStatus => 2,
            Intent::Remember => 3,
            Intent::Recall => 4,
            Intent::SearchFiles => 5,
            Intent::ReadFile => 6,
            Intent::GitStatus => 7,
            Intent::RunTests => 8,
            Intent::OpenUrl => 9,
            Intent::StartApplication => 10,
            Intent::StopApplication => 11,
            Intent::Cancel => 12,
            Intent::Unknown => 13,
        }
    }

    /// Inverse of [`Intent::index`].
    pub fn from_index(idx: usize) -> Option<Intent> {
        Intent::ALL.get(idx).copied()
    }

    /// Parse a stable identifier back into an intent.
    pub fn parse(s: &str) -> Option<Intent> {
        Intent::ALL.into_iter().find(|i| i.as_str() == s)
    }

    /// Does this intent request an executable tool action?
    ///
    /// This is the canonical answer, and the only one. Three definitions used
    /// to disagree:
    ///
    /// * `Intent::needs_tool` said `cancel` was not actionable;
    /// * `safety::action_tier` said it was, and read-only;
    /// * `tool_for_intent` mapped it to a real registered tool.
    ///
    /// `cancel` is actionable: the agent runs a tool for it, and the tool is
    /// registered and read-only. `greeting`, `help` and `unknown` request no
    /// action at all — the agent answers them in words.
    pub fn requests_action(self) -> bool {
        !matches!(self, Intent::Greeting | Intent::Help | Intent::Unknown)
    }

    /// A short, honest description used by `aaron tools list` and `aaron help`.
    pub fn description(self) -> &'static str {
        match self {
            Intent::Greeting => "Greet the assistant",
            Intent::Help => "Describe available capabilities",
            Intent::SystemStatus => "Report CPU, memory, GPU and audio health",
            Intent::Remember => "Store a user-approved fact in long-term memory",
            Intent::Recall => "Search long-term memory",
            Intent::SearchFiles => "Find files matching a pattern inside allowed roots",
            Intent::ReadFile => "Read the first bytes of a text file",
            Intent::GitStatus => "Show the git status of a repository",
            Intent::RunTests => "Run a registered test command in a project",
            Intent::OpenUrl => "Open a URL in the default browser",
            Intent::StartApplication => "Launch a registered application",
            Intent::StopApplication => "Terminate a registered application",
            Intent::Cancel => "Abort the current operation",
            Intent::Unknown => "Input did not map to a known command",
        }
    }
}

impl fmt::Display for Intent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Entities extracted from an utterance.
///
/// Every field is bounded in length by the entity extractor so that a hostile
/// transcript cannot inflate tool arguments beyond the policy limits.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entities {
    /// Free text payload: the thing to remember, the search query, the gist of
    /// the request. Bounded by `config.limits.max_entity_chars`.
    #[serde(default)]
    pub text: Option<String>,
    /// A filesystem path or glob, already validated for shape (sandbox
    /// resolution happens later, inside the tool).
    #[serde(default)]
    pub path: Option<String>,
    /// An application name; must resolve against the configured application
    /// allow-list before any process is spawned.
    #[serde(default)]
    pub application: Option<String>,
    /// An `http`/`https` URL.
    #[serde(default)]
    pub url: Option<String>,
    /// A named, registered test suite (never a raw shell command).
    #[serde(default)]
    pub suite: Option<String>,
    /// Explicit count argument, e.g. "read the first 20 lines".
    #[serde(default)]
    pub count: Option<u32>,
}

impl Entities {
    /// Reject oversized entity payloads and NUL/control characters.
    ///
    /// This runs before *any* tool dispatch. Tool implementations repeat the
    /// check, but the agent refuses to build a command at all if it fails.
    pub fn validate(&self, max_chars: usize) -> crate::Result<()> {
        let fields: [(&str, &Option<String>); 5] = [
            ("text", &self.text),
            ("path", &self.path),
            ("application", &self.application),
            ("url", &self.url),
            ("suite", &self.suite),
        ];
        for (name, value) in fields {
            if let Some(v) = value {
                if v.chars().count() > max_chars {
                    return Err(crate::Error::LimitExceeded(format!(
                        "entity `{name}` is {} chars, limit is {max_chars}",
                        v.chars().count()
                    )));
                }
                if v.contains('\0') {
                    return Err(crate::Error::LimitExceeded(format!(
                        "entity `{name}` contains a NUL byte"
                    )));
                }
            }
        }
        if let Some(url) = &self.url {
            if !(url.starts_with("http://") || url.starts_with("https://")) {
                return Err(crate::Error::LimitExceeded(
                    "only http and https URLs are allowed".into(),
                ));
            }
        }
        Ok(())
    }
}

/// Whether a classified request carries enough information to act on.
///
/// This is a *second, independent* output of the classifier. The intent says
/// **what** the user wants; the readiness says **whether the request says
/// enough to do it**.
///
/// # Readiness is not authorisation
///
/// A readiness verdict can only ever turn "proceed" into "ask". It can never
/// turn "refuse" into "proceed", and it never grants a permission the
/// argument validator, the policy engine, the allow-list and the confirmation
/// gate would not have granted anyway. A request whose readiness output is
/// missing or unusable is treated as having no verdict, which changes
/// nothing: the remaining controls decide.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Readiness {
    /// The request names everything the operation needs.
    Ready,
    /// A required argument or reference is unresolved. The assistant should
    /// ask rather than guess.
    NeedsClarification,
    /// The utterance does not request an executable tool action.
    NotApplicable,
    /// No readiness verdict is available.
    ///
    /// A backend that does not produce one, a payload written before the
    /// field existed, and a verdict that failed to decode all report this.
    /// It is not an error and it is not a permission.
    #[default]
    Undetermined,
}

impl Readiness {
    /// A short, stable identifier for reports and JSON output.
    pub fn as_str(self) -> &'static str {
        match self {
            Readiness::Ready => "ready",
            Readiness::NeedsClarification => "needs_clarification",
            Readiness::NotApplicable => "not_applicable",
            Readiness::Undetermined => "undetermined",
        }
    }

    /// Parse a stable identifier back into a verdict.
    pub fn parse(text: &str) -> Option<Readiness> {
        match text {
            "ready" => Some(Readiness::Ready),
            "needs_clarification" => Some(Readiness::NeedsClarification),
            "not_applicable" => Some(Readiness::NotApplicable),
            "undetermined" => Some(Readiness::Undetermined),
            _ => None,
        }
    }

    /// Does this verdict ask the assistant to ask a question?
    pub fn needs_clarification(self) -> bool {
        matches!(self, Readiness::NeedsClarification)
    }
}

impl fmt::Display for Readiness {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A classified command: intent, entities and the classifier's confidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Command {
    /// Predicted intent.
    pub intent: Intent,
    /// Extracted entities (empty when the intent carries no argument).
    pub entities: Entities,
    /// Softmax probability of `intent`, in `[0, 1]`.
    pub confidence: f32,
    /// Which backend produced this command.
    pub source: ClassifySource,
    /// Whether the request is sufficiently specified to act on.
    ///
    /// `Undetermined` when the backend has no readiness output, which is not
    /// the same as `Ready`: a missing verdict grants nothing.
    #[serde(default)]
    pub readiness: Readiness,
}

impl Default for Command {
    fn default() -> Self {
        Command {
            intent: Intent::Unknown,
            entities: Entities::default(),
            confidence: 0.0,
            source: ClassifySource::Explicit,
            readiness: Readiness::Undetermined,
        }
    }
}

impl Command {
    /// Build an `unknown` command carrying the original text.
    pub fn unknown(text: impl Into<String>, source: ClassifySource) -> Self {
        Command {
            intent: Intent::Unknown,
            entities: Entities {
                text: Some(text.into()),
                ..Default::default()
            },
            confidence: 0.0,
            source,
            // The utterance could not be classified, so there is no verdict
            // on whether it is complete either.
            readiness: Readiness::Undetermined,
        }
    }
}

/// Provenance of a classification, so the assistant can be honest about whether
/// a neural model or a deterministic rule made the decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClassifySource {
    /// The custom 1M-parameter neural intent model.
    Neural,
    /// The deterministic keyword/grammar router.
    Heuristic,
    /// Explicitly supplied by the caller (CLI `aaron ask --intent ...`).
    Explicit,
}

impl ClassifySource {
    pub fn as_str(self) -> &'static str {
        match self {
            ClassifySource::Neural => "neural",
            ClassifySource::Heuristic => "heuristic",
            ClassifySource::Explicit => "explicit",
        }
    }
}

impl fmt::Display for ClassifySource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Pluggable intent classification.
///
/// Both the neural brain and the deterministic fallback implement this trait so
/// the agent never depends on Candle, and so tests can inject a stub.
pub trait IntentClassifier: Send + Sync {
    /// Human readable backend name, e.g. `aaron-brain/1M` or `heuristic-rules`.
    fn name(&self) -> &'static str;

    /// Classify a raw utterance.
    fn classify(&self, utterance: &str) -> crate::Result<Command>;

    /// Whether this backend needs a trained checkpoint on disk.
    fn requires_checkpoint(&self) -> bool {
        false
    }
}

/// Shared description of a classifier's confidence calibration, shown by
/// `aaron model info`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Calibration {
    /// Probabilities below this value are coerced to `Intent::Unknown`.
    pub confidence_threshold: f32,
    /// Softmax temperature applied to logits before thresholding.
    pub temperature: f32,
}
