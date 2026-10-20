//! The readiness contract: is a request specific enough to act on?
//!
//! This module is the single source of truth for two questions that used to
//! be answered in two places, with two answers:
//!
//! 1. **Which arguments does an intent need?** The agent's argument builder
//!    (`aaron-agent::agent::build_arguments`) knows, but the classifier
//!    (`aaron-brain`) cannot call it: the dependency runs the other way.
//! 2. **Is a request complete?** The agent decides by trying to build the
//!    arguments; a classifier that wants to report a readiness verdict has to
//!    answer the same question without being able to build the arguments.
//!
//! So the answer lives here, in the crate both depend on, and the agent's
//! builder is checked against it by a test that asserts the two agree on
//! every intent (`aaron-app/tests/readiness.rs`).
//!
//! # Readiness is not authorisation
//!
//! A readiness verdict can only ever turn "proceed" into "ask". It never
//! grants a permission that the argument validator, the policy engine, the
//! allow-list and the confirmation gate would not have granted anyway. A
//! missing verdict is [`Readiness::Undetermined`], which changes nothing:
//! the remaining controls decide.

use crate::intent::{Entities, Intent, Readiness};
use crate::text;

/// How an argument's value is checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentKind {
    /// A free-text payload: what to remember, what to look for.
    ///
    /// This is *content*, not a reference to be resolved. "recall my name"
    /// names the thing the user wants, so the generic-noun rule must not
    /// apply to it. The only failure worth catching is an argument that
    /// carries no content at all — "remember that", where the extractor left
    /// the bare filler "that" behind.
    Content,
    /// A reference to a particular object: a file, an application, a search
    /// pattern.
    ///
    /// A generic noun here is a failure: the user named a category ("the
    /// file") rather than a thing ("the readme"), and the assistant must ask
    /// which one.
    Referent,
    /// A value with its own shape rules, such as a URL.
    ///
    /// The shape is enforced by the tool's own validator, so readiness has
    /// nothing to say about it.
    Value,
}

/// Which [`Entities`] field supplies an argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgumentField {
    Text,
    Path,
    Application,
    Url,
    Suite,
}

impl ArgumentField {
    /// The value of this field, if the extractor produced one.
    pub fn get<'a>(&self, entities: &'a Entities) -> Option<&'a String> {
        match self {
            ArgumentField::Text => entities.text.as_ref(),
            ArgumentField::Path => entities.path.as_ref(),
            ArgumentField::Application => entities.application.as_ref(),
            ArgumentField::Url => entities.url.as_ref(),
            ArgumentField::Suite => entities.suite.as_ref(),
        }
    }
}

/// One argument slot of an intent.
#[derive(Debug, Clone, Copy)]
pub struct ArgumentSlot {
    /// Which field supplies it.
    pub field: ArgumentField,
    /// Is it required?
    ///
    /// An optional slot that is present but unusable still makes the request
    /// incomplete: "run the suite" names a test suite that does not exist,
    /// and dispatching it would be guessing.
    pub required: bool,
    /// How the value is checked.
    pub kind: ArgumentKind,
    /// What the assistant calls it when it is missing, in the phrasing the
    /// agent already uses: "which file to read".
    pub what: &'static str,
}

const fn required(field: ArgumentField, kind: ArgumentKind, what: &'static str) -> ArgumentSlot {
    ArgumentSlot {
        field,
        required: true,
        kind,
        what,
    }
}

const fn optional(field: ArgumentField, kind: ArgumentKind, what: &'static str) -> ArgumentSlot {
    ArgumentSlot {
        field,
        required: false,
        kind,
        what,
    }
}

/// The argument slots of every intent that requests an action.
///
/// `greeting`, `help` and `unknown` have none: they request no action.
/// `system_status` and `cancel` have none either, because they are fully
/// determined by the utterance — there is nothing to clarify.
///
/// The table mirrors the agent's argument builder exactly: same fields, same
/// required flags, same kinds. `aaron-app/tests/readiness.rs` asserts that
/// the two agree on every intent, so the table cannot drift away from the
/// code that actually builds the arguments.
static SEARCH_FILES_SLOTS: [ArgumentSlot; 2] = [
    required(
        ArgumentField::Text,
        ArgumentKind::Referent,
        "what to search for",
    ),
    optional(ArgumentField::Path, ArgumentKind::Value, "which directory"),
];
static READ_FILE_SLOTS: [ArgumentSlot; 1] = [required(
    ArgumentField::Path,
    ArgumentKind::Referent,
    "which file to read",
)];
static OPEN_URL_SLOTS: [ArgumentSlot; 1] = [required(
    ArgumentField::Url,
    ArgumentKind::Value,
    "which URL to open",
)];
static START_APPLICATION_SLOTS: [ArgumentSlot; 2] = [
    required(
        ArgumentField::Application,
        ArgumentKind::Referent,
        "which application",
    ),
    optional(ArgumentField::Url, ArgumentKind::Value, "which URL"),
];
static STOP_APPLICATION_SLOTS: [ArgumentSlot; 1] = [required(
    ArgumentField::Application,
    ArgumentKind::Referent,
    "which application",
)];
static REMEMBER_SLOTS: [ArgumentSlot; 3] = [
    required(
        ArgumentField::Text,
        ArgumentKind::Content,
        "what to remember",
    ),
    optional(ArgumentField::Path, ArgumentKind::Value, "a short label"),
    optional(ArgumentField::Application, ArgumentKind::Value, "some tags"),
];
static RECALL_SLOTS: [ArgumentSlot; 1] = [required(
    ArgumentField::Text,
    ArgumentKind::Content,
    "what to look for",
)];
static GIT_STATUS_SLOTS: [ArgumentSlot; 1] = [optional(
    ArgumentField::Path,
    ArgumentKind::Value,
    "which repository",
)];
static RUN_TESTS_SLOTS: [ArgumentSlot; 2] = [
    optional(
        ArgumentField::Suite,
        ArgumentKind::Value,
        "which test suite",
    ),
    optional(ArgumentField::Path, ArgumentKind::Value, "which directory"),
];

/// The argument slots of every intent that requests an action.
pub fn argument_slots(intent: Intent) -> &'static [ArgumentSlot] {
    match intent {
        Intent::SearchFiles => &SEARCH_FILES_SLOTS,
        Intent::ReadFile => &READ_FILE_SLOTS,
        Intent::OpenUrl => &OPEN_URL_SLOTS,
        Intent::StartApplication => &START_APPLICATION_SLOTS,
        Intent::StopApplication => &STOP_APPLICATION_SLOTS,
        Intent::Remember => &REMEMBER_SLOTS,
        Intent::Recall => &RECALL_SLOTS,
        Intent::GitStatus => &GIT_STATUS_SLOTS,
        Intent::RunTests => &RUN_TESTS_SLOTS,
        Intent::SystemStatus
        | Intent::Cancel
        | Intent::Greeting
        | Intent::Help
        | Intent::Unknown => &[],
    }
}

/// Is this value usable in a slot of the given kind?
fn value_is_usable(kind: ArgumentKind, value: &str) -> bool {
    let value = value.trim();
    if value.is_empty() {
        return false;
    }
    match kind {
        ArgumentKind::Content => !text::is_contentless(value),
        ArgumentKind::Referent => !text::is_generic_referent(value),
        // The shape is the tool's business, not readiness's.
        ArgumentKind::Value => true,
    }
}

/// The readiness verdict for a classified request.
///
/// This is a property of `(intent, entities)`, decided by exactly the same
/// rules the agent's argument builder applies, so the two cannot drift
/// without a test failing.
pub fn readiness_of(intent: Intent, entities: &Entities) -> Readiness {
    if !intent.requests_action() {
        return Readiness::NotApplicable;
    }
    for slot in argument_slots(intent) {
        match slot.field.get(entities) {
            None => {
                if slot.required {
                    return Readiness::NeedsClarification;
                }
            }
            Some(value) => {
                if !value_is_usable(slot.kind, value) {
                    return Readiness::NeedsClarification;
                }
            }
        }
    }
    Readiness::Ready
}

/// The readiness verdict for an utterance, given the intent it was
/// classified as.
///
/// The entities are extracted with the *predicted* intent, which is what the
/// agent does: the extraction is a function of the intent, so the verdict has
/// to be recomputed rather than read from a label.
pub fn readiness_for_text(intent: Intent, utterance: &str) -> Readiness {
    let entities = text::extract_entities(intent, utterance);
    readiness_of(intent, &entities)
}
