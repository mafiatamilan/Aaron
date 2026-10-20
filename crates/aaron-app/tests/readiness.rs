//! The readiness verdict and the argument builder must agree.
//!
//! Release 0.5 introduced a readiness verdict: a typed answer to "is this
//! request specific enough to act on?". It is computed in
//! `aaron-core::readiness`, from a table of argument slots, because the
//! classifier cannot call the agent's argument builder (the dependency runs
//! the other way).
//!
//! Two independent implementations of the same question is a drift risk. This
//! file is the detector: for every intent and a wide spread of utterances,
//! the verdict and the builder must reach the same conclusion.
//!
//! * `readiness == NeedsClarification` ⟺ the builder refuses.
//! * `readiness == Ready` ⟺ the builder accepts.
//!
//! When they disagree, one of them is wrong, and the readiness verdict —
//! which can refuse a request the builder would have run — is the more
//! dangerous of the two.

use aaron_agent::agent::build_arguments;
use aaron_core::intent::{Entities, Intent, Readiness};
use aaron_core::text::extract_entities;

/// A generous limit, so the builder's length check cannot fire and the
/// comparison is only about completeness.
const MAX_CHARS: usize = 4_000;

/// The utterances the two implementations are compared on.
///
/// The spread matters: the same intent appears with a missing argument, a
/// generic one, a real one, an over-long one and a nonsense one.
fn utterances() -> Vec<&'static str> {
    vec![
        "",
        "read",
        "read the file",
        "read the document",
        "read the readme",
        "read main.rs",
        "read the first 20 lines",
        "read the file called",
        "open",
        "open the file",
        "open the browser",
        "open chrome",
        "search",
        "search for files",
        "search for the file",
        "search for *.rs in the project",
        "search for *.rs in my code",
        "search for large files in the project",
        "launch",
        "launch the app",
        "launch code",
        "kill",
        "kill the process",
        "kill the browser process",
        "kill spotify",
        "remember",
        "remember that",
        "remember my birthday is in June",
        "recall",
        "recall it",
        "recall my name",
        "open the url",
        "open https://example.com",
        "run the tests",
        "run the suite",
        "run the test suite",
        "run cargo test",
        "git status",
        "git status in the repository",
        "check system status",
        "cancel",
        "hello",
        "help",
        "what is the meaning of life",
        "the bus was late this morning",
    ]
}

#[test]
fn the_readiness_verdict_and_the_argument_builder_agree() {
    let mut disagreements: Vec<String> = Vec::new();

    for intent in Intent::ALL {
        for text in utterances() {
            let entities = extract_entities(intent, text);
            let verdict = aaron_core::readiness::readiness_of(intent, &entities);
            let built = build_arguments(intent, &entities, MAX_CHARS);

            // The builder only refuses for a missing or unusable argument
            // when the limit is generous, so a refusal and a
            // `NeedsClarification` verdict must be the same event.
            let builder_refused = built.is_err();
            let verdict_refused = verdict == Readiness::NeedsClarification;

            if builder_refused != verdict_refused {
                disagreements.push(format!(
                    "{intent} / {text:?}: verdict {verdict}, builder {}",
                    if builder_refused {
                        format!("refused ({})", built.err().unwrap())
                    } else {
                        "accepted".to_string()
                    }
                ));
            }
        }
    }

    assert!(
        disagreements.is_empty(),
        "the readiness verdict and the argument builder disagree on {} case(s):\n  {}",
        disagreements.len(),
        disagreements.join("\n  ")
    );
}

#[test]
fn an_intent_with_no_arguments_is_always_ready() {
    // `system_status` and `cancel` are executable actions with nothing to
    // clarify. `greeting`, `help` and `unknown` request no action at all, so
    // they are *not applicable* rather than ready — the distinction is the
    // point of having three states.
    for intent in [Intent::SystemStatus, Intent::Cancel] {
        let entities = Entities::default();
        assert_eq!(
            aaron_core::readiness::readiness_of(intent, &entities),
            Readiness::Ready,
            "{intent} has nothing to clarify"
        );
    }
}

#[test]
fn an_intent_that_requests_no_action_is_not_applicable() {
    for intent in [Intent::Greeting, Intent::Help, Intent::Unknown] {
        let entities = Entities::default();
        assert_eq!(
            aaron_core::readiness::readiness_of(intent, &entities),
            Readiness::NotApplicable,
            "{intent} requests no tool action"
        );
    }
}

#[test]
fn a_missing_required_argument_needs_clarification() {
    let entities = Entities::default();
    assert_eq!(
        aaron_core::readiness::readiness_of(Intent::ReadFile, &entities),
        Readiness::NeedsClarification
    );
    assert_eq!(
        aaron_core::readiness::readiness_of(Intent::StartApplication, &entities),
        Readiness::NeedsClarification
    );
}

#[test]
fn a_specific_argument_is_ready() {
    let entities = Entities {
        path: Some("main.rs".into()),
        ..Default::default()
    };
    assert_eq!(
        aaron_core::readiness::readiness_of(Intent::ReadFile, &entities),
        Readiness::Ready
    );
}

#[test]
fn a_generic_argument_needs_clarification() {
    // A path that names a category rather than a thing.
    let entities = Entities {
        path: Some("the file".into()),
        ..Default::default()
    };
    assert_eq!(
        aaron_core::readiness::readiness_of(Intent::ReadFile, &entities),
        Readiness::NeedsClarification
    );
}

#[test]
fn a_free_text_argument_that_names_a_topic_is_ready() {
    // "recall my name" names the thing the user wants. It is content, not a
    // reference to be resolved, and the generic-noun rule must not apply to
    // it. Release 0.4's rule refused it, and 0.5 found that.
    let entities = Entities {
        text: Some("my name".into()),
        ..Default::default()
    };
    assert_eq!(
        aaron_core::readiness::readiness_of(Intent::Recall, &entities),
        Readiness::Ready
    );
}

#[test]
fn a_contentless_free_text_argument_needs_clarification() {
    // "remember that" leaves the bare filler "that" behind. Storing it would
    // be storing noise.
    let entities = Entities {
        text: Some("that".into()),
        ..Default::default()
    };
    assert_eq!(
        aaron_core::readiness::readiness_of(Intent::Remember, &entities),
        Readiness::NeedsClarification
    );
}

#[test]
fn the_readiness_enum_round_trips_through_its_identifier() {
    for verdict in [
        Readiness::Ready,
        Readiness::NeedsClarification,
        Readiness::NotApplicable,
        Readiness::Undetermined,
    ] {
        assert_eq!(
            Readiness::parse(verdict.as_str()),
            Some(verdict),
            "{verdict} must round-trip"
        );
    }
    assert_eq!(Readiness::parse("nonsense"), None);
}

#[test]
fn a_command_without_a_readiness_field_deserialises_as_undetermined() {
    // A payload written before the field existed must not come back as
    // `Ready`: a missing verdict grants nothing, and `Ready` would be a
    // claim the payload never made.
    let json = r#"{"intent":"read_file","entities":{},"confidence":0.9,"source":"neural"}"#;
    let command: aaron_core::Command = serde_json::from_str(json).unwrap();
    assert_eq!(command.readiness, Readiness::Undetermined);
    assert_eq!(command.intent, Intent::ReadFile);
}
