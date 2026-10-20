//! Verification of two claims made by the independent 0.4 review.
//!
//! 1. The argument layer does **not** refuse an incomplete request: it
//!    dispatches a tool with a placeholder argument, so the "I understood
//!    `read_file` but not which file" path is unreachable.
//! 2. The regression suite is (almost) part of the training data, so "42/42"
//!    measures memorisation rather than generalisation.
//!
//! This file is a *review artefact*. It asserts the current behaviour, and
//! the reviewer's findings were verified by running it.

use std::sync::{Arc, Mutex};

use aaron_agent::{Agent, AgentDeps, AutoConfirm};
use aaron_core::config::Config;
use aaron_memory::MemoryStore;
use aaron_security::{AuditLog, PathSandbox, PolicyEngine, PolicyFile};
use aaron_tools::{ToolContext, ToolRegistry};

const ALLOW_ALL: &str = r#"
version = 1
[tools.system_status]
effect = "allow"
[tools.search_files]
effect = "allow"
[tools.read_file]
effect = "allow"
[tools.git_status]
effect = "allow"
[tools.run_tests]
effect = "allow"
[tools.open_url]
effect = "allow"
[tools.start_application]
effect = "allow"
[tools.stop_application]
effect = "allow"
[tools.remember]
effect = "allow"
[tools.recall]
effect = "allow"
"#;

fn agent_in(root: &std::path::Path) -> Agent {
    let mut config = Config::default();
    config.tts.echo_to_terminal = false;
    let memory = Arc::new(Mutex::new(
        MemoryStore::open_in_memory(Default::default()).unwrap(),
    ));
    let ctx = ToolContext::new(
        Arc::new(PathSandbox::new([root]).unwrap()),
        Arc::clone(&memory),
        Arc::new(config.clone()),
        root.to_path_buf(),
    );
    Agent::new(AgentDeps {
        classifier: Arc::new(aaron_brain::HeuristicClassifier::new(0.5)),
        registry: ToolRegistry::builtin(),
        policy: PolicyEngine::new(PolicyFile::from_toml(ALLOW_ALL).unwrap(), Vec::new()),
        audit: AuditLog::disabled(),
        context: ctx,
        memory,
        tts: Arc::new(aaron_tts::TextOnlyTts),
        confirmer: Arc::new(AutoConfirm),
        config,
    })
}

#[test]
fn claim_1_an_incomplete_request_dispatches_a_placeholder_argument() {
    // "read the file" has no file. The extractor leaves "the file" as the
    // path, the argument builder sees a non-empty string, and the tool runs
    // with "the file" as its argument.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let mut agent = agent_in(&root);

    let response = agent.handle_text("read the file").unwrap();
    // Before the 0.4 fix the tool was *dispatched* with the placeholder "the
    // file" as its path and was then refused by the sandbox, so the user was
    // told "path rejected: `the file` does not exist inside any allowed
    // root". Nothing unsafe happened, but the user was never asked which file
    // they meant.
    assert_eq!(
        response.outcome,
        aaron_agent::Outcome2::Denied,
        "expected a clarification, got {:?} {}",
        response.outcome,
        response.reply
    );
    assert!(
        response.reply.contains("read_file") && response.reply.contains("Try naming it"),
        "expected a clarification naming the intent: {}",
        response.reply
    );
}

#[test]
fn claim_1b_an_incomplete_stop_request_dispatches_a_placeholder_application() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let mut agent = agent_in(&root);

    let response = agent.handle_text("kill the process").unwrap();
    // Before the fix this was dispatched with application="the process" and
    // refused by the application allow-list. Now it is refused at the
    // argument layer, with a clarification.
    assert_eq!(
        response.outcome,
        aaron_agent::Outcome2::Denied,
        "expected a clarification, got {:?} {}",
        response.outcome,
        response.reply
    );
    assert!(
        response.reply.contains("stop_application") && response.reply.contains("Try naming it"),
        "expected a clarification naming the intent: {}",
        response.reply
    );
}

#[test]
fn the_argument_layer_asks_instead_of_dispatching_a_placeholder() {
    // The 0.4 fix: a generic reference ("the file", "it", "the process")
    // produces the clarification, not a tool call with a placeholder.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let mut agent = agent_in(&root);

    for (text, expected) in [
        ("read the file", "read_file"),
        ("kill the process", "stop_application"),
        ("launch the app", "start_application"),
        ("remember that", "remember"),
    ] {
        let response = agent.handle_text(text).unwrap();
        assert_eq!(
            response.outcome,
            aaron_agent::Outcome2::Denied,
            "`{text}` should be refused with a clarification, got {:?} {}",
            response.outcome,
            response.reply
        );
        assert!(
            response.reply.contains(expected),
            "`{text}` should name the intent it understood: {}",
            response.reply
        );
        assert!(
            response.reply.to_lowercase().contains("try naming it"),
            "`{text}` should ask for the missing argument: {}",
            response.reply
        );
    }
}

#[test]
fn the_generic_referent_rule_does_not_over_refuse() {
    // The 0.4 rule fires on "name", "project" and "code" as well as on "file"
    // and "app". Those words are generic *categories* in the abstract, but
    // here they are what the user actually asked for.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let mut agent = agent_in(&root);

    for text in ["recall my name", "search your memory for my project"] {
        let response = agent.handle_text(text).unwrap();
        assert_ne!(
            response.outcome,
            aaron_agent::Outcome2::Denied,
            "`{text}` was refused even though the user named the thing: {}",
            response.reply
        );
    }
}

#[test]
fn a_specific_argument_still_works() {
    // The rule must not fire on a real argument.
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("ok.txt"), "content").unwrap();
    let mut agent = agent_in(&root);

    let response = agent.handle_text("read the file ok.txt").unwrap();
    assert_eq!(
        response.outcome,
        aaron_agent::Outcome2::Executed,
        "a specific argument must still dispatch: {:?} {}",
        response.outcome,
        response.reply
    );
    assert!(response.reply.contains("content"), "{}", response.reply);
}

#[test]
fn claim_2_the_regression_suite_is_mostly_inside_the_training_data() {
    let dataset = aaron_brain::Dataset::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../training/dataset/current.jsonl"
    ))
    .unwrap();
    let regression = aaron_brain::Dataset::load(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../training/evaluation/regression.jsonl"
    ))
    .unwrap();

    let mut in_training = 0usize;
    let mut in_train_split = 0usize;
    for r in &regression.examples {
        let key = aaron_core::text::normalize(&r.text);
        if let Some(match_) = dataset
            .examples
            .iter()
            .find(|e| aaron_core::text::normalize(&e.text) == key)
        {
            in_training += 1;
            if dataset.split_of(match_) == aaron_brain::Split::Train {
                in_train_split += 1;
            }
        }
    }
    println!(
        "regression suite: {}/{} are in the training dataset, {}/{} in the TRAIN split",
        in_training,
        regression.examples.len(),
        in_train_split,
        regression.examples.len()
    );
    // After the 0.4 hold-out fix the regression suite is *out* of the training
    // data. Before the fix it was 41/42 inside it, which is why "42/42" was
    // measuring memorisation.
    assert_eq!(
        in_training,
        0,
        "the regression suite must not be in the training data; found {in_training}/{}",
        regression.examples.len()
    );
    assert_eq!(in_train_split, 0);
}
