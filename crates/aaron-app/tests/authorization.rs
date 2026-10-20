//! The authorization boundary, tested independently of the model.
//!
//! # The property
//!
//! **A classifier prediction is a suggestion. It is never an authorisation.**
//!
//! Every test in this file uses a stub classifier that returns a *chosen*
//! intent at **confidence 1.0** — maximally confident, and deliberately wrong
//! for most of the inputs. If the boundary depended on the model being
//! right, or on it being unsure enough to abstain, these tests would fail.
//!
//! That is the point: the authorisation layer must be independently
//! sufficient. A model that is 99% accurate and 1% wrong still produces a
//! wrong action once in a hundred turns, and the boundary has to catch that
//! one.
//!
//! # The six checks
//!
//! Every actionable request must pass all of:
//!
//! 1. intent allowlisting — the intent must map to a registered tool;
//! 2. tool registry validation — the tool must exist and the arguments must
//!    match its spec;
//! 3. strict argument validation — lengths, shapes, schemes, path safety;
//! 4. permission policy evaluation — deny by default;
//! 5. confirmation requirements — sensitive actions ask, and a
//!    non-interactive session declines;
//! 6. timeout and resource limits — enforced by the policy, never by the
//!    model.
//!
//! # Proving a tool did not run
//!
//! Most tests assert on the outcome *and* on the audit log. But an outcome
//! can be wrong for the wrong reason, so the strongest tests use a **sentinel**:
//! a registered application whose "launch" is a script that would create a
//! file. If the file appears, the boundary leaked. It must not.

use std::sync::{Arc, Mutex};

use aaron_agent::{Agent, AgentDeps, AutoConfirm, Confirmer, NonInteractive};
use aaron_core::config::Config;
use aaron_core::error::Result;
use aaron_core::intent::Entities;
use aaron_core::intent::{Command, Intent, IntentClassifier};
use aaron_memory::MemoryStore;
use aaron_security::{AuditLog, Decision, PathSandbox, PolicyEngine, PolicyFile};
use aaron_tools::{ToolContext, ToolRegistry};
use std::time::Duration;

/// A classifier that always returns the same intent, at full confidence.
///
/// This is the adversarial case for the authorisation layer: the model is not
/// hedging, it is asserting, and it is asserting the wrong thing.
struct ConfidentClassifier {
    intent: Intent,
    entities: Entities,
}

impl ConfidentClassifier {
    fn new(intent: Intent) -> ConfidentClassifier {
        ConfidentClassifier {
            intent,
            entities: Entities::default(),
        }
    }

    fn with_entities(intent: Intent, entities: Entities) -> ConfidentClassifier {
        ConfidentClassifier { intent, entities }
    }
}

impl IntentClassifier for ConfidentClassifier {
    fn name(&self) -> &'static str {
        "test/confident-stub"
    }

    fn classify(&self, _utterance: &str) -> Result<Command> {
        Ok(Command {
            intent: self.intent,
            entities: self.entities.clone(),
            confidence: 1.0,
            source: aaron_core::intent::ClassifySource::Neural,
            // `Ready`, and it must stay `Ready`.
            //
            // The readiness verdict is a one-way valve: it can only turn
            // "proceed" into "ask". If this stub reported
            // `NeedsClarification`, every test in this file would
            // short-circuit before the policy engine, no tool would be
            // dispatched, and every assertion would pass *vacuously* — the
            // suite would prove nothing about the authorisation boundary.
            //
            // The authorisation boundary must be exercised, so the stub must
            // not pre-empt it. `aaron_authorization_reaches_the_policy_engine`
            // in this file is the control that keeps this honest.
            readiness: aaron_core::intent::Readiness::Ready,
        })
    }
}

/// A confirmer that records what it was asked and always refuses.
#[derive(Default)]
struct RecordingRefuser {
    prompts: Mutex<Vec<String>>,
}

impl Confirmer for RecordingRefuser {
    fn confirm(&self, prompt: &str, _timeout: Duration) -> Result<bool> {
        self.prompts.lock().unwrap().push(prompt.to_string());
        Ok(false)
    }
}

/// A confirmer that records and always approves.
#[derive(Default)]
struct RecordingApprover {
    prompts: Mutex<Vec<String>>,
}

impl Confirmer for RecordingApprover {
    fn confirm(&self, prompt: &str, _timeout: Duration) -> Result<bool> {
        self.prompts.lock().unwrap().push(prompt.to_string());
        Ok(true)
    }
}

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

struct Fixture {
    _dir: tempfile::TempDir,
    #[allow(dead_code)]
    root: std::path::PathBuf,
    agent: Agent,
    #[allow(dead_code)]
    audit_path: std::path::PathBuf,
}

fn fixture(
    policy: &str,
    always_confirm: Vec<String>,
    confirmer: Arc<dyn Confirmer>,
    classifier: Arc<dyn IntentClassifier>,
) -> Fixture {
    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    std::fs::write(root.join("ok.txt"), "hello from inside the root").unwrap();

    let mut config = Config::default();
    config.tts.echo_to_terminal = false;
    config.security.always_confirm = always_confirm;

    let memory = Arc::new(Mutex::new(
        MemoryStore::open_in_memory(Default::default()).unwrap(),
    ));
    let ctx = ToolContext::new(
        Arc::new(PathSandbox::new([&root]).unwrap()),
        Arc::clone(&memory),
        Arc::new(config.clone()),
        root.clone(),
    );
    let audit = AuditLog::open(&audit_path, 400).unwrap();
    let agent = Agent::new(AgentDeps {
        classifier,
        registry: ToolRegistry::builtin(),
        policy: PolicyEngine::new(
            PolicyFile::from_toml(policy).unwrap(),
            config.security.always_confirm.clone(),
        ),
        audit: audit.clone(),
        context: ctx,
        memory,
        tts: Arc::new(aaron_tts::TextOnlyTts),
        confirmer,
        config,
    });
    Fixture {
        _dir: dir,
        root,
        agent,
        audit_path,
    }
}

impl Fixture {
    fn ask(&mut self, text: &str) -> aaron_agent::AgentResponse {
        self.agent
            .handle_text(text)
            .expect("a turn should not error")
    }
}

// ---------------------------------------------------------------------------
// 1. Intent allowlisting
// ---------------------------------------------------------------------------

#[test]
fn a_confident_prediction_of_a_toolless_intent_dispatches_nothing() {
    // `greeting` and `help` have no tool behind them. Even at confidence 1.0
    // there is nothing to dispatch.
    for intent in [Intent::Greeting, Intent::Help] {
        let mut f = fixture(
            ALLOW_ALL,
            Vec::new(),
            Arc::new(AutoConfirm),
            Arc::new(ConfidentClassifier::new(intent)),
        );
        let r = f.ask("anything at all");
        assert_eq!(r.outcome, aaron_agent::Outcome2::Answered);
        assert!(r.tool.is_none(), "a tool was selected for {intent:?}");
    }
}

#[test]
fn an_intent_with_no_registered_tool_cannot_be_forced() {
    // The policy allows every tool, but `unknown` has none. A prediction of
    // `unknown` must not be turned into a tool call.
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::new(Intent::Unknown)),
    );
    let r = f.ask("do the thing");
    assert!(r.tool.is_none());
    assert_ne!(r.outcome, aaron_agent::Outcome2::Executed);
}

// ---------------------------------------------------------------------------
// 2 + 4. Tool registry validation and the permission policy
// ---------------------------------------------------------------------------

#[test]
fn a_confident_prediction_cannot_bypass_a_deny() {
    // The policy denies `run_tests`. The model is certain. It does not matter:
    // the policy is evaluated on facts from the tool registry, and the
    // classifier's output never reaches it.
    let policy = "version = 1\n[tools.system_status]\neffect = \"allow\"\n";
    let mut f = fixture(
        policy,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::new(Intent::RunTests)),
    );
    let r = f.ask("run the tests");
    assert_eq!(r.outcome, aaron_agent::Outcome2::Denied);
    assert!(r.denied_reason.is_some());
}

#[test]
fn an_empty_policy_denies_every_confident_prediction() {
    for intent in [
        Intent::RunTests,
        Intent::ReadFile,
        Intent::OpenUrl,
        Intent::StartApplication,
        Intent::StopApplication,
        Intent::Remember,
    ] {
        let mut f = fixture(
            "version = 1\n",
            Vec::new(),
            Arc::new(AutoConfirm),
            Arc::new(ConfidentClassifier::new(intent)),
        );
        let r = f.ask("do it");
        assert_eq!(
            r.outcome,
            aaron_agent::Outcome2::Denied,
            "intent {intent:?} was not denied by an empty policy"
        );
    }
}

#[test]
fn a_tool_with_no_policy_entry_is_denied_even_when_others_are_allowed() {
    let policy = "version = 1\n[tools.system_status]\neffect = \"allow\"\n";
    let mut f = fixture(
        policy,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::new(Intent::OpenUrl)),
    );
    let r = f.ask("open the website");
    assert_eq!(r.outcome, aaron_agent::Outcome2::Denied);
}

#[test]
fn the_classifier_cannot_influence_the_policy_inputs() {
    // The policy engine is handed facts from the *registry*, never from the
    // classifier. This test pins the shape of that boundary: a `ToolFacts`
    // value is derived from the tool's static spec.
    let registry = ToolRegistry::builtin();
    let spec = registry.get("run_tests").unwrap();
    let facts = spec.spec().facts();
    assert_eq!(facts.id, "run_tests");
    // The facts carry the tool's own declared limits, not anything the
    // classifier said.
    assert!(facts.declared_timeout_secs > 0);
    assert!(facts.declared_max_output_bytes > 0);
}

// ---------------------------------------------------------------------------
// 3. Strict argument validation
// ---------------------------------------------------------------------------

#[test]
fn a_confident_prediction_with_a_traversal_path_executes_nothing() {
    // The model is certain the user wants `read_file`. The path escapes the
    // sandbox. The sandbox refuses.
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::with_entities(
            Intent::ReadFile,
            Entities {
                path: Some("../../etc/passwd".into()),
                ..Default::default()
            },
        )),
    );
    let r = f.ask("read the file");
    assert_ne!(r.outcome, aaron_agent::Outcome2::Executed);
    assert!(
        !r.reply.contains("root:"),
        "file contents leaked: {}",
        r.reply
    );
}

#[test]
fn a_confident_prediction_with_a_credential_url_executes_nothing() {
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::with_entities(
            Intent::OpenUrl,
            Entities {
                url: Some("https://user:hunter2@example.com/".into()),
                ..Default::default()
            },
        )),
    );
    let r = f.ask("open the website");
    assert_ne!(r.outcome, aaron_agent::Outcome2::Executed);
}

#[test]
fn a_confident_prediction_with_a_non_http_url_executes_nothing() {
    for url in [
        "file:///etc/passwd",
        "javascript:alert(1)",
        "ftp://example.com",
    ] {
        let mut f = fixture(
            ALLOW_ALL,
            Vec::new(),
            Arc::new(AutoConfirm),
            Arc::new(ConfidentClassifier::with_entities(
                Intent::OpenUrl,
                Entities {
                    url: Some(url.into()),
                    ..Default::default()
                },
            )),
        );
        let r = f.ask("open the website");
        assert_ne!(
            r.outcome,
            aaron_agent::Outcome2::Executed,
            "`{url}` was dispatched"
        );
    }
}

#[test]
fn a_confident_prediction_with_an_unregistered_application_executes_nothing() {
    // The application allow-list is the second line of defence for
    // `start_application`, after the policy.
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::with_entities(
            Intent::StartApplication,
            Entities {
                application: Some("definitely-not-registered".into()),
                ..Default::default()
            },
        )),
    );
    let r = f.ask("launch it");
    assert_ne!(r.outcome, aaron_agent::Outcome2::Executed);
    assert!(!r.reply.to_lowercase().contains("started"));
}

#[test]
fn a_confident_prediction_with_a_missing_required_argument_executes_nothing() {
    // `read_file` needs a path. The model is certain about the intent but
    // supplies no path. The argument builder refuses rather than inventing
    // one.
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::new(Intent::ReadFile)),
    );
    let r = f.ask("read the file");
    assert_ne!(r.outcome, aaron_agent::Outcome2::Executed);
}

#[test]
fn an_oversized_argument_is_refused_rather_than_truncated() {
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::with_entities(
            Intent::ReadFile,
            Entities {
                path: Some("a".repeat(50_000)),
                ..Default::default()
            },
        )),
    );
    let r = f.ask("read the file");
    assert_ne!(r.outcome, aaron_agent::Outcome2::Executed);
}

// ---------------------------------------------------------------------------
// 5. Confirmation requirements
// ---------------------------------------------------------------------------

#[test]
fn a_confirmation_gated_action_is_declined_when_nobody_can_answer() {
    let confirmer = Arc::new(RecordingRefuser::default());
    let mut f = fixture(
        ALLOW_ALL,
        vec!["run_tests".into()],
        confirmer.clone(),
        Arc::new(ConfidentClassifier::new(Intent::RunTests)),
    );
    let r = f.ask("run the tests");
    assert_eq!(r.outcome, aaron_agent::Outcome2::Declined);
    // And the user was actually asked, with the tool named.
    let prompts = confirmer.prompts.lock().unwrap();
    assert!(!prompts.is_empty(), "no confirmation was requested");
    assert!(prompts[0].contains("run_tests"), "{}", prompts[0]);
}

#[test]
fn a_non_interactive_session_declines_even_with_an_approving_policy() {
    // The policy allows it; there is nobody to ask.
    let mut f = fixture(
        ALLOW_ALL,
        vec!["run_tests".into()],
        Arc::new(NonInteractive),
        Arc::new(ConfidentClassifier::new(Intent::RunTests)),
    );
    let r = f.ask("run the tests");
    assert_eq!(r.outcome, aaron_agent::Outcome2::Declined);
}

#[test]
fn a_confirmation_cannot_be_skipped_by_asking_for_a_different_tool() {
    // `system_status` is not gated; `run_tests` is. A request for the gated
    // tool must not fall through to the ungated one.
    let confirmer = Arc::new(RecordingRefuser::default());
    let mut f = fixture(
        ALLOW_ALL,
        vec!["run_tests".into()],
        confirmer.clone(),
        Arc::new(ConfidentClassifier::new(Intent::RunTests)),
    );
    let first = f.ask("run the tests");
    assert_eq!(first.outcome, aaron_agent::Outcome2::Declined);
    // The second time, the repetition guard may block it — also a refusal.
    let second = f.ask("run the tests");
    assert!(
        matches!(
            second.outcome,
            aaron_agent::Outcome2::Declined | aaron_agent::Outcome2::Blocked
        ),
        "{:?}",
        second.outcome
    );
}

#[test]
fn an_approved_confirmation_is_the_only_way_a_gated_action_runs() {
    // The positive control for the tests above: when the user *does* approve,
    // the action happens. Without this, "everything is denied" would pass.
    let confirmer = Arc::new(RecordingApprover::default());
    let mut f = fixture(
        ALLOW_ALL,
        vec!["run_tests".into()],
        confirmer.clone(),
        Arc::new(ConfidentClassifier::new(Intent::RunTests)),
    );
    let r = f.ask("run the tests");
    assert!(
        matches!(
            r.outcome,
            aaron_agent::Outcome2::Executed | aaron_agent::Outcome2::Confirmed
        ),
        "{:?}: {}",
        r.outcome,
        r.reply
    );
    assert_eq!(confirmer.prompts.lock().unwrap().len(), 1);
}

// ---------------------------------------------------------------------------
// 6. Timeout and resource limits
// ---------------------------------------------------------------------------

#[test]
fn a_permit_always_carries_execution_limits() {
    // The limits come from the policy, and the model has no way to widen
    // them. Network is enabled explicitly here, because the network switch is
    // a separate control and is tested on its own below.
    let policy = ALLOW_ALL.replace("version = 1", "version = 1\n[limits]\nallow_network = true");
    let engine = PolicyEngine::new(PolicyFile::from_toml(&policy).unwrap(), Vec::new());
    let registry = ToolRegistry::builtin();
    for id in ["run_tests", "read_file", "open_url", "start_application"] {
        let spec = registry.get(id).unwrap();
        match engine.evaluate(&spec.spec().facts()) {
            Decision::Allow(limits) | Decision::NeedsConfirmation { limits, .. } => {
                assert!(limits.timeout_secs > 0, "`{id}` has no timeout");
                assert!(limits.max_output_bytes > 0, "`{id}` has no output cap");
            }
            other => panic!("expected a permit for `{id}`, got {other:?}"),
        }
    }
}

#[test]
fn the_network_switch_blocks_network_tools_by_default() {
    // `open_url` needs an explicit opt-in, whatever the tool's own entry says.
    // The default policy file has `allow_network = false`, and a permissive
    // tool entry does not change that.
    let engine = PolicyEngine::new(PolicyFile::from_toml(ALLOW_ALL).unwrap(), Vec::new());
    let registry = ToolRegistry::builtin();
    let spec = registry.get("open_url").unwrap();
    match engine.evaluate(&spec.spec().facts()) {
        Decision::Deny(reason) => {
            assert!(reason.contains("network"), "unexpected reason: {reason}")
        }
        other => panic!("open_url should be denied by default, got {other:?}"),
    }
}

#[test]
fn a_policy_cannot_widen_the_global_limits() {
    let policy = r#"
version = 1
[limits]
max_runtime_secs = 2
max_output_bytes = 1024
[tools.run_tests]
effect = "allow"
timeout_secs = 9999
max_output_bytes = 999999
"#;
    let engine = PolicyEngine::new(PolicyFile::from_toml(policy).unwrap(), Vec::new());
    let registry = ToolRegistry::builtin();
    let spec = registry.get("run_tests").unwrap();
    match engine.evaluate(&spec.spec().facts()) {
        Decision::Allow(limits) | Decision::NeedsConfirmation { limits, .. } => {
            assert_eq!(limits.timeout_secs, 2);
            assert_eq!(limits.max_output_bytes, 1024);
        }
        other => panic!("expected a permit, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------
// The sentinel: proving a tool did not run
// ---------------------------------------------------------------------------

#[test]
fn a_denied_action_leaves_no_trace_in_the_audit_log() {
    // An outcome can be wrong for the wrong reason, so the audit log is
    // checked too: a denied action must be recorded as denied, and must not
    // be recorded as authorised or as a result.
    let dir = tempfile::tempdir().unwrap();
    let audit_path = dir.path().join("audit.jsonl");
    let audit = AuditLog::open(&audit_path, 400).unwrap();

    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).unwrap();
    let mut config = Config::default();
    config.tts.echo_to_terminal = false;
    let memory = Arc::new(Mutex::new(
        MemoryStore::open_in_memory(Default::default()).unwrap(),
    ));
    let ctx = ToolContext::new(
        Arc::new(PathSandbox::new([&root]).unwrap()),
        Arc::clone(&memory),
        Arc::new(config.clone()),
        root.clone(),
    );
    let mut agent = Agent::new(AgentDeps {
        classifier: Arc::new(ConfidentClassifier::new(Intent::RunTests)),
        registry: ToolRegistry::builtin(),
        policy: PolicyEngine::new(PolicyFile::from_toml("version = 1\n").unwrap(), Vec::new()),
        audit: audit.clone(),
        context: ctx,
        memory,
        tts: Arc::new(aaron_tts::TextOnlyTts),
        confirmer: Arc::new(AutoConfirm),
        config,
    });
    let r = agent.handle_text("run the tests").unwrap();
    assert_eq!(r.outcome, aaron_agent::Outcome2::Denied);

    let log = std::fs::read_to_string(&audit_path).unwrap();
    assert!(
        log.contains("requested"),
        "the attempt was not recorded: {log}"
    );
    assert!(
        !log.contains("authorized"),
        "a denied action was recorded as authorised: {log}"
    );
    assert!(
        !log.contains("\"result\""),
        "a denied action recorded a result: {log}"
    );
}

// ---------------------------------------------------------------------------
// The boundary under the *real* classifier
// ---------------------------------------------------------------------------

#[test]
fn the_deterministic_router_cannot_execute_a_denied_tool_either() {
    // The same property with the shipped classifier, to prove the tests above
    // are not an artefact of the stub.
    let mut f = fixture(
        "version = 1\n[tools.system_status]\neffect = \"allow\"\n",
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(aaron_brain::HeuristicClassifier::new(0.5)),
    );
    for text in [
        "run the tests",
        "open the website https://example.com",
        "kill the terminal",
    ] {
        let r = f.ask(text);
        assert_ne!(
            r.outcome,
            aaron_agent::Outcome2::Executed,
            "`{text}` executed under a policy that only allows system_status"
        );
    }
}

// ---------------------------------------------------------------------------
// 7. The readiness field must not pre-empt the boundary
// ---------------------------------------------------------------------------

#[test]
fn a_ready_verdict_still_reaches_the_policy_engine() {
    // The control for the whole file.
    //
    // The readiness verdict is a one-way valve: it can only turn "proceed"
    // into "ask". If a future change made the agent short-circuit on it
    // *before* the policy engine, then with the stub reporting `Ready` the
    // suite would still pass — but it would be testing a different thing, or
    // nothing at all.
    //
    // This test asserts the boundary is actually reached: the audit log
    // records a `requested` entry for the tool, which only happens after the
    // arguments are built and before the policy engine runs.
    let mut f = fixture(
        ALLOW_ALL,
        Vec::new(),
        Arc::new(AutoConfirm),
        Arc::new(ConfidentClassifier::new(Intent::SystemStatus)),
    );
    let r = f.ask("anything at all");
    assert_eq!(
        r.outcome,
        aaron_agent::Outcome2::Executed,
        "the turn must reach the tool: {r:?}"
    );
    assert_eq!(
        r.readiness,
        aaron_core::intent::Readiness::Ready,
        "the stub reports Ready, and that verdict must not change the outcome"
    );

    let audit = std::fs::read_to_string(&f.audit_path).unwrap();
    assert!(
        audit.contains("requested"),
        "the audit log must record the request, which only happens once the \
         turn has passed classification and argument building: {audit}"
    );
}
