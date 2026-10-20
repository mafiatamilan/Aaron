//! What does the readiness head do to the agent's behaviour on the
//! regression suite?
//!
//! The regression suite is 42 complete requests. The readiness head does not
//! change the *intent* it predicts, so the suite's pass count cannot move;
//! what can move is whether the assistant *acts* on a case or asks a
//! question about it.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aaron-agent --example regression_behaviour -- \
//!     <checkpoint> [regression.jsonl]
//! ```

use std::sync::Arc;

use aaron_agent::{Agent, AgentDeps, NonInteractive};
use aaron_core::config::Config;
use aaron_memory::MemoryStore;
use aaron_security::{AuditLog, PathSandbox, PolicyEngine, PolicyFile};
use aaron_tools::{ToolContext, ToolRegistry};

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("usage: regression_behaviour <checkpoint> [regression.jsonl]");
    let path = args.next().unwrap_or_else(|| {
        format!(
            "{}/../../training/evaluation/regression.jsonl",
            env!("CARGO_MANIFEST_DIR")
        )
    });

    let classifier = aaron_brain::NeuralClassifier::load(&checkpoint, 0.60, 1.0, 256)
        .expect("the checkpoint must load");
    println!("checkpoint: {checkpoint}");
    println!(
        "  architecture {}   parameters {}",
        classifier.manifest().arch,
        classifier.manifest().param_count
    );

    let raw = std::fs::read_to_string(&path).expect("the regression file must read");
    let mut texts: Vec<String> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
                texts.push(text.to_string());
            }
        }
    }

    let dir = tempfile::tempdir().expect("a temporary directory");
    let root = dir.path().join("root");
    std::fs::create_dir_all(&root).expect("the root must exist");
    let memory = Arc::new(std::sync::Mutex::new(
        MemoryStore::open_in_memory(Default::default()).expect("memory"),
    ));
    let config = Config::default();
    let ctx = ToolContext::new(
        Arc::new(PathSandbox::new([&root]).expect("the sandbox")),
        Arc::clone(&memory),
        Arc::new(config.clone()),
        root.clone(),
    );
    let mut agent = Agent::new(AgentDeps {
        classifier: Arc::new(classifier),
        registry: ToolRegistry::builtin(),
        policy: PolicyEngine::new(PolicyFile::default(), Vec::new()),
        audit: AuditLog::disabled(),
        context: ctx,
        memory,
        tts: Arc::new(aaron_tts::TextOnlyTts),
        confirmer: Arc::new(NonInteractive),
        config,
    });

    let mut acted = 0usize;
    let mut asked = 0usize;
    let mut answered = 0usize;
    let mut denied = 0usize;
    for text in &texts {
        let response = agent.handle_text(text).expect("a turn should not error");
        match response.outcome {
            aaron_agent::Outcome2::Executed | aaron_agent::Outcome2::Confirmed => {
                acted += 1;
                println!("  ACTED    {text:?}");
            }
            aaron_agent::Outcome2::NeedsClarification => {
                asked += 1;
                println!("  ASKED    {text:?}");
            }
            aaron_agent::Outcome2::Denied | aaron_agent::Outcome2::Failed => {
                denied += 1;
                println!("  REFUSED  {text:?}");
            }
            _ => {
                answered += 1;
                println!("  ANSWERED {text:?}");
            }
        }
    }
    println!();
    println!("{} cases", texts.len());
    println!("  acted on    {acted}");
    println!("  asked about {asked}");
    println!("  refused     {denied}");
    println!("  answered    {answered}");
}
