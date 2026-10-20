//! What does the readiness head change about the agent's behaviour on a
//! benchmark?
//!
//! Scores every row of a JSONL file with the *deployed* classifier and with
//! a checkpoint that carries the readiness head, and reports where the two
//! differ.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aaron-brain --example readiness_effect -- \
//!     <checkpoint-with-head> <benchmark.jsonl>
//! ```

use aaron_core::intent::IntentClassifier;

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = args.next().expect("usage: readiness_effect <ckpt> <jsonl>");
    let benchmark = args.next().expect("usage: readiness_effect <ckpt> <jsonl>");

    let with_head = aaron_brain::NeuralClassifier::load(&checkpoint, 0.60, 1.0, 256)
        .expect("the checkpoint must load");

    let raw = std::fs::read_to_string(&benchmark).expect("the benchmark must read");
    let mut rows: Vec<(String, String)> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
            let category = value
                .get("category")
                .and_then(|c| c.as_str())
                .unwrap_or("unclassified")
                .to_string();
            rows.push((text.to_string(), category));
        }
    }

    println!("checkpoint: {checkpoint}");
    println!("benchmark:  {benchmark}");
    println!("rows: {}", rows.len());
    println!();

    let mut fires = 0usize;
    let mut fires_on_actionable = 0usize;
    let mut by_intent: std::collections::BTreeMap<String, usize> = Default::default();
    for (text, _category) in &rows {
        let command = with_head
            .classify(text)
            .expect("the classification must succeed");
        if command.readiness == aaron_core::Readiness::NeedsClarification {
            fires += 1;
            *by_intent
                .entry(command.intent.as_str().to_string())
                .or_insert(0) += 1;
            if command.intent.requests_action() {
                fires_on_actionable += 1;
            }
        }
    }
    println!(
        "the head asks for clarification on {fires} of {} rows",
        rows.len()
    );
    println!("  of those, {fires_on_actionable} have an actionable intent");
    println!("  by intent:");
    for (intent, count) in &by_intent {
        println!("    {intent:<20} {count}");
    }
}
