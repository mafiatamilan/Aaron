//! Compute the readiness labels for the shipped dataset, using the *real*
//! extraction and argument-validation code rather than a re-implementation.
//!
//! The label is a property of `(intent, entities)`, decided by exactly the
//! code the agent dispatches through:
//!
//! * an intent with no required argument → Ready
//! * an intent with a required argument that is absent or generic
//!   → NeedsClarification
//! * an intent that requests no executable action (greeting, help, unknown)
//!   → NotApplicable
//!
//! The point of the output is to answer one question: **is there enough
//! `NeedsClarification` supervision in the dataset for a readiness head to
//! learn anything?** A head trained on a near-constant label is a head that
//! always says "Ready", and that is worse than no head at all.

use std::collections::BTreeMap;

use aaron_agent::agent::build_arguments;
use aaron_core::intent::Intent;

/// The intents that request an executable tool action.
const ACTIONABLE: [Intent; 11] = [
    Intent::SystemStatus,
    Intent::SearchFiles,
    Intent::ReadFile,
    Intent::GitStatus,
    Intent::RunTests,
    Intent::OpenUrl,
    Intent::StartApplication,
    Intent::StopApplication,
    Intent::Remember,
    Intent::Recall,
    Intent::Cancel,
];

fn label(intent: Intent, text: &str) -> &'static str {
    if !ACTIONABLE.contains(&intent) {
        return "NotApplicable";
    }
    let entities = aaron_core::text::extract_entities(intent, text);
    match build_arguments(intent, &entities, 2000) {
        // The argument builder refuses when a required argument is missing or
        // generic. That refusal *is* the NeedsClarification signal.
        Err(_) => "NeedsClarification",
        Ok(_) => "Ready",
    }
}

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "training/dataset/current.jsonl".into());
    let dataset = aaron_brain::Dataset::load(&path).unwrap();

    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    let mut per_intent: BTreeMap<String, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut per_split: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    let mut examples: BTreeMap<&str, Vec<String>> = BTreeMap::new();

    for e in &dataset.examples {
        let l = label(e.intent, &e.text);
        *counts.entry(l).or_insert(0) += 1;
        *per_intent
            .entry(e.intent.as_str().to_string())
            .or_default()
            .entry(l)
            .or_insert(0) += 1;
        *per_split
            .entry(match dataset.split_of(e) {
                aaron_brain::Split::Train => "train",
                aaron_brain::Split::Validation => "validation",
                aaron_brain::Split::Test => "test",
            })
            .or_default()
            .entry(l)
            .or_insert(0) += 1;
        if l == "NeedsClarification" {
            examples.entry(l).or_default().push(e.text.clone());
        }
    }

    println!("dataset: {} examples", dataset.len());
    println!();
    println!("readiness label distribution");
    let total = dataset.len() as f64;
    for (l, n) in &counts {
        println!("  {:<22} {:>5}  {:>6.2}%", l, n, *n as f64 / total * 100.0);
    }
    println!();
    println!("per split");
    for (split, c) in &per_split {
        let t: usize = c.values().sum();
        println!(
            "  {split:<12} Ready {:>4}  NeedsClarification {:>4}  NotApplicable {:>4}   (of {t})",
            c.get("Ready").copied().unwrap_or(0),
            c.get("NeedsClarification").copied().unwrap_or(0),
            c.get("NotApplicable").copied().unwrap_or(0),
        );
    }
    println!();
    println!("per intent");
    for (intent, c) in &per_intent {
        let t: usize = c.values().sum();
        println!(
            "  {intent:<20} Ready {:>4}  NeedsClarification {:>4}  NotApplicable {:>4}   (of {t})",
            c.get("Ready").copied().unwrap_or(0),
            c.get("NeedsClarification").copied().unwrap_or(0),
            c.get("NotApplicable").copied().unwrap_or(0),
        );
    }
    println!();
    let nc = examples
        .get("NeedsClarification")
        .cloned()
        .unwrap_or_default();
    println!("NeedsClarification examples: {}", nc.len());
    for t in nc.iter().take(40) {
        println!("  {t:?}");
    }
}
