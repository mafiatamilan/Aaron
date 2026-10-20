//! The readiness corpus must hold still.
//!
//! The corpus is partly machine-written, which makes it exactly the kind of
//! artefact that rots quietly: a near-duplicate of the shipped dataset leaks
//! the hold-out, a pronoun sneaks in and teaches the head the wrong thing, a
//! repeated frame makes a rate look precise when it is not.
//!
//! These tests are the guard. They are about the *data*, not the model.

use std::collections::{BTreeMap, BTreeSet};

use aaron_brain::readiness::{ReadinessCorpus, Split};

/// The corpus under test. Overridable so that a candidate corpus can be
/// checked before it is adopted.
fn corpus_path() -> String {
    std::env::var("AARON_READINESS_CORPUS")
        .unwrap_or_else(|_| "training/readiness/corpus-v3.jsonl".to_string())
}

/// The sets the corpus must not overlap.
const PROTECTED: &[&str] = &[
    "training/dataset/current.jsonl",
    "training/evaluation/benchmark-v3.jsonl",
    "training/evaluation/safety-benchmark-v1.jsonl",
    "training/evaluation/safety-benchmark-v2.jsonl",
    "training/evaluation/regression.jsonl",
];

fn repo(relative: &str) -> String {
    format!("{}/../../{relative}", env!("CARGO_MANIFEST_DIR"))
}

/// The builder's similarity metric, copied from `training/build_dataset_v2.py`.
fn normalise(text: &str) -> String {
    let text = text.to_lowercase();
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        if c.is_ascii_alphanumeric() || c.is_whitespace() {
            out.push(c);
        } else {
            out.push(' ');
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn shingles(text: &str) -> BTreeSet<String> {
    let normalised = normalise(text);
    let words: Vec<&str> = normalised.split_whitespace().collect();
    if words.len() < 3 {
        return words.iter().map(|w| w.to_string()).collect();
    }
    (0..=words.len() - 3)
        .map(|i| words[i..i + 3].join(" "))
        .collect()
}

fn containment(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let shared = a.intersection(b).count();
    shared as f64 / a.len().min(b.len()) as f64
}

fn read_texts(path: &str) -> Vec<String> {
    let raw = std::fs::read_to_string(repo(path)).expect("the file must exist");
    let mut out = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(line) {
            if let Some(text) = value.get("text").and_then(|t| t.as_str()) {
                out.push(text.to_string());
            }
        }
    }
    out
}

#[test]
fn the_corpus_loads_and_declares_a_split_for_every_row() {
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    assert!(
        corpus.len() > 100,
        "the corpus is suspiciously small: {}",
        corpus.len()
    );
    let total: usize = [Split::Train, Split::Validation, Split::Test]
        .iter()
        .map(|s| corpus.split(*s).len())
        .sum();
    assert_eq!(
        total,
        corpus.len(),
        "every row must be in exactly one split"
    );
}

#[test]
fn the_holdout_is_big_enough_to_measure_a_rate() {
    // One row is worth several points on a small split, so the held-out
    // measurement needs rows to spare.
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    for split in [Split::Validation, Split::Test] {
        let rows = corpus.split(split);
        let incomplete = rows
            .iter()
            .filter(|r| r.readiness == aaron_core::Readiness::NeedsClarification)
            .count();
        let complete = rows
            .iter()
            .filter(|r| r.readiness != aaron_core::Readiness::NeedsClarification)
            .count();
        assert!(
            incomplete >= 20,
            "{} has only {incomplete} incomplete rows; a rate over that is not a measurement",
            split.as_str()
        );
        assert!(
            complete >= 20,
            "{} has only {complete} complete rows; a false-not-ready rate over that is not a measurement",
            split.as_str()
        );
    }
}

#[test]
fn the_corpus_is_disjoint_from_every_protected_set() {
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    let mut overlaps: Vec<String> = Vec::new();
    for path in PROTECTED {
        for other in read_texts(path) {
            let other_shingles = shingles(&other);
            for row in &corpus.examples {
                if normalise(&row.text) == normalise(&other) {
                    overlaps.push(format!("{path}: {:?} is identical", row.text));
                } else if containment(&shingles(&row.text), &other_shingles) >= 0.8 {
                    overlaps.push(format!("{path}: {:?} is a near-duplicate", row.text));
                }
            }
        }
    }
    assert!(
        overlaps.is_empty(),
        "the corpus overlaps a protected set in {} case(s):\n  {}",
        overlaps.len(),
        overlaps.join("\n  ")
    );
}

#[test]
fn no_row_uses_a_bare_pronoun_as_its_argument() {
    // A pronoun is only resolvable from the conversation history, which the
    // head does not see. Release 0.6 takes context resolution up separately.
    //
    // The check is on the **extracted argument**, not on the utterance: "it
    // is raining outside" is a statement whose subject is "it", and "read the
    // manuscript chapter-one.md" contains "one" inside a filename. Neither is
    // a pronoun standing in for the thing the action would act on.
    //
    // The list is the unambiguous one. "that" and "this" are excluded because
    // they are the frame of a content request ("remember that ...", "recall
    // this ..."), and "one" because it collides with filenames.
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    let pronouns = ["it", "them", "those", "these"];
    let mut offenders: Vec<String> = Vec::new();
    for row in &corpus.examples {
        if !row.intent.requests_action() {
            continue;
        }
        let entities = aaron_core::text::extract_entities(row.intent, &row.text);
        let argument = entities
            .path
            .as_deref()
            .or(entities.text.as_deref())
            .or(entities.application.as_deref())
            .or(entities.url.as_deref())
            .unwrap_or("");
        // The pronoun stands in for the object, and the object is the last
        // thing said. "remember it" is a pronoun; "remember it is in the
        // drawer" is a proposition whose subject happens to be "it".
        let normalised_argument = normalise(argument);
        let last = normalised_argument
            .split_whitespace()
            .next_back()
            .unwrap_or("");
        if pronouns.contains(&last) {
            offenders.push(format!("{argument:?} ({})", row.text));
        }
    }
    assert!(
        offenders.is_empty(),
        "{} row(s) use a bare pronoun as the argument: {offenders:?}",
        offenders.len()
    );
}

#[test]
fn no_text_is_repeated_for_one_intent() {
    // A repeated row makes a rate look precise when it is not: the same
    // input counted ten times is one input.
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    let mut seen: BTreeMap<(String, String), usize> = BTreeMap::new();
    for row in &corpus.examples {
        *seen
            .entry((row.intent.as_str().to_string(), normalise(&row.text)))
            .or_insert(0) += 1;
    }
    let repeated: Vec<(&(String, String), &usize)> = seen.iter().filter(|(_, n)| **n > 1).collect();
    assert!(
        repeated.is_empty(),
        "{} text(s) are repeated: {repeated:?}",
        repeated.len()
    );
}

#[test]
fn a_text_is_never_incomplete_for_one_intent_and_complete_for_another() {
    // The head reads the text, not the intent. A text with two different
    // labels teaches it two different things about the same input.
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    let mut by_text: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for row in &corpus.examples {
        by_text
            .entry(normalise(&row.text))
            .or_default()
            .insert(row.readiness.as_str().to_string());
    }
    let contradictions: Vec<_> = by_text
        .iter()
        .filter(|(_, labels)| labels.len() > 1)
        .map(|(key, labels)| (key.clone(), labels.clone()))
        .collect();
    assert!(
        contradictions.is_empty(),
        "{} text(s) carry contradictory labels: {contradictions:?}",
        contradictions.len()
    );
}

#[test]
fn the_corpus_holds_both_sides_of_every_contrast() {
    // The corpus is contrastive: a family with only incomplete rows teaches
    // the head that a shape is incomplete without ever showing what complete
    // looks like.
    let corpus = ReadinessCorpus::load(repo(&corpus_path())).expect("the corpus must load");
    let mut families: BTreeMap<(String, String), BTreeSet<String>> = BTreeMap::new();
    for row in &corpus.examples {
        families
            .entry((row.intent.as_str().to_string(), row.family.clone()))
            .or_default()
            .insert(row.readiness.as_str().to_string());
    }
    let one_sided: Vec<_> = families
        .iter()
        .filter(|(_, labels)| labels.len() == 1 && labels.contains("needs_clarification"))
        .map(|(key, labels)| (key.clone(), labels.clone()))
        .collect();
    assert!(
        one_sided.is_empty(),
        "{} intent/family combination(s) have only incomplete rows: {one_sided:?}",
        one_sided.len()
    );
}
