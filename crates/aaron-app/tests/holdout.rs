//! The frozen hold-out must be frozen *end to end*.
//!
//! Release 0.4 froze the hold-out in `training/dataset/holdout.json` and wrote
//! a `split` into every row of the shipped dataset. The trainer read neither:
//! `Dataset::split_of` re-derived the splits from the id hash, per intent,
//! while the builder hashes the lowest id of a near-duplicate group. The two
//! algorithms disagree on every row that belongs to a near-duplicate group.
//!
//! Measured on the shipped dataset before the fix: 32 rows disagreed, and 12
//! rows that the frozen hold-out pins as validation or test were in the
//! *training* split. The validation macro-F1 that 0.4 reported was therefore
//! measured on a set that differed from the frozen hold-out in 26 of its 214
//! rows, and the early-stopping decision was steered by the wrong rows.
//!
//! These tests pin the fix: the dataset file's declared split is
//! authoritative, and the trainer's hold-out is exactly the pinned set.

use std::collections::BTreeMap;

use aaron_brain::{Dataset, Split};

fn dataset_path(relative: &str) -> String {
    format!("{}/../../{relative}", env!("CARGO_MANIFEST_DIR"))
}

/// The pinned hold-out: `{id: "validation" | "test"}`.
fn frozen_holdout() -> BTreeMap<String, String> {
    let raw = std::fs::read_to_string(dataset_path("training/dataset/holdout.json"))
        .expect("training/dataset/holdout.json must exist");
    let parsed: serde_json::Value = serde_json::from_str(&raw).expect("holdout.json must be JSON");
    parsed["rows"]
        .as_object()
        .expect("holdout.json must have a `rows` object")
        .iter()
        .map(|(id, split)| {
            (
                id.clone(),
                split.as_str().expect("split must be a string").to_string(),
            )
        })
        .collect()
}

#[test]
fn the_shipped_dataset_declares_a_split_for_every_row() {
    let dataset = Dataset::load(dataset_path("training/dataset/current.jsonl")).unwrap();
    let undeclared: Vec<&str> = dataset
        .examples
        .iter()
        .filter(|e| e.split.is_none())
        .map(|e| e.id.as_str())
        .collect();
    assert!(
        undeclared.is_empty(),
        "every row of the shipped dataset must declare its split; {} rows do not: {undeclared:?}",
        undeclared.len()
    );
}

#[test]
fn the_trainers_split_is_the_declared_split() {
    let dataset = Dataset::load(dataset_path("training/dataset/current.jsonl")).unwrap();
    let mut wrong: Vec<String> = Vec::new();
    for e in &dataset.examples {
        let declared = e.split.expect("the shipped dataset declares every split");
        if dataset.split_of(e) != declared {
            wrong.push(format!(
                "{}: declared {declared:?}, used {:?}",
                e.id,
                dataset.split_of(e)
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "the trainer must use the split the dataset declares; {} rows disagree: {wrong:?}",
        wrong.len()
    );
}

#[test]
fn the_trainers_holdout_is_exactly_the_frozen_holdout() {
    let dataset = Dataset::load(dataset_path("training/dataset/current.jsonl")).unwrap();
    let pinned = frozen_holdout();

    let mut actual: BTreeMap<String, String> = BTreeMap::new();
    for e in &dataset.examples {
        if dataset.split_of(e) != Split::Train {
            actual.insert(e.id.clone(), dataset.split_of(e).as_str().to_string());
        }
    }

    assert_eq!(
        actual.len(),
        pinned.len(),
        "the trainer's hold-out has {} rows, the frozen hold-out pins {}",
        actual.len(),
        pinned.len()
    );

    let mut mismatched: Vec<String> = Vec::new();
    for (id, split) in &pinned {
        match actual.get(id) {
            Some(seen) if seen == split => {}
            Some(seen) => mismatched.push(format!("{id}: pinned {split}, trainer has {seen}")),
            None => mismatched.push(format!("{id}: pinned {split}, trainer trained on it")),
        }
    }
    for (id, split) in &actual {
        if !pinned.contains_key(id) {
            mismatched.push(format!("{id}: trainer holds it out as {split}, not pinned"));
        }
    }
    assert!(
        mismatched.is_empty(),
        "the trainer's hold-out must be exactly the frozen hold-out; {} rows differ: {mismatched:?}",
        mismatched.len()
    );
}

#[test]
fn no_pinned_holdout_row_is_trained_on() {
    let dataset = Dataset::load(dataset_path("training/dataset/current.jsonl")).unwrap();
    let pinned = frozen_holdout();
    let leaked: Vec<&String> = pinned
        .keys()
        .filter(|id| {
            dataset
                .examples
                .iter()
                .find(|e| &e.id == *id)
                .map(|e| dataset.split_of(e) == Split::Train)
                .unwrap_or(false)
        })
        .collect();
    assert!(
        leaked.is_empty(),
        "{} pinned hold-out row(s) are in the training split: {leaked:?}",
        leaked.len()
    );
}

#[test]
fn a_partially_declared_split_is_refused() {
    // A file that declares a split for some rows but not others would mix two
    // different hold-out definitions in one dataset. It must be an error, not
    // a variant to accommodate.
    let mut examples = vec![
        aaron_brain::Example::new("hello there", aaron_core::Intent::Greeting, "test"),
        aaron_brain::Example::new("good morning to you", aaron_core::Intent::Greeting, "test"),
    ];
    examples[0].split = Some(Split::Validation);
    let err = Dataset::new("partial", examples).unwrap_err();
    let message = err.to_string();
    assert!(
        message.contains("declare a split"),
        "the error must explain the partial declaration: {message}"
    );
}

#[test]
fn a_dataset_with_no_declared_splits_still_splits_stratified() {
    // The fallback must keep working: datasets built in memory, and the
    // evaluation files, which are not split at all.
    let examples: Vec<aaron_brain::Example> = ["alpha", "beta", "gamma", "delta", "epsilon"]
        .iter()
        .map(|t| aaron_brain::Example::new(*t, aaron_core::Intent::Greeting, "test"))
        .collect();
    let dataset = Dataset::new("fallback", examples).unwrap();
    assert!(!dataset.has_declared_splits());
    let splits: Vec<Split> = dataset
        .examples
        .iter()
        .map(|e| dataset.split_of(e))
        .collect();
    assert!(
        splits.contains(&Split::Train) && splits.contains(&Split::Test),
        "the stratified fallback must produce a train and a test split: {splits:?}"
    );
}
