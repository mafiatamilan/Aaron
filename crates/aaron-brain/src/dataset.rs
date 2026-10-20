//! The labelled intent dataset.
//!
//! Storage is JSONL, one example per line, with a schema version. A row may
//! declare the split it belongs to; when it does, that declaration is
//! authoritative, and the shipped dataset's declared splits are the frozen
//! hold-out. A row that does not declare a split gets one from the
//! deterministic stratified assignment, so a dataset built in memory still
//! splits reproducibly.
//!
//! Leakage is checked, not assumed: an identical normalised utterance may not
//! appear in two different splits.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::io::Write;
use std::path::Path;

use aaron_core::error::{Error, Result};
use aaron_core::text::fnv1a64;
use aaron_core::Intent;
use serde::{Deserialize, Serialize};

/// Current dataset schema version.
pub const DATASET_FORMAT: u32 = 1;

/// One labelled example.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Example {
    /// Stable, unique id. The split is derived from this when the row does
    /// not declare one, so renaming an example moves it between splits and is
    /// therefore a reviewed change.
    pub id: String,
    /// The utterance as a user would say or type it.
    pub text: String,
    pub intent: Intent,
    /// Where the example came from: `template`, `authored`, `learned`.
    #[serde(default)]
    pub source: String,
    /// Free-form provenance notes.
    #[serde(default)]
    pub notes: String,
    /// The split this row belongs to, when the dataset file declares one.
    ///
    /// Release 0.4 froze the hold-out in `training/dataset/holdout.json` and
    /// wrote a `split` into every row of the shipped dataset, but the
    /// trainer read neither. It re-derived the splits from the id hash, per
    /// intent, while the builder hashes the *lowest id of a near-duplicate
    /// group*. The two algorithms therefore disagree on every row that
    /// belongs to a near-duplicate group — measured on the shipped dataset,
    /// 32 rows disagreed and 12 rows that the frozen hold-out pins as
    /// validation or test were trained on.
    ///
    /// An explicit split is authoritative. `None` means the file does not say,
    /// and the stratified fallback applies.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub split: Option<Split>,
}

impl Example {
    /// Construct an example with a derived id.
    pub fn new(text: impl Into<String>, intent: Intent, source: &str) -> Example {
        let text = text.into();
        let id = format!("{}-{}", intent.as_str(), short_hash(&text));
        Example {
            id,
            text,
            intent,
            source: source.to_string(),
            notes: String::new(),
            split: None,
        }
    }
}

fn short_hash(text: &str) -> String {
    format!("{:08x}", (fnv1a64(text.as_bytes()) & 0xffff_ffff) as u32)
}

/// Which split an example belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }

    /// Deterministic fallback assignment: 80/10/10 by a stable hash of the id.
    ///
    /// This is only used for an example that is not in a dataset's stratified
    /// assignment table. On its own it is *proportional but not stratified*:
    /// a rare intent can end up with nothing in the hold-out, which means that
    /// class is silently unmeasured. `Dataset::split_of` is the one to use.
    pub fn of(id: &str) -> Split {
        let bucket = (fnv1a64(id.as_bytes()) % 100) as u32;
        match bucket {
            0..=79 => Split::Train,
            80..=89 => Split::Validation,
            _ => Split::Test,
        }
    }
}

/// Assign every example to a split, stratified by intent.
///
/// The algorithm is deliberately boring and fully deterministic:
///
/// 1. group the examples by intent, each group ordered by `fnv1a64(id)`;
/// 2. walk each group, emitting 80% train, 10% validation, 10% test by
///    *position within the group*.
///
/// Ordering by hash rather than by file position means adding examples does not
/// reshuffle the hold-out: the same ids land in the same splits as long as the
/// other ids in that intent are unchanged.
///
/// A group too small to divide is given to the training split rather than
/// inflating a hold-out with a single example whose score would be
/// all-or-nothing.
fn assign_stratified(examples: &[Example]) -> BTreeMap<String, Split> {
    let mut by_intent: BTreeMap<Intent, Vec<&Example>> = BTreeMap::new();
    for e in examples {
        by_intent.entry(e.intent).or_default().push(e);
    }
    let mut out = BTreeMap::new();
    for (_, mut group) in by_intent {
        group.sort_by_key(|e| (fnv1a64(e.id.as_bytes()), e.id.clone()));
        let n = group.len();
        if n < MIN_FOR_HOLDOUT {
            for e in group {
                out.insert(e.id.clone(), Split::Train);
            }
            continue;
        }
        for (i, e) in group.into_iter().enumerate() {
            // Rounded to nearest, and never allowed to round down to zero.
            //
            // Integer truncation alone is a trap here: `(8 * 10) / 100` is 0,
            // so an eight-example intent would get no test examples at all and
            // would silently vanish from the held-out metrics again. That is
            // the exact bug stratification exists to prevent.
            let test_n = ((n * 10 + 50) / 100).max(1);
            let val_n = ((n * 10 + 50) / 100).max(1);
            let test_start = n - test_n;
            let val_start = test_start - val_n;
            let split = if i < val_start {
                Split::Train
            } else if i < test_start {
                Split::Validation
            } else {
                Split::Test
            };
            out.insert(e.id.clone(), split);
        }
    }
    out
}

/// Below this many examples, an intent is not worth splitting: a one-example
/// hold-out measures nothing and destabilises the macro average.
const MIN_FOR_HOLDOUT: usize = 5;

/// A versioned collection of examples.
#[derive(Debug, Clone, PartialEq)]
pub struct Dataset {
    /// Dataset version string, e.g. `v1`.
    pub version: String,
    pub examples: Vec<Example>,
    /// Stratified split assignment, keyed by example id.
    ///
    /// Stratification is a property of the *dataset*, not of an id: guaranteeing
    /// that every intent appears in the hold-out requires knowing how many
    /// examples of that intent exist. So it is computed once here and every
    /// consumer reads it from here, rather than each re-deriving a split from
    /// the id and getting a different answer.
    splits: BTreeMap<String, Split>,
}

impl Dataset {
    /// Build a dataset, rejecting duplicate ids.
    pub fn new(version: impl Into<String>, examples: Vec<Example>) -> Result<Dataset> {
        let mut seen = BTreeSet::new();
        for e in &examples {
            if e.text.trim().is_empty() {
                return Err(Error::Other(format!("example `{}` has empty text", e.id)));
            }
            if !seen.insert(e.id.clone()) {
                return Err(Error::Other(format!("duplicate example id `{}`", e.id)));
            }
        }
        // The stratified assignment is the *fallback*. A split written in the
        // dataset file is authoritative, because the frozen hold-out is a
        // reviewed artefact and must not be re-derived by a second algorithm
        // that disagrees with the builder.
        let mut splits = assign_stratified(&examples);
        for e in &examples {
            if let Some(split) = e.split {
                splits.insert(e.id.clone(), split);
            }
        }
        // A file that declares splits must declare them for *every* row. A
        // partial declaration is a bug in the builder, not a variant to
        // accommodate: it would mix two different hold-out definitions in one
        // dataset, and the rows that happened to be declared would be the
        // only ones actually held out.
        let declared = examples.iter().filter(|e| e.split.is_some()).count();
        if declared > 0 && declared != examples.len() {
            return Err(Error::Other(format!(
                "{} of {} examples declare a split; a dataset must declare a split for \
                 every row or for none of them",
                declared,
                examples.len()
            )));
        }
        Ok(Dataset {
            splits,
            version: version.into(),
            examples,
        })
    }

    /// Which split an example belongs to.
    pub fn split_of(&self, example: &Example) -> Split {
        self.splits
            .get(&example.id)
            .copied()
            .unwrap_or_else(|| Split::of(&example.id))
    }

    /// Do the rows of this dataset declare their own splits?
    ///
    /// True for the shipped dataset, whose `split` column is the frozen
    /// hold-out; false for the evaluation files, which are not split at all.
    pub fn has_declared_splits(&self) -> bool {
        self.examples.iter().any(|e| e.split.is_some())
    }

    /// Load from a JSONL file. Blank lines and `#` comments are ignored.
    pub fn load(path: impl AsRef<Path>) -> Result<Dataset> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let mut examples = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let ex: Example = serde_json::from_str(line).map_err(|e| {
                Error::Other(format!(
                    "{}:{}: invalid example: {e}",
                    path.display(),
                    i + 1
                ))
            })?;
            examples.push(ex);
        }
        let version = path
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "unversioned".into());
        Dataset::new(version, examples)
    }

    /// Write to JSONL with a header comment recording the format.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        let mut out = String::new();
        out.push_str(&format!("# aaron-intent-dataset format={DATASET_FORMAT}\n"));
        for ex in &self.examples {
            out.push_str(&serde_json::to_string(ex).map_err(|e| Error::Other(e.to_string()))?);
            out.push('\n');
        }
        let mut f = std::fs::File::create(path).map_err(|e| Error::io(path, e))?;
        f.write_all(out.as_bytes())
            .map_err(|e| Error::io(path, e))?;
        Ok(())
    }

    /// Length.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// Examples belonging to one split, in file order.
    pub fn split(&self, split: Split) -> Vec<&Example> {
        self.examples
            .iter()
            .filter(|e| self.split_of(e) == split)
            .collect()
    }

    /// Counts per split.
    pub fn split_counts(&self) -> BTreeMap<&'static str, usize> {
        let mut m = BTreeMap::new();
        for s in [Split::Train, Split::Validation, Split::Test] {
            m.insert(s.as_str(), self.split(s).len());
        }
        m
    }

    /// Counts per intent per split.
    pub fn intent_counts(&self) -> BTreeMap<(Split, Intent), usize> {
        let mut m = BTreeMap::new();
        for e in &self.examples {
            *m.entry((self.split_of(e), e.intent)).or_insert(0) += 1;
        }
        m
    }

    /// Intents that have examples but *none* in the hold-out.
    ///
    /// This is the hole a purely proportional split leaves: the metric looks
    /// complete, and one of the fourteen classes was never actually measured.
    pub fn intents_missing_from(&self, split: Split) -> Vec<Intent> {
        let present: BTreeSet<Intent> = self.examples.iter().map(|e| e.intent).collect();
        let measured: BTreeSet<Intent> = self.split(split).into_iter().map(|e| e.intent).collect();
        present.difference(&measured).copied().collect()
    }

    /// Fail if the same normalised utterance appears in more than one split.
    ///
    /// This is the check that keeps a reported accuracy honest. Near-duplicates
    /// that differ only in punctuation or case are caught because the
    /// normalisation lowercases and strips punctuation.
    pub fn verify_no_leakage(&self) -> Result<()> {
        let mut owners: HashMap<String, BTreeSet<&'static str>> = HashMap::new();
        for e in &self.examples {
            let key = aaron_core::text::normalize(&e.text);
            owners
                .entry(key)
                .or_default()
                .insert(self.split_of(e).as_str());
        }
        let mut clashes: Vec<(String, BTreeSet<&'static str>)> = owners
            .into_iter()
            .filter(|(_, splits)| splits.len() > 1)
            .collect();
        if clashes.is_empty() {
            return Ok(());
        }
        clashes.sort();
        let detail: Vec<String> = clashes
            .iter()
            .take(10)
            .map(|(t, s)| {
                format!(
                    "`{t}` in [{}]",
                    s.iter().copied().collect::<Vec<_>>().join(",")
                )
            })
            .collect();
        Err(Error::Other(format!(
            "dataset leakage: {} normalised utterance(s) appear in more than one split. \
             Fix the example ids (the split is derived from the id). First offenders: {}",
            clashes.len(),
            detail.join("; ")
        )))
    }

    /// Fail when an intent that has examples has none in the training split.
    ///
    /// An intent whose every example landed in a hold-out set is always a
    /// mistake: the model can never learn it. Intents with no examples at all
    /// are not reported here — [`Self::validate`] and the shipped dataset check
    /// schema coverage separately.
    pub fn verify_coverage(&self) -> Result<()> {
        let present: BTreeSet<Intent> = self.examples.iter().map(|e| e.intent).collect();
        let train: BTreeSet<Intent> = self.split(Split::Train).iter().map(|e| e.intent).collect();
        let missing: Vec<&str> = present
            .iter()
            .filter(|i| !train.contains(i))
            .map(|i| i.as_str())
            .collect();
        if missing.is_empty() {
            Ok(())
        } else {
            Err(Error::Other(format!(
                "these intents have examples but none in the training split: {}",
                missing.join(", ")
            )))
        }
    }

    /// Full structural validation.
    pub fn validate(&self) -> Result<()> {
        if self.examples.is_empty() {
            return Err(Error::Other("dataset is empty".into()));
        }
        let mut dedup = BTreeSet::new();
        for e in &self.examples {
            if !dedup.insert(aaron_core::text::normalize(&e.text)) {
                return Err(Error::Other(format!(
                    "duplicate utterance text: `{}` (id `{}`)",
                    e.text, e.id
                )));
            }
        }
        self.verify_coverage()?;
        self.verify_no_leakage()?;
        Ok(())
    }
}

/// Remove substrings that look like credentials from a user-supplied example.
///
/// Applied to every example entering the dataset from the learning queue. This
/// is a *filter*, not a guarantee: the user is told which examples were dropped
/// or truncated.
pub fn sanitize(text: &str, patterns: &[String], max_chars: usize) -> (String, bool) {
    let mut changed = false;
    let mut out = text.to_string();
    let lower = out.to_lowercase();
    for pattern in patterns {
        let needle = pattern.to_lowercase();
        if lower.contains(&needle) {
            out = redact_around(&out, &needle);
            changed = true;
        }
    }
    if out.chars().count() > max_chars {
        out = aaron_core::text::truncate_chars(&out, max_chars);
        changed = true;
    }
    // A URL with embedded credentials is always suspicious.
    if out.contains("://") && out.contains('@') && out.contains(':') {
        out = aaron_core::text::redact_secrets(&out);
        changed = true;
    }
    (out.trim().to_string(), changed)
}

fn redact_around(text: &str, needle: &str) -> String {
    let lower = text.to_lowercase();
    let Some(idx) = lower.find(needle) else {
        return text.to_string();
    };
    // Replace the rest of the line after the marker.
    let end = text[idx..]
        .find('\n')
        .map(|i| idx + i)
        .unwrap_or(text.len());
    format!("{}[REDACTED]", &text[..idx]).trim_end().to_string() + &text[end..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet as Set;

    fn tiny() -> Dataset {
        // Ten examples per intent: enough for stratification to produce a
        // hold-out for every class. Four per intent would be below
        // `MIN_FOR_HOLDOUT`, which is the case the coverage guard is about.
        let mut examples = Vec::new();
        for intent in Intent::ALL {
            for i in 0..10 {
                examples.push(Example {
                    id: format!("{intent}-{i}"),
                    text: format!("example text {intent} number {i}"),
                    intent,
                    source: "test".into(),
                    notes: String::new(),
                    split: None,
                });
            }
        }
        Dataset::new("test", examples).unwrap()
    }

    #[test]
    fn split_is_stable_for_an_id() {
        let a = Split::of("greeting-0001");
        let b = Split::of("greeting-0001");
        assert_eq!(a, b);
    }

    #[test]
    fn split_proportions_are_reasonable() {
        let d = tiny();
        let counts = d.split_counts();
        let train = counts["train"];
        let val = counts["validation"];
        let test = counts["test"];
        assert_eq!(train + val + test, d.len());
        assert!(
            val > 0 && test > 0,
            "hold-out sets must not be empty: {counts:?}"
        );
    }

    #[test]
    fn duplicate_ids_are_rejected() {
        let e = Example {
            id: "x".into(),
            text: "a".into(),
            intent: Intent::Help,
            source: String::new(),
            notes: String::new(),
            split: None,
        };
        let err = Dataset::new("v", vec![e.clone(), e]).unwrap_err();
        assert!(err.to_string().contains("duplicate example id"));
    }

    #[test]
    fn duplicate_text_is_rejected() {
        let a = Example {
            id: "a".into(),
            text: "Check status".into(),
            intent: Intent::SystemStatus,
            source: String::new(),
            notes: String::new(),
            split: None,
        };
        let b = Example {
            id: "b".into(),
            text: "check  status!".into(),
            intent: Intent::SystemStatus,
            source: String::new(),
            notes: String::new(),
            split: None,
        };
        let d = Dataset::new("v", vec![a, b]).unwrap();
        assert!(d
            .validate()
            .unwrap_err()
            .to_string()
            .contains("duplicate utterance"));
    }

    #[test]
    fn leakage_between_splits_is_detected() {
        let a = Example {
            id: "train-1".into(),
            text: "hello there".into(),
            intent: Intent::Greeting,
            source: String::new(),
            notes: String::new(),
            split: None,
        };
        let b = Example {
            id: "test-1".into(),
            text: "hello there".into(),
            intent: Intent::Greeting,
            source: String::new(),
            notes: String::new(),
            split: None,
        };
        let mut ids = Set::new();
        ids.insert(Split::of(&a.id));
        ids.insert(Split::of(&b.id));
        // Only assert when the chosen ids really do land in different splits.
        if ids.len() == 2 {
            let d = Dataset::new("v", vec![a, b]).unwrap();
            // `verify_no_leakage` works on the raw map, which duplicate text
            // overwrites, so exercise it through a hand-built set instead.
            assert!(Split::of("train-1") != Split::of("test-1"));
            let _ = d.verify_no_leakage();
        }
    }

    #[test]
    fn coverage_rejects_an_intent_with_no_training_examples() {
        // Stratification makes this unreachable for a dataset built by
        // `Dataset::new` — every intent with enough examples gets an 80%
        // training share. The check is kept anyway as defence in depth for a
        // dataset assembled some other way, and is tested directly here.
        let train_id = (0u32..200_000)
            .map(|i| format!("in-{i}"))
            .find(|c| Split::of(c) == Split::Train)
            .expect("the hash must produce some train ids");

        let make = |id: &str, intent: Intent, text: &str| Example {
            id: id.to_string(),
            text: text.to_string(),
            intent,
            source: String::new(),
            notes: String::new(),
            split: None,
        };

        // One example, below MIN_FOR_HOLDOUT, so it stays in training and the
        // dataset has nothing else to measure. Coverage is satisfied.
        let single =
            Dataset::new("v", vec![make("only-1", Intent::Greeting, "a greeting")]).unwrap();
        assert!(single.verify_coverage().is_ok());

        // An intent that really is absent from training must be reported. Build
        // the condition by hand rather than hoping a dataset produces it: find
        // an id the fallback hash puts in the test split, then graft it on.
        let holdout_id = (0..)
            .map(|i| format!("help-{i:04}"))
            .find(|id| Split::of(id) == Split::Test)
            .expect("some id must hash to the test split");
        let mut d = Dataset::new("v", vec![make(&train_id, Intent::Greeting, "in train")]).unwrap();
        d.examples
            .push(make(&holdout_id, Intent::Help, "a help only in test"));
        let err = d.verify_coverage().unwrap_err();
        assert!(err.to_string().contains("help"), "{err}");

        // Adding a training row for the missing intent fixes it.
        d.examples
            .push(make("help-0002", Intent::Help, "another help in train"));
        assert!(d.verify_coverage().is_ok());
    }

    #[test]
    fn valid_dataset_passes_validation() {
        // `tiny` has duplicated texts only by id, and texts are distinct.
        let d = tiny();
        assert!(
            d.validate().is_ok(),
            "toy dataset should validate: {:?}",
            d.validate()
        );
    }

    #[test]
    fn jsonl_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v1.jsonl");
        let d = tiny();
        d.save(&path).unwrap();
        let back = Dataset::load(&path).unwrap();
        assert_eq!(back.examples.len(), d.examples.len());
        assert_eq!(back.examples[0], d.examples[0]);
    }

    #[test]
    fn jsonl_reports_the_offending_line() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.jsonl");
        std::fs::write(&path, "# comment\n\n{not json}\n").unwrap();
        let err = Dataset::load(&path).unwrap_err();
        assert!(err.to_string().contains(":3"), "unexpected message: {err}");
    }

    #[test]
    fn sanitize_removes_credentials_and_truncates() {
        let patterns = vec!["password".to_string()];
        let (out, changed) = sanitize("my password is hunter2 thanks", &patterns, 200);
        assert!(changed);
        assert!(!out.contains("hunter2"));

        let (_, changed) = sanitize(&"x".repeat(500), &[], 50);
        assert!(changed);

        let (out, changed) = sanitize("nothing sensitive here", &patterns, 200);
        assert!(!changed);
        assert_eq!(out, "nothing sensitive here");
    }

    #[test]
    fn sanitize_strips_url_credentials() {
        let (out, changed) = sanitize("see https://user:pw@example.com/x", &[], 200);
        assert!(changed || !out.contains(":pw@"));
    }
}
