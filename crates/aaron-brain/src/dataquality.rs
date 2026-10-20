//! Dataset quality checks.
//!
//! A classifier can only be as good as its labels. This module answers the
//! questions that decide whether a reported metric means anything:
//!
//! * Is the file schema valid and are the labels in the closed `Intent` set?
//! * Are there exact or near-duplicate examples?
//! * Does any normalised utterance appear in more than one split?
//! * How many examples does each intent have in each split?
//! * Which intents have too little hold-out data to be *measured*?
//!
//! # Why duplicates matter more than they look
//!
//! An exact duplicate inflates the score of whichever split it lands in. A
//! *near*-duplicate is worse: two sentences that differ by one function word
//! will be hashed to overlapping feature buckets, so the model can score well
//! on both while having learned nothing generalisable.
//!
//! # Why coverage matters
//!
//! A macro average over intents that were never measured is not a quality
//! score, it is an average over a subset that happens to look complete. The
//! report says which intents were actually evaluated.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::dataset::{Dataset, Example, Split};

/// One problem found in the data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Finding {
    /// Machine-readable category.
    pub kind: FindingKind,
    /// What is wrong.
    pub detail: String,
    /// How many examples are affected.
    pub count: usize,
    /// Whether this must be fixed before the dataset is usable for training.
    pub blocking: bool,
}

/// Categories of data defect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FindingKind {
    /// Duplicate example id.
    DuplicateId,
    /// Two examples whose normalised text is identical.
    DuplicateText,
    /// Two examples that are very similar without being identical.
    NearDuplicate,
    /// The same utterance in more than one split.
    Leakage,
    /// An intent with too few examples to be trained or measured.
    InsufficientCoverage,
    /// An example with text that is empty or only punctuation.
    EmptyText,
    /// An example whose intent has no examples anywhere.
    OrphanIntent,
    /// Split sizes that are wildly unbalanced.
    SkewedSplit,
}

impl FindingKind {
    pub fn as_str(self) -> &'static str {
        match self {
            FindingKind::DuplicateId => "duplicate_id",
            FindingKind::DuplicateText => "duplicate_text",
            FindingKind::NearDuplicate => "near_duplicate",
            FindingKind::Leakage => "leakage",
            FindingKind::InsufficientCoverage => "insufficient_coverage",
            FindingKind::EmptyText => "empty_text",
            FindingKind::OrphanIntent => "orphan_intent",
            FindingKind::SkewedSplit => "skewed_split",
        }
    }
}

/// Below this many hold-out examples an intent's metrics are not reported.
pub const MIN_HOLDOUT_FOR_REPORTING: usize = 5;

/// Below this many training examples an intent cannot be learned.
pub const MIN_TRAIN_FOR_LEARNING: usize = 10;

/// Similarity above which two sentences count as near-duplicates.
///
/// Tuned by inspection against the shipped dataset, not by taste.
pub const NEAR_DUPLICATE_THRESHOLD: f64 = 0.8;

/// A full quality report for one dataset.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct QualityReport {
    /// Dataset version string.
    pub version: String,
    /// SHA-256 of the canonical example content: ids, texts and labels.
    ///
    /// Two datasets with the same hash train the same model. Two reports with
    /// different hashes are not comparable, and this is what makes that
    /// checkable.
    pub content_sha256: String,
    /// Total examples.
    pub total: usize,
    /// Examples per split.
    pub split_counts: BTreeMap<String, usize>,
    /// Examples per intent per split.
    pub per_intent: BTreeMap<String, IntentCounts>,
    /// `source` field distribution, so authored and generated data are visible.
    pub provenance: BTreeMap<String, usize>,
    /// Intents with fewer than [`MIN_HOLDOUT_FOR_REPORTING`] test examples.
    pub intents_under_measured: Vec<String>,
    /// Intents with fewer than [`MIN_TRAIN_FOR_LEARNING`] training examples.
    pub intents_under_trained: Vec<String>,
    /// Everything that is wrong.
    pub findings: Vec<Finding>,
    /// Convenience: true when no finding is blocking.
    pub usable_for_training: bool,
}

/// Per-intent split counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntentCounts {
    pub train: usize,
    pub validation: usize,
    pub test: usize,
}

impl IntentCounts {
    fn new() -> IntentCounts {
        IntentCounts {
            train: 0,
            validation: 0,
            test: 0,
        }
    }

    fn add(&mut self, split: Split) {
        match split {
            Split::Train => self.train += 1,
            Split::Validation => self.validation += 1,
            Split::Test => self.test += 1,
        }
    }

    /// Total across all splits.
    pub fn total(&self) -> usize {
        self.train + self.validation + self.test
    }
}

/// Content hash of a dataset: ids, normalised texts and labels.
///
/// Deliberately *not* a hash of the file bytes: reformatting the JSONL or
/// reordering it must not change the hash, because none of those change the
/// training set. It *does* change when a label or a text changes.
pub fn content_sha256(dataset: &Dataset) -> String {
    let mut lines: Vec<String> = dataset
        .examples
        .iter()
        .map(|e| {
            format!(
                "{}\u{1}{}\u{1}{}",
                e.id,
                aaron_core::text::normalize(&e.text),
                e.intent.as_str()
            )
        })
        .collect();
    lines.sort();
    let joined = lines.join("\n");
    sha256_hex(joined.as_bytes())
}

/// Minimal SHA-256, so this module needs no extra dependency.
///
/// Written out rather than pulled in because the dataset report is one of the
/// few things that must keep working in a trimmed checkout.
fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|w| format!("{w:08x}")).collect()
}

/// Word 3-gram shingles, lower-cased and stripped.
fn shingles(text: &str) -> BTreeSet<String> {
    let words: Vec<String> = aaron_core::text::normalize(text)
        .split_whitespace()
        .map(|s| s.to_string())
        .collect();
    if words.is_empty() {
        return BTreeSet::new();
    }
    if words.len() < 3 {
        return words.iter().cloned().collect();
    }
    words.windows(3).map(|w| w.join(" ")).collect()
}

/// Containment of the smaller shingle set in the larger.
///
/// Jaccard is the wrong measure for short commands. "read the file main.rs"
/// has three 3-grams; adding the word "please" produces four, of which three
/// are shared — Jaccard says 0.75, containment says 1.0. Every paraphrase in a
/// command set is of that shape: *the same sentence plus a filler word*.
/// Containment catches exactly that and leaves genuinely different commands
/// alone, because two unrelated commands share no 3-grams at all.
pub fn containment(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    a.intersection(b).count() as f64 / a.len().min(b.len()) as f64
}

/// Near-duplicate similarity: the stronger of Jaccard and containment.
///
/// Both are reported separately in the test suite because they fail in
/// different ways, and picking only one hides the other's blind spot.
pub fn similarity(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    jaccard(a, b).max(containment(a, b))
}

/// Jaccard similarity of two shingle sets.
pub fn jaccard(a: &BTreeSet<String>, b: &BTreeSet<String>) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let inter = a.intersection(b).count();
    let union = a.union(b).count();
    inter as f64 / union as f64
}

/// Analyse a dataset and produce a full quality report.
pub fn analyse(dataset: &Dataset) -> QualityReport {
    let mut findings: Vec<Finding> = Vec::new();

    // --- schema and empty text -------------------------------------------
    let empty: Vec<&Example> = dataset
        .examples
        .iter()
        .filter(|e| aaron_core::text::normalize(&e.text).is_empty())
        .collect();
    if !empty.is_empty() {
        findings.push(Finding {
            kind: FindingKind::EmptyText,
            detail: format!(
                "{} example(s) have no words after normalisation, e.g. `{}`",
                empty.len(),
                empty[0].id
            ),
            count: empty.len(),
            blocking: true,
        });
    }

    // --- duplicate ids ----------------------------------------------------
    let mut by_id: BTreeMap<&str, usize> = BTreeMap::new();
    for e in &dataset.examples {
        *by_id.entry(e.id.as_str()).or_insert(0) += 1;
    }
    let dup_ids: Vec<(&str, usize)> = by_id
        .iter()
        .filter(|(_, n)| **n > 1)
        .map(|(k, v)| (*k, *v))
        .collect();
    if !dup_ids.is_empty() {
        findings.push(Finding {
            kind: FindingKind::DuplicateId,
            detail: format!(
                "{} id(s) appear more than once, e.g. `{}`",
                dup_ids.len(),
                dup_ids[0].0
            ),
            count: dup_ids.len(),
            blocking: true,
        });
    }

    // --- duplicate normalised text ---------------------------------------
    let mut by_text: BTreeMap<String, Vec<&str>> = BTreeMap::new();
    for e in &dataset.examples {
        by_text
            .entry(aaron_core::text::normalize(&e.text))
            .or_default()
            .push(e.id.as_str());
    }
    let dup_text: Vec<(String, Vec<&str>)> = by_text
        .into_iter()
        .filter(|(_, ids)| ids.len() > 1)
        .collect();
    if !dup_text.is_empty() {
        let affected: usize = dup_text.iter().map(|(_, ids)| ids.len()).sum();
        findings.push(Finding {
            kind: FindingKind::DuplicateText,
            detail: format!(
                "{} normalised text(s) occur more than once, e.g. `{}` ({} ids)",
                dup_text.len(),
                dup_text[0].0,
                dup_text[0].1.len()
            ),
            count: affected,
            blocking: true,
        });
    }

    // --- leakage ----------------------------------------------------------
    let mut owner: BTreeMap<String, BTreeSet<&'static str>> = BTreeMap::new();
    for e in &dataset.examples {
        owner
            .entry(aaron_core::text::normalize(&e.text))
            .or_default()
            .insert(dataset.split_of(e).as_str());
    }
    let leaks: Vec<(&String, &BTreeSet<&'static str>)> =
        owner.iter().filter(|(_, s)| s.len() > 1).collect();
    if !leaks.is_empty() {
        findings.push(Finding {
            kind: FindingKind::Leakage,
            detail: format!(
                "{} utterance(s) appear in more than one split, e.g. `{}` in {:?}",
                leaks.len(),
                leaks[0].0,
                leaks[0].1
            ),
            count: leaks.len(),
            blocking: true,
        });
    }

    // --- near duplicates --------------------------------------------------
    // Only compare within a split. A near-duplicate across splits is already
    // reported as leakage when the texts match exactly; for fuzzy matches the
    // interesting case is "the same sentence twice in one split", which
    // inflates that split's score.
    let mut shingle_cache: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for e in &dataset.examples {
        shingle_cache.insert(e.id.as_str(), shingles(&e.text));
    }
    let mut near: Vec<(String, String, f64, Split)> = Vec::new();
    let by_split: BTreeMap<&'static str, Vec<&Example>> = {
        let mut m: BTreeMap<&'static str, Vec<&Example>> = BTreeMap::new();
        for e in &dataset.examples {
            m.entry(dataset.split_of(e).as_str()).or_default().push(e);
        }
        m
    };
    for (split_name, group) in &by_split {
        for i in 0..group.len() {
            for j in (i + 1)..group.len() {
                let a = &shingle_cache[group[i].id.as_str()];
                let b = &shingle_cache[group[j].id.as_str()];
                let sim = similarity(a, b);
                if sim >= NEAR_DUPLICATE_THRESHOLD
                    && aaron_core::text::normalize(&group[i].text)
                        != aaron_core::text::normalize(&group[j].text)
                {
                    near.push((
                        group[i].text.clone(),
                        group[j].text.clone(),
                        sim,
                        Split::of(split_name),
                    ));
                }
            }
        }
    }
    if !near.is_empty() {
        near.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        findings.push(Finding {
            kind: FindingKind::NearDuplicate,
            detail: format!(
                "{} near-duplicate pair(s) within a split, most similar at {:.2}: `{}` / `{}`",
                near.len(),
                near[0].2,
                truncate(&near[0].0, 60),
                truncate(&near[0].1, 60)
            ),
            count: near.len(),
            // Not blocking: a near-duplicate is a quality problem, but the
            // dataset is still trainable and the report makes it visible.
            blocking: false,
        });
    }

    // --- per-intent counts -------------------------------------------------
    let mut per_intent: BTreeMap<String, IntentCounts> = BTreeMap::new();
    let mut provenance: BTreeMap<String, usize> = BTreeMap::new();
    for e in &dataset.examples {
        per_intent
            .entry(e.intent.as_str().to_string())
            .or_insert_with(IntentCounts::new)
            .add(dataset.split_of(e));
        *provenance.entry(e.source.clone()).or_insert(0) += 1;
    }
    // Intents with no examples at all are still reported, so a class added to
    // the enum cannot quietly vanish from the table.
    for intent in aaron_core::Intent::ALL {
        per_intent
            .entry(intent.as_str().to_string())
            .or_insert_with(IntentCounts::new);
    }

    let mut intents_under_measured: Vec<String> = per_intent
        .iter()
        .filter(|(_, c)| c.test < MIN_HOLDOUT_FOR_REPORTING)
        .map(|(k, _)| k.clone())
        .collect();
    intents_under_measured.sort();

    let mut intents_under_trained: Vec<String> = per_intent
        .iter()
        .filter(|(_, c)| c.train < MIN_TRAIN_FOR_LEARNING)
        .map(|(k, _)| k.clone())
        .collect();
    intents_under_trained.sort();

    if !intents_under_trained.is_empty() {
        findings.push(Finding {
            kind: FindingKind::InsufficientCoverage,
            detail: format!(
                "{} intent(s) have fewer than {MIN_TRAIN_FOR_LEARNING} training examples: {}",
                intents_under_trained.len(),
                intents_under_trained.join(", ")
            ),
            count: intents_under_trained.len(),
            blocking: false,
        });
    }

    // --- split balance ------------------------------------------------------
    let split_counts: BTreeMap<String, usize> = by_split
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.len()))
        .collect();
    let largest = split_counts.values().copied().max().unwrap_or(0).max(1);
    let smallest = split_counts.values().copied().min().unwrap_or(0);
    if smallest * 10 < largest {
        findings.push(Finding {
            kind: FindingKind::SkewedSplit,
            detail: format!(
                "split sizes are badly unbalanced: {split_counts:?}; \
                 a near-empty split makes early stopping and hold-out metrics noisy"
            ),
            count: 1,
            blocking: false,
        });
    }

    // Sort so the report is stable regardless of hash-map iteration order.
    findings.sort_by(|a, b| {
        (a.blocking == b.blocking, a.kind, a.detail.clone())
            .cmp(&(false, b.kind, b.detail.clone()))
            .then(a.blocking.cmp(&b.blocking))
    });

    QualityReport {
        version: dataset.version.clone(),
        content_sha256: content_sha256(dataset),
        total: dataset.len(),
        split_counts,
        per_intent,
        provenance,
        intents_under_measured,
        intents_under_trained,
        usable_for_training: !findings.iter().any(|f| f.blocking),
        findings,
    }
}

fn truncate(s: &str, n: usize) -> String {
    aaron_core::text::truncate_chars(s, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use aaron_core::Intent;

    fn example(id: &str, text: &str, intent: Intent, source: &str) -> Example {
        Example {
            id: id.into(),
            text: text.into(),
            intent,
            source: source.into(),
            notes: String::new(),
            split: None,
        }
    }

    fn balanced(intent: Intent, n: usize) -> Vec<Example> {
        (0..n)
            .map(|i| {
                example(
                    &format!("{}-{i:04}", intent.as_str()),
                    &format!("{} request number {i}", intent.as_str()),
                    intent,
                    "authored",
                )
            })
            .collect()
    }

    /// Sixty examples per intent: the stratification gives 48 train / 6 val /
    /// 6 test, which clears both coverage floors. A smaller fixture silently
    /// produces a dataset that is *itself* flagged, which is how this fixture
    /// first failed.
    fn healthy() -> Dataset {
        let mut ex = Vec::new();
        for intent in Intent::ALL {
            ex.extend(balanced(intent, 60));
        }
        Dataset::new("test-v1", ex).unwrap()
    }

    /// An id that lands in the requested split, so a test can place examples
    /// deliberately rather than hoping the hash cooperates.
    fn id_in_split(prefix: &str, want: Split) -> String {
        (0..100_000)
            .map(|i| format!("{prefix}-{i:06}"))
            .find(|id| Split::of(id) == want)
            .expect("some id must hash to the requested split")
    }

    // ------------------------------------------------------------ hashing

    #[test]
    fn the_content_hash_is_stable_and_label_sensitive() {
        let d = healthy();
        let a = content_sha256(&d);
        assert_eq!(a, content_sha256(&d), "hashing must be deterministic");

        // Reordering the file must not change the hash: the training set is the
        // same set.
        let mut reordered = d.clone();
        reordered.examples.reverse();
        assert_eq!(a, content_sha256(&reordered), "file order must not matter");

        // Changing a label must change it, or two different training sets would
        // be reported as the same experiment.
        let mut relabelled = d.clone();
        relabelled.examples[0].intent = Intent::Help;
        assert_ne!(a, content_sha256(&relabelled));
    }

    #[test]
    fn the_content_hash_matches_known_vectors() {
        // Pinned so a refactor of the hash cannot silently change every
        // recorded dataset identity.
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    // ------------------------------------------------------------ findings

    #[test]
    fn a_healthy_dataset_reports_no_blocking_findings() {
        let report = analyse(&healthy());
        assert!(report.usable_for_training, "{:?}", report.findings);
        assert_eq!(report.total, 60 * Intent::ALL.len());
        assert!(
            report.intents_under_measured.is_empty(),
            "{:?}",
            report.intents_under_measured
        );
        assert!(
            report.intents_under_trained.is_empty(),
            "{:?}",
            report.intents_under_trained
        );
    }

    #[test]
    fn exact_duplicate_text_is_blocking() {
        let mut ex = balanced(Intent::Greeting, 30);
        ex.push(example(
            "dup-1",
            "hello there",
            Intent::Greeting,
            "authored",
        ));
        ex.push(example(
            "dup-2",
            "Hello there!",
            Intent::Greeting,
            "authored",
        ));
        let d = Dataset::new("v", ex).unwrap();
        let r = analyse(&d);
        assert!(!r.usable_for_training);
        assert!(
            r.findings
                .iter()
                .any(|f| f.kind == FindingKind::DuplicateText),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn duplicate_ids_are_rejected_at_construction() {
        // `analyse` also checks for duplicate ids, but `Dataset::new` makes
        // that state unreachable: the constructor is the enforcement point, so
        // this is the contract that actually holds. The check in `analyse`
        // stays as defence in depth for a dataset assembled by another route.
        let mut ex = balanced(Intent::Greeting, 30);
        ex.push(example(
            "greeting-0000",
            "a brand new sentence",
            Intent::Greeting,
            "authored",
        ));
        let err = Dataset::new("v", ex).unwrap_err();
        assert!(err.to_string().contains("duplicate example id"), "{err}");
    }

    #[test]
    fn empty_text_is_blocking() {
        let mut ex = balanced(Intent::Greeting, 30);
        ex.push(example("empty-1", "   ...  ", Intent::Greeting, "authored"));
        let d = Dataset::new("v", ex).unwrap();
        let r = analyse(&d);
        assert!(r.findings.iter().any(|f| f.kind == FindingKind::EmptyText));
    }

    #[test]
    fn leakage_is_detected_and_blocking() {
        let mut ex = balanced(Intent::Help, 30);
        // Re-use an existing normalised text under a different id. The split is
        // derived from the id, so this is exactly how leakage happens in
        // practice.
        let victim = balanced(Intent::Help, 1)[0].text.clone();
        ex.push(example("leak-1", &victim, Intent::Help, "authored"));
        let d = Dataset::new("v", ex).unwrap();
        let r = analyse(&d);
        assert!(
            r.findings.iter().any(|f| f.kind == FindingKind::Leakage),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn near_duplicates_within_a_split_are_reported_but_not_blocking() {
        let mut ex = balanced(Intent::ReadFile, 60);
        // Place all three in the training split on purpose: a near-duplicate
        // only inflates a score when both copies are scored together.
        for (i, text) in [
            "read the file main.rs",
            "read the file main.rs please",
            "read the file main.rs now",
        ]
        .iter()
        .enumerate()
        {
            let id = id_in_split(&format!("near{i}"), Split::Train);
            ex.push(example(&id, text, Intent::ReadFile, "generated"));
        }
        let d = Dataset::new("v", ex).unwrap();
        let r = analyse(&d);
        let f = r
            .findings
            .iter()
            .find(|f| f.kind == FindingKind::NearDuplicate)
            .unwrap_or_else(|| panic!("no near-duplicate finding in {:?}", r.findings));
        assert!(
            !f.blocking,
            "near-duplicates are a quality signal, not a hard error"
        );
        assert!(r.usable_for_training);
        assert!(f.count >= 1);
    }

    #[test]
    fn genuinely_different_sentences_are_not_near_duplicates() {
        let a = shingles("find files named main.rs");
        let b = shingles("read the file config.toml");
        assert!(
            jaccard(&a, &b) < NEAR_DUPLICATE_THRESHOLD,
            "unrelated commands scored {:.2}",
            jaccard(&a, &b)
        );
    }

    #[test]
    fn containment_catches_the_filler_word_paraphrase_that_jaccard_misses() {
        let a = shingles("read the file main.rs");
        let b = shingles("read the file main.rs please");
        assert!(
            jaccard(&a, &b) < NEAR_DUPLICATE_THRESHOLD,
            "jaccard should miss this one, otherwise the containment test proves nothing"
        );
        assert_eq!(containment(&a, &b), 1.0);
        assert!(similarity(&a, &b) >= NEAR_DUPLICATE_THRESHOLD);
    }

    #[test]
    fn containment_does_not_fire_on_different_commands() {
        let a = shingles("find files named main.rs");
        let b = shingles("read the file config.toml");
        assert_eq!(containment(&a, &b), 0.0);
        assert!(similarity(&a, &b) < NEAR_DUPLICATE_THRESHOLD);
    }

    #[test]
    fn containment_of_identical_sets_is_one() {
        let a = shingles("check the system status");
        assert_eq!(containment(&a, &a), 1.0);
        let empty: BTreeSet<String> = BTreeSet::new();
        assert_eq!(containment(&empty, &a), 0.0);
    }

    #[test]
    fn jaccard_is_one_for_identical_and_zero_for_disjoint() {
        let a = shingles("check the system status please");
        assert_eq!(jaccard(&a, &a), 1.0);
        assert_eq!(jaccard(&BTreeSet::new(), &BTreeSet::new()), 1.0);
        let empty: BTreeSet<String> = BTreeSet::new();
        assert_eq!(jaccard(&empty, &a), 0.0);
    }

    #[test]
    fn short_sentences_still_get_shingles() {
        // Fewer than three words must not produce an empty set, or every short
        // command would look like every other short command.
        let s = shingles("cancel");
        assert!(!s.is_empty());
        assert!(s.contains("cancel"));
    }

    #[test]
    fn every_intent_appears_in_the_table_even_with_no_examples() {
        let d = healthy();
        let r = analyse(&d);
        for intent in Intent::ALL {
            assert!(
                r.per_intent.contains_key(intent.as_str()),
                "intent `{}` missing from the report",
                intent.as_str()
            );
        }
    }

    #[test]
    fn an_intent_with_too_little_hold_out_is_flagged_as_unmeasured() {
        // A thin intent must be named, because a macro average that silently
        // skips it is not the average it appears to be.
        let mut ex = Vec::new();
        for intent in Intent::ALL {
            if intent == Intent::GitStatus {
                // Below MIN_FOR_HOLDOUT, so the whole group stays in training.
                ex.extend(balanced(intent, 4));
            } else {
                ex.extend(balanced(intent, 60));
            }
        }
        let d = Dataset::new("v", ex).unwrap();
        let r = analyse(&d);
        assert!(
            r.intents_under_measured.contains(&"git_status".to_string()),
            "git_status has 0 test examples and was not reported as unmeasured: {:?}",
            r.intents_under_measured
        );
        assert!(r.intents_under_trained.contains(&"git_status".to_string()));
        assert!(
            r.findings
                .iter()
                .any(|f| f.kind == FindingKind::InsufficientCoverage),
            "{:?}",
            r.findings
        );
    }

    #[test]
    fn provenance_is_counted_by_source() {
        let mut ex = balanced(Intent::Greeting, 60);
        ex.push(example(
            "gen-1",
            "a generated greeting",
            Intent::Greeting,
            "generated",
        ));
        let d = Dataset::new("v", ex).unwrap();
        let r = analyse(&d);
        assert_eq!(r.provenance.get("authored"), Some(&60));
        assert_eq!(r.provenance.get("generated"), Some(&1));
    }

    #[test]
    fn the_report_is_stable_across_runs() {
        let d = healthy();
        let a = serde_json::to_string(&analyse(&d)).unwrap();
        let b = serde_json::to_string(&analyse(&d)).unwrap();
        assert_eq!(
            a, b,
            "report ordering must not depend on hash-map iteration"
        );
    }
}
