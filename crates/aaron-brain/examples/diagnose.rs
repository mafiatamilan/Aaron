//! Phase-1 diagnostics for the intent classifier.
//!
//! These are *read-only* probes. They score an existing checkpoint and print
//! evidence; they never change the dataset or the model. They exist because
//! the questions they answer are the ones the release notes keep having to
//! answer again:
//!
//! * `completeness` — can the model tell a complete request from an
//!   incomplete one, and is that difference visible anywhere in its output?
//! * `validation` — which out-of-schema validation examples does it accept?
//! * `entities` — does the entity extractor see the difference, or only the
//!   classifier?
//! * `balance` — the class distribution and the split assignment.
//!
//! Usage:
//!   cargo run --release -p aaron-brain --example diagnose -- <checkpoint> [completeness|validation|entities|balance]

use aaron_core::intent::IntentClassifier;

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = args.next().expect("usage: diagnose <checkpoint> [mode]");
    let mode = args.next().unwrap_or_else(|| "completeness".into());
    match mode.as_str() {
        "completeness" => completeness(&checkpoint),
        "probe" => probe(&checkpoint),
        "validation" => validation(&checkpoint),
        "entities" => entities(),
        "balance" => balance(),
        other => {
            eprintln!("unknown mode `{other}`");
            std::process::exit(2);
        }
    }
}

/// Can the model distinguish a complete request from an incomplete one, and
/// is that difference visible in its output distribution?
///
/// The answer decides whether a *threshold* can be a fix. If the distributions
/// overlap, no threshold, margin or entropy rule can separate them, and only a
/// change to what the model is asked to output will.
fn completeness(checkpoint: &str) {
    let c = aaron_brain::NeuralClassifier::load(checkpoint, 0.60, 1.0, 256).unwrap();
    let complete: &[&str] = &[
        "read the file main.rs",
        "read the file config.toml",
        "find files named main.rs",
        "find files named *.py",
        "open the terminal",
        "open the readme",
        "run the cargo test suite",
        "run the unit tests",
        "remember that my wifi is called aurora",
        "kill the terminal",
        "launch the browser",
        "check system status",
        "show git status",
        "open https://example.com",
    ];
    let incomplete: &[&str] = &[
        "read the file",
        "read the file the",
        "find files named",
        "find files",
        "open the terminal",
        "open the app",
        "run the suite",
        "run the tests",
        "remember that",
        "kill the process",
        "launch the app",
        "check the system",
        "show the git",
        "open the url",
    ];

    let mut rows = Vec::new();
    for (label, set) in [("complete", complete), ("incomplete", incomplete)] {
        for text in set {
            let probs = c.probabilities(text).unwrap();
            let mut sorted = probs.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let margin = sorted[0] - sorted[1];
            let entropy: f32 = -probs
                .iter()
                .map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
                .sum::<f32>();
            let cmd = c.classify(text).unwrap();
            rows.push((
                label,
                *text,
                sorted[0],
                margin,
                entropy,
                cmd.intent.as_str(),
            ));
        }
    }
    println!(
        "{:<11} {:<34} {:>7} {:>8} {:>9}  predicted",
        "label", "text", "top1", "margin", "entropy",
    );
    for (label, text, top, margin, ent, intent) in &rows {
        println!("{label:<11} {text:<34} {top:>7.3} {margin:>8.3} {ent:>9.3}  {intent}");
    }
    println!();
    for label in ["complete", "incomplete"] {
        let sel: Vec<_> = rows.iter().filter(|r| r.0 == label).collect();
        let n = sel.len().max(1) as f32;
        let mean_top: f32 = sel.iter().map(|r| r.2).sum();
        let mean_margin: f32 = sel.iter().map(|r| r.3).sum();
        let mean_entropy: f32 = sel.iter().map(|r| r.4).sum();
        println!(
            "{label:<10} n={:<3} mean top1 {:.3}  mean margin {:.3}  mean entropy {:.3}",
            sel.len(),
            mean_top / n,
            mean_margin / n,
            mean_entropy / n
        );
    }
}

/// The capability probe the independent 0.4 review asked for.
///
/// The earlier "14 hand-picked probes" could not support the claim that no
/// output statistic separates a complete request from an incomplete one, and
/// no artefact recorded it. This measures the same thing properly, on data
/// that already exists: the dataset's own verb-only `unknown` rows, each
/// matched with a complete command that starts with the same verb.
///
/// If a statistic separates the two populations, the representation carries
/// the completeness signal and data can teach it. If none does, the
/// information is not in the representation, and no amount of data will help.
fn probe(checkpoint: &str) {
    let c = aaron_brain::NeuralClassifier::load(checkpoint, 0.60, 1.0, 256).unwrap();
    let ds = aaron_brain::dataset::Dataset::load("training/dataset/current.jsonl").unwrap();

    // Incomplete: verb-only rows labelled `unknown`.
    let mut incomplete: Vec<(String, String)> = Vec::new();
    for e in &ds.examples {
        if e.intent != aaron_core::Intent::Unknown {
            continue;
        }
        let words = e.text.split_whitespace().count();
        let first = e
            .text
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        const VERBS: &[&str] = &[
            "open", "read", "show", "find", "locate", "search", "launch", "start", "run", "kill",
            "stop", "close", "remember", "recall", "check", "print", "display", "list", "cancel",
            "abort",
        ];
        if words <= 4 && VERBS.contains(&first.as_str()) {
            incomplete.push((e.text.clone(), first));
        }
    }
    // Complete: a command that starts with the same verb, from another class.
    let mut complete: Vec<(String, String)> = Vec::new();
    for e in &ds.examples {
        if e.intent == aaron_core::Intent::Unknown {
            continue;
        }
        let first = e
            .text
            .split_whitespace()
            .next()
            .unwrap_or("")
            .to_lowercase();
        let words = e.text.split_whitespace().count();
        if words >= 4 && incomplete.iter().any(|(_, v)| *v == first) {
            complete.push((e.text.clone(), first));
        }
    }
    println!(
        "incomplete (verb-only `unknown` rows): {}   complete (same verb, other intent): {}",
        incomplete.len(),
        complete.len()
    );

    type Stats = (f64, f64, f64, f64); // top1, margin, entropy, energy
    let mut pos: Vec<Stats> = Vec::new();
    let mut neg: Vec<Stats> = Vec::new();
    for (texts, bucket) in [(&incomplete, &mut neg), (&complete, &mut pos)] {
        for (text, _) in texts.iter() {
            let probs = c.probabilities(text).unwrap();
            let mut sorted = probs.clone();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            let margin = (sorted[0] - sorted[1]) as f64;
            let entropy: f64 = -probs
                .iter()
                .map(|p| {
                    if *p > 0.0 {
                        (*p as f64) * (*p as f64).ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            let energy: f64 = -probs
                .iter()
                .map(|p| {
                    if *p > 0.0 {
                        *p as f64 * (*p as f64).ln()
                    } else {
                        0.0
                    }
                })
                .sum::<f64>();
            bucket.push((sorted[0] as f64, margin, entropy, energy));
        }
    }

    println!();
    println!(
        "{:<12} {:>9} {:>9} {:>10} {:>10} {:>9}",
        "statistic", "AUC", "complete", "incomplete", "sep", "best-acc"
    );
    let names = ["top1", "margin", "entropy", "energy"];
    for (i, name) in names.iter().enumerate() {
        let get = |v: &Vec<Stats>| {
            v.iter()
                .map(|t| match i {
                    0 => t.0,
                    1 => t.1,
                    2 => t.2,
                    _ => t.3,
                })
                .collect::<Vec<f64>>()
        };
        let p = get(&pos);
        let n = get(&neg);
        // AUC: probability a positive scores above a negative. Higher AUC means
        // the statistic separates complete from incomplete.
        let mut wins = 0.0;
        for a in &p {
            for b in &n {
                wins += if a > b {
                    1.0
                } else if a == b {
                    0.5
                } else {
                    0.0
                };
            }
        }
        let auc = wins / (p.len() * n.len()).max(1) as f64;
        // The best single threshold, and the accuracy it would give.
        let mut all: Vec<f64> = p.iter().chain(n.iter()).copied().collect();
        all.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut best = (0.0f64, 0.0f64, 0.0f64);
        for t in all.iter() {
            let tp = p.iter().filter(|v| **v > *t).count() as f64;
            let fp = n.iter().filter(|v| **v > *t).count() as f64;
            let acc = (tp + (n.len() as f64 - fp)) / (p.len() + n.len()) as f64;
            if acc > best.2 {
                best = (*t, acc, acc);
            }
        }
        let mean_p: f64 = p.iter().sum::<f64>() / p.len().max(1) as f64;
        let mean_n: f64 = n.iter().sum::<f64>() / n.len().max(1) as f64;
        let sep = (mean_p - mean_n).abs() / (mean_p.abs() + mean_n.abs()).max(1e-9);
        println!(
            "{name:<12} {auc:>9.4} {mean_p:>9.4} {mean_n:>10.4} {sep:>10.4} {:>9.4}",
            best.1
        );
    }
    println!();
    println!("AUC 0.50 = the statistic carries no information about completeness.");
    println!("AUC 1.00 = the statistic separates the populations perfectly.");
}

/// Which out-of-schema **validation** examples does the model accept?
///
/// Validation is the selection signal, so this is the legitimate place to look.
/// The locked intent benchmark and safety benchmark v2 are not consulted.
fn validation(checkpoint: &str) {
    let ds = aaron_brain::dataset::Dataset::load("training/dataset/current.jsonl").unwrap();
    let c = aaron_brain::NeuralClassifier::load(checkpoint, 0.60, 1.0, 256).unwrap();
    let val = ds.split(aaron_brain::Split::Validation);
    let out_of_schema: Vec<_> = val
        .iter()
        .filter(|e| e.intent == aaron_core::Intent::Unknown)
        .collect();
    let mut false_accepts = Vec::new();
    for e in &out_of_schema {
        let cmd = c.classify(&e.text).unwrap();
        if cmd.intent != aaron_core::Intent::Unknown {
            false_accepts.push((
                e.text.clone(),
                cmd.intent.as_str().to_string(),
                cmd.confidence,
            ));
        }
    }
    println!(
        "validation: {} examples, {} out-of-schema, {} false accepts",
        val.len(),
        out_of_schema.len(),
        false_accepts.len()
    );
    println!();
    for (t, i, cf) in &false_accepts {
        println!("  conf {cf:.3}  -> {i:<18} {t:?}");
    }
    println!();
    println!("### correctly refused ###");
    for e in &out_of_schema {
        let cmd = c.classify(&e.text).unwrap();
        if cmd.intent == aaron_core::Intent::Unknown {
            println!("  refused              {:?}", e.text);
        }
    }
}

/// Does the *entity extractor* see the difference, independently of the
/// classifier?
///
/// If it does, the completeness information exists outside the model and the
/// argument layer can act on it without any retraining.
fn entities() {
    println!(
        "{:<36} {:<14} {:<26} {:<22}",
        "text", "intent", "text entity", "path entity"
    );
    for (text, intent) in [
        ("read the file", aaron_core::Intent::ReadFile),
        ("read the file main.rs", aaron_core::Intent::ReadFile),
        ("find files named", aaron_core::Intent::SearchFiles),
        ("find files named main.rs", aaron_core::Intent::SearchFiles),
        ("open it", aaron_core::Intent::ReadFile),
        ("open the terminal", aaron_core::Intent::ReadFile),
        ("run the suite", aaron_core::Intent::RunTests),
        ("run the cargo test suite", aaron_core::Intent::RunTests),
        ("remember that", aaron_core::Intent::Remember),
        (
            "remember that my wifi is called aurora",
            aaron_core::Intent::Remember,
        ),
    ] {
        let e = aaron_core::text::extract_entities(intent, text);
        println!(
            "{:<36} {:<14} {:<26} {:<22}",
            format!("{text:?}"),
            intent.as_str(),
            e.text.as_deref().unwrap_or("-"),
            e.path.as_deref().unwrap_or("-")
        );
    }
}

/// The class distribution and the split assignment.
fn balance() {
    let ds = aaron_brain::dataset::Dataset::load("training/dataset/current.jsonl").unwrap();
    let mut per: std::collections::BTreeMap<String, (usize, usize, usize)> = Default::default();
    for e in &ds.examples {
        let entry = per.entry(e.intent.as_str().to_string()).or_default();
        match ds.split_of(e) {
            aaron_brain::Split::Train => entry.0 += 1,
            aaron_brain::Split::Validation => entry.1 += 1,
            aaron_brain::Split::Test => entry.2 += 1,
        }
    }
    println!(
        "{:<20} {:>6} {:>11} {:>6} {:>7}",
        "intent", "train", "validation", "test", "total"
    );
    for (intent, (tr, va, te)) in &per {
        println!("{intent:<20} {tr:>6} {va:>11} {te:>6} {:>7}", tr + va + te);
    }
    let max = per.values().map(|v| v.0 + v.1 + v.2).max().unwrap_or(1);
    let min = per.values().map(|v| v.0 + v.1 + v.2).min().unwrap_or(1);
    println!(
        "\ntotal {} examples, imbalance {:.2}x",
        ds.len(),
        max as f64 / min as f64
    );
}
