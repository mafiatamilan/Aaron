//! A paired comparison of two checkpoints on the same examples.
//!
//! Comparing two models' false-accept *rates* as independent proportions
//! throws away most of the information. The right test is paired: only the
//! examples where the two models *disagree* carry evidence. McNemar's test
//! uses exactly those, and it is far more powerful at these sample sizes.
//!
//! Usage: mcnemar <checkpoint A> <checkpoint B> [validation|locked]

use aaron_core::intent::IntentClassifier;

fn main() {
    let a = std::env::args().nth(1).unwrap();
    let b = std::env::args().nth(2).unwrap();
    let which = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "validation".into());
    let ca = aaron_brain::NeuralClassifier::load(&a, 0.60, 1.0, 256).unwrap();
    let cb = aaron_brain::NeuralClassifier::load(&b, 0.60, 1.0, 256).unwrap();

    let pairs: Vec<(String, aaron_core::Intent)> = if which == "locked" || which == "safety-v1" {
        let path = if which == "locked" {
            "training/evaluation/benchmark-v3.jsonl"
        } else {
            "training/evaluation/safety-benchmark-v1.jsonl"
        };
        // The safety benchmark contains a whitespace-only example, which the
        // `Dataset` type rightly refuses, so it is read directly.
        let mut pairs = Vec::new();
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let v: serde_json::Value = serde_json::from_str(line).unwrap();
            // The intent is stored as its name, not its index.
            let name = v["intent"].as_str().unwrap();
            let intent = aaron_core::Intent::ALL
                .iter()
                .find(|i| i.as_str() == name)
                .copied()
                .unwrap_or(aaron_core::Intent::Unknown);
            pairs.push((v["text"].as_str().unwrap().to_string(), intent));
        }
        pairs
    } else {
        let d = aaron_brain::dataset::Dataset::load("training/dataset/current.jsonl").unwrap();
        d.split(aaron_brain::Split::Validation)
            .iter()
            .map(|e| (e.text.clone(), e.intent))
            .collect()
    };

    let mut both_right = 0;
    let mut both_wrong = 0;
    let mut a_only = 0; // A correct, B wrong
    let mut b_only = 0; // B correct, A wrong
    let mut fa_a = 0;
    let mut fa_b = 0;
    let mut n_unk = 0;

    for (text, label) in &pairs {
        let ok_a = ca.classify(text).unwrap().intent == *label;
        let ok_b = cb.classify(text).unwrap().intent == *label;
        match (ok_a, ok_b) {
            (true, true) => both_right += 1,
            (false, false) => both_wrong += 1,
            (true, false) => a_only += 1,
            (false, true) => b_only += 1,
        }
        if *label == aaron_core::Intent::Unknown {
            n_unk += 1;
            if ca.classify(text).unwrap().intent != aaron_core::Intent::Unknown {
                fa_a += 1;
            }
            if cb.classify(text).unwrap().intent != aaron_core::Intent::Unknown {
                fa_b += 1;
            }
        }
    }

    let n = pairs.len();
    println!("set: {which}   examples: {n}   out-of-schema: {n_unk}");
    println!(
        "  A (0.2 baseline)   correct {}/{}   false accepts {}/{}",
        both_right + a_only,
        n,
        fa_a,
        n_unk
    );
    println!(
        "  B (0.3 shipped)    correct {}/{}   false accepts {}/{}",
        both_right + b_only,
        n,
        fa_b,
        n_unk
    );
    println!();
    println!("  paired: both right {both_right}, both wrong {both_wrong}, A only {a_only}, B only {b_only}");
    let (c, p) = mcnemar(a_only, b_only);
    println!("  McNemar: chi2 = {c:.3} (continuity-corrected), p ~= {p:.4}");
    println!(
        "  => the difference is {}",
        if p < 0.05 {
            "STATISTICALLY SIGNIFICANT"
        } else {
            "not significant at this sample size"
        }
    );
}

/// McNemar's test with a continuity correction, exact binomial p-value.
fn mcnemar(a_only: usize, b_only: usize) -> (f64, f64) {
    let n = a_only + b_only;
    if n == 0 {
        return (0.0, 1.0);
    }
    let chi2 = ((a_only as f64 - b_only as f64).abs() - 1.0).powi(2) / n as f64;
    // Exact two-sided binomial test on the discordant pairs.
    let k = a_only.min(b_only) as u64;
    let p = 2.0 * binomial_tail(n as u64, k, 0.5).min(1.0);
    (chi2, p)
}

fn binomial_tail(n: u64, k: u64, p: f64) -> f64 {
    let mut acc = 0.0;
    for i in 0..=k {
        acc += binomial_pmf(n, i, p);
    }
    acc
}

fn binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    let mut c = 1.0f64;
    for i in 0..k {
        c = c * (n - i) as f64 / (i + 1) as f64;
    }
    c * p.powi(k as i32) * (1.0 - p).powi((n - k) as i32)
}
