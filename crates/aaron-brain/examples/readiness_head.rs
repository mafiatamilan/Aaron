//! Train the readiness head on a frozen trunk and report what it can do.
//!
//! This is the diagnostic that decides whether the readiness head is worth
//! building at all. It is deliberately cheap:
//!
//! 1. the intent classifier is already trained (the base checkpoint);
//! 2. the readiness head is trained on top of it, with every other
//!    parameter frozen, so the intent classifier's outputs are bit-identical
//!    before and after;
//! 3. the head is scored on the readiness corpus's held-out splits, against
//!    the deterministic rule on exactly the same rows.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aaron-brain --example readiness_head -- \
//!     <base-checkpoint> [corpus] [out-checkpoint]
//! ```

use std::collections::HashMap;

use aaron_brain::checkpoint::{CheckpointInfo, Manifest};
use aaron_brain::model::Heads;
use aaron_brain::readiness::{
    merge_head, train_head, HeadTrainingConfig, ReadinessConfusion, ReadinessCorpus, Split,
};

fn describe(name: &str, confusion: &ReadinessConfusion) {
    let fr = confusion
        .false_ready_rate()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "undefined".into());
    let fnr = confusion
        .false_not_ready_rate()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "undefined".into());
    let precision = confusion
        .precision()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "undefined".into());
    let recall = confusion
        .recall()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "undefined".into());
    let f1 = confusion
        .f1()
        .map(|v| format!("{v:.4}"))
        .unwrap_or_else(|| "undefined".into());
    println!("  {name}");
    println!(
        "    incomplete: {}   complete: {}   total: {}",
        confusion.incomplete(),
        confusion.complete(),
        confusion.total()
    );
    println!(
        "    true incomplete {}   false ready {}",
        confusion.true_incomplete, confusion.false_ready
    );
    println!(
        "    false not ready {}   true complete {}",
        confusion.false_not_ready, confusion.true_complete
    );
    println!("    false-ready rate     {fr}");
    println!("    false-not-ready rate {fnr}");
    println!("    precision {precision}   recall {recall}   F1 {f1}");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let base_path = args.next().unwrap_or_else(|| {
        "/tmp/opencode/aaron-04/data/checkpoints/candidate-seed2690307351-e100.ckpt".into()
    });
    let corpus_path = args
        .next()
        .unwrap_or_else(|| "training/readiness/corpus-v1.jsonl".into());
    let out_path = args.next();

    println!("base checkpoint: {base_path}");
    println!("corpus:          {corpus_path}");

    let (manifest, tensors): (Manifest, HashMap<String, candle_core::Tensor>) =
        aaron_brain::checkpoint::load(&base_path).expect("the base checkpoint must load");
    println!(
        "base architecture: {}   parameters: {}",
        manifest.arch, manifest.param_count
    );
    assert_eq!(
        manifest.param_count,
        aaron_brain::expected_parameter_count(Heads::IntentOnly),
        "the base checkpoint must be the intent-only architecture"
    );

    let corpus = ReadinessCorpus::load(&corpus_path).expect("the corpus must load");
    println!(
        "corpus rows: {}   train {}   validation {}   test {}",
        corpus.len(),
        corpus.split(Split::Train).len(),
        corpus.split(Split::Validation).len(),
        corpus.split(Split::Test).len()
    );

    let config = HeadTrainingConfig::default();
    println!(
        "training the head: epochs {}   lr {}   batch {}   seed {}",
        config.epochs, config.learning_rate, config.batch_size, config.seed
    );
    let outcome = train_head(tensors.clone(), &corpus, &config).expect("the head must train");

    println!();
    println!(
        "threshold selected on the training split: {:.2}",
        outcome.threshold
    );
    println!("best epoch: {}", outcome.best_epoch);
    println!();
    for metrics in &outcome.metrics {
        println!("=== {} ===", metrics.split.as_str());
        println!("  loss {:.6}", metrics.loss);
        describe("readiness head", &metrics.confusion);
        println!();
        describe("deterministic rule (the baseline)", &metrics.rule_confusion);
        println!();
    }

    // The threshold sweep, from the library. The library computes it from the
    // same model and the same probabilities as the metrics above, so the two
    // cannot disagree.
    println!("threshold sweep (the head's operating points, against the rule)");
    for metrics in &outcome.metrics {
        println!();
        println!("=== {} ===", metrics.split.as_str());
        let rule = &metrics.rule_confusion;
        println!(
            "  the rule: false-ready {:.4}   false-not-ready {:.4}   F1 {}",
            rule.false_ready_rate().unwrap_or(f64::NAN),
            rule.false_not_ready_rate().unwrap_or(f64::NAN),
            rule.f1()
                .map(|v| format!("{v:.4}"))
                .unwrap_or("undefined".into())
        );
        println!("  threshold   false-ready   false-not-ready   precision   recall   F1");
        for (threshold, confusion) in &metrics.sweep {
            println!(
                "  {:.2}        {:.4}        {:.4}          {}      {}    {}",
                threshold,
                confusion.false_ready_rate().unwrap_or(f64::NAN),
                confusion.false_not_ready_rate().unwrap_or(f64::NAN),
                confusion
                    .precision()
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or("undefined".into()),
                confusion
                    .recall()
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or("undefined".into()),
                confusion
                    .f1()
                    .map(|v| format!("{v:.4}"))
                    .unwrap_or("undefined".into()),
            );
        }
    }

    if let Some(out) = out_path {
        let merged = merge_head(&tensors, &outcome.tensors);
        let info = CheckpointInfo {
            dataset_version: format!("{} + readiness corpus", manifest.dataset_version),
            seed: config.seed,
            epochs: outcome.best_epoch,
            metrics: serde_json::json!({
                "readiness": {
                    "threshold": outcome.threshold,
                    "best_epoch": outcome.best_epoch,
                    "splits": outcome.metrics.iter().map(|m| serde_json::json!({
                        "split": m.split.as_str(),
                        "loss": m.loss,
                        "head": {
                            "false_ready_rate": m.confusion.false_ready_rate(),
                            "false_not_ready_rate": m.confusion.false_not_ready_rate(),
                            "precision": m.confusion.precision(),
                            "recall": m.confusion.recall(),
                            "f1": m.confusion.f1(),
                        },
                        "rule": {
                            "false_ready_rate": m.rule_confusion.false_ready_rate(),
                            "false_not_ready_rate": m.rule_confusion.false_not_ready_rate(),
                            "precision": m.rule_confusion.precision(),
                            "recall": m.rule_confusion.recall(),
                            "f1": m.rule_confusion.f1(),
                        },
                    })).collect::<Vec<_>>(),
                }
            }),
        };
        let bytes =
            aaron_brain::checkpoint::encode(&merged, &info).expect("the checkpoint must encode");
        std::fs::write(&out, &bytes).expect("the checkpoint must write");
        println!("wrote {out}");
    }
}
