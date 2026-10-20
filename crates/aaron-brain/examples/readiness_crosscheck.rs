//! Cross-check: does the readiness head's verdict agree with its probability?
//!
//! The head produces a **logit** and the threshold is on the **probability**.
//! Getting that wrong does not look like a bug: a logit of 0.30 is a
//! probability of 0.57, so every threshold below 0.5 fires on almost
//! everything and the numbers still look plausible.
//!
//! This scores the corpus rows three ways — through the classifier, through
//! the model's probability, and through the probability computed from the
//! logit — and asserts they agree. It exists because a version of this
//! release measured the corpus with the logit compared against a probability
//! threshold, and the two disagreed by more than a factor of two.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aaron-brain --example readiness_crosscheck -- \
//!     <checkpoint-with-head> <corpus.jsonl>
//! ```

use aaron_brain::readiness::{ReadinessCorpus, Split};
use aaron_core::intent::IntentClassifier;

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("usage: readiness_crosscheck <ckpt> <corpus>");
    let corpus_path = args
        .next()
        .expect("usage: readiness_crosscheck <ckpt> <corpus>");

    let classifier = aaron_brain::NeuralClassifier::load_with_readiness_threshold(
        &checkpoint,
        0.60,
        1.0,
        256,
        0.50,
    )
    .expect("the checkpoint must load");
    let corpus = ReadinessCorpus::load(&corpus_path).expect("the corpus must load");

    let mut failures = 0usize;
    for split in [Split::Validation, Split::Test] {
        let rows = corpus.split(split);
        let texts: Vec<String> = rows.iter().map(|e| e.text.clone()).collect();

        // The classifier's own verdict.
        let mut via_classifier = 0usize;
        for row in &rows {
            let command = classifier
                .classify(&row.text)
                .expect("the classification must succeed");
            if command.readiness == aaron_core::Readiness::NeedsClarification {
                via_classifier += 1;
            }
        }

        // The same rows, scored as a batch through the model.
        let batch = aaron_brain::FeatureBatch::from_texts(&texts);
        let model = aaron_brain::IntentModel::from_tensors(
            aaron_brain::checkpoint::load(&checkpoint)
                .expect("the checkpoint must load")
                .1,
            &candle_core::Device::Cpu,
            aaron_brain::Heads::WithReadiness,
        )
        .expect("the model must build");
        let (_, ready) = model
            .forward_with_readiness(&batch, None, None)
            .expect("the forward pass must run");
        let ready = ready.expect("the head must exist");
        let logits = ready
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .expect("the logits must be readable");
        let mut via_probability = 0usize;
        for logit in &logits {
            let probability = 1.0 / (1.0 + (-logit).exp());
            if probability >= 0.50 {
                via_probability += 1;
            }
        }

        // And the same threshold applied to the *logit*, which is what the
        // bug did. It must not agree.
        let mut via_logit = 0usize;
        for logit in &logits {
            if *logit >= 0.50 {
                via_logit += 1;
            }
        }

        println!("{}: {} rows", split.as_str(), rows.len());
        println!("  asked, via the classifier:     {via_classifier}");
        println!("  asked, via the probability:      {via_probability}");
        println!("  asked, via the logit (the bug):  {via_logit}");

        if via_classifier != via_probability {
            failures += 1;
            println!("  MISMATCH: the classifier and the probability disagree");
        }
        if via_logit == via_probability {
            failures += 1;
            println!("  MISMATCH: the logit and the probability agree, so this corpus cannot");
            println!("           distinguish the two — pick rows where they differ");
        }
    }

    if failures > 0 {
        std::process::exit(1);
    }
    println!();
    println!("the classifier's verdict and the head's probability agree");
}
