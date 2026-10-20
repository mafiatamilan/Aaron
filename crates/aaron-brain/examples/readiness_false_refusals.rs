//! Which complete requests does the readiness head wrongly refuse?
//!
//! The head's false-not-ready rate is the price of the dispatch-rate
//! reduction, and a rate on its own does not say what the user experiences.
//! This lists the actual utterances.
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aaron-brain --example readiness_false_refusals -- \
//!     <checkpoint> <corpus.jsonl> <threshold>
//! ```

use aaron_brain::readiness::{ReadinessCorpus, Split};
use aaron_core::intent::IntentClassifier;

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("usage: readiness_false_refusals <ckpt> <corpus> <threshold>");
    let corpus_path = args
        .next()
        .expect("usage: readiness_false_refusals <ckpt> <corpus> <threshold>");
    let threshold: f32 = args
        .next()
        .unwrap_or_else(|| "0.3".into())
        .parse()
        .expect("the threshold must be a number");

    let classifier = aaron_brain::NeuralClassifier::load_with_readiness_threshold(
        &checkpoint,
        0.60,
        1.0,
        256,
        threshold,
    )
    .expect("the checkpoint must load");
    let corpus = ReadinessCorpus::load(&corpus_path).expect("the corpus must load");

    println!("checkpoint: {checkpoint}");
    println!("threshold:  {threshold:.2}");
    println!();
    for split in [Split::Validation, Split::Test] {
        let rows = corpus.split(split);
        let mut refused = Vec::new();
        let mut missed = Vec::new();
        for row in &rows {
            let command = classifier
                .classify(&row.text)
                .expect("the classification must succeed");
            let asked = command.readiness == aaron_core::Readiness::NeedsClarification;
            let incomplete = row.readiness == aaron_core::Readiness::NeedsClarification;
            if asked && !incomplete {
                refused.push((
                    row.text.clone(),
                    row.family.clone(),
                    row.intent.as_str().to_string(),
                ));
            }
            if !asked && incomplete {
                missed.push((
                    row.text.clone(),
                    row.family.clone(),
                    row.intent.as_str().to_string(),
                ));
            }
        }
        println!("=== {} ===", split.as_str());
        println!("  {} complete rows wrongly refused:", refused.len());
        for (text, family, intent) in &refused {
            println!("    [{family}/{intent}] {text:?}");
        }
        println!("  {} incomplete rows wrongly accepted:", missed.len());
        for (text, family, intent) in &missed {
            println!("    [{family}/{intent}] {text:?}");
        }
        println!();
    }
}
