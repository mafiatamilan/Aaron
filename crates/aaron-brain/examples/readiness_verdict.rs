//! What does the readiness head say about particular utterances?
//!
//! Usage:
//!
//! ```text
//! cargo run --release -p aaron-brain --example readiness_verdict -- <checkpoint> [text...]
//! ```

use aaron_core::intent::IntentClassifier;

fn main() {
    let mut args = std::env::args().skip(1);
    let checkpoint = args
        .next()
        .expect("usage: readiness_verdict <checkpoint> [text...]");
    let classifier = aaron_brain::NeuralClassifier::load(&checkpoint, 0.60, 1.0, 256)
        .expect("the checkpoint must load");
    let mut texts: Vec<String> = args.collect();
    // A single argument that names a JSONL file is read as a list of
    // utterances, so a whole benchmark can be scored at once.
    if texts.len() == 1 && std::path::Path::new(&texts[0]).is_file() {
        let raw = std::fs::read_to_string(&texts[0]).expect("the file must read");
        texts = raw
            .lines()
            .filter(|l| !l.trim().is_empty() && !l.trim_start().starts_with('#'))
            .map(|l| {
                serde_json::from_str::<serde_json::Value>(l)
                    .ok()
                    .and_then(|v| v.get("text").and_then(|t| t.as_str()).map(String::from))
                    .unwrap_or_default()
            })
            .filter(|t| !t.is_empty())
            .collect();
    }
    let texts = if texts.is_empty() {
        vec![
            "read the document".to_string(),
            "read main.rs".to_string(),
            "read".to_string(),
            "read the first 20 lines".to_string(),
            "open the plugin".to_string(),
            "open vscode".to_string(),
            "what do you remember about".to_string(),
            "show me the file".to_string(),
            "read the file ../../etc/passwd".to_string(),
            "hello there".to_string(),
            "check system status".to_string(),
        ]
    } else {
        texts
    };

    println!("checkpoint: {checkpoint}");
    println!(
        "  architecture {}   parameters {}",
        classifier.manifest().arch,
        classifier.manifest().param_count
    );
    println!();
    println!(
        "{:<44} {:<18} {:<22} confidence",
        "text", "intent", "readiness"
    );
    for text in &texts {
        let command = classifier
            .classify(text)
            .expect("the classification must succeed");
        println!(
            "{:<44} {:<18} {:<22} {:.2}",
            format!("{text:?}"),
            command.intent.as_str(),
            command.readiness.as_str(),
            command.confidence
        );
    }

    // Latency: the head's cost is one 168-wide dot product per utterance.
    let warm: Vec<String> = texts.iter().take(10).cloned().collect();
    for text in &warm {
        let _ = classifier.classify(text);
    }
    let iterations = 200;
    let started = std::time::Instant::now();
    for i in 0..iterations {
        let text = &texts[i % texts.len()];
        let _ = classifier
            .classify(text)
            .expect("the classification must succeed");
    }
    let elapsed = started.elapsed();
    println!();
    println!(
        "latency: {:.3} ms per classification over {iterations} iterations",
        elapsed.as_secs_f64() * 1000.0 / iterations as f64
    );
}
