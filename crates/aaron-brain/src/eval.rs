//! Evaluation: accuracy, macro-F1, per-intent precision/recall and a full
//! confusion matrix.
//!
//! Nothing here is computed from training data. [`evaluate`] takes a predictor
//! function and a set of held-out examples, so the same code path is used by
//! the CI evaluation test and by `aaron learning evaluate`.

use std::collections::BTreeMap;

use aaron_core::Intent;
use serde::{Deserialize, Serialize};

use crate::model::{predict, softmax, CLASSES};

/// Per-intent scores.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IntentMetrics {
    pub intent: Intent,
    pub support: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f64,
    pub recall: f64,
    pub f1: f64,
}

impl IntentMetrics {
    /// Recall is the number CI cares about most: an intent the model never
    /// produces is a feature the user cannot use.
    pub fn is_dead(&self) -> bool {
        self.support > 0 && self.recall == 0.0
    }
}

/// Full evaluation report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metrics {
    /// Examples scored.
    pub n: usize,
    pub correct: usize,
    pub accuracy: f64,
    /// Unweighted mean of per-intent F1 over every class in the schema.
    pub macro_f1: f64,
    /// Unweighted mean of per-intent F1 restricted to classes present in the
    /// evaluation set. This is the number to watch when a hold-out split does
    /// not happen to contain every intent.
    pub macro_f1_supported: f64,
    pub per_intent: Vec<IntentMetrics>,
    /// `confusion[actual][predicted]`, both indexed by [`Intent::index`].
    pub confusion: Vec<Vec<u64>>,
    /// How often the model fell back to `unknown`.
    pub unknown_rate: f64,
    /// Mean probability of the predicted class.
    pub mean_confidence: f64,
    /// Examples the model got wrong, for building the regression suite.
    pub errors: Vec<ErrorExample>,
}

/// One misclassification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ErrorExample {
    pub text: String,
    pub expected: Intent,
    pub predicted: Intent,
    pub confidence: f64,
}

/// A function mapping text to `(intent, confidence)`.
pub type Predictor<'a> = &'a dyn Fn(&str) -> (Intent, f64);

/// Score a predictor against a labelled set.
pub fn evaluate<F>(examples: &[(String, Intent)], predict_fn: F, unknown: Intent) -> Metrics
where
    F: Fn(&str) -> (Intent, f64),
{
    let mut confusion = vec![vec![0u64; CLASSES]; CLASSES];
    let mut correct = 0usize;
    let mut confidence_sum = 0.0f64;
    let mut unknown_count = 0usize;
    let mut errors = Vec::new();

    for (text, expected) in examples {
        let (predicted, confidence) = predict_fn(text);
        confidence_sum += confidence;
        if predicted == unknown {
            unknown_count += 1;
        }
        confusion[expected.index()][predicted.index()] += 1;
        if predicted == *expected {
            correct += 1;
        } else {
            errors.push(ErrorExample {
                text: text.clone(),
                expected: *expected,
                predicted,
                confidence,
            });
        }
    }

    let n = examples.len();
    let mut per_intent = Vec::with_capacity(CLASSES);
    for intent in Intent::ALL {
        let a = intent.index();
        let support: u64 = confusion[a].iter().sum();
        let tp = confusion[a][a];
        let predicted_total: u64 = (0..CLASSES).map(|p| confusion[p][a]).sum();
        let fp = predicted_total.saturating_sub(tp);
        let fnn = support.saturating_sub(tp);
        let precision = if predicted_total == 0 {
            0.0
        } else {
            tp as f64 / predicted_total as f64
        };
        let recall = if support == 0 {
            0.0
        } else {
            tp as f64 / support as f64
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        per_intent.push(IntentMetrics {
            intent,
            support: support as usize,
            true_positives: tp as usize,
            false_positives: fp as usize,
            false_negatives: fnn as usize,
            precision,
            recall,
            f1,
        });
    }

    let macro_f1 = if per_intent.is_empty() {
        0.0
    } else {
        per_intent.iter().map(|m| m.f1).sum::<f64>() / per_intent.len() as f64
    };
    let supported: Vec<&IntentMetrics> = per_intent.iter().filter(|m| m.support > 0).collect();
    let macro_f1_supported = if supported.is_empty() {
        0.0
    } else {
        supported.iter().map(|m| m.f1).sum::<f64>() / supported.len() as f64
    };

    Metrics {
        n,
        correct,
        accuracy: if n == 0 {
            0.0
        } else {
            correct as f64 / n as f64
        },
        macro_f1,
        macro_f1_supported,
        per_intent,
        confusion,
        unknown_rate: if n == 0 {
            0.0
        } else {
            unknown_count as f64 / n as f64
        },
        mean_confidence: if n == 0 {
            0.0
        } else {
            confidence_sum / n as f64
        },
        errors,
    }
}

/// Turn model logits into `(intent, confidence)` using a threshold.
pub fn logits_to_intent(logits: &[f32], temperature: f32, threshold: f32) -> (Intent, f64) {
    match softmax(logits, temperature) {
        None => (Intent::Unknown, 0.0),
        Some(probs) => {
            let p = predict(&probs);
            let intent = Intent::from_index(p.index).unwrap_or(Intent::Unknown);
            if p.probability < threshold {
                (Intent::Unknown, p.probability as f64)
            } else {
                (intent, p.probability as f64)
            }
        }
    }
}

/// Out-of-distribution probe: how often does the model confidently claim to
/// understand something it was never shown?
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OodReport {
    pub n: usize,
    /// Utterances correctly reported as `unknown`.
    pub rejected: usize,
    pub rejection_rate: f64,
    /// Mean confidence on the rejected utterances.
    pub mean_confidence: f64,
    /// The most confident wrong answers, for adding to the regression suite.
    pub worst: Vec<ErrorExample>,
}

/// Score behaviour on utterances that are not commands.
pub fn evaluate_ood<F>(utterances: &[&str], predict_fn: F) -> OodReport
where
    F: Fn(&str) -> (Intent, f64),
{
    let mut rejected = 0usize;
    let mut conf_sum = 0.0;
    let mut wrong: Vec<ErrorExample> = Vec::new();
    for text in utterances {
        let (intent, conf) = predict_fn(text);
        conf_sum += conf;
        if intent == Intent::Unknown {
            rejected += 1;
        } else {
            wrong.push(ErrorExample {
                text: (*text).to_string(),
                expected: Intent::Unknown,
                predicted: intent,
                confidence: conf,
            });
        }
    }
    wrong.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    OodReport {
        n: utterances.len(),
        rejected,
        rejection_rate: if utterances.is_empty() {
            0.0
        } else {
            rejected as f64 / utterances.len() as f64
        },
        mean_confidence: if utterances.is_empty() {
            0.0
        } else {
            conf_sum / utterances.len() as f64
        },
        worst: wrong.into_iter().take(10).collect(),
    }
}

/// Render a confusion matrix as fixed-width text for the CLI.
pub fn render_confusion(metrics: &Metrics) -> String {
    let labels: Vec<String> = Intent::ALL.iter().map(|i| i.as_str().to_string()).collect();
    let short: Vec<String> = labels.iter().map(|l| l.chars().take(7).collect()).collect();
    let width = short.iter().map(|s| s.len()).max().unwrap_or(7).max(7) + 1;

    let mut out = String::new();
    out.push_str("rows = expected, columns = predicted\n");
    out.push_str(&format!("{:<width$}", "expected\\pred", width = width + 12));
    for s in &short {
        out.push_str(&format!("{s:>8}"));
    }
    out.push('\n');
    for (i, intent) in Intent::ALL.iter().enumerate() {
        out.push_str(&format!("{:<width$}", intent.as_str(), width = width + 4));
        for j in 0..CLASSES {
            let v = metrics.confusion[i][j];
            // The diagonal needs no special case: it is just another cell, and
            // highlighting it is the reader's job, not the formatter's.
            if v > 0 {
                out.push_str(&format!("{v:>8}"));
            } else {
                out.push_str(&format!("{:>8}", "."));
            }
        }
        out.push('\n');
    }
    let mut labels_line = BTreeMap::new();
    for (i, l) in short.iter().enumerate() {
        labels_line.insert(i, l.clone());
    }
    out.push_str("\nclasses: ");
    for (i, l) in labels_line {
        out.push_str(&format!("{i}={l} "));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn examples() -> Vec<(String, Intent)> {
        vec![
            ("check system status".to_string(), Intent::SystemStatus),
            ("how is the machine".to_string(), Intent::SystemStatus),
            ("help".to_string(), Intent::Help),
            ("what can you do".to_string(), Intent::Help),
        ]
    }

    #[test]
    fn perfect_predictor_scores_one() {
        let m = evaluate(
            &examples(),
            |t| {
                let i = if t.contains("status") || t.contains("machine") {
                    Intent::SystemStatus
                } else {
                    Intent::Help
                };
                (i, 0.99)
            },
            Intent::Unknown,
        );
        assert_eq!(m.accuracy, 1.0);
        assert_eq!(m.correct, 4);
        // Only two intents appear in this tiny set, so the schema-wide
        // macro-F1 is diluted while the supported macro-F1 is 1.0.
        assert_eq!(m.macro_f1_supported, 1.0);
        assert!(m.macro_f1 < 1.0);
        assert!(m.errors.is_empty());
    }

    #[test]
    fn always_unknown_has_zero_accuracy_but_is_reported_honestly() {
        let m = evaluate(&examples(), |_| (Intent::Unknown, 0.0), Intent::Unknown);
        assert_eq!(m.accuracy, 0.0);
        assert_eq!(m.unknown_rate, 1.0);
        assert_eq!(m.errors.len(), 4);
    }

    #[test]
    fn confusion_matrix_places_errors_in_the_right_cell() {
        let m = evaluate(
            &examples(),
            |t| {
                // Everything is misclassified as `help`, so every
                // system_status example must land in the help column.
                let _ = t;
                (Intent::Help, 0.5)
            },
            Intent::Unknown,
        );
        let help = Intent::Help.index();
        let status = Intent::SystemStatus.index();
        assert_eq!(m.confusion[status][help], 2);
        assert_eq!(m.confusion[help][help], 2);
    }

    #[test]
    fn dead_intents_are_identifiable() {
        let m = evaluate(&examples(), |_| (Intent::Help, 0.9), Intent::Unknown);
        let dead: Vec<Intent> = m
            .per_intent
            .iter()
            .filter(|i| i.is_dead())
            .map(|i| i.intent)
            .collect();
        assert!(dead.contains(&Intent::SystemStatus));
        assert!(!dead.contains(&Intent::Help));
    }

    #[test]
    fn empty_input_does_not_divide_by_zero() {
        let m = evaluate(&[], |_| (Intent::Help, 0.5), Intent::Unknown);
        assert_eq!(m.accuracy, 0.0);
        assert_eq!(m.macro_f1, 0.0);
        assert_eq!(m.macro_f1_supported, 0.0);
        assert_eq!(m.mean_confidence, 0.0);
    }

    #[test]
    fn logits_below_the_threshold_become_unknown() {
        let logits = [10.0f32, 0.0, 0.0];
        assert_eq!(logits_to_intent(&logits, 1.0, 0.1).0, Intent::Greeting);
        // A threshold of exactly 1.0 can never be met by a real distribution.
        assert_eq!(logits_to_intent(&logits, 1.0, 1.0).0, Intent::Unknown);
        assert_eq!(logits_to_intent(&logits, 1.0, 0.5).0, Intent::Greeting);
        assert_eq!(
            logits_to_intent(&[f32::NAN, 0.0], 1.0, 0.1).0,
            Intent::Unknown
        );
    }

    #[test]
    fn ood_report_measures_rejection() {
        let inputs = [
            "the weather is nice today",
            "tell me a joke about cats",
            "what is the capital of france",
        ];
        let r = evaluate_ood(&inputs, |t| {
            if t.contains("weather") {
                (Intent::Unknown, 0.2)
            } else {
                (Intent::OpenUrl, 0.8)
            }
        });
        assert_eq!(r.n, 3);
        assert_eq!(r.rejected, 1);
        assert!((r.rejection_rate - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(r.worst.len(), 2);
        assert!(r.worst[0].confidence >= r.worst[1].confidence);
    }

    #[test]
    fn confusion_renders_every_class() {
        let m = evaluate(&examples(), |_| (Intent::Help, 0.5), Intent::Unknown);
        let text = render_confusion(&m);
        for intent in Intent::ALL {
            assert!(
                text.contains(intent.as_str()),
                "confusion table omits {intent}"
            );
        }
    }
}
