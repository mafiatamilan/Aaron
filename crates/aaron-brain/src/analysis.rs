//! Systematic error analysis.
//!
//! Produces a machine-readable report and a readable Markdown summary from
//! one evaluation run: accuracy, macro-F1, per-intent precision/recall/F1,
//! the confusion matrix, per-split class counts, calibration, false-accept
//! and false-reject rates, entity-extraction error rate, regression results,
//! and the misclassifications themselves, grouped into actionable categories.
//!
//! # Why categories, and not a list of mistakes
//!
//! "Eight utterances were misclassified" is not actionable. "Seven of the
//! eight are `open <noun>` where the noun is an application and the model
//! chose `read_file`" is a data problem with a specific fix. Each category
//! below names a different remedy, which is the whole point of sorting them.
//!
//! # Privacy
//!
//! Every text that reaches a report goes through
//! [`aaron_core::text::redact_secrets`] first, and is truncated. A report is
//! something a user may paste into a bug tracker, so it must not carry
//! credentials — and it must not carry the user's memories either, which is
//! why only dataset and regression examples are ever included. No transcript
//! and no stored memory is read by this module.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use aaron_core::intent::Intent;
use aaron_core::text::redact_secrets;

/// How many characters of example text a report may contain.
pub const REPORT_TEXT_LIMIT: usize = 120;

/// A single misclassification.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Misclassification {
    /// The (redacted, truncated) utterance.
    pub text: String,
    /// The label it should have had.
    pub expected: Intent,
    /// The label it got.
    pub predicted: Intent,
    /// The model's confidence in its wrong answer.
    pub confidence: f32,
    /// The confidence of the correct class.
    pub correct_confidence: f32,
    /// The margin between them. A small margin means the model nearly
    /// disagreed with itself; a large one means it was confidently wrong.
    pub margin: f32,
    /// Which category this mistake falls into.
    pub category: MistakeCategory,
}

/// The actionable buckets a mistake can fall into.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MistakeCategory {
    /// The utterance matches another intent's phrasing as well as this one's.
    IntentOverlap,
    /// The phrasing or vocabulary is simply not in the training data.
    MissingCoverage,
    /// The intent was right but the entity was extracted wrongly.
    EntityExtraction,
    /// An out-of-schema request that was accepted as a command.
    OutOfSchemaAccepted,
    /// An in-schema request that was rejected as out-of-schema.
    OutOfSchemaRejected,
    /// The model was right at the top but fell below the confidence threshold.
    LowConfidence,
    /// The dataset and the deployed path disagree about preprocessing.
    PreprocessingMismatch,
    /// The label in the data is probably wrong.
    LabelError,
}

impl MistakeCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            MistakeCategory::IntentOverlap => "intent_overlap",
            MistakeCategory::MissingCoverage => "missing_coverage",
            MistakeCategory::EntityExtraction => "entity_extraction",
            MistakeCategory::OutOfSchemaAccepted => "out_of_schema_accepted",
            MistakeCategory::OutOfSchemaRejected => "out_of_schema_rejected",
            MistakeCategory::LowConfidence => "low_confidence",
            MistakeCategory::PreprocessingMismatch => "preprocessing_mismatch",
            MistakeCategory::LabelError => "label_error",
        }
    }

    /// What to do about it.
    pub fn remedy(self) -> &'static str {
        match self {
            MistakeCategory::IntentOverlap => {
                "add explicit minimal pairs for the two intents, and make sure the \
                 discriminating token appears in both"
            }
            MistakeCategory::MissingCoverage => {
                "add examples of this phrasing for the expected intent"
            }
            MistakeCategory::EntityExtraction => "fix the entity extractor, not the classifier",
            MistakeCategory::OutOfSchemaAccepted => {
                "add this as a hard negative; a false accept can run the wrong tool"
            }
            MistakeCategory::OutOfSchemaRejected => {
                "add a positive example of this shape; the model is too eager to \
                 reject"
            }
            MistakeCategory::LowConfidence => {
                "the top class was right but below the threshold; raise the \
                 training signal for this class, do not lower the threshold"
            }
            MistakeCategory::PreprocessingMismatch => {
                "the deployed path disagrees with the evaluated path; this is a \
                 bug, not a data problem"
            }
            MistakeCategory::LabelError => "the label is probably wrong; review it with a human",
        }
    }
}

/// A confusion-matrix cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConfusionCell {
    pub expected: Intent,
    pub predicted: Intent,
    pub count: usize,
}

/// A calibration bin: predictions whose top probability fell in this range.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CalibrationBin {
    /// Lower bound of the confidence range, inclusive.
    pub low: f32,
    /// Upper bound of the confidence range, exclusive.
    pub high: f32,
    /// How many predictions fell in this range.
    pub count: usize,
    /// How many of them were correct.
    pub correct: usize,
}

impl CalibrationBin {
    /// Mean confidence of the predictions in this bin.
    pub fn mean_confidence(&self) -> f32 {
        (self.low + self.high) / 2.0
    }

    /// Fraction of the predictions in this bin that were correct.
    pub fn accuracy(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.correct as f32 / self.count as f32
    }
}

/// Per-intent metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PerIntent {
    pub intent: Intent,
    /// Examples of this intent in the evaluated split.
    pub support: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub false_negatives: usize,
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

/// A full analysis report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnalysisReport {
    /// Total examples evaluated.
    pub total: usize,
    pub accuracy: f32,
    pub macro_f1: f32,
    /// Examples per intent in the evaluated split.
    pub support: BTreeMap<String, usize>,
    pub per_intent: Vec<PerIntent>,
    /// The confusion matrix, as non-zero cells only.
    pub confusion: Vec<ConfusionCell>,
    /// Calibration bins.
    pub calibration: Vec<CalibrationBin>,
    /// Expected calibration error: the mean absolute gap between confidence and
    /// accuracy, weighted by bin population.
    pub expected_calibration_error: f32,
    /// `unknown` precision and recall.
    pub unknown_precision: f32,
    pub unknown_recall: f32,
    /// Fraction of out-of-schema examples that were wrongly accepted as a
    /// command. This is the number that matters for safety.
    pub false_accept_rate: f32,
    /// Fraction of in-schema examples that were wrongly rejected.
    pub false_reject_rate: f32,
    /// How many examples had the right intent but the wrong entity.
    pub entity_extraction_errors: usize,
    /// Fraction of examples with the right intent that had a wrong entity.
    pub entity_extraction_error_rate: f32,
    /// Every misclassification, most confident-wrong first.
    pub misclassifications: Vec<Misclassification>,
    /// Counts per mistake category.
    pub by_category: BTreeMap<String, usize>,
    /// The regression suite result, if one was supplied.
    pub regression_passed: usize,
    pub regression_total: usize,
    pub regression_failures: Vec<String>,
}

/// One classification, with enough detail to tell *why* it was wrong.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    /// The intent after the confidence threshold was applied. This is
    /// `Unknown` when the model's answer was not confident enough.
    pub intent: Intent,
    /// The probability of the top class.
    pub confidence: f32,
    /// The top class *before* thresholding.
    ///
    /// Without this, "the model ranked the right class first but was not
    /// confident enough" and "the model had no idea" look identical from the
    /// outside, and they have completely different remedies.
    pub top_class: Intent,
    /// The probability of the top class before thresholding.
    pub top_confidence: f32,
}

impl Prediction {
    /// A prediction whose thresholded intent is `intent`.
    pub fn new(intent: Intent, confidence: f32) -> Prediction {
        Prediction {
            intent,
            confidence,
            top_class: intent,
            top_confidence: confidence,
        }
    }
}

/// A scorer: text in, prediction out.
pub type Scorer<'a> = &'a dyn Fn(&str) -> Prediction;

/// Run a full analysis.
///
/// `entity_of` extracts the expected entities for an example, so entity
/// errors can be separated from intent errors. Passing `None` disables the
/// entity check.
pub fn analyse<F, G>(
    examples: &[(String, Intent)],
    predict: F,
    entity_of: Option<G>,
    regression: Option<&[(String, Intent)]>,
) -> AnalysisReport
where
    F: Fn(&str) -> Prediction,
    G: Fn(&str) -> Option<String>,
{
    let n = examples.len();
    let mut per: BTreeMap<Intent, (usize, usize, usize)> = BTreeMap::new();
    let mut confusion: BTreeMap<(Intent, Intent), usize> = BTreeMap::new();
    let mut misclassifications: Vec<Misclassification> = Vec::new();
    let mut correct = 0usize;
    let mut entity_errors = 0usize;
    let mut intent_right_entity_wrong = 0usize;

    // Calibration bins of width 0.1: (count, correct, confidence sum in
    // thousandths, so the mean is exact without floating-point drift).
    let mut bins: Vec<(usize, usize, u64)> = vec![(0, 0, 0); 10];

    let mut false_accepts = 0usize;
    let mut false_rejects = 0usize;
    let mut out_of_schema = 0usize;
    let mut in_schema = 0usize;

    for (text, expected) in examples {
        let prediction = predict(text);
        let predicted = prediction.intent;
        let confidence = prediction.confidence;
        let is_unknown_truth = *expected == Intent::Unknown;
        if is_unknown_truth {
            out_of_schema += 1;
        } else {
            in_schema += 1;
        }

        let entry = per.entry(*expected).or_insert((0, 0, 0));
        entry.0 += 1;
        if predicted == *expected {
            correct += 1;
            entry.1 += 1;
        } else {
            entry.2 += 1;
            if is_unknown_truth {
                // An out-of-schema request was accepted as a command.
                false_accepts += 1;
            } else if predicted == Intent::Unknown {
                // A real command was rejected.
                false_rejects += 1;
            }
        }
        if predicted != *expected {
            *confusion.entry((*expected, predicted)).or_insert(0) += 1;
        }

        // Calibration.
        let bin = ((confidence * 10.0).floor() as usize).min(9);
        bins[bin].0 += 1;
        bins[bin].2 += (confidence * 1000.0) as u64;
        if predicted == *expected {
            bins[bin].1 += 1;
        }

        // Entity extraction.
        if predicted == *expected {
            if let Some(getter) = &entity_of {
                let expected_entity = getter(text);
                let got = extract_entity_for(predicted, text);
                if expected_entity != got {
                    entity_errors += 1;
                    if expected_entity.is_some() {
                        intent_right_entity_wrong += 1;
                    }
                }
            }
        }

        if predicted != *expected {
            // When the top class was the expected one, the margin between the
            // two confidences is zero by construction — the model agreed with
            // the label and was merely not confident. That case is the
            // `LowConfidence` category, and its remedy is different.
            let correct_confidence = if prediction.top_class == *expected {
                prediction.top_confidence
            } else {
                0.0
            };
            misclassifications.push(Misclassification {
                text: redact_and_clip(text),
                expected: *expected,
                predicted,
                confidence,
                correct_confidence,
                margin: confidence - correct_confidence,
                category: classify_mistake(*expected, prediction),
            });
        }
    }

    // Most confident-wrong first: the mistakes that are most dangerous.
    misclassifications.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut per_intent = Vec::new();
    for intent in Intent::ALL {
        let (support, tp, fn_) = per.get(&intent).copied().unwrap_or((0, 0, 0));
        // False positives: how many times this intent was predicted for
        // something that was not it.
        let fp: usize = confusion
            .iter()
            .filter(|((_, predicted), _)| *predicted == intent)
            .map(|(_, count)| *count)
            .sum();
        let precision = if tp + fp == 0 {
            0.0
        } else {
            tp as f32 / (tp + fp) as f32
        };
        let recall = if support == 0 {
            0.0
        } else {
            tp as f32 / support as f32
        };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        per_intent.push(PerIntent {
            intent,
            support,
            true_positives: tp,
            false_positives: fp,
            false_negatives: fn_,
            precision,
            recall,
            f1,
        });
    }

    let calibration: Vec<CalibrationBin> = bins
        .iter()
        .enumerate()
        .filter(|(_, (count, _, _))| *count > 0)
        .map(|(i, (count, correct, _))| CalibrationBin {
            low: i as f32 / 10.0,
            high: (i + 1) as f32 / 10.0,
            count: *count,
            correct: *correct,
        })
        .collect();

    // Expected calibration error, computed from the bins directly: the
    // population-weighted mean gap between mean confidence and accuracy.
    let mut ece = 0.0f32;
    for (count, correct, conf_sum) in bins.iter() {
        if *count == 0 {
            continue;
        }
        let mean_conf = *conf_sum as f32 / *count as f32 / 1000.0;
        let acc = *correct as f32 / *count as f32;
        ece += (*count as f32 / n.max(1) as f32) * (mean_conf - acc).abs();
    }

    let mut by_category: BTreeMap<String, usize> = BTreeMap::new();
    for m in &misclassifications {
        *by_category
            .entry(m.category.as_str().to_string())
            .or_insert(0) += 1;
    }

    let unknown_metrics = per_intent.iter().find(|p| p.intent == Intent::Unknown);
    let unknown_precision = unknown_metrics.map(|p| p.precision).unwrap_or(0.0);
    let unknown_recall = unknown_metrics.map(|p| p.recall).unwrap_or(0.0);

    let (regression_passed, regression_total, regression_failures) = match regression {
        None => (0, 0, Vec::new()),
        Some(cases) => {
            let mut passed = 0usize;
            let mut failures = Vec::new();
            for (text, expected) in cases {
                let predicted = predict(text).intent;
                if predicted == *expected {
                    passed += 1;
                } else {
                    failures.push(format!(
                        "`{}` expected {} got {}",
                        redact_and_clip(text),
                        expected.as_str(),
                        predicted.as_str()
                    ));
                }
            }
            (passed, cases.len(), failures)
        }
    };

    AnalysisReport {
        total: n,
        accuracy: if n == 0 {
            0.0
        } else {
            correct as f32 / n as f32
        },
        macro_f1: {
            let measured: Vec<f32> = per_intent
                .iter()
                .filter(|p| p.support > 0)
                .map(|p| p.f1)
                .collect();
            if measured.is_empty() {
                0.0
            } else {
                measured.iter().sum::<f32>() / measured.len() as f32
            }
        },
        support: per_intent
            .iter()
            .map(|p| (p.intent.as_str().to_string(), p.support))
            .collect(),
        per_intent,
        confusion: confusion
            .iter()
            .map(|((expected, predicted), count)| ConfusionCell {
                expected: *expected,
                predicted: *predicted,
                count: *count,
            })
            .collect(),
        calibration,
        expected_calibration_error: ece,
        unknown_precision,
        unknown_recall,
        false_accept_rate: if out_of_schema == 0 {
            0.0
        } else {
            false_accepts as f32 / out_of_schema as f32
        },
        false_reject_rate: if in_schema == 0 {
            0.0
        } else {
            false_rejects as f32 / in_schema as f32
        },
        entity_extraction_errors: entity_errors,
        entity_extraction_error_rate: if intent_right_entity_wrong == 0 {
            0.0
        } else {
            entity_errors as f32 / (entity_errors + intent_right_entity_wrong) as f32
        },
        misclassifications,
        by_category,
        regression_passed,
        regression_total,
        regression_failures,
    }
}

/// Decide which category a mistake belongs to.
///
/// The order matters: the first matching rule wins, and the rules are ordered
/// from most specific to least.
fn classify_mistake(expected: Intent, prediction: Prediction) -> MistakeCategory {
    let predicted = prediction.intent;
    // A correct top-1 answer that fell below the threshold is a calibration
    // problem, not a classification problem, and the remedy is completely
    // different. This is checked first because it is the most specific
    // diagnosis.
    if predicted == Intent::Unknown
        && expected != Intent::Unknown
        && prediction.top_class == expected
    {
        return MistakeCategory::LowConfidence;
    }
    // Out-of-schema text accepted as a command is the dangerous direction.
    if expected == Intent::Unknown && predicted != Intent::Unknown {
        return MistakeCategory::OutOfSchemaAccepted;
    }
    // A real command rejected as out-of-schema.
    if expected != Intent::Unknown && predicted == Intent::Unknown {
        return MistakeCategory::OutOfSchemaRejected;
    }
    // Two command intents confused with each other.
    if expected != Intent::Unknown && predicted != Intent::Unknown {
        return MistakeCategory::IntentOverlap;
    }
    MistakeCategory::MissingCoverage
}

/// The entity the extractor would produce for an intent.
fn extract_entity_for(intent: Intent, text: &str) -> Option<String> {
    let entities = aaron_core::text::extract_entities(intent, text);
    entities.text.or(entities.path).or(entities.url)
}

/// Redact and clip text for a report.
pub fn redact_and_clip(text: &str) -> String {
    let redacted = redact_secrets(text);
    aaron_core::text::truncate_chars(&redacted, REPORT_TEXT_LIMIT)
}

/// Render the report as Markdown.
pub fn to_markdown(report: &AnalysisReport) -> String {
    let mut out = String::new();
    out.push_str("# Error analysis\n\n");
    out.push_str(&format!(
        "Evaluated **{}** example(s). Accuracy **{:.4}**, macro-F1 **{:.4}**.\n\n",
        report.total, report.accuracy, report.macro_f1
    ));

    out.push_str("## Safety-relevant rates\n\n");
    out.push_str(&format!(
        "| metric | value | meaning |\n|---|---|---|\n\
         | false-accept rate | {:.4} | out-of-schema text accepted as a command |\n\
         | false-reject rate | {:.4} | real commands rejected as out-of-schema |\n\
         | unknown precision | {:.4} | of the times it said `unknown`, how often it was right |\n\
         | unknown recall | {:.4} | of the out-of-schema text, how much it caught |\n\
         | expected calibration error | {:.4} | mean gap between confidence and accuracy |\n\n",
        report.false_accept_rate,
        report.false_reject_rate,
        report.unknown_precision,
        report.unknown_recall,
        report.expected_calibration_error
    ));

    out.push_str("## Per-intent metrics\n\n");
    out.push_str("| intent | support | precision | recall | f1 |\n|---|---|---|---|---|\n");
    for p in &report.per_intent {
        out.push_str(&format!(
            "| {} | {} | {:.3} | {:.3} | {:.3} |\n",
            p.intent.as_str(),
            p.support,
            p.precision,
            p.recall,
            p.f1
        ));
    }
    out.push('\n');

    out.push_str("## Calibration\n\n");
    out.push_str("| confidence band | count | correct | accuracy |\n|---|---|---|---|\n");
    for b in &report.calibration {
        out.push_str(&format!(
            "| {:.1}–{:.1} | {} | {} | {:.3} |\n",
            b.low,
            b.high,
            b.count,
            b.correct,
            b.accuracy()
        ));
    }
    out.push('\n');

    out.push_str("## Confusions\n\n");
    if report.confusion.is_empty() {
        out.push_str("None.\n\n");
    } else {
        out.push_str("| expected | predicted | count |\n|---|---|---|\n");
        for c in &report.confusion {
            out.push_str(&format!(
                "| {} | {} | {} |\n",
                c.expected.as_str(),
                c.predicted.as_str(),
                c.count
            ));
        }
        out.push('\n');
    }

    out.push_str("## Mistakes by category\n\n");
    if report.by_category.is_empty() {
        out.push_str("No misclassifications.\n\n");
    } else {
        out.push_str("| category | count | what to do |\n|---|---|---|\n");
        for (name, count) in &report.by_category {
            let remedy = match name.as_str() {
                "intent_overlap" => MistakeCategory::IntentOverlap.remedy(),
                "missing_coverage" => MistakeCategory::MissingCoverage.remedy(),
                "entity_extraction" => MistakeCategory::EntityExtraction.remedy(),
                "out_of_schema_accepted" => MistakeCategory::OutOfSchemaAccepted.remedy(),
                "out_of_schema_rejected" => MistakeCategory::OutOfSchemaRejected.remedy(),
                "low_confidence" => MistakeCategory::LowConfidence.remedy(),
                "preprocessing_mismatch" => MistakeCategory::PreprocessingMismatch.remedy(),
                _ => MistakeCategory::LabelError.remedy(),
            };
            out.push_str(&format!("| {name} | {count} | {remedy} |\n"));
        }
        out.push('\n');
    }

    out.push_str("## Representative misclassifications\n\n");
    out.push_str("Most confident-wrong first. Text is redacted and truncated.\n\n");
    if report.misclassifications.is_empty() {
        out.push_str("None.\n\n");
    } else {
        out.push_str(
            "| text | expected | predicted | confidence | category |\n|---|---|---|---|---|\n",
        );
        for m in report.misclassifications.iter().take(40) {
            out.push_str(&format!(
                "| {} | {} | {} | {:.3} | {} |\n",
                m.text.replace('|', "\\|"),
                m.expected.as_str(),
                m.predicted.as_str(),
                m.confidence,
                m.category.as_str()
            ));
        }
        out.push('\n');
    }

    if report.regression_total > 0 {
        out.push_str(&format!(
            "## Regression suite\n\n{}/{} passed.\n\n",
            report.regression_passed, report.regression_total
        ));
        for f in &report.regression_failures {
            out.push_str(&format!("- {f}\n"));
        }
        out.push('\n');
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(text: &str, intent: Intent) -> (String, Intent) {
        (text.to_string(), intent)
    }

    #[test]
    fn a_perfect_scorer_produces_no_mistakes() {
        let examples = vec![
            ex("hello", Intent::Greeting),
            ex("status", Intent::SystemStatus),
        ];
        let predict = |t: &str| {
            if t == "hello" {
                Prediction::new(Intent::Greeting, 0.9)
            } else {
                Prediction::new(Intent::SystemStatus, 0.9)
            }
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(r.accuracy, 1.0);
        assert!(r.misclassifications.is_empty());
        assert_eq!(r.false_accept_rate, 0.0);
        assert_eq!(r.false_reject_rate, 0.0);
    }

    #[test]
    fn false_accepts_and_rejects_are_counted_separately() {
        // One out-of-schema utterance accepted as a command, and one real
        // command rejected. These are opposite errors and must not be summed.
        let examples = vec![
            ex("what is the capital of peru", Intent::Unknown),
            ex("check system status", Intent::SystemStatus),
        ];
        let predict = |t: &str| {
            if t.contains("peru") {
                Prediction::new(Intent::SystemStatus, 0.8)
            } else {
                Prediction::new(Intent::Unknown, 0.4)
            }
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(
            r.false_accept_rate, 1.0,
            "the out-of-schema one was accepted"
        );
        assert_eq!(r.false_reject_rate, 1.0, "the real command was rejected");
    }

    #[test]
    fn a_correct_top_one_below_the_threshold_is_a_low_confidence_mistake() {
        // The model ranked the right class first but not confidently enough.
        // That is a calibration problem with a different remedy from a
        // ranking problem, so the category must distinguish them.
        let examples = vec![ex("cancel that", Intent::Cancel)];
        // The model ranked `cancel` first at 0.40 — below the 0.55 threshold —
        // so the threshold turned it into `unknown`.
        let predict = |_: &str| Prediction {
            intent: Intent::Unknown,
            confidence: 0.40,
            top_class: Intent::Cancel,
            top_confidence: 0.40,
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(r.misclassifications.len(), 1);
        assert_eq!(
            r.misclassifications[0].category,
            MistakeCategory::LowConfidence
        );
    }

    #[test]
    fn an_out_of_schema_acceptance_is_its_own_category() {
        let examples = vec![ex("play some music", Intent::Unknown)];
        let predict = |_: &str| Prediction::new(Intent::StartApplication, 0.9);
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(
            r.misclassifications[0].category,
            MistakeCategory::OutOfSchemaAccepted
        );
    }

    #[test]
    fn an_out_of_schema_rejection_is_its_own_category() {
        let examples = vec![ex("check system status", Intent::SystemStatus)];
        // The model ranked `unknown` first at 0.9, so this is a genuine
        // over-rejection, not a threshold artefact.
        let predict = |_: &str| Prediction::new(Intent::Unknown, 0.9);
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(
            r.misclassifications[0].category,
            MistakeCategory::OutOfSchemaRejected
        );
    }

    #[test]
    fn two_command_intents_confused_is_intent_overlap() {
        let examples = vec![ex("open the terminal", Intent::StartApplication)];
        let predict = |_: &str| Prediction::new(Intent::ReadFile, 0.7);
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(
            r.misclassifications[0].category,
            MistakeCategory::IntentOverlap
        );
    }

    #[test]
    fn mistakes_are_ordered_most_confident_wrong_first() {
        let examples = vec![ex("a", Intent::Greeting), ex("b", Intent::Help)];
        let predict = |t: &str| {
            if t == "a" {
                Prediction::new(Intent::Unknown, 0.95)
            } else {
                Prediction::new(Intent::Unknown, 0.30)
            }
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(r.misclassifications[0].confidence, 0.95);
        assert_eq!(r.misclassifications[1].confidence, 0.30);
    }

    #[test]
    fn text_in_a_report_is_redacted() {
        let examples = vec![ex(
            "open https://user:hunter2@example.com/",
            Intent::OpenUrl,
        )];
        let predict = |_: &str| Prediction::new(Intent::SystemStatus, 0.9);
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        let text = &r.misclassifications[0].text;
        assert!(
            !text.contains("hunter2"),
            "credential survived into the report: {text}"
        );
    }

    #[test]
    fn text_in_a_report_is_truncated() {
        let long = "x".repeat(5000);
        let examples = vec![ex(&long, Intent::Greeting)];
        let predict = |_: &str| Prediction::new(Intent::Help, 0.9);
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert!(
            r.misclassifications[0].text.chars().count() <= REPORT_TEXT_LIMIT,
            "report text was {} chars",
            r.misclassifications[0].text.chars().count()
        );
    }

    #[test]
    fn calibration_is_computed_over_the_bins() {
        let examples = vec![ex("a", Intent::Greeting), ex("b", Intent::Help)];
        let predict = |t: &str| {
            if t == "a" {
                Prediction::new(Intent::Greeting, 0.95)
            } else {
                Prediction::new(Intent::Help, 0.15)
            }
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert!(!r.calibration.is_empty());
        // Both predictions were right, so the error is |confidence - 1| per bin.
        assert!(r.expected_calibration_error > 0.0);
        assert!(r.expected_calibration_error < 0.5);
    }

    #[test]
    fn the_confusion_matrix_records_only_non_zero_cells() {
        let examples = vec![ex("a", Intent::Greeting), ex("b", Intent::Greeting)];
        let predict = |t: &str| {
            if t == "a" {
                Prediction::new(Intent::Greeting, 0.9)
            } else {
                Prediction::new(Intent::Help, 0.9)
            }
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        assert_eq!(r.confusion.len(), 1);
        assert_eq!(r.confusion[0].expected, Intent::Greeting);
        assert_eq!(r.confusion[0].predicted, Intent::Help);
        assert_eq!(r.confusion[0].count, 1);
    }

    #[test]
    fn per_intent_precision_and_recall_are_computed_correctly() {
        let examples = vec![
            ex("a", Intent::Greeting),
            ex("b", Intent::Greeting),
            ex("c", Intent::Greeting),
            ex("d", Intent::Help),
        ];
        let predict = |t: &str| {
            if t == "c" {
                Prediction::new(Intent::Help, 0.9)
            } else {
                Prediction::new(Intent::Greeting, 0.9)
            }
        };
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        let g = r
            .per_intent
            .iter()
            .find(|p| p.intent == Intent::Greeting)
            .unwrap();
        assert_eq!(g.support, 3);
        assert_eq!(g.true_positives, 2);
        assert_eq!(g.false_negatives, 1);
        assert!((g.recall - 2.0 / 3.0).abs() < 1e-5);
        let h = r
            .per_intent
            .iter()
            .find(|p| p.intent == Intent::Help)
            .unwrap();
        assert_eq!(h.support, 1);
        assert_eq!(h.false_positives, 1, "the `c` utterance was predicted help");
        assert_eq!(h.true_positives, 0);
        assert!((h.precision - 0.0).abs() < 1e-5);
        assert!((h.recall - 0.0).abs() < 1e-5);
    }

    #[test]
    fn the_regression_suite_is_scored_and_failures_are_listed() {
        let examples = vec![ex("a", Intent::Greeting)];
        let predict = |t: &str| {
            if t == "a" {
                Prediction::new(Intent::Greeting, 0.9)
            } else {
                Prediction::new(Intent::Unknown, 0.9)
            }
        };
        let regression = vec![ex("a", Intent::Greeting), ex("b", Intent::Help)];
        let r = analyse(
            &examples,
            predict,
            None::<fn(&str) -> Option<String>>,
            Some(&regression),
        );
        assert_eq!(r.regression_total, 2);
        assert_eq!(r.regression_passed, 1);
        assert_eq!(r.regression_failures.len(), 1);
        assert!(r.regression_failures[0].contains("expected help got unknown"));
    }

    #[test]
    fn the_markdown_report_contains_every_required_section() {
        let examples = vec![ex("a", Intent::Greeting), ex("b", Intent::Unknown)];
        let predict = |t: &str| {
            if t == "a" {
                Prediction::new(Intent::Greeting, 0.9)
            } else {
                Prediction::new(Intent::SystemStatus, 0.9)
            }
        };
        let r = analyse(
            &examples,
            predict,
            None::<fn(&str) -> Option<String>>,
            Some(&examples),
        );
        let md = to_markdown(&r);
        for section in [
            "## Safety-relevant rates",
            "## Per-intent metrics",
            "## Calibration",
            "## Confusions",
            "## Mistakes by category",
            "## Representative misclassifications",
            "## Regression suite",
        ] {
            assert!(md.contains(section), "missing section {section}:\n{md}");
        }
    }

    #[test]
    fn the_report_round_trips_through_json() {
        let examples = vec![ex("a", Intent::Greeting)];
        let predict = |_: &str| Prediction::new(Intent::Help, 0.9);
        let r = analyse(&examples, predict, None::<fn(&str) -> Option<String>>, None);
        let json = serde_json::to_string(&r).unwrap();
        let back: AnalysisReport = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
