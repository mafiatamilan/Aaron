//! The neural intent classifier.
//!
//! This is the only component that uses the custom 1M-parameter model. It
//! implements [`aaron_core::IntentClassifier`], so the agent depends on the
//! trait and not on Candle.

use std::path::{Path, PathBuf};

use candle_core::Device;

use aaron_core::error::{Error, Result};
use aaron_core::intent::{ClassifySource, Command, Readiness};
use aaron_core::text::extract_entities;
use aaron_core::{Intent, IntentClassifier};

use crate::checkpoint::{self, Manifest};
use crate::eval::logits_to_intent;
use crate::model::Heads;
use crate::model::{expected_parameter_count, FeatureBatch, IntentModel};

/// The default probability above which the readiness head says the request
/// is not specific enough to act on.
///
/// A single uncalibrated logit, so the probability is not a frequency and
/// 0.5 is a starting point rather than a measured value. Release 0.5
/// selects the threshold on the readiness corpus's *training* split and
/// reports on its held-out split.
pub const DEFAULT_READINESS_THRESHOLD: f32 = 0.5;

/// Where to look for a checkpoint when the config does not name one.
pub const DEFAULT_CHECKPOINTS: &[&str] = &[
    "models/brain/aaron-intent-1m.ckpt",
    "data/checkpoints/aaron-intent-1m.ckpt",
];

/// The model-backed classifier.
pub struct NeuralClassifier {
    model: IntentModel,
    manifest: Manifest,
    path: PathBuf,
    threshold: f32,
    temperature: f32,
    max_features: usize,
    /// The probability above which the readiness head says the request is
    /// not specific enough to act on.
    readiness_threshold: f32,
}

impl NeuralClassifier {
    /// Load a checkpoint from an explicit path.
    pub fn load(
        path: impl AsRef<Path>,
        threshold: f32,
        temperature: f32,
        max_features: usize,
    ) -> Result<NeuralClassifier> {
        Self::load_with_readiness_threshold(
            path,
            threshold,
            temperature,
            max_features,
            DEFAULT_READINESS_THRESHOLD,
        )
    }

    /// Load a checkpoint with an explicit readiness threshold.
    ///
    /// The readiness threshold is a threshold on a single uncalibrated
    /// logit, so it has to be sweepable: the operating point is a trade-off
    /// between refusing an under-specified request and refusing a perfectly
    /// good one, and the only way to choose it is to measure both sides.
    pub fn load_with_readiness_threshold(
        path: impl AsRef<Path>,
        threshold: f32,
        temperature: f32,
        max_features: usize,
        readiness_threshold: f32,
    ) -> Result<NeuralClassifier> {
        let path = path.as_ref().to_path_buf();
        let (manifest, tensors) = checkpoint::load(&path)?;
        let mut classifier = Self::from_parts(
            manifest,
            tensors,
            path,
            threshold,
            temperature,
            max_features,
        )?;
        classifier.readiness_threshold = readiness_threshold;
        Ok(classifier)
    }

    /// Search the default locations for a checkpoint.
    pub fn discover<P: AsRef<Path>>(
        explicit: Option<&str>,
        roots: &[P],
        threshold: f32,
        temperature: f32,
        max_features: usize,
    ) -> Result<NeuralClassifier> {
        let path = find_checkpoint(explicit, roots).ok_or_else(|| {
            Error::missing_with_hint(
                "the 1M-parameter intent checkpoint",
                "run `aaron brain train --dataset training/dataset/current.jsonl` to build one, \
                 or set brain.backend = \"heuristic\" to use the deterministic router",
            )
        })?;
        NeuralClassifier::load(path, threshold, temperature, max_features)
    }

    fn from_parts(
        manifest: Manifest,
        tensors: std::collections::HashMap<String, candle_core::Tensor>,
        path: PathBuf,
        threshold: f32,
        temperature: f32,
        max_features: usize,
    ) -> Result<NeuralClassifier> {
        // The checkpoint says which heads it carries; the model is built to
        // match. Guessing would either fail to find the tensors or, worse,
        // load weights into a model of the wrong shape.
        let heads = checkpoint::manifest_heads(&manifest);
        if manifest.param_count != expected_parameter_count(heads) {
            return Err(Error::Model(format!(
                "checkpoint declares {} parameters, architecture requires {}",
                manifest.param_count,
                expected_parameter_count(heads)
            )));
        }
        let model = IntentModel::from_tensors(tensors, &Device::Cpu, heads)
            .map_err(|e| Error::Model(format!("could not rebuild the model: {e}")))?;
        Ok(NeuralClassifier {
            model,
            manifest,
            path,
            threshold,
            temperature,
            max_features,
            readiness_threshold: DEFAULT_READINESS_THRESHOLD,
        })
    }

    /// The checkpoint manifest, for `aaron model info`.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Where the weights came from.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Raw logits for one utterance.
    ///
    /// Public so the error-analysis tool can score a candidate and also see
    /// the pre-threshold argmax, which is what distinguishes "wrong" from
    /// "right but not confident enough".
    pub fn logits_for(&self, text: &str) -> Result<Vec<f32>> {
        self.logits(text)
    }

    /// Raw logits for one utterance.
    fn logits(&self, text: &str) -> Result<Vec<f32>> {
        Ok(self.forward(text)?.0)
    }

    /// One forward pass, returning the intent logits and the readiness logit.
    ///
    /// Both come from the same pass on purpose. Computing the readiness in a
    /// second forward pass doubled the classification latency — measured at
    /// 1.49 ms against 0.54 ms — for a head that costs 168 multiply-adds.
    fn forward(&self, text: &str) -> Result<(Vec<f32>, Option<candle_core::Tensor>)> {
        let batch = FeatureBatch::from_texts(&[text.to_string()]);
        let (logits, ready) = self
            .model
            .forward_with_readiness(&batch, None, None)
            .map_err(|e| Error::Brain(format!("forward pass failed: {e}")))?;
        let values = logits
            .flatten_all()
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| Error::Brain(e.to_string()))?;
        Ok((values, ready))
    }

    /// The readiness verdict for one utterance.
    ///
    /// `Undetermined` when the checkpoint has no readiness head, when the
    /// head's output is not a finite number, or when the probability cannot
    /// be trusted. A verdict the classifier cannot vouch for is reported as
    /// *no verdict*, never as `Ready`: a missing verdict must not be the
    /// reason a tool runs.
    fn readiness_from(&self, ready: Option<candle_core::Tensor>) -> Readiness {
        if self.model.heads() != Heads::WithReadiness {
            return Readiness::Undetermined;
        }
        let ready = match ready {
            Some(t) => t,
            None => return Readiness::Undetermined,
        };
        let logit = match ready.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
            Ok(values) => values.first().copied().unwrap_or(f32::NAN),
            Err(_) => return Readiness::Undetermined,
        };
        if !logit.is_finite() {
            return Readiness::Undetermined;
        }
        let probability = 1.0 / (1.0 + (-logit).exp());
        if !probability.is_finite() || !(0.0..=1.0).contains(&probability) {
            return Readiness::Undetermined;
        }
        if probability >= self.readiness_threshold {
            Readiness::NeedsClarification
        } else {
            Readiness::Ready
        }
    }

    /// Class probabilities for one utterance.
    ///
    /// These are *probabilities*, suitable for reporting. Feeding them back
    /// into [`logits_to_intent`] would apply a second softmax — see
    /// `classification_is_not_double_softmaxed` for what that costs.
    pub fn probabilities(&self, text: &str) -> Result<Vec<f32>> {
        crate::model::softmax(&self.logits(text)?, self.temperature).ok_or_else(|| {
            Error::Brain(
                "the model produced non-finite logits; the checkpoint may be corrupt".into(),
            )
        })
    }

    /// Inference latency for a single utterance, for the benchmark suite.
    pub fn time_one(&self, text: &str) -> Result<u64> {
        let start = std::time::Instant::now();
        let _ = self.probabilities(text)?;
        Ok(start.elapsed().as_micros() as u64)
    }
}

impl IntentClassifier for NeuralClassifier {
    fn name(&self) -> &'static str {
        "aaron-brain/1M"
    }

    fn requires_checkpoint(&self) -> bool {
        true
    }

    fn classify(&self, utterance: &str) -> Result<Command> {
        // `logits_to_intent` applies the softmax itself, so it must be given
        // logits. Handing it probabilities flattened the distribution until
        // every class scored ~0.17 and the confidence threshold rejected
        // everything.
        let (logits, ready) = self.forward(utterance)?;
        let (intent, confidence) = logits_to_intent(&logits, self.temperature, self.threshold);
        // Entity extraction always runs on the original utterance, even when
        // the intent is `unknown`, so a correction can still be offered.
        let entities = extract_entities(intent, &self.truncate(utterance));
        // The readiness verdict, from the head when the checkpoint has one.
        let readiness = self.readiness_from(ready);
        Ok(Command {
            intent,
            entities,
            confidence: confidence as f32,
            source: ClassifySource::Neural,
            readiness,
        })
    }
}

impl NeuralClassifier {
    fn truncate(&self, text: &str) -> String {
        let cap = self.max_features.saturating_mul(8).max(64);
        aaron_core::text::truncate_chars(text, cap)
    }
}

/// Find a checkpoint, preferring the configured path, then the well-known
/// locations under each root.
pub fn find_checkpoint<P: AsRef<Path>>(explicit: Option<&str>, roots: &[P]) -> Option<PathBuf> {
    if let Some(p) = explicit.filter(|p| !p.trim().is_empty()) {
        let candidate = PathBuf::from(p);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    for root in roots {
        for rel in DEFAULT_CHECKPOINTS {
            let candidate = root.as_ref().join(rel);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }
    None
}

/// Build the best available classifier for the configuration.
///
/// `auto` prefers the neural model and falls back to the deterministic router
/// when no checkpoint is installed, so a fresh install works with zero
/// downloads and still improves once the user trains a model.
pub fn build(
    backend: aaron_core::config::BrainBackend,
    checkpoint_path: &str,
    threshold: f32,
    temperature: f32,
    max_features: usize,
    roots: &[std::path::PathBuf],
) -> Result<std::sync::Arc<dyn IntentClassifier>> {
    use aaron_core::config::BrainBackend;
    match backend {
        BrainBackend::Heuristic => Ok(std::sync::Arc::new(
            crate::heuristic::HeuristicClassifier::new(threshold),
        )),
        BrainBackend::Neural => Ok(std::sync::Arc::new(NeuralClassifier::load(
            find_checkpoint(Some(checkpoint_path), roots)
                .unwrap_or_else(|| PathBuf::from(checkpoint_path)),
            threshold,
            temperature,
            max_features,
        )?)),
        BrainBackend::Auto => match find_checkpoint(Some(checkpoint_path), roots) {
            Some(path) => match NeuralClassifier::load(&path, threshold, temperature, max_features)
            {
                Ok(c) => Ok(std::sync::Arc::new(c)),
                Err(e) => {
                    tracing::warn!(error = %e, "checkpoint present but unusable; using the deterministic router");
                    Ok(std::sync::Arc::new(
                        crate::heuristic::HeuristicClassifier::new(threshold),
                    ))
                }
            },
            None => Ok(std::sync::Arc::new(
                crate::heuristic::HeuristicClassifier::new(threshold),
            )),
        },
    }
}

/// Which intents are in the schema, for diagnostics.
pub fn schema_summary() -> Vec<(&'static str, &'static str)> {
    Intent::ALL
        .iter()
        .map(|i| (i.as_str(), i.description()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::initial_tensors;
    use std::collections::HashMap;

    fn write_checkpoint(dir: &Path) -> PathBuf {
        let path = dir
            .join("models")
            .join("brain")
            .join("aaron-intent-1m.ckpt");
        let tensors: HashMap<String, candle_core::Tensor> =
            initial_tensors(3, crate::model::Heads::IntentOnly).unwrap();
        crate::checkpoint::save(
            &path,
            &tensors,
            &crate::checkpoint::CheckpointInfo {
                dataset_version: "test".into(),
                seed: 3,
                epochs: 1,
                metrics: serde_json::json!({"accuracy": 1.0}),
            },
        )
        .unwrap();
        path
    }

    #[test]
    fn loads_a_valid_checkpoint_and_classifies() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_checkpoint(dir.path());
        let c = NeuralClassifier::load(&path, 0.0, 1.0, 256).unwrap();
        let cmd = c.classify("check system status").unwrap();
        assert_eq!(cmd.source, ClassifySource::Neural);
        assert!(cmd.confidence >= 0.0 && cmd.confidence <= 1.0);
        assert_eq!(c.manifest().param_count, 1_000_000);
    }

    #[test]
    fn classification_is_not_double_softmaxed() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_checkpoint(dir.path());
        let c = NeuralClassifier::load(&path, 0.0, 1.0, 256).unwrap();
        let text = "check system status";

        // `classify` must apply exactly one softmax. `probabilities` is already
        // a distribution, so its values must be used directly as confidence
        // rather than being fed through `logits_to_intent` again.
        let probs = c.probabilities(text).unwrap();
        let top = probs.iter().copied().fold(0.0f32, f32::max);
        let cmd = c.classify(text).unwrap();
        assert!(
            (cmd.confidence - top).abs() < 1e-5,
            "confidence {} should equal the top probability {top}; a second \
             softmax would compress everything toward 1/{}",
            cmd.confidence,
            crate::model::CLASSES
        );
    }

    /// The contract that actually matters: the metric reported by training is
    /// the behaviour you get from the deployed checkpoint.
    ///
    /// This is the test that catches a divergence between the scoring path
    /// (raw logits) and the classification path (whatever `classify` does with
    /// them). A double softmax looked like a working model in training and a
    /// model that answered `unknown` to everything once deployed.
    #[test]
    fn the_deployed_checkpoint_reproduces_the_training_metric() {
        let dataset_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../training/dataset/current.jsonl"
        );
        let Ok(dataset) = crate::dataset::Dataset::load(dataset_path) else {
            // The shipped dataset is absent in a trimmed checkout; there is
            // nothing to assert against, and failing would be noise.
            return;
        };
        let config = crate::train::TrainingConfig {
            epochs: 40,
            learning_rate: 3e-3,
            weight_decay: 1e-2,
            patience: 40,
            ..Default::default()
        };
        let outcome = crate::train::train(&dataset, &config).unwrap();
        assert!(
            outcome.test_metrics.accuracy > 0.5,
            "the model did not learn at all in this test (accuracy {}); \
             the fixture needs attention before this test means anything",
            outcome.test_metrics.accuracy
        );

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("candidate.ckpt");
        crate::checkpoint::save(
            &path,
            &outcome.tensors,
            &crate::checkpoint::CheckpointInfo {
                dataset_version: outcome.dataset_version.clone(),
                seed: config.seed,
                epochs: config.epochs,
                metrics: serde_json::json!({"accuracy": outcome.test_metrics.accuracy}),
            },
        )
        .unwrap();
        let c = NeuralClassifier::load(&path, 0.0, config.temperature, 256).unwrap();

        let test = dataset.split(crate::dataset::Split::Test);
        let agree: usize = test
            .iter()
            .filter(|e| c.classify(&e.text).unwrap().intent == e.intent)
            .count();
        let deployed = agree as f64 / test.len().max(1) as f64;

        // `score` uses the threshold, so allow the small slack the two paths
        // differ by, then require the deployed model to be in the same league.
        // A pipeline that quietly breaks the deployed path shows up here as a
        // number near 1/CLASSES.
        assert!(
            deployed > outcome.test_metrics.accuracy - 0.15,
            "deployed accuracy {deployed:.4} does not match the trained \
             metric {:.4}; the scoring and classification paths have diverged",
            outcome.test_metrics.accuracy
        );
    }

    #[test]
    fn probabilities_are_a_distribution() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_checkpoint(dir.path());
        let c = NeuralClassifier::load(&path, 0.5, 1.0, 256).unwrap();
        let p = c.probabilities("hello there").unwrap();
        assert_eq!(p.len(), crate::model::CLASSES);
        let sum: f32 = p.iter().sum();
        assert!((sum - 1.0).abs() < 1e-4, "probabilities sum to {sum}");
    }

    #[test]
    fn a_high_threshold_forces_unknown() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_checkpoint(dir.path());
        let c = NeuralClassifier::load(&path, 1.01, 1.0, 256).unwrap();
        let cmd = c.classify("check system status").unwrap();
        assert_eq!(cmd.intent, Intent::Unknown);
    }

    #[test]
    fn corrupt_checkpoint_produces_a_model_error_not_a_panic() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_checkpoint(dir.path());
        let mut bytes = std::fs::read(&path).unwrap();
        let n = bytes.len();
        bytes[n - 2] ^= 0xFF;
        std::fs::write(&path, &bytes).unwrap();
        let Err(err) = NeuralClassifier::load(&path, 0.5, 1.0, 256) else {
            panic!("a corrupt checkpoint must be refused");
        };
        assert!(matches!(err, Error::Model(_)), "got {err}");
    }

    #[test]
    fn missing_checkpoint_is_a_clear_install_hint() {
        let Err(err) =
            NeuralClassifier::discover(Some("no/such/file.ckpt"), &[] as &[PathBuf], 0.5, 1.0, 256)
        else {
            panic!("a missing checkpoint must be reported");
        };
        assert!(matches!(err, Error::MissingDependency(_)));
        assert!(err.to_string().contains("aaron brain train"));
    }

    #[test]
    fn auto_backend_falls_back_to_the_deterministic_router() {
        let roots: Vec<std::path::PathBuf> = vec![std::path::PathBuf::from("/nonexistent")];
        let c = build(
            aaron_core::config::BrainBackend::Auto,
            "",
            0.5,
            1.0,
            256,
            &roots,
        )
        .unwrap();
        assert_eq!(c.name(), crate::heuristic::NAME);
    }

    #[test]
    fn auto_backend_uses_the_checkpoint_when_present() {
        let dir = tempfile::tempdir().unwrap();
        write_checkpoint(dir.path());
        let roots = vec![dir.path().to_path_buf()];
        let c = build(
            aaron_core::config::BrainBackend::Auto,
            "",
            0.0,
            1.0,
            256,
            &roots,
        )
        .unwrap();
        assert_eq!(c.name(), "aaron-brain/1M");
    }

    #[test]
    fn discovery_prefers_the_explicit_path() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_checkpoint(dir.path());
        let found = find_checkpoint(Some(path.to_str().unwrap()), &[dir.path()]).unwrap();
        assert_eq!(found, path);
        // A bad explicit path falls back to the well-known locations, which do
        // contain a checkpoint here.
        assert_eq!(
            find_checkpoint(Some("nope.ckpt"), &[dir.path()]),
            Some(dir.path().join("models/brain/aaron-intent-1m.ckpt"))
        );
        // With no checkpoint anywhere, discovery fails cleanly.
        assert!(find_checkpoint(Some("nope.ckpt"), &[&dir.path().join("empty")]).is_none());
    }
}
