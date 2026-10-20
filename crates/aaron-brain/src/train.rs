//! Offline training for the custom intent model.
//!
//! Training runs entirely in Rust. It is deterministic given `(dataset,
//! seed, config)`: initialisation comes from [`crate::model::SplitMix64`], batch
//! order comes from the same PRNG, and every reduction is a fixed-size CPU
//! kernel.
//!
//! The result is *not* written into the live model. [`train`] returns metrics
//! and tensors; the caller must explicitly save a checkpoint, and the deployed
//! checkpoint is only replaced by an explicit promotion step in `aaron-learning`.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};

use aaron_core::error::{Error, Result};
use aaron_core::Intent;
use serde::{Deserialize, Serialize};

use crate::dataset::{Dataset, Split};
use crate::eval::{evaluate, logits_to_intent, Metrics};
use crate::model::{
    expected_parameter_count, initial_tensors, FeatureBatch, Heads, IntentModel, SplitMix64,
};

/// Training hyperparameters.
#[derive(Debug, Clone, PartialEq)]
pub struct TrainingConfig {
    pub seed: u64,
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    /// Stop early when validation macro-F1 has not improved for this many epochs.
    pub patience: u32,
    /// Decision threshold used when scoring during training.
    pub confidence_threshold: f32,
    pub temperature: f32,
    /// Dropout probability on the hidden layer. Adds no parameters.
    pub dropout: f64,
    /// Dropout probability on the *input features*. Adds no parameters.
    ///
    /// For a bag-of-hashed-features model this is the strongest augmentation
    /// available: it forces the model to rely on several cues instead of
    /// memorising one.
    pub feature_dropout: f64,
    /// Label smoothing, in `[0, 1)`. Adds no parameters.
    pub label_smoothing: f64,
    /// How to weight the classes in the loss.
    pub class_weighting: ClassWeighting,
}

/// How to weight classes in the cross-entropy loss.
///
/// # Why
///
/// The release-0.3 dataset is measurably imbalanced: after the hard negatives
/// were added, `unknown` had 187 training examples and the smallest intent
/// had 45 — a ratio of 4.2x. An unbalanced loss makes the majority class
/// cheap and the minority class expensive, and the model learns the
/// majority.
///
/// # The form
///
/// The weight for a class with `n_c` examples is
///
/// ```text
/// w_c = (mean_class_size / n_c) ^ p
/// ```
///
/// normalised so the *mean* weight over the training set is exactly 1. That
/// normalisation matters: it keeps the loss on the same scale as the
/// unweighted loss, so the learning rate does not have to be retuned and the
/// experiment changes one thing rather than two.
///
/// * `None` — every class weighs 1. The baseline.
/// * `Balanced` (`p = 1`) — the fully balanced loss.
/// * `SqrtBalanced` (`p = 0.5`) — a middle ground, and the one the sweep
///   usually prefers: full balancing over-corrects when the imbalance is
///   mild and the classes are genuinely different sizes.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum ClassWeighting {
    /// Every class weighs 1.
    #[default]
    None,
    /// `w_c = (mean / n_c) ^ 1`.
    Balanced,
    /// `w_c = (mean / n_c) ^ 0.5`.
    SqrtBalanced,
    /// `w_c = (mean / n_c) ^ p`, for an arbitrary `p`.
    Power(f32),
}

impl ClassWeighting {
    /// The exponent, or `None` for no weighting.
    pub fn exponent(self) -> Option<f32> {
        match self {
            ClassWeighting::None => None,
            ClassWeighting::Balanced => Some(1.0),
            ClassWeighting::SqrtBalanced => Some(0.5),
            ClassWeighting::Power(p) => Some(p),
        }
    }
}

/// Per-class loss weights, derived from the training split.
///
/// The weights are normalised so their mean over the training examples is 1.
pub fn class_weights(examples: &[&crate::dataset::Example], weighting: ClassWeighting) -> Vec<f32> {
    let Some(power) = weighting.exponent() else {
        return vec![1.0; crate::model::CLASSES];
    };
    let mut counts = [0usize; crate::model::CLASSES];
    for e in examples {
        counts[e.intent.index()] += 1;
    }
    let present: Vec<usize> = counts.iter().copied().filter(|c| *c > 0).collect();
    if present.is_empty() {
        return vec![1.0; crate::model::CLASSES];
    }
    let mean = present.iter().sum::<usize>() as f32 / present.len() as f32;
    let mut weights = vec![1.0f32; crate::model::CLASSES];
    for (i, count) in counts.iter().enumerate() {
        if *count > 0 {
            weights[i] = (mean / *count as f32).powf(power);
        }
    }
    // Normalise so the mean weight over the *examples* is 1, not the mean over
    // the classes. Those differ, and using the wrong one would change the
    // effective loss scale.
    let total: f32 = examples.iter().map(|e| weights[e.intent.index()]).sum();
    if total > 0.0 {
        let scale = examples.len() as f32 / total;
        for w in weights.iter_mut() {
            *w *= scale;
        }
    }
    weights
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            // Defaults chosen by measurement on the shipped v1 dataset, not
            // by convention. See docs/model-training.md for the sweep; a
            // larger learning rate or weight decay destabilised training once
            // pooling was corrected.
            seed: 0xA0_40_2D_17,
            epochs: 100,
            batch_size: 16,
            // Selected by measurement on the release-0.2 dataset, not by
            // convention. The learning rate was 3e-3 when the dataset had 641
            // examples; with 1210 it moved to 1e-2, which is worth +3 points
            // of macro-F1 on the locked benchmark. The sweep is in
            // `training/experiments.jsonl` and the method in
            // `training/run_experiments.py`.
            learning_rate: 1e-2,
            weight_decay: 1e-2,
            patience: 20,
            confidence_threshold: 0.5,
            temperature: 1.0,
            // Regularisation was added after measuring train accuracy of
            // exactly 1.0000: the model was memorising 968 examples with
            // 860,160 embedding parameters. All three are off by default
            // until the sweep says otherwise — see the experiment log.
            dropout: 0.0,
            feature_dropout: 0.0,
            label_smoothing: 0.0,
            class_weighting: ClassWeighting::None,
        }
    }
}

/// Per-epoch progress.
#[derive(Debug, Clone, PartialEq)]
pub struct EpochLog {
    pub epoch: u32,
    pub loss: f64,
    pub val_accuracy: f64,
    pub val_macro_f1: f64,
    pub elapsed_ms: u64,
}

/// Everything a training run produced.
#[derive(Debug, Clone)]
pub struct TrainingOutcome {
    /// Weights after the final epoch.
    pub tensors: HashMap<String, Tensor>,
    /// Metrics on the held-out test split, scored with the final weights.
    pub test_metrics: Metrics,
    /// Metrics on the validation split at the best epoch.
    pub best_validation: Metrics,
    pub best_epoch: u32,
    pub log: Vec<EpochLog>,
    pub config: TrainingConfig,
    pub dataset_version: String,
    pub elapsed_ms: u64,
}

impl TrainingOutcome {
    /// The JSON blob stored in the checkpoint manifest.
    pub fn manifest_metrics(&self) -> serde_json::Value {
        serde_json::json!({
            "accuracy": self.test_metrics.accuracy,
            "macro_f1": self.test_metrics.macro_f1,
            "unknown_rate": self.test_metrics.unknown_rate,
            "mean_confidence": self.test_metrics.mean_confidence,
            "test_examples": self.test_metrics.n,
            "val_accuracy": self.best_validation.accuracy,
            "val_macro_f1": self.best_validation.macro_f1,
            "best_epoch": self.best_epoch,
            "epochs_run": self.log.len(),
            "elapsed_ms": self.elapsed_ms,
            "per_intent_recall": self
                .test_metrics
                .per_intent
                .iter()
                .map(|m| (m.intent.as_str().to_string(), m.recall))
                .collect::<std::collections::BTreeMap<_, _>>(),
        })
    }
}

fn to_batch(texts: &[String]) -> FeatureBatch {
    FeatureBatch::from_texts(texts)
}

fn labels_for(intents: &[Intent], device: &Device) -> Result<Tensor> {
    let raw: Vec<u32> = intents.iter().map(|i| i.index() as u32).collect();
    Tensor::new(raw.as_slice(), device)
        .and_then(|t| t.reshape(intents.len()))
        .map_err(|e| Error::Brain(e.to_string()))
}

/// Train the model on a dataset.
/// Train several models and keep the best one **by validation score**.
///
/// # Why
///
/// The measured seed variance of this model is large: on the release-0.2
/// dataset, three seeds spanned 0.669 to 0.754 on the locked benchmark. A
/// single training run is therefore a lottery, and a number reported from one
/// run is partly seed luck.
///
/// # How the selection is kept honest
///
/// The winner is chosen on the **validation** split, which is also what
/// early stopping uses. The held-out test split and the locked benchmark are
/// never consulted. Selecting on validation is a standard model-selection
/// step; selecting on the test split would not be, and the code makes that
/// distinction structural by only ever computing validation metrics here.
pub fn train_best_of(
    dataset: &Dataset,
    config: &TrainingConfig,
    seeds: &[u64],
) -> Result<(TrainingOutcome, Vec<SeedScore>)> {
    if seeds.is_empty() {
        return Err(Error::Brain("no seeds given".into()));
    }
    let mut best: Option<(TrainingOutcome, SeedScore)> = None;
    let mut all = Vec::with_capacity(seeds.len());
    for seed in seeds {
        let mut cfg = config.clone();
        cfg.seed = *seed;
        let outcome = train(dataset, &cfg)?;
        // The selection metric: validation macro-F1, then validation accuracy.
        // Both come from the validation split only.
        let score = SeedScore {
            seed: *seed,
            validation_macro_f1: outcome.best_validation.macro_f1,
            validation_accuracy: outcome.best_validation.accuracy,
            best_epoch: outcome.best_epoch,
            test_accuracy: outcome.test_metrics.accuracy,
            test_macro_f1: outcome.test_metrics.macro_f1,
        };
        let better = match &best {
            None => true,
            Some((_, b)) => {
                (score.validation_macro_f1, score.validation_accuracy)
                    > (b.validation_macro_f1, b.validation_accuracy)
            }
        };
        if better {
            best = Some((outcome, score));
        }
        all.push(score);
    }
    let (outcome, _) = best.expect("at least one seed was trained");
    Ok((outcome, all))
}

/// What one seed produced.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SeedScore {
    pub seed: u64,
    pub validation_macro_f1: f64,
    pub validation_accuracy: f64,
    pub best_epoch: u32,
    /// Recorded for the experiment log only. Never used for selection.
    pub test_accuracy: f64,
    pub test_macro_f1: f64,
}

pub fn train(dataset: &Dataset, config: &TrainingConfig) -> Result<TrainingOutcome> {
    dataset.validate()?;

    if config.batch_size == 0 {
        return Err(Error::Brain("batch_size must be > 0".into()));
    }
    if config.epochs == 0 {
        return Err(Error::Brain("epochs must be > 0".into()));
    }
    if config.learning_rate <= 0.0 || !config.learning_rate.is_finite() {
        return Err(Error::Brain(
            "learning_rate must be positive and finite".into(),
        ));
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    let train_examples: Vec<&crate::dataset::Example> = dataset.split(Split::Train);
    // Per-class loss weights, derived from the training split only.
    let weights = class_weights(&train_examples, config.class_weighting);
    if train_examples.is_empty() {
        return Err(Error::Brain("training split is empty".into()));
    }
    let val_examples: Vec<&crate::dataset::Example> = dataset.split(Split::Validation);
    let test_examples: Vec<&crate::dataset::Example> = dataset.split(Split::Test);
    if val_examples.is_empty() || test_examples.is_empty() {
        return Err(Error::Brain(
            "validation and test splits must both be non-empty; add more examples".into(),
        ));
    }

    let started = std::time::Instant::now();

    // Deterministic initial weights, then a VarMap so AdamW can update them.
    // Build the model against a VarMap first so that every parameter exists as
    // a `Var`, then overwrite the values with the deterministic initialisation.
    // `Var::set` copies into the existing storage in place, so the `Tensor`s the
    // model already holds stay valid and AdamW sees the same variables.
    let mut var_map = candle_nn::VarMap::new();
    let vb = candle_nn::VarBuilder::from_varmap(&var_map, dtype, &device);
    let model =
        IntentModel::load(vb, Heads::IntentOnly).map_err(|e| Error::Brain(e.to_string()))?;

    let initial =
        initial_tensors(config.seed, Heads::IntentOnly).map_err(|e| Error::Brain(e.to_string()))?;
    var_map
        .set(initial.into_iter())
        .map_err(|e| Error::Brain(format!("could not install initial weights: {e}")))?;

    if model.parameter_count() != expected_parameter_count(Heads::IntentOnly) {
        return Err(Error::Brain(format!(
            "model reports {} parameters, architecture declares {}",
            model.parameter_count(),
            expected_parameter_count(Heads::IntentOnly)
        )));
    }

    let vars = var_map.all_vars();
    let mut optimizer = AdamW::new(
        vars,
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )
    .map_err(|e| Error::Brain(e.to_string()))?;

    // Deterministic batch order: reshuffled every epoch from a PRNG stream
    // that depends only on the seed, so two runs visit identical batches.
    let mut order: Vec<usize> = (0..train_examples.len()).collect();
    let mut best_f1 = f64::MIN;
    let mut best_epoch = 0u32;
    let mut best_weights: Option<HashMap<String, Tensor>> = None;
    let mut log: Vec<EpochLog> = Vec::new();
    let mut since_improvement = 0u32;

    for epoch in 1..=config.epochs {
        let epoch_start = std::time::Instant::now();
        let mut rng = SplitMix64::new(config.seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9));
        fisher_yates(&mut order, &mut rng);

        let mut loss_sum = 0.0f64;
        let mut batches = 0usize;
        for chunk in order.chunks(config.batch_size) {
            let texts: Vec<String> = chunk
                .iter()
                .map(|i| train_examples[*i].text.clone())
                .collect();
            let labels: Vec<Intent> = chunk.iter().map(|i| train_examples[*i].intent).collect();

            let batch = to_batch(&texts);
            // Regularisation masks, drawn from this run's own PRNG stream so
            // the whole training run stays reproducible.
            let feature_mask = if config.feature_dropout > 0.0 {
                Some(
                    dropout_mask(
                        &mut rng,
                        (batch.len(), crate::model::SEQ_LEN),
                        config.feature_dropout,
                        false,
                    )
                    .map_err(|e| Error::Brain(e.to_string()))?,
                )
            } else {
                None
            };
            let hidden_mask = if config.dropout > 0.0 {
                Some(
                    dropout_mask(
                        &mut rng,
                        (batch.len(), crate::model::HIDDEN),
                        config.dropout,
                        true,
                    )
                    .map_err(|e| Error::Brain(e.to_string()))?,
                )
            } else {
                None
            };

            let logits = model
                .forward_with(&batch, feature_mask.as_ref(), hidden_mask.as_ref())
                .map_err(|e| Error::Brain(e.to_string()))?;
            // The weights are per *class*; the loss needs them per *example*.
            let example_weights: Option<Vec<f32>> = if weights.iter().any(|w| *w != 1.0) {
                Some(
                    labels
                        .iter()
                        .map(|intent| weights[intent.index()])
                        .collect(),
                )
            } else {
                None
            };
            let loss = smoothed_cross_entropy(
                &logits,
                &labels,
                config.label_smoothing,
                example_weights.as_deref(),
            )
            .map_err(|e| Error::Brain(e.to_string()))?;
            let loss_value = loss
                .to_scalar::<f32>()
                .map_err(|e| Error::Brain(e.to_string()))?;
            loss_sum += loss_value as f64;
            batches += 1;

            let grads = loss.backward().map_err(|e| Error::Brain(e.to_string()))?;
            optimizer
                .step(&grads)
                .map_err(|e| Error::Brain(e.to_string()))?;
        }

        let val = score(&model, &val_examples, config);
        log.push(EpochLog {
            epoch,
            loss: if batches == 0 {
                0.0
            } else {
                loss_sum / batches as f64
            },
            val_accuracy: val.accuracy,
            val_macro_f1: val.macro_f1,
            elapsed_ms: epoch_start.elapsed().as_millis() as u64,
        });

        if val.macro_f1 > best_f1 + 1e-6 {
            best_f1 = val.macro_f1;
            best_epoch = epoch;
            best_weights = Some(snapshot(&var_map)?);
            since_improvement = 0;
        } else {
            since_improvement += 1;
            if config.patience > 0 && since_improvement >= config.patience {
                tracing::info!(epoch, "early stopping: validation macro-F1 plateaued");
                break;
            }
        }
    }

    let final_weights = match best_weights {
        Some(w) => w,
        // Every epoch failed to improve on the initial -inf sentinel only when
        // there was exactly one epoch; fall back to the live weights.
        None => snapshot(&var_map)?,
    };

    let final_model = IntentModel::from_tensors(final_weights.clone(), &device, Heads::IntentOnly)
        .map_err(|e| Error::Brain(e.to_string()))?;
    let test_metrics = score(&final_model, &test_examples, config);
    let best_validation = score(&final_model, &val_examples, config);

    Ok(TrainingOutcome {
        tensors: final_weights,
        test_metrics,
        best_validation,
        best_epoch,
        log,
        config: config.clone(),
        dataset_version: dataset.version.clone(),
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn snapshot(var_map: &candle_nn::VarMap) -> Result<HashMap<String, Tensor>> {
    let mut out = HashMap::new();
    for spec in crate::model::parameter_specs(Heads::IntentOnly) {
        let tensor = var_map
            .get(
                spec.shape.clone(),
                &spec.name,
                candle_nn::Init::Const(0.0),
                DType::F32,
                &Device::Cpu,
            )
            .map_err(|e| Error::Brain(format!("missing var `{}`: {e}", spec.name)))?;
        out.insert(spec.name.clone(), tensor);
    }
    Ok(out)
}

fn score(
    model: &IntentModel,
    examples: &[&crate::dataset::Example],
    config: &TrainingConfig,
) -> Metrics {
    // Score every example once, then walk the precomputed predictions in the
    // same order `evaluate` iterates. Doing the forward passes up front keeps
    // this O(n) instead of re-classifying inside the scoring closure.
    let preds: Vec<(Intent, f64)> = examples
        .iter()
        .map(|e| classify_one(model, &e.text, config))
        .collect();
    let pairs: Vec<(String, Intent)> = examples
        .iter()
        .map(|e| (e.text.clone(), e.intent))
        .collect();
    let cursor = std::cell::Cell::new(0usize);
    evaluate(
        &pairs,
        |_| {
            let i = cursor.get();
            cursor.set(i + 1);
            preds.get(i).copied().unwrap_or((Intent::Unknown, 0.0))
        },
        Intent::Unknown,
    )
}

/// Run one utterance through the model.
pub fn classify_one(model: &IntentModel, text: &str, config: &TrainingConfig) -> (Intent, f64) {
    let batch = FeatureBatch::from_texts(&[text.to_string()]);
    match model.forward(&batch) {
        Ok(logits) => {
            let row: Vec<f32> = match logits.flatten_all().and_then(|t| t.to_vec1::<f32>()) {
                Ok(v) => v,
                Err(_) => return (Intent::Unknown, 0.0),
            };
            logits_to_intent(&row, config.temperature, config.confidence_threshold)
        }
        Err(_) => (Intent::Unknown, 0.0),
    }
}

/// Cross-entropy with optional label smoothing.
///
/// With label smoothing `eps`, the target for the true class is
/// `1 - eps + eps / K` and every other class gets `eps / K`. This stops the
/// model from driving the true class's probability to 1.0, which is exactly
/// the behaviour that produced a 0.99 confidence on every prediction,
/// including the wrong ones.
fn smoothed_cross_entropy(
    logits: &candle_core::Tensor,
    intents: &[Intent],
    eps: f64,
    weights: Option<&[f32]>,
) -> candle_core::Result<candle_core::Tensor> {
    let device = logits.device();
    let index_tensor =
        labels_for(intents, device).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    // The unweighted, un-smoothed case is candle's own loss: it is the
    // baseline every other variant is compared against, so it is used
    // verbatim rather than re-implemented.
    if eps <= 0.0 && weights.is_none() {
        return candle_nn::loss::cross_entropy(logits, &index_tensor);
    }
    // With weights but no smoothing, the per-example loss is plain
    // cross-entropy; it is computed here so the two features compose.
    let plain_only = eps <= 0.0;
    let classes = logits.dim(1)?;
    let eps = eps.clamp(0.0, 0.99);
    let _ = &index_tensor;
    // Softmax over the **class** axis, which is axis 1 for a [B, K] tensor.
    // Axis 0 is the batch: normalising over it mixes examples together and
    // produces a loss that looks plausible and trains nothing.
    let log_probs = candle_nn::ops::log_softmax(logits, 1)?;

    // -log p(true class), computed from an explicit one-hot rather than a
    // gather: candle's gather is implemented with a scatter whose shape
    // rules are stricter than the documentation suggests, and building the
    // one-hot in Rust is both simple and exact.
    let rows = intents.len();
    let mut one_hot = vec![0.0f32; rows * classes];
    for (row, intent) in intents.iter().enumerate() {
        one_hot[row * classes + intent.index()] = 1.0;
    }
    let one_hot = candle_core::Tensor::new(one_hot.as_slice(), device)?.reshape((rows, classes))?;
    // [B, K] * [B, K] -> [B, 1]: the log-probability of the true class.
    let picked = log_probs.neg()?.broadcast_mul(&one_hot)?.sum(1)?;

    let per_example = if plain_only {
        // No smoothing: the loss is plain cross-entropy.
        picked
    } else {
        // Mean over classes of -log p(k): the uniform part of the smoothed
        // target.
        let mean_neg_log_lik = log_probs.neg()?.mean(1)?;
        let scale = candle_core::Tensor::new(&[(1.0 - eps) as f32], device)?;
        let mix = candle_core::Tensor::new(&[(eps / classes as f64) as f32], device)?;
        let scaled = picked.broadcast_mul(&scale)?;
        let mixed = mean_neg_log_lik.broadcast_mul(&mix)?;
        scaled.add(&mixed)?
    };

    // Per-example class weights, applied to the per-example loss and then
    // averaged. The weights are already normalised so the mean weight is 1,
    // so this changes *which* examples count for more, not the loss scale.
    match weights {
        Some(w) => {
            let weight_tensor = candle_core::Tensor::new(w, device)?;
            per_example.broadcast_mul(&weight_tensor)?.mean(0)
        }
        None => per_example.mean(0),
    }
}

/// A deterministic dropout mask of the given shape.
///
/// `inverted` selects the convention: for the hidden layer the mask keeps a
/// unit and scales it by `1 / (1 - p)`; for the input features it keeps a
/// unit and drops to zero.
fn dropout_mask(
    rng: &mut SplitMix64,
    (rows, cols): (usize, usize),
    p: f64,
    inverted: bool,
) -> candle_core::Result<candle_core::Tensor> {
    // Everything here is f32: the model is f32 throughout, and an f64 mask
    // fails the multiply with a dtype mismatch.
    let p = p.clamp(0.0, 0.99) as f32;
    let keep = 1.0f32 / (1.0f32 - p);
    let mut values: Vec<f32> = Vec::with_capacity(rows * cols);
    for _ in 0..rows * cols {
        // 24 bits of mantissa is plenty for a Bernoulli draw and keeps the
        // stream cheap.
        let u = (rng.next_u64() >> 40) as f32 / (1u32 << 24) as f32;
        let dropped = u < p;
        let value = if inverted {
            if dropped {
                0.0f32
            } else {
                keep
            }
        } else if dropped {
            0.0f32
        } else {
            1.0f32
        };
        values.push(value);
    }
    candle_core::Tensor::new(values.as_slice(), &candle_core::Device::Cpu)?.reshape((rows, cols))
}

/// Deterministic in-place Fisher-Yates shuffle.
fn fisher_yates(items: &mut [usize], rng: &mut SplitMix64) {
    for i in (1..items.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Example;

    fn toy_dataset() -> Dataset {
        // Two intents, a handful of examples each. Enough to prove the loop
        // runs, decreases loss and produces finite weights; real accuracy is
        // measured by the full dataset in `aaron learning evaluate`.
        let mut examples = Vec::new();
        for i in 0..12 {
            examples.push(Example {
                id: format!("status-train-{i}"),
                text: format!("check the system status please number {i}"),
                intent: Intent::SystemStatus,
                source: "test".into(),
                notes: String::new(),
                split: None,
            });
            examples.push(Example {
                id: format!("help-test-{i}"),
                text: format!("tell me what you can do number {i}"),
                intent: Intent::Help,
                source: "test".into(),
                notes: String::new(),
                split: None,
            });
        }
        Dataset::new("toy", examples).unwrap()
    }

    fn fast_config() -> TrainingConfig {
        TrainingConfig {
            epochs: 3,
            batch_size: 4,
            learning_rate: 5e-3,
            patience: 0,
            ..Default::default()
        }
    }

    #[test]
    fn training_runs_and_produces_finite_weights() {
        let d = toy_dataset();
        let outcome = train(&d, &fast_config()).unwrap();
        assert!(!outcome.log.is_empty());
        assert!(outcome.test_metrics.n > 0);
        for t in outcome.tensors.values() {
            let vals: Vec<f32> = t.flatten_all().unwrap().to_vec1().unwrap();
            assert!(
                vals.iter().all(|v| v.is_finite()),
                "a weight became non-finite"
            );
        }
    }

    #[test]
    fn training_is_deterministic() {
        let d = toy_dataset();
        let a = train(&d, &fast_config()).unwrap();
        let b = train(&d, &fast_config()).unwrap();
        assert_eq!(a.test_metrics.accuracy, b.test_metrics.accuracy);
        assert_eq!(a.test_metrics.macro_f1, b.test_metrics.macro_f1);
        for (name, ta) in &a.tensors {
            let tb = b.tensors.get(name).unwrap();
            let va: Vec<f32> = ta.flatten_all().unwrap().to_vec1().unwrap();
            let vb: Vec<f32> = tb.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(va, vb, "tensor `{name}` is not reproducible");
        }
    }

    #[test]
    fn different_seeds_produce_different_weights() {
        let d = toy_dataset();
        let mut cfg_a = fast_config();
        cfg_a.seed = 1;
        let mut cfg_b = fast_config();
        cfg_b.seed = 2;
        let a = train(&d, &cfg_a).unwrap();
        let b = train(&d, &cfg_b).unwrap();
        let va: Vec<f32> = a.tensors["fc1.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let vb: Vec<f32> = b.tensors["fc1.weight"]
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert_ne!(va, vb);
    }

    #[test]
    fn loss_decreases_over_a_short_run() {
        let d = toy_dataset();
        let cfg = TrainingConfig {
            epochs: 12,
            batch_size: 4,
            learning_rate: 1e-2,
            patience: 0,
            ..Default::default()
        };
        let outcome = train(&d, &cfg).unwrap();
        let first = outcome.log.first().unwrap().loss;
        let last = outcome.log.last().unwrap().loss;
        assert!(last < first, "loss did not decrease: {first} -> {last}");
    }

    #[test]
    fn rejects_impossible_hyperparameters() {
        let d = toy_dataset();
        let mut cfg = fast_config();
        cfg.batch_size = 0;
        assert!(train(&d, &cfg).is_err());
        let mut cfg = fast_config();
        cfg.epochs = 0;
        assert!(train(&d, &cfg).is_err());
        let mut cfg = fast_config();
        cfg.learning_rate = -1.0;
        assert!(train(&d, &cfg).is_err());
    }

    #[test]
    fn refuses_a_dataset_with_leakage() {
        let mut d = toy_dataset();
        d.examples[1].text = d.examples[0].text.clone();
        let err = train(&d, &fast_config()).unwrap_err();
        assert!(err.to_string().contains("duplicate utterance"));
    }

    #[test]
    fn manifest_metrics_are_json_serialisable() {
        let d = toy_dataset();
        let outcome = train(&d, &fast_config()).unwrap();
        let v = outcome.manifest_metrics();
        assert!(v["accuracy"].is_number());
        assert!(v["per_intent_recall"].is_object());
        let _ = serde_json::to_string(&v).unwrap();
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut items: Vec<usize> = (0..32).collect();
        let mut rng = SplitMix64::new(5);
        fisher_yates(&mut items, &mut rng);
        assert_ne!(items, (0..32).collect::<Vec<_>>());
        items.sort();
        assert_eq!(items, (0..32).collect::<Vec<_>>());
    }
}
