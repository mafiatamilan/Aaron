//! The readiness head: its data, its training, and its metrics.
//!
//! # What the head is
//!
//! A `Linear(2D -> 1)` on the model's **pooled** representation — the
//! 168-dimensional mean-and-RMS vector, computed before `ln1` — producing one
//! logit. 168 weights and one bias: 169 parameters.
//!
//! It answers one question: *is this request specific enough to act on?* The
//! verdict is binary. `NotApplicable` is not the head's business; it is
//! derived from the intent, because "the utterance requests no action" is a
//! fact about the intent, not about the argument.
//!
//! # What the head is measured against
//!
//! The deterministic rule (`aaron_core::readiness`), which reads the extracted
//! argument. The rule is exact where the extractor is right. The head's
//! whole opportunity is the cases the rule misses:
//!
//! * the argument is the verb itself ("read" → the extractor hands back
//!   "read");
//! * the argument is a determiner plus a quantity ("the first 20 lines");
//! * the argument is a category noun the rule's closed list does not contain
//!   ("the podcast", "the plugin").
//!
//! A third of those are caught by extending the closed list, which costs
//! nothing. The head has to be better than *that*, not merely better than
//! doing nothing.
//!
//! # The training data
//!
//! `training/readiness/corpus-v1.jsonl`: contrastive pairs of incomplete and
//! complete utterances, with ground-truth labels, disjoint from the intent
//! dataset and from every held-out evaluation set.
//!
//! # The two-stage design
//!
//! The head is trained on a **frozen trunk**:
//!
//! 1. the intent classifier is trained normally, producing a checkpoint;
//! 2. the readiness head is then trained on top of it, with every other
//!    parameter frozen.
//!
//! That is a *probe*, and it is the cheapest possible answer to the design
//! question. It also has a property the end-to-end variant does not: the
//! intent classifier is provably unchanged, so any difference in the intent
//! metrics is zero by construction, and the comparison is clean.
//!
//! A probe that fails is a lower bound on what the head can do, not a proof:
//! a jointly trained trunk could reorganise to make the classes linearly
//! separable. It is a diagnostic, not a gate.

use std::collections::HashMap;
use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer, ParamsAdamW};
use candle_nn::{VarBuilder, VarMap};
use serde::{Deserialize, Serialize};

use aaron_core::error::{Error, Result};
use aaron_core::intent::{Intent, Readiness};

use crate::model::{initial_tensors, Heads, IntentModel, SplitMix64};

/// One labelled readiness example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReadinessExample {
    /// Stable id.
    pub id: String,
    /// The utterance.
    pub text: String,
    /// The intent it carries, used to extract the argument.
    pub intent: Intent,
    /// The ground truth: is the request specific enough to act on?
    pub readiness: Readiness,
    /// The shape of the utterance, for per-family reporting.
    #[serde(default)]
    pub family: String,
    /// Which split this row belongs to.
    #[serde(default = "default_split")]
    pub split: Split,
}

/// The default split for a row that does not declare one.
fn default_split() -> Split {
    Split::Train
}

/// The split a readiness row belongs to.
///
/// The readiness corpus is split 70/15/15 rather than the intent dataset's
/// 80/10/10: it is small, and a 10% hold-out would be too thin to measure a
/// rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}

impl Split {
    /// A short identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Split::Train => "train",
            Split::Validation => "validation",
            Split::Test => "test",
        }
    }
}

/// The readiness corpus.
#[derive(Debug, Clone, Default)]
pub struct ReadinessCorpus {
    pub examples: Vec<ReadinessExample>,
}

impl ReadinessCorpus {
    /// Load a corpus from a JSONL file.
    pub fn load(path: impl AsRef<Path>) -> Result<ReadinessCorpus> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
        let mut examples = Vec::new();
        for (number, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let example: ReadinessExample = serde_json::from_str(line).map_err(|e| {
                Error::Other(format!(
                    "{}:{}: invalid readiness example: {e}",
                    path.display(),
                    number + 1
                ))
            })?;
            examples.push(example);
        }
        Ok(ReadinessCorpus { examples })
    }

    /// How many rows the corpus has.
    pub fn len(&self) -> usize {
        self.examples.len()
    }

    /// Is the corpus empty?
    pub fn is_empty(&self) -> bool {
        self.examples.is_empty()
    }

    /// The rows in one split.
    pub fn split(&self, split: Split) -> Vec<&ReadinessExample> {
        self.examples.iter().filter(|e| e.split == split).collect()
    }

    /// The label the head is trained to predict: 1 for
    /// `NeedsClarification`, 0 for everything else.
    ///
    /// `NotApplicable` and `Ready` are both "the head has nothing to say":
    /// the request either names what it needs or requests no action.
    fn target(example: &ReadinessExample) -> f32 {
        if example.readiness == Readiness::NeedsClarification {
            1.0
        } else {
            0.0
        }
    }
}

/// How the readiness head is trained.
#[derive(Debug, Clone, PartialEq)]
pub struct HeadTrainingConfig {
    pub seed: u64,
    pub epochs: u32,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub weight_decay: f64,
    /// Stop early when the validation loss has not improved for this many
    /// epochs.
    pub patience: u32,
    /// The decision threshold, selected on the corpus's training split.
    pub threshold: f32,
}

impl Default for HeadTrainingConfig {
    fn default() -> Self {
        HeadTrainingConfig {
            seed: 7,
            epochs: 200,
            batch_size: 32,
            learning_rate: 1e-2,
            weight_decay: 1e-2,
            patience: 30,
            threshold: 0.5,
        }
    }
}

/// The confusion of the readiness head, over the two classes the head
/// actually distinguishes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReadinessConfusion {
    /// Incomplete, called incomplete. Correct.
    pub true_incomplete: usize,
    /// Incomplete, called complete. A false-ready.
    pub false_ready: usize,
    /// Complete, called incomplete. A false-not-ready.
    pub false_not_ready: usize,
    /// Complete, called complete. Correct.
    pub true_complete: usize,
}

impl ReadinessConfusion {
    /// Total rows.
    pub fn total(&self) -> usize {
        self.true_incomplete + self.false_ready + self.false_not_ready + self.true_complete
    }

    /// How many rows are actually incomplete.
    pub fn incomplete(&self) -> usize {
        self.true_incomplete + self.false_ready
    }

    /// How many rows are actually complete.
    pub fn complete(&self) -> usize {
        self.false_not_ready + self.true_complete
    }

    /// The fraction of incomplete rows the head called complete.
    ///
    /// `None` when there are no incomplete rows: a rate over an empty
    /// denominator is undefined, not zero.
    pub fn false_ready_rate(&self) -> Option<f64> {
        if self.incomplete() == 0 {
            return None;
        }
        Some(self.false_ready as f64 / self.incomplete() as f64)
    }

    /// The fraction of complete rows the head called incomplete.
    pub fn false_not_ready_rate(&self) -> Option<f64> {
        if self.complete() == 0 {
            return None;
        }
        Some(self.false_not_ready as f64 / self.complete() as f64)
    }

    /// Precision: of the rows called incomplete, how many were.
    pub fn precision(&self) -> Option<f64> {
        let called = self.true_incomplete + self.false_not_ready;
        if called == 0 {
            return None;
        }
        Some(self.true_incomplete as f64 / called as f64)
    }

    /// Recall: of the incomplete rows, how many were called incomplete.
    pub fn recall(&self) -> Option<f64> {
        if self.incomplete() == 0 {
            return None;
        }
        Some(self.true_incomplete as f64 / self.incomplete() as f64)
    }

    /// The F1 score, or `None` when it is undefined.
    pub fn f1(&self) -> Option<f64> {
        let precision = self.precision()?;
        let recall = self.recall()?;
        if precision + recall == 0.0 {
            return Some(0.0);
        }
        Some(2.0 * precision * recall / (precision + recall))
    }
}

/// The head's performance on one split.
#[derive(Debug, Clone, PartialEq)]
pub struct ReadinessMetrics {
    pub split: Split,
    pub confusion: ReadinessConfusion,
    /// The threshold the verdict used.
    pub threshold: f32,
    /// The head's confusion at every threshold from 0.05 to 0.95, computed
    /// from the same model and the same probabilities as [`Self::confusion`].
    ///
    /// The sweep lives here rather than in a caller because a caller that
    /// rebuilds the model to sweep it can disagree with the metrics — and
    /// when two pieces of code report two different numbers for one
    /// measurement, at least one of them is wrong and there is no way to
    /// tell which. There is one number now.
    pub sweep: Vec<(f32, ReadinessConfusion)>,
    /// The same confusion, computed from the deterministic rule instead of
    /// the head, on exactly the same rows.
    ///
    /// This is the comparison that decides the release: the head has to be
    /// better than the rule on the rule's *own* task, measured on the same
    /// rows, not better than nothing.
    pub rule_confusion: ReadinessConfusion,
    /// The head's loss on this split.
    pub loss: f64,
}

/// The outcome of training the readiness head.
#[derive(Debug, Clone)]
pub struct HeadTrainingOutcome {
    /// The trained head's tensors, ready to be merged into a checkpoint.
    pub tensors: HashMap<String, Tensor>,
    /// The head's metrics on each split.
    pub metrics: Vec<ReadinessMetrics>,
    /// The threshold the head was selected at.
    pub threshold: f32,
    /// The epoch with the best validation loss.
    pub best_epoch: u32,
    /// Per-epoch losses.
    pub epoch_losses: Vec<(u32, f64)>,
}

/// Add the readiness head's freshly initialised tensors to a base
/// checkpoint's tensors.
///
/// The base checkpoint was trained without the head, so it has no
/// `fc_ready` tensors. The head starts from the same deterministic
/// initialisation every run would use, seeded from the training seed, so a
/// probe is reproducible.
pub fn head_tensors(seed: u64) -> Result<HashMap<String, Tensor>> {
    let all =
        initial_tensors(seed, Heads::WithReadiness).map_err(|e| Error::Brain(e.to_string()))?;
    let mut head = HashMap::new();
    for (name, tensor) in all {
        if name.starts_with("fc_ready") {
            head.insert(name, tensor);
        }
    }
    Ok(head)
}

/// Score one batch of examples with the head: the probability that the
/// request is under-specified.
///
/// The head produces a **logit** and the threshold is on the **probability**.
/// Returning the logit here and comparing it against a probability threshold
/// is a bug that does not look like one: a logit of 0.30 is a probability of
/// 0.57, so every threshold below 0.5 fires on almost everything, and the
/// numbers it produces look plausible while describing a different model.
///
/// The conversion is written out here, rather than left to the caller, so
/// that there is exactly one place where a logit becomes a probability.
fn head_probabilities(model: &IntentModel, texts: &[String]) -> Result<Vec<f32>> {
    let batch = crate::model::FeatureBatch::from_texts(texts);
    let (_, ready) = model
        .forward_with_readiness(&batch, None, None)
        .map_err(|e| Error::Brain(e.to_string()))?;
    let ready = match ready {
        Some(t) => t,
        None => return Err(Error::Brain("the model has no readiness head".into())),
    };
    let logits = ready
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| Error::Brain(e.to_string()))?;
    Ok(logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect())
}

/// The loss of the head on a set of examples.
fn head_loss(model: &IntentModel, examples: &[&ReadinessExample]) -> Result<f64> {
    if examples.is_empty() {
        return Ok(0.0);
    }
    let texts: Vec<String> = examples.iter().map(|e| e.text.clone()).collect();
    let batch = crate::model::FeatureBatch::from_texts(&texts);
    let (_, ready) = model
        .forward_with_readiness(&batch, None, None)
        .map_err(|e| Error::Brain(e.to_string()))?;
    let ready = match ready {
        Some(t) => t,
        None => return Err(Error::Brain("the model has no readiness head".into())),
    };
    let targets: Vec<f32> = examples
        .iter()
        .map(|e| ReadinessCorpus::target(e))
        .collect();
    // The loss is elementwise, so the target must have the same shape as
    // the logit: `[B, 1]`, not `[B]`.
    let target = Tensor::new(targets.as_slice(), &Device::Cpu)
        .map_err(|e| Error::Brain(e.to_string()))?
        .reshape((targets.len(), 1))
        .map_err(|e| Error::Brain(e.to_string()))?;
    let loss = candle_nn::loss::binary_cross_entropy_with_logit(&ready, &target)
        .map_err(|e| Error::Brain(e.to_string()))?;
    let value = loss
        .to_scalar::<f32>()
        .map_err(|e| Error::Brain(e.to_string()))?;
    Ok(value as f64)
}

/// The confusion matrix of a set of verdicts against the ground truth.
fn confusion<F>(examples: &[&ReadinessExample], mut verdict: F) -> ReadinessConfusion
where
    F: FnMut(&ReadinessExample) -> bool,
{
    let mut out = ReadinessConfusion::default();
    for example in examples {
        let called_incomplete = verdict(example);
        match (
            example.readiness == Readiness::NeedsClarification,
            called_incomplete,
        ) {
            (true, true) => out.true_incomplete += 1,
            (true, false) => out.false_ready += 1,
            (false, true) => out.false_not_ready += 1,
            (false, false) => out.true_complete += 1,
        }
    }
    out
}

/// The deterministic rule's verdict on a corpus row.
///
/// The rule is computed with the row's *own* intent, because that is what
/// the corpus records and what the agent would use at runtime.
fn rule_verdict(example: &ReadinessExample) -> bool {
    let entities = aaron_core::text::extract_entities(example.intent, &example.text);
    aaron_core::readiness::readiness_of(example.intent, &entities) == Readiness::NeedsClarification
}

/// Choose the decision threshold on the corpus's **training** split.
///
/// The threshold is a threshold on a single uncalibrated logit, so it has to
/// be picked somewhere. Picking it on the split the metrics are reported on
/// would be selection on the test set, and the resulting F1 would be
/// optimistic — the same mistake the release-0.4 threshold sweep made, which
/// is why the sweep now runs on the validation split only.
pub fn select_threshold(model: &IntentModel, examples: &[&ReadinessExample]) -> Result<f32> {
    if examples.is_empty() {
        return Ok(0.5);
    }
    let texts: Vec<String> = examples.iter().map(|e| e.text.clone()).collect();
    let probabilities = head_probabilities(model, &texts)?;
    let mut best = (f64::NEG_INFINITY, 0.5f32);
    // A hundredth of a step is finer than the head's output resolution.
    for step in 1..=99 {
        let threshold = step as f32 / 100.0;
        let confusion = confusion(examples, |example| {
            let index = examples
                .iter()
                .position(|r| r.id == example.id)
                .unwrap_or_default();
            probabilities
                .get(index)
                .map(|p| *p >= threshold)
                .unwrap_or(false)
        });
        let score = confusion.f1().unwrap_or(0.0);
        if score > best.0 {
            best = (score, threshold);
        }
    }
    Ok(best.1)
}

/// Train the readiness head on a frozen trunk.
///
/// The base tensors are the trained intent classifier's. Every parameter
/// except the head's is frozen: the optimiser holds only `fc_ready`, and the
/// intent classifier's outputs are therefore bit-identical before and after.
pub fn train_head(
    base_tensors: HashMap<String, Tensor>,
    corpus: &ReadinessCorpus,
    config: &HeadTrainingConfig,
) -> Result<HeadTrainingOutcome> {
    if config.batch_size == 0 {
        return Err(Error::Brain("batch_size must be > 0".into()));
    }
    if config.epochs == 0 {
        return Err(Error::Brain("epochs must be > 0".into()));
    }
    let train = corpus.split(Split::Train);
    let validation = corpus.split(Split::Validation);
    let _test = corpus.split(Split::Test);
    if train.is_empty() {
        return Err(Error::Brain(
            "the readiness corpus has no training rows".into(),
        ));
    }

    let device = Device::Cpu;
    let dtype = DType::F32;

    // The base tensors plus a freshly initialised head.
    let mut tensors = base_tensors.clone();
    for (name, tensor) in head_tensors(config.seed)? {
        tensors.insert(name, tensor);
    }

    // A VarMap so the head's parameters are `Var`s the optimiser can see.
    // The base tensors are installed with `set`, which copies into the
    // existing storage, so the model keeps pointing at the same variables.
    let mut var_map = VarMap::new();
    let vb = VarBuilder::from_varmap(&var_map, dtype, &device);
    let model =
        IntentModel::load(vb, Heads::WithReadiness).map_err(|e| Error::Brain(e.to_string()))?;
    var_map
        .set(tensors.clone().into_iter())
        .map_err(|e| Error::Brain(format!("could not install the base weights: {e}")))?;

    // Only the head is trained.
    let head_vars = head_parameters(&var_map)?;
    if head_vars.is_empty() {
        return Err(Error::Brain("the readiness head has no parameters".into()));
    }
    let mut optimizer = AdamW::new(
        head_vars.into_iter().map(|(_, var)| var).collect(),
        ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..Default::default()
        },
    )
    .map_err(|e| Error::Brain(e.to_string()))?;

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut best_loss = f64::INFINITY;
    let mut best_epoch = 0u32;
    let mut best_head: Option<HashMap<String, Tensor>> = None;
    let mut epoch_losses: Vec<(u32, f64)> = Vec::new();
    let mut since_improvement = 0u32;

    for epoch in 1..=config.epochs {
        let mut rng = SplitMix64::new(config.seed ^ (epoch as u64).wrapping_mul(0x9E37_79B9));
        fisher_yates(&mut order, &mut rng);

        let mut loss_sum = 0.0f64;
        let mut batches = 0usize;
        for chunk in order.chunks(config.batch_size) {
            let texts: Vec<String> = chunk.iter().map(|i| train[*i].text.clone()).collect();
            let targets: Vec<f32> = chunk
                .iter()
                .map(|i| ReadinessCorpus::target(train[*i]))
                .collect();
            let batch = crate::model::FeatureBatch::from_texts(&texts);
            let (_, ready) = model
                .forward_with_readiness(&batch, None, None)
                .map_err(|e| Error::Brain(e.to_string()))?;
            let ready = match ready {
                Some(t) => t,
                None => return Err(Error::Brain("the model has no readiness head".into())),
            };
            let target = Tensor::new(targets.as_slice(), &device)
                .map_err(|e| Error::Brain(e.to_string()))?
                .reshape((targets.len(), 1))
                .map_err(|e| Error::Brain(e.to_string()))?;
            let loss = candle_nn::loss::binary_cross_entropy_with_logit(&ready, &target)
                .map_err(|e| Error::Brain(e.to_string()))?;
            let grads = loss.backward().map_err(|e| Error::Brain(e.to_string()))?;
            optimizer
                .step(&grads)
                .map_err(|e| Error::Brain(e.to_string()))?;
            loss_sum += loss
                .to_scalar::<f32>()
                .map_err(|e| Error::Brain(e.to_string()))? as f64;
            batches += 1;
        }

        let epoch_loss = if batches == 0 {
            0.0
        } else {
            loss_sum / batches as f64
        };
        epoch_losses.push((epoch, epoch_loss));

        // The selection metric is the validation *loss*, not the
        // validation F1: the threshold is chosen separately, and tuning a
        // threshold on the metric it is scored with would be selection on the
        // test set.
        let val_loss = head_loss(&model, &validation)?;
        if val_loss < best_loss {
            best_loss = val_loss;
            best_epoch = epoch;
            since_improvement = 0;
            let mut head = HashMap::new();
            for (name, var) in head_parameters(&var_map)? {
                head.insert(name, copy_tensor(var.as_tensor())?);
            }
            best_head = Some(head);
        } else {
            since_improvement += 1;
            if since_improvement >= config.patience {
                break;
            }
        }
    }

    // The best head, as a copy.
    //
    // It has to be a copy: a variable's tensor shares its storage, so a
    // snapshot taken at the best epoch would keep changing as training
    // continued, and the "best" weights would be the last ones.
    let head = best_head
        .ok_or_else(|| Error::Brain("the readiness head produced no usable weights".into()))?;

    // The threshold is chosen on the training split, and the metrics are
    // reported on the held-out splits. Selecting it anywhere else would make
    // the reported F1 optimistic.
    // The final model is the base trunk with the best head installed, built
    // from tensors so it is a plain constant model: it is only used for
    // scoring.
    let final_tensors = merge_head(&base_tensors, &head);
    let final_model =
        IntentModel::from_tensors(final_tensors.clone(), &device, Heads::WithReadiness)
            .map_err(|e| Error::Brain(e.to_string()))?;
    let model = &final_model;

    // The threshold is selected on the training split, *with the head that
    // will actually be shipped*, and the metrics are reported on the
    // held-out splits. Selecting it anywhere else would make the reported
    // F1 optimistic.
    let threshold = select_threshold(model, &train)?;

    let metrics = [Split::Validation, Split::Test]
        .iter()
        .map(|split| {
            let rows = corpus.split(*split);
            let texts: Vec<String> = rows.iter().map(|e| e.text.clone()).collect();
            let probabilities = if rows.is_empty() {
                Vec::new()
            } else {
                head_probabilities(model, &texts)?
            };
            let head_confusion = confusion(&rows, |example| {
                let index = rows
                    .iter()
                    .position(|r| r.id == example.id)
                    .unwrap_or_default();
                probabilities
                    .get(index)
                    .map(|p| *p >= threshold)
                    .unwrap_or(false)
            });
            // The sweep, from the same probabilities.
            let sweep = (1..=19)
                .map(|step| {
                    let at = step as f32 / 20.0;
                    let c = confusion(&rows, |example| {
                        let index = rows
                            .iter()
                            .position(|r| r.id == example.id)
                            .unwrap_or_default();
                        probabilities.get(index).map(|p| *p >= at).unwrap_or(false)
                    });
                    (at, c)
                })
                .collect();
            let rule_confusion = confusion(&rows, rule_verdict);
            let loss = head_loss(model, &rows)?;
            Ok(ReadinessMetrics {
                split: *split,
                confusion: head_confusion,
                threshold,
                sweep,
                rule_confusion,
                loss,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(HeadTrainingOutcome {
        tensors: final_tensors,
        metrics,
        threshold,
        best_epoch,
        epoch_losses,
    })
}

/// Copy a tensor's contents into a new tensor.
///
/// A `Tensor` clone shares its storage, so cloning a variable's tensor and
/// keeping it does not snapshot anything: it keeps a window onto a moving
/// value.
fn copy_tensor(tensor: &Tensor) -> Result<Tensor> {
    let values = tensor
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| Error::Brain(e.to_string()))?;
    Tensor::new(values.as_slice(), tensor.device())
        .map_err(|e| Error::Brain(e.to_string()))?
        .reshape(tensor.shape())
        .map_err(|e| Error::Brain(e.to_string()))
}

/// The readiness head's parameters, by name.
///
/// The variable map holds the whole model; the head is the part whose name
/// starts with `fc_ready`. Only these are handed to the optimiser, which is
/// what "frozen trunk" means: the intent classifier's weights are in the
/// map, but the optimiser never sees them, so they cannot change.
fn head_parameters(var_map: &VarMap) -> Result<Vec<(String, candle_core::Var)>> {
    let data = var_map
        .data()
        .lock()
        .map_err(|e| Error::Brain(e.to_string()))?;
    let mut out: Vec<(String, candle_core::Var)> = data
        .iter()
        .filter(|(name, _)| name.starts_with("fc_ready"))
        .map(|(name, var)| (name.clone(), var.clone()))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Shuffle a slice, in place, with a deterministic PRNG.
fn fisher_yates<T>(items: &mut [T], rng: &mut SplitMix64) {
    if items.len() < 2 {
        return;
    }
    for i in (1..items.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        items.swap(i, j);
    }
}

/// Merge the trained head into a base checkpoint's tensors.
pub fn merge_head(
    base: &HashMap<String, Tensor>,
    head: &HashMap<String, Tensor>,
) -> HashMap<String, Tensor> {
    let mut out = base.clone();
    for (name, tensor) in head {
        out.insert(name.clone(), tensor.clone());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_logit_becomes_a_probability() {
        // The head's output is a logit and the threshold is on the
        // probability. Release 0.5 measured the corpus with the logit
        // compared against a probability threshold, and the error was silent:
        // the numbers looked plausible and described a different model.
        //
        // A logit of zero is a probability of one half, and the conversion
        // has to be monotone and bounded.
        for (logit, expected) in [
            (0.0f32, 0.5f32),
            (-1.0, 1.0 / (1.0 + std::f32::consts::E)),
            (1.0, 1.0 / (1.0 + 1.0 / std::f32::consts::E)),
        ] {
            let probability = 1.0 / (1.0 + (-logit).exp());
            assert!(
                (probability - expected).abs() < 1e-6,
                "logit {logit} should be a probability of {expected}, got {probability}"
            );
        }
    }

    #[test]
    fn the_confusion_rates_are_undefined_on_an_empty_denominator() {
        // A rate over no rows is undefined, not zero. "Demonstrated zero
        // false accepts" on a set with no incomplete rows is a claim about
        // nothing.
        let empty = ReadinessConfusion::default();
        assert_eq!(empty.false_ready_rate(), None);
        assert_eq!(empty.false_not_ready_rate(), None);
        assert_eq!(empty.precision(), None);
        assert_eq!(empty.recall(), None);
        assert_eq!(empty.f1(), None);
        assert_eq!(empty.total(), 0);
    }

    #[test]
    fn the_confusion_rates_are_computed_over_the_right_denominators() {
        let confusion = ReadinessConfusion {
            true_incomplete: 3,
            false_ready: 1,
            false_not_ready: 2,
            true_complete: 6,
        };
        assert_eq!(confusion.total(), 12);
        assert_eq!(confusion.incomplete(), 4);
        assert_eq!(confusion.complete(), 8);
        assert!((confusion.false_ready_rate().unwrap() - 0.25).abs() < 1e-9);
        assert!((confusion.false_not_ready_rate().unwrap() - 0.25).abs() < 1e-9);
        assert!((confusion.precision().unwrap() - 0.6).abs() < 1e-9);
        assert!((confusion.recall().unwrap() - 0.75).abs() < 1e-9);
        assert!((confusion.f1().unwrap() - 2.0 * 0.6 * 0.75 / 1.35).abs() < 1e-9);
    }

    #[test]
    fn the_readiness_head_costs_exactly_169_parameters() {
        // A `Linear(2D -> 1)` on the pooled vector: 168 weights and a bias.
        assert_eq!(Heads::WithReadiness.readiness_parameters(), 169);
        assert_eq!(Heads::IntentOnly.readiness_parameters(), 0);
    }
}
