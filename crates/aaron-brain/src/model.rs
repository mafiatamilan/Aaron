//! The custom intent network.
//!
//! # Architecture
//!
//! ```text
//! utterance
//!   -> hashed features (word unigrams, word bigrams, padded char trigrams)
//!   -> Embedding(10240 x 84)            860,160 params
//!   -> masked mean-pool  ++  masked RMS-pool            -> 168
//!   -> LayerNorm(168)                       336 params
//!   -> Linear(168 -> 754) + GELU       127,426 params
//!   -> LayerNorm(754)                     1,508 params
//!   -> Linear(754 -> 14)                 10,570 params
//!                                            ----------
//!                                            1,000,000 trainable parameters
//! ```
//!
//! The budget is enforced by [`IntentModel::parameter_count`] and pinned by
//! [`tests::parameter_count_is_exactly_one_million`], which walks the real
//! tensors rather than a hand-written constant.
//!
//! # Why these dimensions
//!
//! `C = 14` is fixed by the closed [`aaron_core::Intent`] enum. That leaves
//! `V*D + 2*D*H + H*C + 3*H + 4*D + C = 1_000_000` to satisfy. `V = 10240` (a
//! power-of-two hash space) and `D = 84` are chosen because they make the
//! residual land on an exact integer for `H = 754`; the search is reproduced in
//! `docs/model-training.md`. Embedding storage is 86% of the budget, which is
//! the correct trade for a 1M-parameter text classifier: the model needs a wide
//! input vocabulary far more than it needs depth.

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Activation, Embedding, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

use aaron_core::intent::INTENT_COUNT;

/// Number of hashed feature buckets: the embedding vocabulary.
pub const VOCAB: usize = 10_240;
/// Embedding width.
pub const DIM: usize = 84;
/// Hidden width.
pub const HIDDEN: usize = 754;
/// Classification head width.
pub const CLASSES: usize = INTENT_COUNT;
/// Padded feature sequence length. Not a parameter: padding is masked out.
pub const SEQ_LEN: usize = 96;
/// Standard deviation of the embedding initialisation.
pub const EMBEDDING_INIT_STD: f32 = 0.05;

/// LayerNorm epsilon.
pub const LAYER_NORM_EPS: f64 = 1e-5;

/// The trainable-parameter budget this architecture is required to hit.
pub const PARAMETER_BUDGET: usize = 1_000_000;

/// One tensor in the architecture, with its shape and element count.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParamSpec {
    /// Fully qualified name, matching the checkpoint manifest.
    pub name: String,
    /// Shape as stored (row-major).
    pub shape: Vec<usize>,
}

impl ParamSpec {
    /// Number of scalars in this tensor.
    pub fn count(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Which output heads the architecture carries.
///
/// The readiness head is a *separate* output that reads the pooled
/// representation: a `Linear(2D -> 1)` on the 168-dimensional pooled vector,
/// which is 168 weights and one bias, 169 parameters.
///
/// It is **additional** to the intent classifier. Adding it to the shipped
/// architecture gives 1,000,169 parameters, which is 169 over the budget.
/// Landing on exactly 1,000,000 means giving the 169 back from somewhere
/// else, and the only solution that keeps the embedding width is
/// `VOCAB = 10249, HIDDEN = 749` — a different hash space, which changes the
/// feature map and invalidates every existing checkpoint.
///
/// So the head is built first as a *diagnostic* on the existing
/// architecture, and the reallocation is only worth doing if the diagnostic
/// says the head works at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Heads {
    /// The intent head only. 1,000,000 parameters.
    #[default]
    IntentOnly,
    /// The intent head plus the 169-parameter readiness head. 1,000,169.
    WithReadiness,
}

impl Heads {
    /// Does the architecture carry the readiness head?
    pub fn has_readiness(self) -> bool {
        matches!(self, Heads::WithReadiness)
    }

    /// The readiness head's parameter count: `2D` weights and one bias.
    pub fn readiness_parameters(self) -> usize {
        if self.has_readiness() {
            2 * DIM + 1
        } else {
            0
        }
    }
}

/// Every tensor in the architecture, in a fixed order.
pub fn parameter_specs(heads: Heads) -> Vec<ParamSpec> {
    let spec = |name: &str, shape: &[usize]| ParamSpec {
        name: name.to_string(),
        shape: shape.to_vec(),
    };
    let mut specs = vec![
        spec("emb.weight", &[VOCAB, DIM]),
        spec("ln1.weight", &[2 * DIM]),
        spec("ln1.bias", &[2 * DIM]),
        spec("fc1.weight", &[HIDDEN, 2 * DIM]),
        spec("fc1.bias", &[HIDDEN]),
        spec("ln2.weight", &[HIDDEN]),
        spec("ln2.bias", &[HIDDEN]),
        spec("fc2.weight", &[CLASSES, HIDDEN]),
        spec("fc2.bias", &[CLASSES]),
    ];
    if heads.has_readiness() {
        // The readiness head reads the *pooled* vector, before `ln1`.
        specs.push(spec("fc_ready.weight", &[1, 2 * DIM]));
        specs.push(spec("fc_ready.bias", &[1]));
    }
    specs
}

/// Exact number of trainable parameters, computed from the architecture.
pub fn expected_parameter_count(heads: Heads) -> usize {
    parameter_specs(heads).iter().map(|s| s.count()).sum()
}

/// Batched, padded feature input.
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureBatch {
    /// `batch x seq_len` bucket indices.
    pub ids: Vec<Vec<u32>>,
    /// `batch x seq_len` 1.0 for real features, 0.0 for padding.
    pub mask: Vec<Vec<f32>>,
}

impl FeatureBatch {
    /// Build a batch from raw utterances.
    pub fn from_texts(texts: &[String]) -> FeatureBatch {
        let ids: Vec<Vec<u32>> = texts.iter().map(|t| encode(t)).collect();
        pad_to(&ids, SEQ_LEN)
    }

    /// Number of utterances in the batch.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Flatten to the `f32` id tensor candle indexes with.
    fn id_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let flat: Vec<u32> = self.ids.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::new(flat.as_slice(), device)?.reshape((self.ids.len(), SEQ_LEN))
    }

    fn mask_tensor(&self, device: &Device) -> candle_core::Result<Tensor> {
        let flat: Vec<f32> = self.mask.iter().flat_map(|r| r.iter().copied()).collect();
        Tensor::new(flat.as_slice(), device)?.reshape((self.mask.len(), SEQ_LEN))
    }
}

/// Hashed feature indices for one utterance, truncated to [`SEQ_LEN`].
///
/// An empty utterance still yields one feature so that the mask is never
/// all-zero (which would make the mean-pool a division by zero).
pub fn encode(text: &str) -> Vec<u32> {
    let mut feats = aaron_core::text::features(text, VOCAB as u32);
    if feats.is_empty() {
        feats.push(aaron_core::text::bucket_of("w:\u{2205}", VOCAB as u32));
    }
    feats.truncate(SEQ_LEN);
    feats
}

fn pad_to(rows: &[Vec<u32>], len: usize) -> FeatureBatch {
    FeatureBatch {
        ids: rows
            .iter()
            .map(|r| {
                let mut v = r.clone();
                v.resize(len, 0);
                v
            })
            .collect(),
        mask: rows
            .iter()
            .map(|r| {
                let mut v = vec![1.0f32; r.len()];
                v.resize(len, 0.0);
                v
            })
            .collect(),
    }
}

/// The intent network.
pub struct IntentModel {
    emb: Embedding,
    ln1: LayerNorm,
    fc1: Linear,
    act: Activation,
    ln2: LayerNorm,
    fc2: Linear,
    /// The readiness head, when the architecture carries one.
    ///
    /// It reads the pooled representation — the 168-dimensional mean-and-RMS
    /// vector, *before* `ln1` — and produces one logit.
    fc_ready: Option<Linear>,
}

impl IntentModel {
    /// Build from a `VarBuilder`. The builder must be able to supply every
    /// tensor named by [`parameter_specs`].
    pub fn load(vb: VarBuilder<'_>, heads: Heads) -> candle_core::Result<IntentModel> {
        let emb = candle_nn::embedding(VOCAB, DIM, vb.pp("emb"))?;
        let ln1 = candle_nn::layer_norm(2 * DIM, LayerNormConfig::default(), vb.pp("ln1"))?;
        let fc1 = candle_nn::linear_b(2 * DIM, HIDDEN, true, vb.pp("fc1"))?;
        let ln2 = candle_nn::layer_norm(HIDDEN, LayerNormConfig::default(), vb.pp("ln2"))?;
        let fc2 = candle_nn::linear_b(HIDDEN, CLASSES, true, vb.pp("fc2"))?;
        let fc_ready = if heads.has_readiness() {
            Some(candle_nn::linear_b(2 * DIM, 1, true, vb.pp("fc_ready"))?)
        } else {
            None
        };
        Ok(IntentModel {
            emb,
            ln1,
            fc1,
            act: Activation::Gelu,
            ln2,
            fc2,
            fc_ready,
        })
    }

    /// Build from an explicit tensor map, as loaded from a checkpoint.
    pub fn from_tensors(
        tensors: HashMap<String, Tensor>,
        device: &Device,
        heads: Heads,
    ) -> candle_core::Result<IntentModel> {
        let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
        IntentModel::load(vb, heads)
    }

    /// Build from freshly seeded, deterministic initial weights.
    pub fn initialised(
        seed: u64,
        device: &Device,
        heads: Heads,
    ) -> candle_core::Result<IntentModel> {
        let tensors = initial_tensors(seed, heads)?;
        IntentModel::from_tensors(tensors, device, heads)
    }

    /// Which heads this model carries.
    pub fn heads(&self) -> Heads {
        if self.fc_ready.is_some() {
            Heads::WithReadiness
        } else {
            Heads::IntentOnly
        }
    }

    /// Forward pass: returns raw logits of shape `batch x CLASSES`.
    pub fn forward(&self, batch: &FeatureBatch) -> candle_core::Result<Tensor> {
        self.forward_with(batch, None, None)
    }

    /// Forward pass with optional training-time regularisation masks.
    ///
    /// # Why the masks exist
    ///
    /// With 1,000,000 parameters and roughly 970 training examples, this
    /// model memorises: measured train accuracy was exactly 1.0000 while the
    /// held-out accuracy was 0.76, and every prediction came back at 0.99
    /// confidence whether it was right or wrong.
    ///
    /// Dropout is the standard answer to that, and it is the answer that
    /// costs nothing here: a dropout mask has **no trainable parameters**, so
    /// the 1,000,000 budget is untouched. The parameter-count test still
    /// passes because it reads the architecture, not the training code.
    ///
    /// * `feature_mask` is `[B, L]` and zeroes individual *input* features.
    ///   For a bag-of-hashed-features model this is the highest-value
    ///   augmentation available: it forces the model to rely on several cues
    ///   rather than memorising one.
    /// * `hidden_mask` is `[B, HIDDEN]` and is ordinary dropout on the
    ///   hidden layer, already scaled by `1 / (1 - p)`.
    ///
    /// Both are `None` at inference, so a trained model and a loaded
    /// checkpoint compute exactly the same thing.
    pub fn forward_with(
        &self,
        batch: &FeatureBatch,
        feature_mask: Option<&Tensor>,
        hidden_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (logits, _) = self.forward_with_readiness(batch, feature_mask, hidden_mask)?;
        Ok(logits)
    }

    /// Forward pass returning the intent logits *and* the readiness logit.
    ///
    /// The readiness logit is `None` for an architecture without the head. It
    /// is read from the **pooled** vector, before `ln1`, which is the
    /// representation the 0.5 plan names.
    pub fn forward_with_readiness(
        &self,
        batch: &FeatureBatch,
        feature_mask: Option<&Tensor>,
        hidden_mask: Option<&Tensor>,
    ) -> candle_core::Result<(Tensor, Option<Tensor>)> {
        let device = batch.id_tensor_device();
        let ids = batch.id_tensor(&device)?;
        let mask = batch.mask_tensor(&device)?;

        // [B, L, D]
        let emb = self.emb.forward(&ids)?;
        let m3 = mask.unsqueeze(2)?; // [B, L, 1]

        // Two pooling statistics, concatenated.
        //
        // RMS was chosen over max-pooling after measuring both: max is a
        // single winning feature per dimension, its scale is an order of
        // magnitude larger than the mean, and LayerNorm then shrinks the mean
        // half until it carries almost no signal — measured test accuracy fell
        // from 0.75 to 0.07. RMS is the same order of magnitude as the mean, so
        // the two halves stay comparable and LayerNorm treats them equally.
        //
        // * mean: the signed average embedding over the utterance's features.
        // * RMS: the per-dimension root-mean-square, i.e. how strongly each
        //   embedding dimension is engaged across the utterance.
        //
        // padded positions are exactly zero, so padding contributes nothing
        // and needs no sentinel.
        //
        // `weighted` is `emb * mask`: padded rows are zeroed, and the gradient
        // through the mask multiplies the embedding rather than a 1e9 sentinel,
        // which keeps training well conditioned.
        let mut weighted = emb.broadcast_mul(&m3)?;
        if let Some(fm) = feature_mask {
            // Dropping a feature is a zero, not a "no bucket" marker, so the
            // masked mean and RMS both see it as absent. Correct: a dropped
            // feature contributes nothing to either statistic.
            //
            // The mask is `[B, L]`; `weighted` is `[B, L, D]`, so the mask
            // needs a trailing axis to broadcast over the embedding.
            let fm = fm.unsqueeze(2)?;
            weighted = weighted.broadcast_mul(&fm)?;
        }
        let count = mask.sum(1)?.clamp(1e-6, f32::MAX)?.unsqueeze(1)?; // [B, 1]
        let sum = weighted.sum(1)?; // [B, D]
        let mean = sum.broadcast_div(&count)?;

        let sum_sq = weighted.sqr()?.sum(1)?; // [B, D]
        let rms = sum_sq.broadcast_div(&count)?.sqrt()?;

        // [B, 2D]
        let pooled = Tensor::cat(&[&mean, &rms], 1)?;
        // The readiness head reads the pooled vector itself, before the
        // trunk touches it. It is a separate output, not a second layer on
        // top of the intent head.
        let ready = match &self.fc_ready {
            Some(head) => Some(head.forward(&pooled)?),
            None => None,
        };
        let x = self.ln1.forward(&pooled)?;
        let x = self.fc1.forward(&x)?;
        let x = self.act.forward(&x)?;
        let x = match hidden_mask {
            Some(hm) => x.broadcast_mul(hm)?,
            None => x,
        };
        let x = self.ln2.forward(&x)?;
        Ok((self.fc2.forward(&x)?, ready))
    }

    /// Total trainable parameters, read from the architecture rather than from
    /// a stored constant.
    pub fn parameter_count(&self) -> usize {
        expected_parameter_count(self.heads())
    }

    /// Architecture fingerprint, used to refuse incompatible checkpoints.
    pub fn arch_id(&self) -> String {
        arch_id(self.heads())
    }
}

impl FeatureBatch {
    fn id_tensor_device(&self) -> Device {
        Device::Cpu
    }
}

/// A stable identifier for this architecture, e.g. `aaron-intent-1m-v1`.
pub fn arch_id(heads: Heads) -> String {
    let version = if heads.has_readiness() { "v1r" } else { "v1" };
    format!("aaron-intent-1m-{version}:{VOCAB}x{DIM}h{HIDDEN}c{CLASSES}L{SEQ_LEN}")
}

/// Deterministic PRNG used for weight initialisation.
///
/// `rand`'s default generators are explicitly not reproducible across
/// versions, so AARON carries its own SplitMix64. Initialisation is therefore
/// byte-identical on any machine, which is what makes a training run
/// reproducible.
#[derive(Debug, Clone)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }

    /// Standard normal via Box-Muller. Rejects zero to avoid `ln(0)`.
    pub fn normal(&mut self) -> f32 {
        loop {
            let u1 = self.next_f32();
            if u1 > 1e-7 {
                let u2 = self.next_f32();
                let r = (-2.0 * u1.ln()).sqrt();
                return r * (std::f32::consts::TAU * u2).cos();
            }
        }
    }
}

/// Build the deterministic initial weight tensors.
pub fn initial_tensors(seed: u64, heads: Heads) -> candle_core::Result<HashMap<String, Tensor>> {
    let device = Device::Cpu;
    let mut rng = SplitMix64::new(seed);
    let mut map: HashMap<String, Tensor> = HashMap::new();

    for spec in parameter_specs(heads) {
        let n = spec.count();
        let data: Vec<f32> = match spec.name.as_str() {
            "ln1.weight" | "ln2.weight" => vec![1.0; n],
            "ln1.bias" | "ln2.bias" | "fc2.bias" | "fc_ready.bias" => vec![0.0; n],
            name => {
                // He initialisation for the ReLU-family stack: keeps the
                // variance of the activations stable across depth.
                //
                // The embedding is initialised far tighter than He would
                // suggest. Its rows are pooled by max as well as by mean, and
                // a max over ~20 draws of N(0, 0.154) dominates the pooled
                // vector; LayerNorm then amplifies that imbalance and early
                // training diverges. 0.05 keeps the pooled halves comparable.
                let std = match name {
                    "emb.weight" => EMBEDDING_INIT_STD,
                    "fc1.weight" => (2.0 / (2 * DIM) as f64).sqrt() as f32,
                    "fc2.weight" => (2.0 / HIDDEN as f64).sqrt() as f32,
                    // The readiness head reads the pooled vector, so it is
                    // initialised like `fc1`, which also reads a `2D`-wide
                    // input.
                    //
                    // It must **not** fall through to the `_ => 0.0` arm: a
                    // zero weight vector makes the head's output constant and
                    // its gradient with respect to the weights zero, so the
                    // head could never learn. That failure is silent — the
                    // parameter count is right and the loss curve is
                    // plausible — so it is worth stating.
                    "fc_ready.weight" => (2.0 / (2 * DIM) as f64).sqrt() as f32,
                    _ => 0.0,
                };
                (0..n).map(|_| rng.normal() * std).collect()
            }
        };
        let shape = spec.shape.clone();
        let tensor = Tensor::new(data.as_slice(), &device)?.reshape(shape)?;
        map.insert(spec.name.clone(), tensor);
    }
    let _ = rng.uniform(0.0, 1.0);
    Ok(map)
}

/// Softmax with a temperature, returning `(intent_index, probability)`.
///
/// Returns `None` when the input is not finite, so a corrupt checkpoint cannot
/// produce a confident wrong answer.
pub fn softmax(logits: &[f32], temperature: f32) -> Option<Vec<f32>> {
    if temperature <= 0.0 || !temperature.is_finite() {
        return None;
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return None;
    }
    let mut probs: Vec<f32> = logits
        .iter()
        .map(|l| ((*l - max) / temperature).exp())
        .collect();
    let sum: f32 = probs.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return None;
    }
    for p in probs.iter_mut() {
        *p /= sum;
    }
    Some(probs)
}

/// The top prediction with its probability.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Prediction {
    pub index: usize,
    pub probability: f32,
}

/// Argmax over class probabilities.
pub fn predict(probs: &[f32]) -> Prediction {
    let mut best = Prediction {
        index: 0,
        probability: 0.0,
    };
    for (i, p) in probs.iter().enumerate() {
        if *p > best.probability {
            best = Prediction {
                index: i,
                probability: *p,
            };
        }
    }
    best
}

/// Marginal over a contiguous class range, used by the evaluation report to
/// show how the model splits "file" intents from "process" intents.
pub fn marginal(probs: &[f32], range: std::ops::Range<usize>) -> f32 {
    probs.iter().skip(range.start).take(range.len()).sum()
}

/// Indices covered by a class range, for documentation purposes.
pub fn range_labels(range: std::ops::Range<usize>) -> Vec<&'static str> {
    aaron_core::Intent::ALL[range]
        .iter()
        .map(|i| i.as_str())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_count_is_exactly_one_million() {
        let total = expected_parameter_count(crate::model::Heads::IntentOnly);
        assert_eq!(
            total, PARAMETER_BUDGET,
            "architecture has {total} trainable parameters, budget is {PARAMETER_BUDGET}"
        );
    }

    #[test]
    fn parameter_breakdown_adds_up() {
        let mut running = 0usize;
        for spec in parameter_specs(crate::model::Heads::IntentOnly) {
            running += spec.count();
        }
        assert_eq!(running, 1_000_000);
        // The documented breakdown must match the code, so that the docs cannot
        // quietly drift away from the architecture.
        assert_eq!(VOCAB * DIM, 860_160);
        assert_eq!(2 * DIM, 168);
        assert_eq!(HIDDEN * 2 * DIM + HIDDEN, 127_426);
        assert_eq!(2 * HIDDEN, 1_508);
        assert_eq!(CLASSES * HIDDEN + CLASSES, 10_570);
    }

    #[test]
    fn model_reports_the_architecture_count() {
        let m = IntentModel::initialised(7, &Device::Cpu, crate::model::Heads::IntentOnly).unwrap();
        assert_eq!(m.parameter_count(), 1_000_000);
    }

    #[test]
    fn forward_produces_one_logit_row_per_utterance() {
        let m = IntentModel::initialised(1, &Device::Cpu, crate::model::Heads::IntentOnly).unwrap();
        let batch = FeatureBatch::from_texts(&[
            "check system status".to_string(),
            "remember that i like tea".to_string(),
            "".to_string(),
        ]);
        let logits = m.forward(&batch).unwrap();
        assert_eq!(logits.dims(), &[3, CLASSES]);
        let flat: Vec<f32> = logits.flatten_all().unwrap().to_vec1().unwrap();
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn padding_does_not_change_the_result() {
        let m = IntentModel::initialised(3, &Device::Cpu, crate::model::Heads::IntentOnly).unwrap();
        let solo = FeatureBatch::from_texts(&["open the file".to_string()]);
        // The same utterance batched with two entirely-padded rows. Row 0 must
        // be bit-comparable with the unbatched forward pass.
        let mut batch = FeatureBatch::from_texts(&[
            "open the file".to_string(),
            "open the file".to_string(),
            "open the file".to_string(),
        ]);
        batch.ids[1] = vec![0; SEQ_LEN];
        batch.mask[1] = vec![0.0; SEQ_LEN];
        batch.ids[2] = vec![0; SEQ_LEN];
        batch.mask[2] = vec![0.0; SEQ_LEN];

        let a: Vec<f32> = m
            .forward(&solo)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let b: Vec<f32> = m
            .forward(&batch)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        let a0 = &a[..CLASSES];
        let b0 = &b[..CLASSES];
        for (x, y) in a0.iter().zip(b0.iter()) {
            assert!(
                (x - y).abs() < 1e-4,
                "batching changed the result: {x} vs {y}"
            );
        }
    }

    #[test]
    fn an_all_padding_row_still_produces_finite_logits() {
        let m = IntentModel::initialised(4, &Device::Cpu, crate::model::Heads::IntentOnly).unwrap();
        let mut batch = FeatureBatch::from_texts(&["hello".to_string(), "hello".to_string()]);
        batch.ids[1] = vec![0; SEQ_LEN];
        batch.mask[1] = vec![0.0; SEQ_LEN];
        let logits: Vec<f32> = m
            .forward(&batch)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        assert!(logits.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn initialisation_is_deterministic() {
        let a = initial_tensors(42, crate::model::Heads::IntentOnly).unwrap();
        let b = initial_tensors(42, crate::model::Heads::IntentOnly).unwrap();
        for (name, ta) in &a {
            let tb = b.get(name).expect("same key set");
            let va: Vec<f32> = ta.flatten_all().unwrap().to_vec1().unwrap();
            let vb: Vec<f32> = tb.flatten_all().unwrap().to_vec1().unwrap();
            assert_eq!(
                va, vb,
                "tensor `{name}` differs between runs with the same seed"
            );
        }
    }

    #[test]
    fn different_seeds_give_different_weights() {
        let a = initial_tensors(1, crate::model::Heads::IntentOnly).unwrap();
        let b = initial_tensors(2, crate::model::Heads::IntentOnly).unwrap();
        let va: Vec<f32> = a["fc1.weight"].flatten_all().unwrap().to_vec1().unwrap();
        let vb: Vec<f32> = b["fc1.weight"].flatten_all().unwrap().to_vec1().unwrap();
        assert_ne!(va, vb);
    }

    #[test]
    fn splitmix64_matches_known_vectors() {
        // Reference values for SplitMix64 seeded with 0.
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xE220A8397B1DCDAF);
        assert_eq!(r.next_u64(), 0x6E789E6AA1B965F4);
        assert_eq!(r.next_u64(), 0x06C45D188009454F);
    }

    #[test]
    fn encode_is_deterministic_and_bounded() {
        let a = encode("check the system status");
        let b = encode("Check the system status!");
        assert_eq!(a, b);
        assert!(a.len() <= SEQ_LEN);
        assert!(a.iter().all(|v| (*v as usize) < VOCAB));
        // An empty utterance still produces one feature.
        assert_eq!(encode("").len(), 1);
    }

    #[test]
    fn softmax_normalises_and_is_temperature_sensitive() {
        let probs = softmax(&[1.0, 2.0, 3.0], 1.0).unwrap();
        assert!((probs.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        let hot = softmax(&[1.0, 2.0, 3.0], 0.5).unwrap();
        assert!(
            hot[2] > probs[2],
            "lower temperature must sharpen the distribution"
        );
        assert!(softmax(&[f32::NAN, 1.0], 1.0).is_none());
        assert!(softmax(&[1.0], 0.0).is_none());
    }

    #[test]
    fn predict_returns_the_argmax() {
        let p = predict(&[0.1, 0.7, 0.2]);
        assert_eq!(p.index, 1);
        assert!((p.probability - 0.7).abs() < 1e-6);
    }

    #[test]
    fn arch_id_encodes_the_dimensions() {
        assert!(arch_id(crate::model::Heads::IntentOnly).contains("10240"));
        assert!(arch_id(crate::model::Heads::IntentOnly).contains("x84"));
    }
}
