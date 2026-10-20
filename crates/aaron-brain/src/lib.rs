//! AARON's custom neural intent classifier.
//!
//! # What this crate is
//!
//! A 1,000,000-trainable-parameter text classifier over a fixed 14-intent
//! schema, plus the deterministic router used when no checkpoint is installed,
//! plus the offline training and evaluation pipeline that produces and
//! validates checkpoints.
//!
//! # What this crate is not
//!
//! It is **not** a language model and it is **not** speech recognition. It maps
//! one utterance to exactly one of 14 closed-set intents. Everything outside
//! that schema becomes `unknown`, and anything the assistant actually *does* is
//! decided by deterministic code in `aaron-agent` and `aaron-tools` after the
//! policy engine has approved it.
//!
//! The 1,000,000 budget covers **this model only**. Any ASR, wake-word or TTS
//! model is a separate, separately licensed and separately sized component; see
//! `docs/model-training.md`.

pub mod analysis;
pub mod checkpoint;
pub mod classifier;
pub mod dataquality;
pub mod dataset;
pub mod eval;
pub mod heuristic;
pub mod model;
pub mod readiness;
pub mod safety;
pub mod train;

pub use checkpoint::{CheckpointInfo, Manifest};
pub use classifier::{find_checkpoint, NeuralClassifier};
pub use dataset::{Dataset, Example, Split, DATASET_FORMAT};
pub use eval::{evaluate, evaluate_ood, Metrics, OodReport};
pub use heuristic::HeuristicClassifier;
pub use model::{
    arch_id, expected_parameter_count, parameter_specs, FeatureBatch, Heads, IntentModel,
    ParamSpec, CLASSES, DIM, HIDDEN, PARAMETER_BUDGET, SEQ_LEN, VOCAB,
};
pub use readiness::{
    train_head, HeadTrainingConfig, HeadTrainingOutcome, ReadinessConfusion, ReadinessCorpus,
    ReadinessExample, ReadinessMetrics,
};
pub use train::{train, TrainingConfig, TrainingOutcome};

/// Version of the model format this crate speaks.
pub const BRAIN_VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn the_enforced_budget_is_one_million() {
        assert_eq!(expected_parameter_count(Heads::IntentOnly), 1_000_000);
        assert_eq!(PARAMETER_BUDGET, 1_000_000);
    }

    #[test]
    fn the_readiness_head_costs_exactly_169_parameters() {
        // A `Linear(2D -> 1)` on the pooled vector: 168 weights and a bias.
        assert_eq!(Heads::WithReadiness.readiness_parameters(), 169);
        assert_eq!(
            expected_parameter_count(Heads::WithReadiness),
            expected_parameter_count(Heads::IntentOnly) + 169
        );
        // Which is 169 over the budget. That is the finding, not a bug: the
        // head has to be paid for out of the budget, and the only exact
        // solution keeps the embedding width at the cost of the hash space.
        assert_eq!(
            expected_parameter_count(Heads::WithReadiness),
            PARAMETER_BUDGET + 169
        );
    }

    #[test]
    fn intent_schema_and_head_width_agree() {
        assert_eq!(CLASSES, aaron_core::INTENT_COUNT);
        assert_eq!(CLASSES, aaron_core::Intent::ALL.len());
        for i in 0..CLASSES {
            assert_eq!(
                aaron_core::Intent::from_index(i).map(|x| x.index()),
                Some(i)
            );
        }
    }

    #[test]
    fn every_declared_tensor_shape_is_consistent() {
        for spec in parameter_specs(Heads::IntentOnly) {
            assert!(!spec.name.is_empty());
            assert!(spec.count() > 0, "{} is empty", spec.name);
            assert!(
                spec.shape.len() <= 2,
                "{} is unexpectedly high rank",
                spec.name
            );
        }
        let names: std::collections::BTreeSet<String> =
            parameter_specs(crate::model::Heads::IntentOnly)
                .into_iter()
                .map(|s| s.name)
                .collect();
        assert_eq!(
            names.len(),
            parameter_specs(crate::model::Heads::IntentOnly).len(),
            "duplicate tensor name"
        );
    }
}
