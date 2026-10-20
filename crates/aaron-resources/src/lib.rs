//! Resource budgeting, runtime sampling, model lifecycle and stress testing.
//!
//! AARON treats memory as a budget, not as an accident. This crate is the
//! single place that measures the machine, decides what AARON may use, decides
//! when to degrade, and proves that repeated work does not leak.
//!
//! # What is guaranteed here
//!
//! * Every queue and retained buffer AARON owns has an explicit, finite bound
//!   that is validated in `aaron-core`'s configuration layer.
//! * A model is never released while a caller holds it. That is a type-level
//!   property ([`ModelGuard`]), not a timing accident.
//! * Low-memory mode degrades AARON's own behaviour. It never kills another
//!   process, never abandons a database write, and never discards user-approved
//!   persistent memory.
//! * Anything that cannot be measured is reported as unmeasured. A missing GPU
//!   is never reported as a full one, and an unreadable `/proc/meminfo` never
//!   invents a crisis.
//!
//! # What is *not* guaranteed here
//!
//! Releasing a model returns the reference; it does not promise the operating
//! system reclaims the pages immediately, and AARON does not clear CUDA caches
//! it does not own. `aaron resources` reports measured RSS so the claim stays
//! checkable.

pub mod context;
pub mod lifecycle;
pub mod manager;
pub mod probe;
pub mod stress;

pub use context::{ConversationContext, Summary, Turn};
pub use lifecycle::{BoxedLoader, ModelGuard, ModelLease, ModelSlot, SlotStats};

/// Guard returned when borrowing the lazily-loaded speech recogniser.
pub type AsrGuard<'a> = ModelGuard<
    'a,
    std::sync::Arc<dyn aaron_core::SpeechRecognizer>,
    BoxedLoader<std::sync::Arc<dyn aaron_core::SpeechRecognizer>>,
>;

/// An owned, pinning handle to the lazily-loaded speech synthesiser.
///
/// A lease rather than a guard: `Agent` owns a synthesiser for its whole
/// lifetime, which outlives any borrow of the application context.
pub type TtsLease = ModelLease<
    std::sync::Arc<dyn aaron_core::SpeechSynthesizer>,
    BoxedLoader<std::sync::Arc<dyn aaron_core::SpeechSynthesizer>>,
>;
pub use manager::{
    degradations_for_low_memory, monitor_interval, resolve_budget, synthetic_sample,
    under_pressure, Budget, Degradation, Mode, ResourceManager, ResourceStatus,
};
pub use probe::{gpu_memory, process_rss_bytes, sample, GpuSample, MemorySample, ResourceSample};
pub use stress::{run as run_stress, StabilityVerdict, StressOutcome};

/// Crate version, surfaced by `aaron resources`.
pub const RESOURCES_VERSION: &str = env!("CARGO_PKG_VERSION");

/// A conservative statement about what AARON itself costs at rest.
///
/// This is a *derived* figure, not a measurement, and is labelled as such
/// wherever it is printed. AARON ships one 4 MB checkpoint, one SQLite
/// connection and one audio buffer; the measured RSS is whatever the allocator
/// decides, which is why `aaron resources` shows the real number instead.
pub const DERIVED_STATIC_FOOTPRINT: DerivedFootprint = DerivedFootprint {
    intent_checkpoint_bytes: 4 * 1024 * 1024,
    intent_params: 1_000_000,
    note: "estimated, not measured; run `aaron resources status` for measured RSS",
};

/// Derived, clearly-labelled size estimates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DerivedFootprint {
    pub intent_checkpoint_bytes: u64,
    pub intent_params: usize,
    pub note: &'static str,
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    use aaron_core::config::ResourceConfig;

    #[test]
    fn the_manager_owns_a_context_and_a_slot_that_share_its_settings() {
        let cfg = ResourceConfig {
            max_conversation_turns_in_context: 3,
            models: aaron_core::config::ModelResourceConfig {
                idle_unload_seconds: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let m = ResourceManager::new(cfg);

        let mut ctx = ConversationContext::from_config(
            m.config().max_conversation_turns_in_context,
            m.config().max_transcript_chars,
        );
        for i in 0..10 {
            ctx.push(&format!("u{i}"), "ok", None, false);
        }
        assert_eq!(ctx.len(), 3);

        let slot = m.model_slot("asr", || Ok(42u32));
        {
            let g = slot.acquire().unwrap();
            assert_eq!(*g.get(), 42);
        }
        assert!(slot.release_now());
    }

    #[test]
    fn low_memory_mode_does_not_affect_persistent_memory() {
        // The rule the specification is really about: freeing RAM must never
        // mean freeing the user's data.
        let m = ResourceManager::new(ResourceConfig::default());
        assert!(m.apply(synthetic_sample(
            4 * 1024 * 1024 * 1024,
            64 * 1024 * 1024,
            0,
            0,
            1 << 20
        )));
        assert_eq!(m.mode(), Mode::LowMemory);
        for d in m.degraded() {
            assert!(
                !d.capability.contains("memory") || d.capability.contains("context"),
                "low-memory mode must not claim to drop stored memory: {d:?}"
            );
        }
    }
}
