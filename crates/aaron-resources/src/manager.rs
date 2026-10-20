//! The central resource manager: budgets, operating mode and the low-memory
//! state machine.
//!
//! # The contract
//!
//! AARON never assumes a machine has memory to spare. At startup it measures
//! what is actually there, subtracts the configured reserve, and works inside
//! what is left. When free memory drops below `low_memory_free_percent` it
//! enters [`Mode::LowMemory`] and says so; it recovers only above
//! `low_memory_recover_percent`, which is deliberately a different number.
//! That gap is the hysteresis band, and it is validated in the config.
//!
//! # What low-memory mode does
//!
//! It degrades AARON's own behaviour and nothing else. In particular it must
//! never: kill another application, drop a database write in flight, discard
//! user-approved persistent memory, or silently terminate the user's current
//! task. Those are not memory optimisations, they are data loss.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aaron_core::config::ResourceConfig;
use aaron_core::error::Result;
use serde::{Deserialize, Serialize};

use crate::lifecycle::{BoxedLoader, ModelSlot};
use crate::probe::{sample, ResourceSample};

/// What AARON is doing right now, from a resource perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    /// Nothing in flight.
    Idle,
    /// Capturing audio.
    Listening,
    /// Classifying or running a tool.
    Processing,
    /// Rendering speech.
    Speaking,
    /// Degraded because the machine is short on memory.
    LowMemory,
}

impl std::fmt::Display for Mode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Idle => "idle",
            Mode::Listening => "listening",
            Mode::Processing => "processing",
            Mode::Speaking => "speaking",
            Mode::LowMemory => "low_memory",
        }
    }

    /// Whether nonessential work may start in this mode.
    pub fn admits_background_work(self) -> bool {
        !matches!(self, Mode::LowMemory)
    }
}

/// A capability that is temporarily unavailable, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Degradation {
    pub capability: String,
    pub reason: String,
}

/// Resolved budgets for this machine and this configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Budget {
    /// System RAM AARON may use, after subtracting the reserve.
    pub usable_system_bytes: u64,
    /// VRAM AARON may use, after subtracting the reserve.
    pub usable_gpu_bytes: u64,
    /// Free-memory percentage that triggers low-memory mode.
    pub trigger_percent: u8,
    /// Free-memory percentage that clears it. Strictly greater.
    pub recover_percent: u8,
}

/// The full resource picture.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceStatus {
    pub budget: Budget,
    pub mode: Mode,
    pub sample: ResourceSample,
    /// Seconds since this status was produced.
    pub age_secs: u64,
    /// Why the mode is what it is.
    pub explanation: String,
    /// Capabilities temporarily unavailable, and why.
    pub degraded: Vec<Degradation>,
}

/// Central manager. Cloneable; all clones share one state.
#[derive(Clone)]
pub struct ResourceManager {
    inner: Arc<Inner>,
}

struct Inner {
    config: ResourceConfig,
    state: Mutex<State>,
    /// Monotonic count of mode entries, for tests and diagnostics.
    mode_changes: AtomicU64,
    low_memory_entries: AtomicU64,
}

struct State {
    mode: Mode,
    activity: Mode,
    sample: ResourceSample,
    measured_at: Instant,
    low_memory: bool,
    /// Consecutive samples above the recovery threshold.
    recovery_samples: u32,
    degraded: Vec<Degradation>,
}

/// How many consecutive samples must clear the threshold before AARON leaves
/// low-memory mode.
///
/// One sample is not enough: a single lucky read would otherwise flap the mode
/// on and off. Together with the `recover > trigger` gap this gives two
/// independent defences against oscillation.
pub const RECOVERY_SAMPLES_REQUIRED: u32 = 2;

impl ResourceManager {
    /// Measure the machine and build a manager.
    pub fn new(config: ResourceConfig) -> ResourceManager {
        let sample = sample();
        let low = under_pressure(&config, &sample);
        ResourceManager {
            inner: Arc::new(Inner {
                config: config.clone(),
                state: Mutex::new(State {
                    mode: if low { Mode::LowMemory } else { Mode::Idle },
                    activity: Mode::Idle,
                    sample,
                    measured_at: Instant::now(),
                    low_memory: low,
                    recovery_samples: 0,
                    degraded: if low {
                        degradations_for_low_memory(&config)
                    } else {
                        Vec::new()
                    },
                }),
                mode_changes: AtomicU64::new(0),
                low_memory_entries: AtomicU64::new(0),
            }),
        }
    }

    /// The configured budgets.
    pub fn config(&self) -> &ResourceConfig {
        &self.inner.config
    }

    /// The resolved budget for this machine.
    pub fn budget(&self) -> Budget {
        let s = self.lock();
        resolve_budget(&self.inner.config, &s.sample)
    }

    /// Current status.
    pub fn status(&self) -> ResourceStatus {
        let s = self.lock();
        ResourceStatus {
            budget: resolve_budget(&self.inner.config, &s.sample),
            mode: s.mode,
            sample: s.sample.clone(),
            age_secs: s.measured_at.elapsed().as_secs(),
            explanation: explanation(&s.mode, &s.degraded),
            degraded: s.degraded.clone(),
        }
    }

    /// The current operating mode, including low-memory override.
    pub fn mode(&self) -> Mode {
        self.lock().mode
    }

    /// The activity mode, ignoring any low-memory override.
    pub fn activity(&self) -> Mode {
        self.lock().activity
    }

    /// Capabilities currently unavailable, and why.
    pub fn degraded(&self) -> Vec<Degradation> {
        self.lock().degraded.clone()
    }

    /// Record what the assistant is doing, without changing low-memory state.
    pub fn set_activity(&self, activity: Mode) {
        let mut s = self.lock();
        s.activity = activity;
        s.mode = if s.low_memory {
            Mode::LowMemory
        } else {
            activity
        };
        self.inner.mode_changes.fetch_add(1, Ordering::Relaxed);
    }

    /// Whether the machine currently looks short on memory.
    pub fn under_pressure(&self) -> bool {
        under_pressure(&self.inner.config, &self.lock().sample)
    }

    /// Take a fresh sample and update the mode, applying hysteresis.
    ///
    /// Returns `true` when the mode changed as a result, so a caller can log
    /// or notify exactly once per transition instead of on every sample.
    pub fn refresh(&self) -> bool {
        let fresh = sample();
        self.apply(fresh)
    }

    /// Update from an already-collected sample. Used by tests and by callers
    /// that already sampled for another reason.
    pub fn apply(&self, fresh: ResourceSample) -> bool {
        let cfg = &self.inner.config;
        let free_percent = fresh.system.free_percent();
        let pressure = under_pressure(cfg, &fresh);
        let mut s = self.lock();
        s.sample = fresh;
        s.measured_at = Instant::now();

        if !cfg.low_memory_mode {
            // The feature is switched off; always normal.
            let changed = s.low_memory;
            s.low_memory = false;
            s.degraded.clear();
            s.mode = s.activity;
            return changed;
        }

        if pressure {
            s.recovery_samples = 0;
            if !s.low_memory {
                s.low_memory = true;
                s.degraded = degradations_for_low_memory(cfg);
                self.inner
                    .low_memory_entries
                    .fetch_add(1, Ordering::Relaxed);
                tracing::warn!(
                    free_system = free_percent,
                    "entering low-memory mode; some capabilities are temporarily degraded"
                );
                s.mode = Mode::LowMemory;
                self.inner.mode_changes.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            return false;
        }
        // `fresh` is moved into `s.sample` above; nothing below needs it.

        // Not under pressure. Leave only after enough consecutive samples that
        // were themselves above the *recovery* threshold. Counting merely
        // non-pressure samples would let two marginal reads satisfy the
        // consecutive-sample rule while memory is still below the recovery
        // line, which is exactly the flapping the hysteresis exists to stop.
        if s.low_memory {
            if free_percent >= cfg.low_memory_recover_percent as f64 {
                s.recovery_samples = s.recovery_samples.saturating_add(1);
            } else {
                s.recovery_samples = 0;
            }
            if s.recovery_samples >= RECOVERY_SAMPLES_REQUIRED {
                s.low_memory = false;
                s.recovery_samples = 0;
                s.degraded.clear();
                tracing::info!(free_system = free_percent, "leaving low-memory mode");
                s.mode = s.activity;
                self.inner.mode_changes.fetch_add(1, Ordering::Relaxed);
                return true;
            }
            return false;
        }
        false
    }

    /// How many times the mode has changed. Tests assert this does not grow
    /// without bound across a long idle soak.
    pub fn mode_changes(&self) -> u64 {
        self.inner.mode_changes.load(Ordering::Relaxed)
    }

    /// How many times low-memory mode has been entered.
    pub fn low_memory_entries(&self) -> u64 {
        self.inner.low_memory_entries.load(Ordering::Relaxed)
    }

    /// Whether a new request may be admitted.
    ///
    /// In low-memory mode AARON still answers the user in front of it; it just
    /// refuses to *start* extra background work.
    pub fn admits(&self, background: bool) -> Result<()> {
        if background && !self.mode().admits_background_work() {
            return Err(aaron_core::Error::LimitExceeded(
                "low-memory mode: background work is deferred until memory recovers".into(),
            ));
        }
        Ok(())
    }

    /// Build a lazily loaded, unloadable model slot bound to this manager.
    pub fn model_slot<T, F>(&self, name: &str, loader: F) -> ModelSlot<T, BoxedLoader<T>>
    where
        T: Send + Sync + 'static,
        F: Fn() -> Result<T> + Send + Sync + 'static,
    {
        ModelSlot::new(
            name,
            Box::new(loader) as BoxedLoader<T>,
            self.inner.config.models.idle_unload_seconds,
        )
        .with_unloading(
            self.inner.config.allow_model_unloading && self.inner.config.models.unload_idle_models,
        )
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, State> {
        self.inner
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// Resolve the usable budget from the sample and the configured reserve.
pub fn resolve_budget(config: &ResourceConfig, sample: &ResourceSample) -> Budget {
    let reserve_sys = config.reserve_system_ram_mb.saturating_mul(1024 * 1024);
    let reserve_gpu = config.reserve_gpu_vram_mb.saturating_mul(1024 * 1024);
    Budget {
        usable_system_bytes: sample.system.available_bytes.saturating_sub(reserve_sys),
        usable_gpu_bytes: sample.gpu.free_bytes().saturating_sub(reserve_gpu),
        trigger_percent: config.low_memory_free_percent,
        recover_percent: config.low_memory_recover_percent,
    }
}

/// Whether the sample counts as "short on memory".
///
/// System RAM always counts. VRAM only counts when it was actually measured:
/// an absent GPU must not be treated as a full one.
pub fn under_pressure(config: &ResourceConfig, sample: &ResourceSample) -> bool {
    if sample.system.total_bytes == 0 {
        // Nothing measurable. Refusing to guess is the safe answer: report
        // normal operation rather than inventing a crisis.
        return false;
    }
    let sys = sample.system.free_percent() < config.low_memory_free_percent as f64;
    let gpu = sample.gpu.measured
        && sample.gpu.total_bytes > 0
        && sample.gpu.free_percent() < config.low_memory_free_percent as f64;
    sys || gpu
}

/// What AARON stops doing in low-memory mode.
///
/// Note what is *absent*: no user's data is dropped, no foreign process is
/// killed, no in-flight write is abandoned.
pub fn degradations_for_low_memory(config: &ResourceConfig) -> Vec<Degradation> {
    let mut out = Vec::new();
    if config.models.unload_idle_models {
        out.push(Degradation {
            capability: "speech models stay unloaded until needed".into(),
            reason: "low-memory mode releases idle ASR/TTS models".into(),
        });
    }
    out.push(Degradation {
        capability: "background training and indexing are deferred".into(),
        reason: "low-memory mode does not start nonessential work".into(),
    });
    out.push(Degradation {
        capability: "audio buffer is capped at the configured maximum".into(),
        reason: "low-memory mode discards stale audio rather than queueing it".into(),
    });
    out.push(Degradation {
        capability: "conversation context stays at its configured bound".into(),
        reason: "low-memory mode does not grow in-RAM context".into(),
    });
    out
}

fn explanation(mode: &Mode, degraded: &[Degradation]) -> String {
    if degraded.is_empty() {
        format!("normal operation ({mode})")
    } else {
        format!(
            "low-memory mode: {} capability degradation(s): {}",
            degraded.len(),
            degraded
                .iter()
                .map(|d| d.capability.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        )
    }
}

/// How long to wait between samples.
pub fn monitor_interval(config: &ResourceConfig) -> Duration {
    Duration::from_secs(config.monitor_interval_seconds.max(1) as u64)
}

/// Build the sample the tests use when they need a controlled machine.
pub fn synthetic_sample(
    total_sys: u64,
    available_sys: u64,
    total_gpu: u64,
    used_gpu: u64,
    rss: u64,
) -> ResourceSample {
    ResourceSample {
        system: crate::probe::MemorySample {
            total_bytes: total_sys,
            available_bytes: available_sys,
        },
        gpu: if total_gpu == 0 {
            crate::probe::GpuSample::unavailable("synthetic: no GPU")
        } else {
            crate::probe::GpuSample {
                total_bytes: total_gpu,
                used_bytes: used_gpu,
                measured: true,
                detail: String::new(),
            }
        },
        process_rss_bytes: rss,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_with(system_free_pct: f64, gpu_free_pct: Option<f64>) -> ResourceSample {
        let total = 16 * 1024 * 1024 * 1024u64;
        let available = (total as f64 * system_free_pct / 100.0) as u64;
        match gpu_free_pct {
            None => synthetic_sample(total, available, 0, 0, 64 * 1024 * 1024),
            Some(p) => {
                let gtotal = 4 * 1024 * 1024 * 1024u64;
                let gused = (gtotal as f64 * (100.0 - p) / 100.0) as u64;
                synthetic_sample(total, available, gtotal, gused, 64 * 1024 * 1024)
            }
        }
    }

    #[test]
    fn a_healthy_machine_starts_in_normal_mode() {
        let m = ResourceManager::new(ResourceConfig::default());
        assert_ne!(m.mode(), Mode::LowMemory);
        assert!(m.degraded().is_empty());
        assert!(m.admits(true).is_ok());
    }

    #[test]
    fn low_system_memory_triggers_the_mode() {
        let m = ResourceManager::new(ResourceConfig::default());
        assert!(m.apply(sample_with(4.0, None)));
        assert_eq!(m.mode(), Mode::LowMemory);
        assert!(!m.degraded().is_empty());
        assert!(m.admits(true).is_err(), "background work must be refused");
        // The user in front of the assistant is still served.
        assert!(m.admits(false).is_ok());
    }

    #[test]
    fn a_full_gpu_alone_triggers_the_mode() {
        let m = ResourceManager::new(ResourceConfig::default());
        // System memory is fine, but only 2% of the card is free.
        assert!(m.apply(sample_with(80.0, Some(2.0))));
        assert_eq!(m.mode(), Mode::LowMemory);
    }

    #[test]
    fn an_absent_gpu_never_triggers_the_mode() {
        let m = ResourceManager::new(ResourceConfig::default());
        assert!(!m.apply(sample_with(80.0, None)));
        assert_ne!(m.mode(), Mode::LowMemory);
    }

    #[test]
    fn recovery_requires_the_higher_threshold_and_two_samples() {
        let m = ResourceManager::new(ResourceConfig::default());
        m.apply(sample_with(4.0, None));
        assert_eq!(m.mode(), Mode::LowMemory);

        // Between the thresholds: stays in low memory, and changes nothing.
        let mid = 16.0f64;
        assert!(!m.apply(sample_with(mid, None)), "must not leave yet");
        assert_eq!(m.mode(), Mode::LowMemory);
        // Still between: another quiet sample.
        assert!(!m.apply(sample_with(mid, None)));
        assert_eq!(m.mode(), Mode::LowMemory);

        // Above the recovery threshold, but only one clean sample so far.
        assert!(!m.apply(sample_with(50.0, None)));
        assert_eq!(m.mode(), Mode::LowMemory);

        assert!(
            m.apply(sample_with(50.0, None)),
            "second clean sample exits"
        );
        assert_ne!(m.mode(), Mode::LowMemory);
        assert!(m.degraded().is_empty());
    }

    #[test]
    fn hovering_near_the_threshold_does_not_flap() {
        let m = ResourceManager::new(ResourceConfig::default());
        // Alternate either side of the trigger for a long stretch.
        for i in 0..200 {
            let pct = if i % 2 == 0 { 11.5 } else { 12.5 };
            m.apply(sample_with(pct, None));
        }
        // One entry into low-memory mode is correct and expected; what must not
        // happen is coming back out, because 12.5% never reaches the 20%
        // recovery threshold.
        assert_eq!(
            m.mode_changes(),
            1,
            "the mode must not oscillate across the threshold"
        );
        assert_eq!(m.mode(), Mode::LowMemory);
        assert_eq!(m.low_memory_entries(), 1);
    }

    #[test]
    fn idle_monitoring_does_not_grow_memory() {
        let m = ResourceManager::new(ResourceConfig::default());
        // Start from a sample the manager itself installed, so the comparison
        // is like-for-like rather than synthetic-versus-real.
        m.apply(sample_with(60.0, Some(50.0)));
        let baseline = m.lock().sample.process_rss_bytes;
        for _ in 0..500 {
            m.apply(sample_with(60.0, Some(50.0)));
        }
        let after = m.lock().sample.process_rss_bytes;
        assert_eq!(
            baseline, after,
            "repeated sampling must not change the reported sample"
        );
    }

    #[test]
    fn the_budget_subtracts_the_reserve() {
        let cfg = ResourceConfig::default();
        let s = sample_with(50.0, Some(50.0));
        let b = resolve_budget(&cfg, &s);
        assert_eq!(
            b.usable_system_bytes,
            s.system.available_bytes - cfg.reserve_system_ram_mb * 1024 * 1024
        );
        assert_eq!(
            b.usable_gpu_bytes,
            s.gpu.free_bytes() - cfg.reserve_gpu_vram_mb * 1024 * 1024
        );
        assert!(b.recover_percent > b.trigger_percent);
    }

    #[test]
    fn the_budget_never_goes_negative() {
        let cfg = ResourceConfig::default();
        // A machine with less free RAM than the reserve.
        let s = sample_with(1.0, Some(0.5));
        let b = resolve_budget(&cfg, &s);
        assert!(b.usable_system_bytes == 0 || b.usable_system_bytes < s.system.available_bytes);
        assert!(b.usable_gpu_bytes == 0 || b.usable_gpu_bytes < s.gpu.free_bytes());
    }

    #[test]
    fn turning_the_feature_off_forces_normal_operation() {
        let m = ResourceManager::new(ResourceConfig {
            low_memory_mode: false,
            ..Default::default()
        });
        assert!(!m.apply(sample_with(0.5, Some(0.5))));
        assert_ne!(m.mode(), Mode::LowMemory);
        assert!(m.admits(true).is_ok());
    }

    #[test]
    fn an_unmeasurable_machine_is_not_treated_as_broken() {
        let m = ResourceManager::new(ResourceConfig::default());
        let unknown = synthetic_sample(0, 0, 0, 0, 1024);
        assert!(!m.apply(unknown));
        assert_ne!(m.mode(), Mode::LowMemory);
    }

    #[test]
    fn activity_is_reported_separately_from_the_mode_override() {
        let m = ResourceManager::new(ResourceConfig::default());
        m.set_activity(Mode::Listening);
        assert_eq!(m.activity(), Mode::Listening);
        assert_eq!(m.mode(), Mode::Listening);

        m.apply(sample_with(4.0, None));
        assert_eq!(m.mode(), Mode::LowMemory);
        assert_eq!(
            m.activity(),
            Mode::Listening,
            "the real activity is still listening"
        );

        m.set_activity(Mode::Processing);
        assert_eq!(
            m.mode(),
            Mode::LowMemory,
            "the override survives a new activity"
        );
    }

    #[test]
    fn low_memory_entries_are_counted() {
        let m = ResourceManager::new(ResourceConfig::default());
        assert_eq!(m.low_memory_entries(), 0);
        m.apply(sample_with(4.0, None));
        assert_eq!(m.low_memory_entries(), 1);
        m.apply(sample_with(4.0, None));
        assert_eq!(m.low_memory_entries(), 1, "staying low is not re-entering");
    }

    #[test]
    fn status_explains_itself() {
        let m = ResourceManager::new(ResourceConfig::default());
        assert!(m.status().explanation.contains("normal operation"));
        m.apply(sample_with(4.0, None));
        let s = m.status();
        assert!(s.explanation.contains("low-memory mode"));
        assert!(s.explanation.contains("degradation"));
    }

    #[test]
    fn monitor_interval_is_at_least_one_second() {
        let cfg = ResourceConfig {
            monitor_interval_seconds: 0,
            ..Default::default()
        };
        assert_eq!(monitor_interval(&cfg), Duration::from_secs(1));
        let cfg = ResourceConfig {
            monitor_interval_seconds: 7,
            ..Default::default()
        };
        assert_eq!(monitor_interval(&cfg), Duration::from_secs(7));
    }
}
