//! Concurrency-safe model lifecycle: load on demand, never unload in use.
//!
//! # The rule that matters
//!
//! A model is only released when **no** handle to it is alive. That is
//! enforced with a lock held for as long as a caller holds the model, not with
//! a timer and a hope.
//!
//! Releasing a model that another thread is still using is the classic way to
//! get a use-after-free in safe Rust: the `Arc` is dropped, the allocator
//! reuses the block, and the other thread keeps a dangling pointer. The
//! [`ModelGuard`] type makes that impossible by construction — it holds the
//! guard for exactly as long as the caller needs the model.
//!
//! What this does **not** claim: releasing a `Tensor` returns the memory to the
//! operating system, or frees GPU memory immediately. Candle's CPU allocator
//! caches freed blocks. The honest statement is "AARON no longer holds a
//! reference"; the OS reclaims what it reclaims. `aaron resources` reports the
//! measured RSS rather than claiming more than it can see.

use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use aaron_core::error::{Error, Result};

/// Counters describing a slot's life, for `aaron resources` and for tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotStats {
    pub loads: u64,
    pub unloads: u64,
    pub load_failures: u64,
    /// Loads refused because the model was already being loaded elsewhere.
    pub coalesced_loads: u64,
    /// Times an unload was refused because the model was in use.
    pub unload_skipped_in_use: u64,
}

struct SlotState<T> {
    model: Option<Arc<T>>,
    /// Live handles to the current model.
    users: usize,
    /// Owners of a lease. A leased model is resident by request, so it is not
    /// idle and must not be released, however long ago it was last called.
    pins: usize,
    last_used: Instant,
    load_in_progress: bool,
}

struct Inner<T, F> {
    name: String,
    loader: F,
    state: Mutex<SlotState<T>>,
    condvar: Condvar,
    unloading_enabled: std::sync::atomic::AtomicBool,
    idle_after: Duration,
    stats: Mutex<SlotStats>,
}

/// A model that is loaded on first use and may be released when idle.
pub struct ModelSlot<T, F> {
    inner: Arc<Inner<T, F>>,
}

impl<T, F> ModelSlot<T, F>
where
    T: Send + Sync + 'static,
    F: Fn() -> Result<T> + Send + Sync + 'static,
{
    /// Create an empty slot.
    pub fn new(name: &str, loader: F, idle_after_secs: u32) -> ModelSlot<T, F> {
        ModelSlot {
            inner: Arc::new(Inner {
                name: name.to_string(),
                loader,
                state: Mutex::new(SlotState {
                    model: None,
                    users: 0,
                    pins: 0,
                    last_used: Instant::now(),
                    load_in_progress: false,
                }),
                condvar: Condvar::new(),
                unloading_enabled: std::sync::atomic::AtomicBool::new(true),
                idle_after: Duration::from_secs(idle_after_secs.max(1) as u64),
                stats: Mutex::new(SlotStats::default()),
            }),
        }
    }

    /// Whether idle models may be released at all.
    pub fn with_unloading(self, enabled: bool) -> Self {
        self.inner
            .unloading_enabled
            .store(enabled, std::sync::atomic::Ordering::Relaxed);
        self
    }

    fn unloading_enabled(&self) -> bool {
        self.inner
            .unloading_enabled
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Name shown in diagnostics.
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    /// Whether the model is currently resident.
    pub fn is_loaded(&self) -> bool {
        self.lock().model.is_some()
    }

    /// How many callers currently hold the model.
    pub fn in_use(&self) -> usize {
        self.lock().users
    }

    /// Counters.
    pub fn stats(&self) -> SlotStats {
        *self.lock_stats()
    }

    /// How long the model has been idle, or `None` when it is resident and
    /// in use.
    pub fn idle_for(&self) -> Option<Duration> {
        let s = self.lock();
        match &s.model {
            None => Some(s.last_used.elapsed()),
            Some(_) if s.users == 0 => Some(s.last_used.elapsed()),
            Some(_) => None,
        }
    }

    /// Borrow the model, loading it if necessary.
    ///
    /// The returned guard keeps the model alive. Dropping it is the only way to
    /// release the reference, so a caller cannot accidentally unload a model
    /// another thread is still using.
    pub fn acquire(&self) -> Result<ModelGuard<'_, T, F>> {
        // Fast path: already resident.
        {
            let mut state = self.lock();
            if let Some(model) = state.model.clone() {
                state.users += 1;
                state.last_used = Instant::now();
                return Ok(ModelGuard { slot: self, model });
            }
        }

        // Slow path: load. Only one thread loads; the rest wait on the
        // condvar rather than each paying the cost.
        let mut state = self.lock();
        while state.load_in_progress {
            // A timed wait rather than an unbounded one, so a loader that
            // panics cannot wedge every caller forever.
            let (next, timeout) = self
                .inner
                .condvar
                .wait_timeout(state, Duration::from_secs(30))
                .unwrap_or_else(|e| e.into_inner());
            state = next;
            if timeout.timed_out() && state.load_in_progress {
                return Err(Error::Other(format!(
                    "timed out waiting for `{}` to finish loading",
                    self.inner.name
                )));
            }
            if let Some(model) = state.model.clone() {
                state.users += 1;
                state.last_used = Instant::now();
                return Ok(ModelGuard { slot: self, model });
            }
        }

        state.load_in_progress = true;
        drop(state);

        let loaded = (self.inner.loader)();
        let mut stats = self.lock_stats();
        match &loaded {
            Ok(_) => stats.loads += 1,
            Err(_) => stats.load_failures += 1,
        }
        drop(stats);

        let mut state = self.lock();
        state.load_in_progress = false;
        match loaded {
            Ok(model) => {
                let arc = Arc::new(model);
                state.model = Some(arc.clone());
                state.users += 1;
                state.last_used = Instant::now();
                self.inner.condvar.notify_all();
                Ok(ModelGuard {
                    slot: self,
                    model: arc,
                })
            }
            Err(e) => {
                self.inner.condvar.notify_all();
                Err(e)
            }
        }
    }

    /// Take an owned, long-lived handle to the model.
    ///
    /// A [`ModelGuard`] borrows the slot, so its model must be dropped before
    /// the slot itself can be released. Some callers need the opposite: an
    /// owned `Arc` that outlives the borrow, because they hand it to a
    /// component with its own lifetime (`Agent` owns a synthesiser, for
    /// instance).
    ///
    /// A lease also *pins* the model. While any lease exists the model is
    /// resident by request, not by idleness, so the unload sweep skips it
    /// however old it is. Without that, a long-running session would reload
    /// its speech engine on every single turn.
    pub fn lease(&self) -> Result<ModelLease<T, F>> {
        let guard = self.acquire()?;
        let model = guard.cloned();
        drop(guard);
        self.lock().pins += 1;
        Ok(ModelLease {
            slot: self.clone(),
            model,
        })
    }

    /// How many outstanding leases are pinning the model.
    pub fn pins(&self) -> usize {
        self.lock().pins
    }

    /// Release the model if it is idle for longer than the configured window.
    ///
    /// Returns `true` when a model was actually released. A model with live
    /// users or an outstanding lease is never released; the attempt is counted
    /// so a caller can see that unloading was skipped rather than silently
    /// lost.
    pub fn release_if_idle(&self) -> bool {
        if !self.unloading_enabled() {
            return false;
        }
        let mut state = self.lock();
        if state.model.is_none() {
            return false;
        }
        if state.users > 0 || state.pins > 0 {
            self.lock_stats().unload_skipped_in_use += 1;
            return false;
        }
        if state.last_used.elapsed() < self.inner.idle_after {
            return false;
        }
        state.model = None;
        self.lock_stats().unloads += 1;
        tracing::debug!(model = %self.inner.name, "released idle model");
        true
    }

    /// Release the model regardless of idleness, still refusing while in use.
    pub fn release_now(&self) -> bool {
        if !self.unloading_enabled() {
            return false;
        }
        let mut state = self.lock();
        if state.model.is_none() {
            return false;
        }
        if state.users > 0 || state.pins > 0 {
            self.lock_stats().unload_skipped_in_use += 1;
            return false;
        }
        state.model = None;
        self.lock_stats().unloads += 1;
        true
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, SlotState<T>> {
        self.inner.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn lock_stats(&self) -> std::sync::MutexGuard<'_, SlotStats> {
        self.inner.stats.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl<T, F> Clone for ModelSlot<T, F> {
    fn clone(&self) -> Self {
        ModelSlot {
            inner: Arc::clone(&self.inner),
        }
    }
}

/// A boxed loader, so a slot's type is nameable in a struct field.
///
/// A closure's type is anonymous. Boxing it here is what lets `AppContext` hold
/// two slots of different model types as ordinary fields rather than hiding
/// them behind trait objects or thread-locals.
pub type BoxedLoader<T> = Box<dyn Fn() -> Result<T> + Send + Sync + 'static>;

/// A live borrow of a model inside its slot.
///
/// While this exists the slot's user count is non-zero, so no unload can
/// happen. Dropping it is the release point.
pub struct ModelGuard<'a, T, F> {
    slot: &'a ModelSlot<T, F>,
    #[allow(dead_code)]
    model: Arc<T>,
}

impl<T, F> std::fmt::Debug for ModelGuard<'_, T, F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelGuard")
            .field("slot", &self.slot.inner.name)
            .finish()
    }
}

impl<'a, T, F> ModelGuard<'a, T, F> {
    /// The model.
    pub fn get(&self) -> &T {
        &self.model
    }

    /// The model as an `Arc`, for handing to a worker thread.
    pub fn cloned(&self) -> Arc<T> {
        Arc::clone(&self.model)
    }
}

/// An owned, pinning handle to a model.
///
/// Unlike [`ModelGuard`] this owns the `Arc`, so it can outlive the borrow of
/// the slot it came from. Holding one keeps the model resident by request: see
/// [`ModelSlot::lease`].
pub struct ModelLease<T, F> {
    slot: ModelSlot<T, F>,
    model: Arc<T>,
}

impl<T, F> ModelLease<T, F> {
    /// The model.
    pub fn get(&self) -> &T {
        &self.model
    }

    /// A further owned handle to the same model.
    pub fn cloned(&self) -> Arc<T> {
        Arc::clone(&self.model)
    }
}

impl<T, F> Drop for ModelLease<T, F> {
    fn drop(&mut self) {
        let mut state = self
            .slot
            .inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.pins = state.pins.saturating_sub(1);
        drop(state);
        self.slot.inner.condvar.notify_all();
    }
}

impl<T, F> Drop for ModelGuard<'_, T, F> {
    fn drop(&mut self) {
        // Deliberately avoids the `F: Fn()` bound that `ModelSlot::lock`
        // carries, so a guard can always be released no matter how the slot
        // was built.
        let mut state = self
            .slot
            .inner
            .state
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        state.users = state.users.saturating_sub(1);
        state.last_used = Instant::now();
        drop(state);
        self.slot.inner.condvar.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A loader closure that counts how many times it ran.
    fn loader(counter: Arc<AtomicUsize>) -> impl Fn() -> Result<usize> + Send + Sync + 'static {
        move || Ok(counter.fetch_add(1, Ordering::SeqCst) + 1)
    }

    fn slot(
        counter: Arc<AtomicUsize>,
    ) -> ModelSlot<usize, impl Fn() -> Result<usize> + Send + Sync + 'static> {
        ModelSlot::new("test-model", loader(counter), 0)
    }

    fn fast() -> ModelSlot<usize, impl Fn() -> Result<usize> + Send + Sync + 'static> {
        slot(Arc::new(AtomicUsize::new(0)))
    }

    fn constant(
        value: usize,
    ) -> ModelSlot<usize, impl Fn() -> Result<usize> + Send + Sync + 'static> {
        ModelSlot::new("constant", move || Ok(value), 0)
    }

    #[test]
    fn loads_on_first_use_and_not_before() {
        let s = fast();
        assert!(!s.is_loaded());
        assert_eq!(s.stats().loads, 0);
        {
            let g = s.acquire().unwrap();
            assert_eq!(*g.get(), 1);
        }
        assert_eq!(s.stats().loads, 1);
        assert!(s.is_loaded());
    }

    #[test]
    fn a_second_acquire_does_not_reload() {
        let counter = Arc::new(AtomicUsize::new(0));
        let s = slot(counter);
        let _a = s.acquire().unwrap();
        let _b = s.acquire().unwrap();
        assert_eq!(s.stats().loads, 1);
        assert_eq!(s.in_use(), 2);
    }

    #[test]
    fn a_failed_load_leaves_the_slot_usable_and_counts_the_failure() {
        let s: ModelSlot<u32, _> = ModelSlot::new(
            "always-fails",
            || Err::<u32, _>(Error::Model("simulated load failure".into())),
            0,
        );
        let err = s.acquire().unwrap_err();
        assert!(matches!(err, Error::Model(_)));
        assert_eq!(s.stats().load_failures, 1);
        assert!(
            !s.is_loaded(),
            "a failed load must not leave a half-built slot"
        );
        assert_eq!(s.in_use(), 0);
        // And it can be retried.
        assert!(s.acquire().is_err());
        assert_eq!(s.stats().load_failures, 2);
    }

    #[test]
    fn a_model_in_use_is_never_unloaded() {
        let mut s = fast();
        s = s.with_unloading(true);
        let guard = s.acquire().unwrap();
        assert!(!s.release_now(), "unloading while in use must be refused");
        assert!(s.is_loaded());
        assert_eq!(s.stats().unload_skipped_in_use, 1);
        drop(guard);
        assert!(s.release_now());
        assert!(!s.is_loaded());
    }

    #[test]
    fn unloading_can_be_disabled_entirely() {
        let s = fast().with_unloading(false);
        {
            let _g = s.acquire().unwrap();
        }
        assert!(!s.release_now());
        assert!(s.is_loaded());
        assert_eq!(s.stats().unloads, 0);
    }

    #[test]
    fn idle_unloading_respects_the_window() {
        let s: ModelSlot<usize, _> = ModelSlot::new("windowed", || Ok(1usize), 60);
        let _ = constant(1);
        {
            let _g = s.acquire().unwrap();
        }
        assert!(!s.release_if_idle(), "still inside the idle window");
        assert!(s.is_loaded());
    }

    #[test]
    fn concurrent_acquires_load_exactly_once() {
        let counter = Arc::new(AtomicUsize::new(0));
        let s = slot(Arc::clone(&counter));
        let mut handles = Vec::new();
        for _ in 0..8 {
            let s = s.clone();
            handles.push(std::thread::spawn(move || {
                let g = s.acquire().unwrap();
                assert_eq!(*g.get(), 1, "every caller must see the same model");
                // Hold it long enough that overlapping is certain.
                std::thread::sleep(Duration::from_millis(5));
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(s.stats().loads, 1, "the loader must run exactly once");
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_clone_shares_the_same_slot() {
        let s = fast();
        let t = s.clone();
        let _g = s.acquire().unwrap();
        assert!(t.is_loaded());
        assert_eq!(t.in_use(), 1);
    }

    #[test]
    fn idle_for_reports_none_while_in_use() {
        let s = fast();
        let g = s.acquire().unwrap();
        assert_eq!(s.idle_for(), None);
        drop(g);
        assert!(s.idle_for().is_some());
    }

    #[test]
    fn releasing_an_always_empty_slot_is_a_no_op() {
        let s = fast();
        assert!(!s.release_now());
        assert!(!s.release_if_idle());
        assert_eq!(s.stats().unloads, 0);
    }

    #[test]
    fn repeated_use_does_not_leak_user_counts() {
        let s = fast();
        for _ in 0..1000 {
            let g = s.acquire().unwrap();
            assert_eq!(*g.get(), 1);
            drop(g);
        }
        assert_eq!(s.in_use(), 0);
        assert_eq!(s.stats().loads, 1);
    }

    #[test]
    fn a_guard_can_be_handed_to_a_worker_thread() {
        let s = fast();
        let g = s.acquire().unwrap();
        let arc = g.cloned();
        let h = std::thread::spawn(move || *arc);
        assert_eq!(h.join().unwrap(), 1);
        // The guard is still alive on this thread, so the slot is still in use.
        assert_eq!(s.in_use(), 1);
        drop(g);
        assert!(s.release_now());
    }

    #[test]
    fn a_lease_pins_the_model_against_the_unload_sweep() {
        let s = fast();
        let lease = s.lease().unwrap();
        assert_eq!(s.pins(), 1);
        assert!(!s.release_if_idle(), "a leased model is not idle");
        assert!(!s.release_now());
        assert!(s.is_loaded());
        assert_eq!(s.stats().unload_skipped_in_use, 2);

        let owned = lease.cloned();
        drop(lease);
        assert_eq!(s.pins(), 0);
        // The Arc outlives the lease, so the value is still usable.
        assert_eq!(*owned, 1);
        assert!(s.release_now());
    }

    #[test]
    fn several_leases_pin_independently() {
        let s = fast();
        let a = s.lease().unwrap();
        let b = s.lease().unwrap();
        assert_eq!(s.pins(), 2);
        drop(a);
        assert_eq!(s.pins(), 1);
        assert!(!s.release_now());
        drop(b);
        assert_eq!(s.pins(), 0);
        assert!(s.release_now());
    }

    #[test]
    fn a_lease_and_a_guard_do_not_interfere() {
        let s = fast();
        let _lease = s.lease().unwrap();
        {
            let g = s.acquire().unwrap();
            assert_eq!(*g.get(), 1);
        }
        assert_eq!(s.in_use(), 0, "the guard released");
        assert_eq!(s.pins(), 1, "the lease still pins");
        assert!(!s.release_now());
    }

    #[test]
    fn a_slots_lifetime_is_bounded_by_its_arc() {
        let s = fast();
        {
            let t = s.clone();
            let _g = t.acquire().unwrap();
            assert!(t.is_loaded());
        }
        // The original handle still keeps it alive.
        assert!(s.is_loaded());
        assert!(s.release_now());
    }

    #[test]
    fn slot_stats_are_shared_across_clones() {
        let s = fast();
        let t = s.clone();
        {
            let _g = s.acquire().unwrap();
        }
        assert_eq!(t.stats().loads, 1);
    }

    #[test]
    fn an_rwlock_backed_model_is_usable_through_the_slot() {
        use std::sync::RwLock;
        // A realistic shape: the model is a lock protecting mutable inference
        // state, and the slot guarantees it is only dropped while nobody holds
        // the outer guard.
        let slot = ModelSlot::new("counter", || Ok(RwLock::new(0)), 1);
        {
            let g = slot.acquire().unwrap();
            *g.get().write().unwrap() += 5;
            assert_eq!(*g.get().read().unwrap(), 5);
        }
        assert!(slot.release_now());
        {
            let g = slot.acquire().unwrap();
            // Reloaded fresh, so the previous value is gone.
            assert_eq!(*g.get().read().unwrap(), 0);
        }
    }
}
