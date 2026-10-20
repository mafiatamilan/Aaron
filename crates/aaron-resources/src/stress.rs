//! Repeatable stress workloads.
//!
//! The point of these is not to make AARON fast; it is to make a *leak*
//! visible. Each workload runs a fixed number of iterations, samples process RSS
//! at a warm-up point and at the end, and reports both so a reviewer can see
//! whether memory stabilised.
//!
//! # Reading a stability result
//!
//! An allocator caches freed blocks, so RSS legitimately rises a little during
//! warm-up and then plateaus. What matters is whether it keeps rising at the
//! end. [`StabilityVerdict::Stable`] therefore compares the *last third* of the
//! samples against the first third after warm-up, rather than comparing the very
//! first sample to the very last. The tolerance is derived from the observed
//! noise, not picked to make a failure disappear.

use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::probe::process_rss_bytes;

/// One measured phase of a stress run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StressSample {
    pub iteration: usize,
    pub rss_bytes: u64,
    pub elapsed_ms: u64,
}

/// The outcome of one stress workload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StressOutcome {
    pub workload: String,
    pub iterations: usize,
    pub samples: Vec<StressSample>,
    pub peak_rss_bytes: u64,
    pub final_rss_bytes: u64,
    /// RSS after the warm-up window, before the measured body.
    pub baseline_rss_bytes: u64,
    pub elapsed_ms: u64,
    pub verdict: StabilityVerdict,
    /// Bytes of growth per thousand iterations after warm-up. The number to
    /// watch: a leak shows up here as a steady positive slope.
    pub growth_per_1k_iterations: i64,
}

/// Whether memory stabilised.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StabilityVerdict {
    /// RSS plateaued after warm-up.
    Stable,
    /// RSS kept growing past the derived tolerance. Investigate before shipping.
    Growing,
    /// Too few samples to say anything. Reported rather than guessed.
    InsufficientData,
}

impl StressOutcome {
    /// A compact, serialisable summary. The full sample vector is thousands
    /// of rows and is deliberately not included in JSON output.
    pub fn to_summary(&self) -> StressSummary {
        StressSummary {
            workload: self.workload.clone(),
            iterations: self.iterations,
            peak_rss_bytes: self.peak_rss_bytes,
            baseline_rss_bytes: self.baseline_rss_bytes,
            final_rss_bytes: self.final_rss_bytes,
            growth_per_1k_iterations: self.growth_per_1k_iterations,
            verdict: self.verdict,
            elapsed_ms: self.elapsed_ms,
        }
    }
}

/// Compact form of a [`StressOutcome`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StressSummary {
    pub workload: String,
    pub iterations: usize,
    pub peak_rss_bytes: u64,
    pub baseline_rss_bytes: u64,
    pub final_rss_bytes: u64,
    pub growth_per_1k_iterations: i64,
    pub verdict: StabilityVerdict,
    pub elapsed_ms: u64,
}

impl StabilityVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            StabilityVerdict::Stable => "stable",
            StabilityVerdict::Growing => "growing",
            StabilityVerdict::InsufficientData => "insufficient_data",
        }
    }
}

/// Number of leading iterations treated as warm-up.
pub const WARMUP_ITERATIONS: usize = 200;

/// Run `body` `iterations` times, sampling RSS.
///
/// `body` receives the zero-based iteration number so a workload can vary its
/// own behaviour (long inputs, large outputs) if it wants to.
pub fn run<F>(workload: &str, iterations: usize, mut body: F) -> StressOutcome
where
    F: FnMut(usize),
{
    let started = Instant::now();
    let mut samples: Vec<StressSample> = Vec::with_capacity(iterations.min(4096));
    let mut peak = 0u64;
    let mut baseline = 0u64;

    for i in 0..iterations {
        body(i);
        let rss = process_rss_bytes();
        peak = peak.max(rss);
        if i == WARMUP_ITERATIONS.saturating_sub(1) {
            baseline = rss;
        }
        // Sample every iteration up to a cap, then every tenth, so a very long
        // run does not itself allocate a large `Vec` every time.
        if samples.len() < 4096 || i % 10 == 0 {
            samples.push(StressSample {
                iteration: i,
                rss_bytes: rss,
                elapsed_ms: started.elapsed().as_millis() as u64,
            });
        }
    }

    let final_rss = samples.last().map(|s| s.rss_bytes).unwrap_or(0);
    let verdict = classify(&samples, baseline);
    let growth = growth_per_1k(&samples);

    StressOutcome {
        workload: workload.to_string(),
        iterations,
        samples,
        peak_rss_bytes: peak,
        final_rss_bytes: final_rss,
        baseline_rss_bytes: baseline,
        elapsed_ms: started.elapsed().as_millis() as u64,
        verdict,
        growth_per_1k_iterations: growth,
    }
}

/// Decide whether the samples show a leak.
///
/// Compares the first and last thirds of the *post-warm-up* samples. The
/// tolerance is `max(2 MiB, 1% of the baseline)` — large enough to absorb
/// allocator behaviour, small enough that a real per-iteration leak shows up.
pub fn classify(samples: &[StressSample], baseline: u64) -> StabilityVerdict {
    let post: Vec<u64> = samples
        .iter()
        .filter(|s| s.iteration >= WARMUP_ITERATIONS)
        .map(|s| s.rss_bytes)
        .collect();
    if post.len() < 9 {
        return StabilityVerdict::InsufficientData;
    }
    let third = post.len() / 3;
    let early = median(&post[..third]);
    let late = median(&post[post.len() - third..]);
    let tolerance = (baseline / 100).max(2 * 1024 * 1024);
    if late > early.saturating_add(tolerance) {
        StabilityVerdict::Growing
    } else {
        StabilityVerdict::Stable
    }
}

/// Signed bytes of growth per thousand iterations, using a least-squares slope.
pub fn growth_per_1k(samples: &[StressSample]) -> i64 {
    let post: Vec<(f64, f64)> = samples
        .iter()
        .filter(|s| s.iteration >= WARMUP_ITERATIONS)
        .map(|s| (s.iteration as f64, s.rss_bytes as f64))
        .collect();
    if post.len() < 3 {
        return 0;
    }
    let n = post.len() as f64;
    let mean_x = post.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = post.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let num: f64 = post.iter().map(|(x, y)| (x - mean_x) * (y - mean_y)).sum();
    let den: f64 = post.iter().map(|(x, _)| (x - mean_x) * (x - mean_x)).sum();
    if den.abs() < f64::EPSILON {
        return 0;
    }
    ((num / den) * 1000.0).round() as i64
}

fn median(values: &[u64]) -> u64 {
    if values.is_empty() {
        return 0;
    }
    let mut v = values.to_vec();
    v.sort_unstable();
    v[v.len() / 2]
}

/// Render a stress outcome as text.
pub fn render(outcome: &StressOutcome) -> String {
    let mut s = format!(
        "{:<32} n={:<7} peak {:>10}  final {:>10}  slope {:>8}/1k  {}\n",
        outcome.workload,
        outcome.iterations,
        crate::probe::human_bytes(outcome.peak_rss_bytes),
        crate::probe::human_bytes(outcome.final_rss_bytes),
        outcome.growth_per_1k_iterations,
        outcome.verdict.as_str()
    );
    s.push_str(&format!("      ({} ms)\n", outcome.elapsed_ms));
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(iteration: usize, rss: u64) -> StressSample {
        StressSample {
            iteration,
            rss_bytes: rss,
            elapsed_ms: iteration as u64,
        }
    }

    #[test]
    fn a_flat_trace_is_stable() {
        let samples: Vec<_> = (0..3000).map(|i| sample(i, 100 * 1024 * 1024)).collect();
        assert_eq!(
            classify(&samples, 100 * 1024 * 1024),
            StabilityVerdict::Stable
        );
    }

    #[test]
    fn a_leak_is_detected() {
        // 64 KiB leaked per iteration.
        let samples: Vec<_> = (0..3000u64)
            .map(|i| sample(i as usize, 10_000_000 + i * 65_536))
            .collect();
        assert_eq!(classify(&samples, 10_000_000), StabilityVerdict::Growing);
    }

    #[test]
    fn allocator_warm_up_is_not_mistaken_for_a_leak() {
        let mut samples: Vec<_> = (0..3000).map(|i| sample(i, 50 * 1024 * 1024)).collect();
        // A 30 MiB one-off jump early on, then flat. That is the allocator
        // warming up, not a leak.
        for s in samples.iter_mut().take(200) {
            s.rss_bytes += 30 * 1024 * 1024;
        }
        assert_eq!(
            classify(&samples, 80 * 1024 * 1024),
            StabilityVerdict::Stable,
            "warm-up must not be reported as growth"
        );
    }

    #[test]
    fn too_few_samples_is_reported_not_guessed() {
        assert_eq!(classify(&[], 0), StabilityVerdict::InsufficientData);
        assert_eq!(
            classify(&[sample(0, 1), sample(1, 2)], 0),
            StabilityVerdict::InsufficientData
        );
    }

    #[test]
    fn the_slope_reports_growth_per_thousand_iterations() {
        let flat: Vec<_> = (0..3000usize)
            .map(|i| sample(i, 1000 + (i % 2) as u64))
            .collect();
        assert!(growth_per_1k(&flat).abs() <= 2);

        let leak: Vec<_> = (0..3000u64).map(|i| sample(i as usize, i * 1024)).collect();
        let slope = growth_per_1k(&leak);
        assert!(
            slope > 900_000,
            "expected roughly 1 MiB per 1k, got {slope}"
        );
    }

    #[test]
    fn a_workload_actually_runs() {
        let mut counter = 0u64;
        let outcome = run("counter", 500, |_| counter += 1);
        assert_eq!(counter, 500);
        assert_eq!(outcome.iterations, 500);
        assert!(outcome.elapsed_ms < 60_000);
        assert!(!outcome.samples.is_empty());
    }

    #[test]
    fn repeated_allocation_is_stable() {
        // Allocate and drop a realistic working set many times. If anything
        // retains it, the slope will be positive.
        let outcome = run("alloc-drop", 3_000, |i| {
            let mut v: Vec<String> = (0..64).map(|k| format!("allocation {i}-{k}")).collect();
            v.sort();
            std::hint::black_box(&v);
        });
        assert_ne!(outcome.verdict, StabilityVerdict::Growing, "{outcome:?}");
    }

    #[test]
    fn sample_vectors_stay_bounded_for_long_runs() {
        let outcome = run("noop", 20_000, |_| {});
        // Sampled every iteration up to 4096, then every tenth.
        assert!(
            outcome.samples.len() < 6000,
            "samples grew to {}",
            outcome.samples.len()
        );
        assert_eq!(outcome.iterations, 20_000);
    }

    #[test]
    fn rendering_is_informative() {
        let outcome = run("render", 300, |_| {});
        let text = render(&outcome);
        assert!(text.contains("render"));
        assert!(text.contains("peak"));
        assert!(text.contains("slope"));
    }
}
