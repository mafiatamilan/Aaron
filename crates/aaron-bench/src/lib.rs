//! Reproducible benchmarks.
//!
//! # What is measured here and what is not
//!
//! Every number this crate produces is measured **on the machine that ran it**,
//! in this process, at the moment `aaron benchmark` was invoked. Nothing is
//! cached, interpolated or hard-coded, and every report carries the machine
//! description, the sample count and the configuration used.
//!
//! GPU-specific benchmarks are deliberately kept separate and opt-in, because a
//! CI runner has no GPU and a benchmark that silently reports "0 fps" on
//! hardware it never touched is worse than no benchmark.
//!
//! Reporting is median + p95, not mean: a single scheduler hiccup should not
//! define the result.

use std::time::{Duration, Instant};

use aaron_core::hardware::HardwareReport;
use serde::{Deserialize, Serialize};

/// Summary statistics for a set of samples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    pub name: String,
    pub samples: usize,
    pub min_us: u64,
    pub median_us: u64,
    pub p95_us: u64,
    pub max_us: u64,
    pub mean_us: u64,
}

impl Stats {
    /// Compute statistics from raw microsecond samples.
    pub fn from_samples(name: &str, mut samples: Vec<u64>) -> Stats {
        samples.sort_unstable();
        let n = samples.len();
        if n == 0 {
            return Stats {
                name: name.to_string(),
                samples: 0,
                min_us: 0,
                median_us: 0,
                p95_us: 0,
                max_us: 0,
                mean_us: 0,
            };
        }
        // Nearest-rank percentile: with small sample counts, interpolation
        // invents a value that was never measured.
        let idx = |p: f64| -> usize {
            let rank = (p * n as f64).ceil() as usize;
            rank.clamp(1, n) - 1
        };
        let sum: u128 = samples.iter().map(|s| *s as u128).sum();
        Stats {
            name: name.to_string(),
            samples: n,
            min_us: samples[0],
            median_us: samples[idx(0.50)],
            p95_us: samples[idx(0.95)],
            max_us: samples[n - 1],
            mean_us: (sum / n as u128) as u64,
        }
    }

    /// Fixed-width text for the CLI.
    pub fn to_row(&self) -> String {
        format!(
            "{:<34} {:>7} {:>10} {:>10} {:>10} {:>10}",
            self.name,
            self.samples,
            fmt_us(self.min_us),
            fmt_us(self.median_us),
            fmt_us(self.p95_us),
            fmt_us(self.max_us)
        )
    }
}

fn fmt_us(us: u64) -> String {
    if us < 1000 {
        format!("{us}us")
    } else if us < 1_000_000 {
        format!("{:.2}ms", us as f64 / 1000.0)
    } else {
        format!("{:.2}s", us as f64 / 1_000_000.0)
    }
}

/// One complete benchmark report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Report {
    /// Unix seconds, so two reports can be ordered.
    pub measured_at_unix: u64,
    pub machine: HardwareReport,
    /// Free-form description of what was configured.
    pub configuration: String,
    pub stats: Vec<Stats>,
    /// Measured memory behaviour, separate from latency.
    pub memory: MemoryReport,
    /// Anything the run could not measure, stated rather than omitted.
    pub unavailable: Vec<String>,
}

/// Memory observations from one benchmark run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MemoryReport {
    /// RSS before anything was measured.
    pub baseline_rss_bytes: u64,
    /// Highest RSS seen during the run.
    pub peak_rss_bytes: u64,
    /// RSS at the end of the run.
    pub final_rss_bytes: u64,
    /// Peak RSS while classifying intents specifically.
    pub intent_peak_rss_bytes: u64,
    /// Peak RSS while running the end-to-end agent turn.
    pub agent_peak_rss_bytes: u64,
    /// GPU memory used when the run started and finished, when measurable.
    pub gpu_used_start_bytes: Option<u64>,
    pub gpu_used_end_bytes: Option<u64>,
    /// GPU total, when measurable.
    pub gpu_total_bytes: Option<u64>,
    /// Why GPU memory is missing, when it is.
    pub gpu_detail: String,
    /// Per-mode memory, measured separately.
    ///
    /// §7 of the release brief asks for text-only, ASR, TTS and combined voice
    /// to be reported *separately*, because a single blended number hides
    /// which component is responsible. A mode that could not be measured on
    /// this machine says so rather than reporting zero.
    pub modes: Vec<ModeMemory>,
}

/// Memory for one operating mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModeMemory {
    /// `text-only`, `asr`, `tts` or `voice`.
    pub mode: String,
    /// RSS before the mode's workload ran.
    pub baseline_rss_bytes: u64,
    /// Highest RSS during the mode's workload.
    pub peak_rss_bytes: u64,
    /// RSS after the mode's workload.
    pub final_rss_bytes: u64,
    /// GPU memory attributable to AARON during the mode, when measurable.
    ///
    /// AARON holds no CUDA context, so this is the *whole card's* usage
    /// before and after, and the delta is what a child process caused. `None`
    /// when the card could not be measured.
    pub gpu_delta_bytes: Option<i64>,
    /// Why this mode could not be fully measured, when it could not.
    pub detail: String,
    /// Whether the mode was actually exercised on this machine.
    pub exercised: bool,
}

impl ModeMemory {
    /// A mode that was not available on this machine.
    pub fn unavailable(mode: &str, detail: impl Into<String>) -> ModeMemory {
        ModeMemory {
            mode: mode.into(),
            baseline_rss_bytes: 0,
            peak_rss_bytes: 0,
            final_rss_bytes: 0,
            gpu_delta_bytes: None,
            detail: detail.into(),
            exercised: false,
        }
    }
}

impl MemoryReport {
    /// A single starting point, for a report that only sampled once.
    pub fn baseline() -> MemoryReport {
        let gpu = aaron_resources::probe::gpu_memory();
        let rss = aaron_resources::probe::process_rss_bytes();
        MemoryReport {
            baseline_rss_bytes: rss,
            peak_rss_bytes: rss,
            final_rss_bytes: rss,
            intent_peak_rss_bytes: rss,
            agent_peak_rss_bytes: rss,
            gpu_used_start_bytes: gpu.measured.then_some(gpu.used_bytes),
            gpu_used_end_bytes: gpu.measured.then_some(gpu.used_bytes),
            gpu_total_bytes: gpu.measured.then_some(gpu.total_bytes),
            gpu_detail: gpu.detail,
            modes: Vec::new(),
        }
    }

    /// Human-readable block for the CLI.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str(
            "memory
",
        );
        out.push_str(&format!(
            "  baseline rss      {}
",
            aaron_core::hardware::human_bytes(self.baseline_rss_bytes)
        ));
        out.push_str(&format!(
            "  peak rss          {}
",
            aaron_core::hardware::human_bytes(self.peak_rss_bytes)
        ));
        out.push_str(&format!(
            "  final rss         {}
",
            aaron_core::hardware::human_bytes(self.final_rss_bytes)
        ));
        out.push_str(&format!(
            "  net change        {}
",
            memory_delta(self.baseline_rss_bytes, self.final_rss_bytes)
        ));
        out.push_str(&format!(
            "  intent peak rss   {}
",
            aaron_core::hardware::human_bytes(self.intent_peak_rss_bytes)
        ));
        out.push_str(&format!(
            "  agent peak rss    {}
",
            aaron_core::hardware::human_bytes(self.agent_peak_rss_bytes)
        ));
        match (
            self.gpu_used_start_bytes,
            self.gpu_used_end_bytes,
            self.gpu_total_bytes,
        ) {
            (Some(a), Some(b), Some(t)) => {
                out.push_str(&format!(
                    "  gpu used          {} -> {} of {}
",
                    aaron_core::hardware::human_bytes(a),
                    aaron_core::hardware::human_bytes(b),
                    aaron_core::hardware::human_bytes(t)
                ));
                out.push_str(&format!(
                    "  gpu delta         {}
",
                    memory_delta(a, b)
                ));
            }
            _ => out.push_str(&format!(
                "  gpu               not measured: {}
",
                self.gpu_detail
            )),
        }
        if !self.modes.is_empty() {
            out.push_str("\nper mode (measured separately)\n");
            for mode in &self.modes {
                out.push_str(&format!(
                    "  {:<12} baseline {:>10}  peak {:>10}  final {:>10}  gpu {}\n",
                    mode.mode,
                    aaron_core::hardware::human_bytes(mode.baseline_rss_bytes),
                    aaron_core::hardware::human_bytes(mode.peak_rss_bytes),
                    aaron_core::hardware::human_bytes(mode.final_rss_bytes),
                    match mode.gpu_delta_bytes {
                        Some(delta) => format!("{:>+} B", delta),
                        None => "not measured".to_string(),
                    },
                ));
                if !mode.detail.is_empty() {
                    out.push_str(&format!("               {}\n", mode.detail));
                }
            }
        }
        out
    }
}

impl Report {
    /// Render as a plain-text table.
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        out.push_str("AARON benchmark\n");
        out.push_str(&format!("  measured at   unix {}\n", self.measured_at_unix));
        out.push_str(&format!(
            "  cpu           {}\n",
            self.machine
                .cpu
                .model
                .clone()
                .unwrap_or_else(|| "unknown".into())
        ));
        out.push_str(&format!(
            "  cores         {}\n",
            self.machine.cpu.logical_cores
        ));
        out.push_str(&format!(
            "  memory        {}\n",
            self.machine
                .memory
                .total_bytes
                .map(aaron_core::hardware::human_bytes)
                .unwrap_or_else(|| "unknown".into())
        ));
        out.push_str(&format!(
            "  gpu           {}\n",
            self.machine
                .gpu
                .name
                .clone()
                .unwrap_or_else(|| "none detected".into())
        ));
        out.push_str(&format!("  configuration {}\n\n", self.configuration));
        out.push_str(&format!(
            "{:<34} {:>7} {:>10} {:>10} {:>10} {:>10}\n",
            "measurement", "n", "min", "median", "p95", "max"
        ));
        out.push_str(&format!("{}\n", "-".repeat(84)));
        for s in &self.stats {
            out.push_str(&s.to_row());
            out.push('\n');
        }
        out.push_str(&self.memory.to_text());
        out.push('\n');
        if !self.unavailable.is_empty() {
            out.push_str("not measured\n");
            for u in &self.unavailable {
                out.push_str(&format!("  - {u}\n"));
            }
        }
        out
    }

    /// Render as JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_else(|e| format!("{{\"error\":\"{e}\"}}"))
    }
}

/// Time `f` `iterations` times, returning microsecond samples.
///
/// `warmup` iterations run first and are discarded, so the first-call cost of
/// lazily initialising something (a CUDA context, a page cache miss) does not
/// dominate the median.
pub fn measure<F: FnMut()>(name: &str, iterations: usize, warmup: usize, mut f: F) -> Stats {
    for _ in 0..warmup {
        f();
    }
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_micros() as u64);
    }
    Stats::from_samples(name, samples)
}

/// Measure total wall-clock time for `iterations` full end-to-end runs.
pub fn measure_end_to_end<F: FnMut(usize)>(name: &str, iterations: usize, mut f: F) -> Stats {
    let mut samples = Vec::with_capacity(iterations);
    for i in 0..iterations {
        let start = Instant::now();
        f(i);
        samples.push(start.elapsed().as_micros() as u64);
    }
    Stats::from_samples(name, samples)
}

/// Utterances used by the intent-inference benchmark.
pub const BENCH_UTTERANCES: &[&str] = &[
    "check system status",
    "hey aaron what can you do",
    "remember that I prefer dark mode in the terminal",
    "find files named main.rs in the project",
    "read the file Cargo.toml",
    "show git status",
    "run the tests",
    "open the website https://example.com/docs",
    "launch the browser",
    "kill the terminal",
    "what do you remember about coffee",
    "cancel that",
    "the mitochondria is the powerhouse of the cell",
    "tell me a joke about penguins",
];

/// Measure intent-classification latency for a classifier.
pub fn bench_classifier(
    classifier: &dyn aaron_core::IntentClassifier,
    iterations: usize,
) -> (Stats, Option<Stats>) {
    let total = bench_classifier_total(classifier, iterations);
    let per = Stats::from_samples(
        "intent inference (mean/1 utterance)",
        vec![total.mean_us / iterations.max(1) as u64],
    );
    (per, Some(total))
}

/// Total time to classify every benchmark utterance, once per iteration.
pub fn bench_classifier_total(
    classifier: &dyn aaron_core::IntentClassifier,
    iterations: usize,
) -> Stats {
    measure(
        &format!("intent inference ({} utterances)", BENCH_UTTERANCES.len()),
        iterations,
        5,
        || {
            for text in BENCH_UTTERANCES {
                let _ = classifier.classify(text);
            }
        },
    )
}

/// Measure ASR latency separately from intent classification.
///
/// A backend that is not installed is reported as unavailable rather than being
/// reported as "very fast".
pub fn bench_asr(
    asr: &dyn aaron_core::SpeechRecognizer,
    iterations: usize,
) -> (Stats, Vec<String>) {
    let info = asr.info();
    if !info.ready {
        return (
            Stats::from_samples("speech recognition", Vec::new()),
            vec![format!("{}: {}", info.backend, info.detail)],
        );
    }
    let audio = match aaron_core::MonoPcm::sine(800, 16_000, 220.0) {
        Ok(a) => a,
        Err(e) => {
            return (
                Stats::from_samples("speech recognition", Vec::new()),
                vec![e.to_string()],
            )
        }
    };
    // A tone is not speech; a real backend will report no text, and that is a
    // valid measurement of its fixed cost, not of its accuracy.
    let stats = measure("speech recognition (800 ms of tone)", iterations, 1, || {
        let _ = asr.transcribe(&audio);
    });
    (stats, Vec::new())
}

/// Measure wake-phrase matching latency.
pub fn bench_wake(iterations: usize) -> Stats {
    let config = aaron_core::config::WakeConfig::default();
    let matcher = aaron_wake::TranscriptMatcher::new(&config);
    measure("wake phrase match", iterations, 100, || {
        for text in BENCH_UTTERANCES {
            let _ = aaron_wake::phrase_similarity(&config.phrase, text);
        }
        let _ = matcher.activations();
    })
}

/// Utterances cycled through the end-to-end agent benchmark.
///
/// The agent's repetition guard blocks an identical action after
/// `max_repeat_identical_actions` in a row — correctly, since that is what
/// stops a misheard phrase re-triggering a tool forever. Benchmarking the same
/// string repeatedly would therefore measure the *guard*, not the agent, so
/// the benchmark varies its input.
pub const AGENT_BENCH_UTTERANCES: &[&str] = &[
    "check system status",
    "what can you do",
    "hello there",
    "remember that the bench marker is set",
    "what do you remember about the bench",
    "find files named Cargo.toml",
    "cancel",
    "show git status",
];

/// Measure the end-to-end text path: classify, policy-check and dispatch.
pub fn bench_agent_text(agent: &mut aaron_agent::Agent, iterations: usize) -> Stats {
    measure_end_to_end("end-to-end text turn", iterations, |i| {
        let text = AGENT_BENCH_UTTERANCES[i % AGENT_BENCH_UTTERANCES.len()];
        let _ = agent.handle_text(text);
    })
}

/// Documented false-activation test.
///
/// AARON has no wake-word neural model, so this measures the *transcript*
/// detector against a labelled set: how many non-wake utterances would trigger
/// it. The number is reported honestly, together with the sample size, and it
/// is specific to the detector under test.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WakeFalseActivation {
    pub detector: String,
    pub negatives: usize,
    pub false_activations: usize,
    pub rate: f64,
    pub positives: usize,
    pub true_activations: usize,
}

/// Utterances that must not wake the assistant.
pub const WAKE_NEGATIVES: &[&str] = &[
    "check the weather tomorrow",
    "the build finished successfully",
    "play some music",
    "what time is it",
    "call mum",
    "this is a test of the emergency broadcast system",
    "hey there everyone",
    "arrr",
    "order me a pizza",
    "the cat is on the table",
];

/// Utterances that must wake the assistant.
pub const WAKE_POSITIVES: &[&str] = &[
    "hey aaron",
    "hey aaron check system status",
    "hey, aaron!",
    "hey aaron what can you do",
    "hey aron",
];

/// Run the documented false-activation test.
pub fn wake_false_activation_test(config: &aaron_core::config::WakeConfig) -> WakeFalseActivation {
    let detector = aaron_wake::build(config);
    let threshold = config.match_threshold;
    let false_activations = WAKE_NEGATIVES
        .iter()
        .filter(|t| aaron_wake::phrase_similarity(&config.phrase, t) >= threshold)
        .count();
    let true_activations = WAKE_POSITIVES
        .iter()
        .filter(|t| aaron_wake::phrase_similarity(&config.phrase, t) >= threshold)
        .count();
    WakeFalseActivation {
        detector: detector.name(),
        negatives: WAKE_NEGATIVES.len(),
        false_activations,
        rate: false_activations as f64 / WAKE_NEGATIVES.len().max(1) as f64,
        positives: WAKE_POSITIVES.len(),
        true_activations,
    }
}

/// Resident memory of this process, in bytes, from `/proc/self/statm`.
pub fn process_resident_bytes() -> Option<u64> {
    aaron_resources::probe::process_rss_bytes().checked_sub(0)
}

/// Measure RSS around a workload.
///
/// Returns `(baseline, peak, final)`. `peak` is sampled inside the loop, so a
/// transient spike between two outer samples would be missed; that is a
/// deliberate trade, because sampling on every iteration would itself dominate
/// the workload's own cost.
pub fn measure_memory<F: FnMut(usize)>(iterations: usize, mut body: F) -> (u64, u64, u64) {
    let baseline = aaron_resources::probe::process_rss_bytes();
    let mut peak = baseline;
    for i in 0..iterations {
        body(i);
        let rss = aaron_resources::probe::process_rss_bytes();
        peak = peak.max(rss);
    }
    (baseline, peak, aaron_resources::probe::process_rss_bytes())
}

/// Measure the *slot bookkeeping* around an optional model.
///
/// This times the acquire/release protocol — lock, reference count, release —
/// using a placeholder payload. It is deliberately labelled as such in the
/// report: the number describes AARON's lifecycle code, not a speech model.
/// Model load cost for the intent checkpoint is measured separately by actually
/// decoding a checkpoint, because that is the one model AARON ships.
pub fn measure_slot_bookkeeping(iterations: usize) -> (Stats, Stats) {
    let slot: aaron_resources::ModelSlot<u32, _> =
        aaron_resources::ModelSlot::new("bench", || Ok(1u32), 0);
    let load = measure(
        "model slot acquire (bookkeeping only)",
        iterations.clamp(1, 50),
        1,
        || {
            let _ = slot.acquire().unwrap();
        },
    );
    let unload = measure_end_to_end(
        "model slot release (bookkeeping only)",
        iterations.clamp(1, 50),
        |_| {
            let _ = slot.release_now();
        },
    );
    (load, unload)
}

/// Format a memory delta for the report.
pub fn memory_delta(baseline: u64, current: u64) -> String {
    let delta = current as i64 - baseline as i64;
    let sign = if delta < 0 { "-" } else { "+" };
    format!(
        "{sign}{}",
        aaron_core::hardware::human_bytes(delta.unsigned_abs())
    )
}

/// A short description of what this process is doing, for the report header.
pub fn configuration_summary(config: &aaron_core::Config) -> String {
    format!(
        "brain={} asr={} tts={} wake={} checkpoint={} threshold={} threads={} sample_rate={}",
        match config.brain.backend {
            aaron_core::config::BrainBackend::Auto => "auto",
            aaron_core::config::BrainBackend::Neural => "neural",
            aaron_core::config::BrainBackend::Heuristic => "heuristic",
        },
        format!("{:?}", config.asr.backend).to_lowercase(),
        format!("{:?}", config.tts.backend).to_lowercase(),
        format!("{:?}", config.wake.strategy).to_lowercase(),
        if config.brain.checkpoint_path.is_empty() {
            "auto-discovered"
        } else {
            &config.brain.checkpoint_path
        },
        config.brain.confidence_threshold,
        config.asr.threads,
        config.audio.sample_rate,
    )
}

/// How long the process took to reach a given point.
pub fn elapsed_since(start: Instant) -> Duration {
    start.elapsed()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn statistics_are_computed_from_real_samples() {
        let s = Stats::from_samples("x", vec![10, 20, 30, 40, 50]);
        assert_eq!(s.samples, 5);
        assert_eq!(s.min_us, 10);
        assert_eq!(s.median_us, 30);
        assert_eq!(s.max_us, 50);
        assert_eq!(s.mean_us, 30);
        assert_eq!(s.p95_us, 50);
    }

    #[test]
    fn empty_samples_do_not_divide_by_zero() {
        let s = Stats::from_samples("x", Vec::new());
        assert_eq!(s.samples, 0);
        assert_eq!(s.median_us, 0);
    }

    #[test]
    fn a_single_sample_is_median_and_p95() {
        let s = Stats::from_samples("x", vec![7]);
        assert_eq!(s.median_us, 7);
        assert_eq!(s.p95_us, 7);
    }

    #[test]
    fn percentile_uses_nearest_rank() {
        let s = Stats::from_samples("x", (1..=100).collect());
        assert_eq!(s.median_us, 50);
        assert_eq!(s.p95_us, 95);
        assert_eq!(s.max_us, 100);
    }

    #[test]
    fn measurement_actually_runs_the_closure() {
        let mut count = 0u32;
        let stats = measure("counter", 5, 2, || {
            count += 1;
            std::hint::black_box(1u64);
        });
        assert_eq!(stats.samples, 5);
        assert_eq!(count, 7, "warmup iterations must also run");
    }

    #[test]
    fn warmup_samples_are_discarded() {
        let stats = measure("x", 3, 10, || {});
        assert_eq!(stats.samples, 3, "warmup must not be counted as a sample");
    }

    #[test]
    fn report_renders_every_measurement() {
        let report = Report {
            measured_at_unix: 1_700_000_000,
            machine: aaron_core::hardware::detect_all(),
            configuration: "test".into(),
            stats: vec![Stats::from_samples("intent inference", vec![1, 2, 3])],
            memory: MemoryReport::baseline(),
            unavailable: vec!["speech recognition: no model".into()],
        };
        let text = report.to_text();
        assert!(text.contains("intent inference"));
        assert!(text.contains("not measured"));
        assert!(text.contains("speech recognition: no model"));
        let json = report.to_json();
        assert!(json.contains("\"machine\""));
    }

    #[test]
    fn the_wake_false_activation_test_is_reported_not_claimed() {
        let config = aaron_core::config::WakeConfig::default();
        let r = wake_false_activation_test(&config);
        assert_eq!(r.negatives, WAKE_NEGATIVES.len());
        assert_eq!(r.positives, WAKE_POSITIVES.len());
        assert!(r.rate >= 0.0 && r.rate <= 1.0);
        // With the default 0.72 threshold the shipped negatives must not wake.
        assert_eq!(
            r.false_activations, 0,
            "false activations: {}",
            r.false_activations
        );
        assert!(
            r.true_activations >= 4,
            "true activations: {}",
            r.true_activations
        );
    }

    #[test]
    fn an_unavailable_asr_is_reported_rather_than_timed() {
        let asr = aaron_asr::DisabledAsr;
        let (stats, unavailable) = bench_asr(&asr, 3);
        assert_eq!(stats.samples, 0);
        assert_eq!(unavailable.len(), 1);
        assert!(unavailable[0].contains("disabled"));
    }

    #[test]
    fn benchmark_utterances_cover_the_intent_space() {
        assert!(BENCH_UTTERANCES.len() >= 10);
        assert!(BENCH_UTTERANCES.iter().any(|t| t.contains("cancel")));
        assert!(BENCH_UTTERANCES
            .iter()
            .any(|t| !aaron_core::text::words(t).is_empty()));
    }

    #[test]
    fn classifier_benchmark_measures_something_real() {
        let c = aaron_brain::HeuristicClassifier::default();
        let (per, total) = bench_classifier(&c, 10);
        assert!(per.samples >= 1);
        assert!(total.unwrap().samples == 10);
    }

    #[test]
    fn resident_memory_is_readable_when_procfs_exists() {
        if std::path::Path::new("/proc/self/statm").exists() {
            assert!(process_resident_bytes().unwrap() > 0);
        }
    }

    #[test]
    fn memory_delta_is_signed_and_readable() {
        assert_eq!(memory_delta(1000, 1500), "+500 B");
        assert_eq!(memory_delta(1500, 1000), "-500 B");
        assert_eq!(memory_delta(1000, 1000), "+0 B");
    }

    #[test]
    fn a_memory_report_is_honest_about_a_missing_gpu() {
        let m = MemoryReport::baseline();
        let text = m.to_text();
        assert!(text.contains("baseline rss"));
        assert!(text.contains("peak rss"));
        if m.gpu_total_bytes.is_none() {
            assert!(text.contains("not measured"));
            assert!(!m.gpu_detail.is_empty());
        }
    }

    #[test]
    fn measuring_memory_around_a_workload_reports_a_baseline_and_a_peak() {
        let (baseline, peak, final_rss) = measure_memory(500, |i| {
            let mut v: Vec<u64> = (0..1024).map(|k| (i * k) as u64).collect();
            v.sort();
            std::hint::black_box(&v);
        });
        assert!(baseline > 0);
        assert!(peak >= baseline);
        assert!(final_rss > 0);
    }

    #[test]
    fn a_slot_lifecycle_can_be_measured_and_released() {
        let (load, unload) = measure_slot_bookkeeping(5);
        assert_eq!(load.samples, 5);
        assert_eq!(unload.samples, 5);
    }

    #[test]
    fn the_agent_benchmark_varies_its_input() {
        // Two identical utterances in a row would be blocked by the repetition
        // guard, so the benchmark set must not repeat.
        assert_eq!(
            AGENT_BENCH_UTTERANCES.len(),
            AGENT_BENCH_UTTERANCES
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            "the agent benchmark must not repeat a single utterance adjacently"
        );
        assert!(AGENT_BENCH_UTTERANCES.len() >= 6);
    }

    #[test]
    fn configuration_summary_names_the_backends() {
        let summary = configuration_summary(&aaron_core::Config::default());
        assert!(summary.contains("brain=auto"));
        assert!(summary.contains("asr=auto"));
    }
}
