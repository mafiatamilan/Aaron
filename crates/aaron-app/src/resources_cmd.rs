//! `aaron resources` — inspect budgets, check stability, prove no leak.
//!
//! Every number printed here is measured on the machine at the moment the
//! command ran. Nothing is estimated, interpolated or hard-coded; the one
//! derived figure AARON has (the static footprint of the intent checkpoint) is
//! labelled as derived wherever it appears.

use std::path::Path;

use aaron_core::error::Result;
use aaron_resources::{probe, StabilityVerdict};

use crate::context::AppContext;

/// `aaron resources status`
pub fn status(ctx: &AppContext, json: bool) -> Result<()> {
    let s = ctx.resources.status();
    if json {
        println!(
            "{}",
            serde_json::json!({
                "mode": s.mode.as_str(),
                "explanation": s.explanation,
                "budget": {
                    "usable_system_bytes": s.budget.usable_system_bytes,
                    "usable_gpu_bytes": s.budget.usable_gpu_bytes,
                    "trigger_percent": s.budget.trigger_percent,
                    "recover_percent": s.budget.recover_percent,
                },
                "system": {
                    "total_bytes": s.sample.system.total_bytes,
                    "available_bytes": s.sample.system.available_bytes,
                    "free_percent": s.sample.system.free_percent(),
                },
                "gpu": {
                    "measured": s.sample.gpu.measured,
                    "total_bytes": s.sample.gpu.total_bytes,
                    "used_bytes": s.sample.gpu.used_bytes,
                    "free_percent": s.sample.gpu.free_percent(),
                    "detail": s.sample.gpu.detail,
                },
                "process_rss_bytes": s.sample.process_rss_bytes,
                "degraded": s.degraded,
            })
        );
        return Ok(());
    }

    println!("mode              {}", s.mode.as_str());
    println!("                  {}", s.explanation);
    println!("sample age        {}s", s.age_secs);
    println!();
    println!("system memory");
    println!(
        "  total           {}",
        probe::human_bytes(s.sample.system.total_bytes)
    );
    println!(
        "  available       {}",
        probe::human_bytes(s.sample.system.available_bytes)
    );
    println!(
        "  free            {:.1}%  (low-memory trigger at {}%)",
        s.sample.system.free_percent(),
        s.budget.trigger_percent
    );
    println!(
        "  usable budget   {}   ({} reserved)",
        probe::human_bytes(s.budget.usable_system_bytes),
        probe::human_bytes(ctx.config.resources.reserve_system_ram_mb * 1024 * 1024)
    );
    println!();
    println!("gpu memory");
    if s.sample.gpu.measured {
        println!(
            "  total           {}",
            probe::human_bytes(s.sample.gpu.total_bytes)
        );
        println!(
            "  used            {}  ({:.1}% free)",
            probe::human_bytes(s.sample.gpu.used_bytes),
            s.sample.gpu.free_percent()
        );
        println!(
            "  usable budget   {}   ({} reserved)",
            probe::human_bytes(s.budget.usable_gpu_bytes),
            probe::human_bytes(ctx.config.resources.reserve_gpu_vram_mb * 1024 * 1024)
        );
    } else {
        // The honest statement when VRAM cannot be measured. AARON does not
        // own a CUDA context, so its own VRAM use is whatever a child process
        // such as whisper.cpp allocates.
        println!("  not measurable  {}", s.sample.gpu.detail);
    }
    println!();
    println!("this process");
    println!(
        "  rss             {}",
        probe::human_bytes(s.sample.process_rss_bytes)
    );
    let fp = aaron_resources::DERIVED_STATIC_FOOTPRINT;
    println!(
        "  checkpoint      {} derived ({} parameters) — {}",
        probe::human_bytes(fp.intent_checkpoint_bytes),
        fp.intent_params,
        fp.note
    );
    println!();
    if !s.degraded.is_empty() {
        println!("temporarily degraded");
        for d in &s.degraded {
            println!("  - {}: {}", d.capability, d.reason);
        }
    } else {
        println!("nothing is degraded.");
    }
    Ok(())
}

/// `aaron resources limits`
pub fn limits(ctx: &AppContext) -> Result<()> {
    let r = &ctx.config.resources;
    println!("{:<44} value", "setting");
    println!("{}", "-".repeat(62));
    let row = |k: &str, v: String| println!("{:<44} {}", k, v);
    row("low_memory_mode", r.low_memory_mode.to_string());
    row(
        "monitor_interval_seconds",
        r.monitor_interval_seconds.to_string(),
    );
    row("reserve_system_ram_mb", r.reserve_system_ram_mb.to_string());
    row("reserve_gpu_vram_mb", r.reserve_gpu_vram_mb.to_string());
    row(
        "low_memory_free_percent",
        r.low_memory_free_percent.to_string(),
    );
    row(
        "low_memory_recover_percent",
        r.low_memory_recover_percent.to_string(),
    );
    row(
        "max_audio_buffer_seconds",
        format!(
            "{} ({} KiB at 16 kHz mono f32)",
            r.max_audio_buffer_seconds,
            r.max_audio_buffer_bytes(16_000) / 1024
        ),
    );
    row("max_pending_requests", r.max_pending_requests.to_string());
    row(
        "max_conversation_turns_in_context",
        r.max_conversation_turns_in_context.to_string(),
    );
    row("max_tool_output_bytes", r.max_tool_output_bytes.to_string());
    row("max_transcript_chars", r.max_transcript_chars.to_string());
    row("allow_model_unloading", r.allow_model_unloading.to_string());
    println!();
    println!("resources.models");
    row(
        "  load_asr_on_demand",
        r.models.load_asr_on_demand.to_string(),
    );
    row(
        "  load_tts_on_demand",
        r.models.load_tts_on_demand.to_string(),
    );
    row(
        "  unload_idle_models",
        r.models.unload_idle_models.to_string(),
    );
    row(
        "  idle_unload_seconds",
        r.models.idle_unload_seconds.to_string(),
    );
    println!();
    println!("resources.memory");
    row("  sqlite_cache_kib", r.memory.sqlite_cache_kib.to_string());
    row(
        "  max_retrieved_memories",
        r.memory.max_retrieved_memories.to_string(),
    );
    row(
        "  max_memory_content_bytes",
        r.memory.max_memory_content_bytes.to_string(),
    );
    row(
        "  max_learning_queue_items_in_memory",
        r.memory.max_learning_queue_items_in_memory.to_string(),
    );
    println!();
    println!("measured live");
    row(
        "  sqlite pages on disk",
        format!("{} KiB", ctx.memory()?.on_disk_bytes() / 1024),
    );
    row(
        "  model slots loaded",
        format!("{}", ctx.model_slots_loaded()),
    );
    let (asr, tts) = ctx.model_slot_stats();
    row(
        "  asr slot loads/releases",
        format!(
            "{} / {} ({} unload(s) skipped while in use)",
            asr.loads, asr.unloads, asr.unload_skipped_in_use
        ),
    );
    row(
        "  tts slot loads/releases",
        format!(
            "{} / {} ({} unload(s) skipped while in use)",
            tts.loads, tts.unloads, tts.unload_skipped_in_use
        ),
    );
    Ok(())
}

/// `aaron resources stress`
pub fn stress(ctx: &AppContext, iterations: usize, json: bool) -> Result<i32> {
    let classifier = ctx.classifier.as_ref();
    let utterances = aaron_bench::BENCH_UTTERANCES;

    // 1. Sustained intent classification: the common case, and the one most
    //    likely to leak if the feature extractor or the tensor arena does.
    let classification = aaron_resources::run_stress("intent classification", iterations, |i| {
        let text = utterances[i % utterances.len()];
        if let Ok(cmd) = classifier.classify(text) {
            std::hint::black_box(cmd.intent);
        }
    });

    // 2. Database churn: open, write, query, close, repeatedly.
    //
    //    Deliberately against a throwaway database, never the user's own. A
    //    stress workload that littered the user's memory with thousands of
    //    synthetic facts would be a serious bug, and one that only shows up
    //    after someone runs the tool they were told to run.
    let scratch = std::env::temp_dir().join(format!("aaron-stress-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&scratch);
    let db_path = scratch.clone();
    let database =
        aaron_resources::run_stress("memory database churn", iterations.min(2_000), |_| {
            if let Ok(mut store) =
                aaron_memory::MemoryStore::open(&db_path, aaron_memory::RetentionPolicy::default())
            {
                let _ = store.add(
                    aaron_memory::MemoryKind::Fact,
                    None,
                    "stress-test entry",
                    &[],
                    "stress",
                );
                let _ = store.search("stress", 5);
                let _ = store.prune();
            }
        });
    let _ = std::fs::remove_file(&scratch);
    let _ = std::fs::remove_file(scratch.with_extension("db-wal"));
    let _ = std::fs::remove_file(scratch.with_extension("db-shm"));

    // 3. Allocation churn: does the working set come back?
    let allocation = aaron_resources::run_stress("allocation churn", iterations.min(5_000), |i| {
        let mut v: Vec<String> = (0..64).map(|k| format!("chunk {i}-{k}")).collect();
        v.sort();
        std::hint::black_box(&v);
    });

    // 4. Audio buffer churn: realistic frame sizes through the bounded path.
    let audio = aaron_resources::run_stress("audio buffer churn", iterations.min(5_000), |_| {
        let pcm = aaron_core::MonoPcm::silence(120, 16_000).unwrap();
        std::hint::black_box(pcm.to_wav_bytes());
    });

    // 5. Idle soak: sample the resource manager the way the monitor loop does.
    let manager = ctx.resources.clone();
    let idle = aaron_resources::run_stress("idle resource sampling", iterations.min(2_000), |_| {
        manager.apply(synthetic_healthy_sample());
    });

    let outcomes = [classification, database, allocation, audio, idle];
    let growing: Vec<&aaron_resources::StressOutcome> = outcomes
        .iter()
        .filter(|o| o.verdict == StabilityVerdict::Growing)
        .collect();

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(
                &outcomes.iter().map(|o| o.to_summary()).collect::<Vec<_>>()
            )
            .unwrap_or_default()
        );
    } else {
        println!(
            "stress: {} iterations per workload, RSS sampled throughout",
            iterations
        );
        println!();
        for o in &outcomes {
            print!("{}", aaron_resources::stress::render(o));
        }
        println!();
        println!("RSS tolerance: max(2 MiB, 1% of baseline) between the first and last thirds");
        println!("of the post-warm-up samples. A verdict of `growing` means investigate.");
    }

    if growing.is_empty() {
        println!();
        println!("no workload showed unbounded growth.");
        Ok(0)
    } else {
        eprintln!();
        for o in growing {
            eprintln!(
                "aaron: `{}` is growing: {} -> {} ({} per 1k iterations)",
                o.workload,
                probe::human_bytes(o.baseline_rss_bytes),
                probe::human_bytes(o.final_rss_bytes),
                o.growth_per_1k_iterations
            );
        }
        Ok(1)
    }
}

/// `aaron resources soak`
pub fn soak(ctx: &AppContext, seconds: u32) -> Result<i32> {
    let manager = ctx.resources.clone();
    let classifier = ctx.classifier.as_ref();
    let utterances = aaron_bench::BENCH_UTTERANCES;
    let deadline =
        std::time::Instant::now() + std::time::Duration::from_secs(seconds.max(1) as u64);
    let baseline = probe::process_rss_bytes();
    let mut turns = 0u64;
    let mut samples = 0u64;
    let low_memory_entries_at_start = manager.low_memory_entries();

    while std::time::Instant::now() < deadline {
        for text in utterances.iter().take(4) {
            let _ = classifier.classify(text);
            turns += 1;
        }
        manager.apply(synthetic_healthy_sample());
        samples += 1;
        std::hint::black_box(manager.mode());
        std::thread::sleep(std::time::Duration::from_millis(5));
    }

    let final_rss = probe::process_rss_bytes();
    let grew = final_rss.saturating_sub(baseline);
    let tolerance = (baseline / 20).max(8 * 1024 * 1024);

    println!("soak");
    println!("  duration        {seconds}s");
    println!("  turns           {turns}");
    println!("  samples         {samples}");
    println!("  baseline rss    {}", probe::human_bytes(baseline));
    println!("  final rss       {}", probe::human_bytes(final_rss));
    println!("  growth          {}", probe::human_bytes(grew));
    println!("  tolerance       {}", probe::human_bytes(tolerance));
    println!("  mode changes    {}", manager.mode_changes());
    println!(
        "  low-memory entries {}",
        manager.low_memory_entries() - low_memory_entries_at_start
    );

    if grew > tolerance {
        eprintln!();
        eprintln!(
            "aaron: RSS grew by {} over the soak, past the {} tolerance.",
            probe::human_bytes(grew),
            probe::human_bytes(tolerance)
        );
        Ok(1)
    } else {
        println!();
        println!("RSS is stable within the documented tolerance.");
        Ok(0)
    }
}

/// `aaron resources checkpoint`
pub fn checkpoint(ctx: &AppContext, path: Option<&Path>) -> Result<()> {
    // Resolve first, and explain what is missing. Reading a path that was
    // never going to exist only produces `No such file or directory`, which
    // tells the user nothing about what to do next.
    //
    // An explicit `--path` is always honoured — it names a specific file the
    // user wants inspected, including a candidate that was never promoted. The
    // auto-discovery fallback only applies when no path was given.
    let p = match path {
        Some(explicit) => explicit.to_path_buf(),
        None => ctx.checkpoint()?.ok_or_else(|| {
            aaron_core::error::Error::missing_with_hint(
                "an installed intent checkpoint",
                "run `aaron brain train` to build one. AARON works without it, using the \
                 deterministic router; `aaron resources checkpoint --path <file>` inspects \
                 a specific file",
            )
        })?,
    };
    if !p.is_file() {
        return Err(aaron_core::error::Error::missing_with_hint(
            format!("an intent checkpoint at {}", p.display()),
            "run `aaron brain train` to build one, or point --path at an existing file",
        ));
    }
    let meta = aaron_brain::checkpoint::read_manifest(&p)?;
    let file = std::fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
    println!("checkpoint       {}", p.display());
    println!("  on disk         {}", probe::human_bytes(file));
    println!("  parameters      {}", meta.param_count);
    println!(
        "  dataset         {} (seed {}, {} epochs)",
        meta.dataset_version, meta.seed, meta.epochs
    );
    println!(
        "  note            the in-memory tensor is the same {} of f32; the file is read once and \
         verified, then released",
        probe::human_bytes(aaron_resources::DERIVED_STATIC_FOOTPRINT.intent_checkpoint_bytes)
    );
    Ok(())
}

/// A sample that keeps the manager in normal operation, used by the stress and
/// soak workloads so they exercise the monitoring path without depending on the
/// machine they run on.
fn synthetic_healthy_sample() -> aaron_resources::ResourceSample {
    aaron_resources::synthetic_sample(
        16 * 1024 * 1024 * 1024,
        8 * 1024 * 1024 * 1024,
        0,
        0,
        probe::process_rss_bytes(),
    )
}
