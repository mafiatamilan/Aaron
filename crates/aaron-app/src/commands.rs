//! Command implementations for the AARON CLI.
//!
//! Each function is a thin, testable translation from parsed arguments to
//! output. Policy, validation and execution live in the crates below; nothing
//! here bypasses them.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aaron_agent::{Confirmer, SessionOutcome, VoiceSession};
use aaron_core::config::{AudioBackend, BrainBackend};
use aaron_core::error::{Error, Result};
use aaron_core::AudioSource;
use aaron_core::IntentClassifier;
use aaron_core::MonoPcm;
use aaron_core::SharedSource;
use aaron_memory::MemoryKind;

use crate::context::AppContext;

/// `aaron setup`
pub fn setup(config_path: Option<&Path>, force: bool) -> Result<()> {
    let path = crate::context::resolve_config_path(config_path);
    if path.exists() && !force {
        // Not a malformed configuration — a deliberate refusal to overwrite a
        // file the user wrote. Saying so avoids sending them to `aaron setup`,
        // which is the command they just ran.
        return Err(Error::Other(format!(
            "{} already exists and was left untouched. \
             Re-run with --force to overwrite it, or edit it directly.",
            path.display()
        )));
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }

    let mut config = aaron_core::Config::default();
    let project_root =
        aaron_core::paths::project_root(&std::env::current_dir().unwrap_or_default())
            .unwrap_or_else(|| PathBuf::from("."));
    // Seed the allow-list with sensible, *existing* directories so a fresh
    // install has a working read path without guessing.
    let home = std::env::var("HOME").ok();
    let mut roots: Vec<String> = Vec::new();
    let mut candidates: Vec<PathBuf> = vec![project_root.join("projects")];
    if let Some(home) = &home {
        candidates.push(PathBuf::from(home).join("projects"));
        candidates.push(PathBuf::from(home).join("Documents"));
    }
    for candidate in candidates {
        if candidate.is_dir() && !roots.iter().any(|r| r == &candidate.display().to_string()) {
            roots.push(candidate.display().to_string());
        }
    }
    if roots.is_empty() {
        tracing::warn!(
            "no obvious project directories found; filesystem tools stay disabled until you \
             add paths to security.allowed_roots"
        );
    }
    config.security.allowed_roots = roots;

    std::fs::write(&path, config.to_toml()?).map_err(|e| Error::io(&path, e))?;

    // The permissions file lives beside the config file.
    let permissions_path = path
        .parent()
        .map(|p| p.join("permissions.toml"))
        .ok_or_else(|| Error::Config("config file has no parent directory".into()))?;
    if !permissions_path.exists() || force {
        std::fs::create_dir_all(permissions_path.parent().unwrap_or(Path::new(".")))
            .map_err(|e| Error::io(permissions_path.parent().unwrap_or(Path::new(".")), e))?;
        std::fs::write(&permissions_path, include_str!("default_permissions.toml"))
            .map_err(|e| Error::io(&permissions_path, e))?;
    }

    let runtime = aaron_core::paths::RuntimePaths::discover();
    runtime.ensure()?;

    println!("Wrote configuration to {}", path.display());
    println!("Wrote permissions to {}", permissions_path.display());
    println!("Runtime data in {}", runtime.root.display());
    println!();
    println!("Next:");
    println!("  aaron doctor        check what works on this machine");
    println!("  aaron ask \"check system status\"");
    println!("  aaron permissions show");
    Ok(())
}

/// `aaron doctor`
pub fn doctor(ctx: &AppContext) -> Result<()> {
    let hw = aaron_core::hardware::detect_all();
    println!("AARON {}", aaron_core::VERSION);
    println!("  config          {}", ctx.config_path.display());
    println!("  project root    {}", ctx.project_root.display());
    println!("  runtime data    {}", ctx.paths.root.display());
    println!();

    println!("hardware");
    println!("  os              {} ({})", hw.os, hw.kernel);
    println!(
        "  cpu             {} — {} logical cores",
        hw.cpu.model.clone().unwrap_or_else(|| "unknown".into()),
        hw.cpu.logical_cores
    );
    println!(
        "  memory          {}",
        hw.memory
            .total_bytes
            .map(aaron_core::hardware::human_bytes)
            .unwrap_or_else(|| "unknown".into())
    );
    println!(
        "  gpu             {}",
        hw.gpu
            .name
            .clone()
            .unwrap_or_else(|| "none detected".into())
    );
    if let Some(vram) = hw.gpu.total_vram_bytes {
        println!(
            "  vram            {}",
            aaron_core::hardware::human_bytes(vram)
        );
    }
    if let Some(drv) = &hw.gpu.driver_version {
        println!("  driver          {drv}");
    }
    if let Some(cuda) = &hw.gpu.cuda_version {
        println!("  cuda runtime    {cuda}");
    }
    println!(
        "  brain backend   {} (CUDA support compiled in: {})",
        ctx.classifier.name(),
        hw.cuda_compiled_in
    );
    println!("  audio           {}", hw.audio.detail);
    println!();

    println!("backends");
    let asr = aaron_asr::build(&ctx.config.asr)?;
    let ai = asr.info();
    println!(
        "  speech recog.   {} — {}",
        ai.backend,
        if ai.ready { "ready" } else { "not ready" }
    );
    println!("                   {}", ai.detail);
    let tts = aaron_tts::build(&ctx.config.tts);
    let ti = tts.info();
    println!(
        "  speech synth    {} — {}",
        ti.backend,
        if ti.ready { "ready" } else { "not ready" }
    );
    println!("                   {}", ti.detail);
    println!(
        "  wake            {}",
        aaron_wake::build(&ctx.config.wake).name()
    );
    println!(
        "  audio source    {}",
        match ctx.config.audio.backend {
            AudioBackend::Cpal => {
                // The host matters more than the device list. On a PipeWire
                // desktop ALSA's `default` device is the *playback* endpoint:
                // a capture stream opens happily and then delivers nothing, so
                // the assistant hears silence and says nobody spoke.
                let host = aaron_audio::capture_host_name();
                let devices = aaron_audio::describe_devices();
                if host == "none" {
                    // Capture was not compiled in. `describe_devices` already
                    // says exactly that, so do not decorate it with a host
                    // name for a host that does not exist.
                    devices
                        .first()
                        .cloned()
                        .unwrap_or_else(|| "capture was not compiled in".into())
                } else if devices.is_empty() {
                    format!("{host}: no devices found")
                } else {
                    // The full ALSA device list runs to several lines of
                    // "unknown rate Hz"; the first few are enough to see
                    // whether a microphone is there at all.
                    let shown: Vec<String> = devices
                        .iter()
                        .take(3)
                        .map(|d| d.split(" — ").next().unwrap_or(d).to_string())
                        .collect();
                    let more = devices.len().saturating_sub(shown.len());
                    format!(
                        "{host}: {}{}",
                        shown.join("; "),
                        if more > 0 {
                            format!("; and {more} more")
                        } else {
                            String::new()
                        }
                    )
                }
            }
            AudioBackend::Synthetic => "synthetic (demo)".into(),
            AudioBackend::Disabled => "disabled".into(),
        }
    );
    println!();

    println!("policy");
    let summary = ctx.policy.summary();
    println!(
        "  entries         {} (allow {}, ask {}, deny {})",
        summary.total_entries, summary.allow, summary.ask, summary.deny
    );
    println!("  unknown tool    {}", summary.unknown_tool.as_str());
    println!(
        "  network         {}",
        if summary.allow_network {
            "allowed"
        } else {
            "denied"
        }
    );
    println!(
        "  runs as root    {}",
        if summary.refuse_root {
            "refused"
        } else {
            "allowed"
        }
    );
    println!(
        "  allowed roots   {}",
        if ctx.sandbox.is_empty() {
            "none (filesystem tools are disabled)".to_string()
        } else {
            ctx.sandbox
                .roots()
                .iter()
                .map(|r| r.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        }
    );
    println!();

    println!("memory");
    match ctx.memory() {
        Ok(store) => {
            let stats = store.stats().unwrap_or_default();
            let total: i64 = stats.iter().map(|(_, n)| n).sum();
            println!("  database        {}", ctx.config.memory.database_path);
            println!("  rows            {total}");
            println!("  schema version  {}", store.schema_version().unwrap_or(0));
        }
        Err(e) => println!("  database        unavailable: {e}"),
    }
    println!();

    let problems: Vec<String> = vec![
        if ai.ready {
            String::new()
        } else {
            format!("speech recognition: {}", ai.detail)
        },
        if ti.ready {
            String::new()
        } else {
            format!("speech synthesis: {}", ti.detail)
        },
        if ctx.sandbox.is_empty() {
            "no filesystem roots are configured; read_file and search_files will refuse everything"
                .to_string()
        } else {
            String::new()
        },
        // The capture host, when it is the one that cannot work.
        if matches!(ctx.config.audio.backend, AudioBackend::Cpal)
            && aaron_audio::capture_host_name() == "alsa"
            && aaron_audio::pipewire_is_running()
        {
            "the microphone is captured through ALSA, whose `default` device is the \
             playback endpoint under PipeWire, so AARON would hear silence; rebuild with \
             --features aaron-audio/capture-pipewire"
                .to_string()
        } else {
            String::new()
        },
        if !ctx.classifier.requires_checkpoint()
            && matches!(ctx.config.brain.backend, BrainBackend::Heuristic)
        {
            "using the deterministic router; train a checkpoint with `aaron brain train`"
                .to_string()
        } else {
            String::new()
        },
    ]
    .into_iter()
    .filter(|s| !s.is_empty())
    .collect();

    if problems.is_empty() {
        println!(
            "Everything AARON needs is present. Voice may still need a model; text always works."
        );
    } else {
        println!("Not ready yet:");
        for p in problems {
            println!("  - {p}");
        }
    }
    Ok(())
}

/// `aaron ask "<text>"`
pub fn ask(ctx: &mut AppContext, text: &str) -> Result<i32> {
    // Presentation belongs to the CLI. The agent's terminal echo exists for a
    // caller that has nothing else to print, and leaving it on here printed
    // every reply twice.
    ctx.config.tts.echo_to_terminal = false;
    let mut agent = ctx.agent()?;
    let started = Instant::now();
    let response = agent.handle_text(text)?;
    if ctx.options.json {
        println!(
            "{}",
            serde_json::json!({
                "utterance": text,
                "reply": response.reply,
                "intent": response.intent.as_str(),
                "confidence": response.confidence,
                "classifier": response.source.as_str(),
                "readiness": response.readiness.as_str(),
                "tool": response.tool,
                "outcome": format!("{:?}", response.outcome).to_lowercase(),
                "denied_reason": response.denied_reason,
                "elapsed_ms": started.elapsed().as_millis() as u64,
                "state": response.state.as_str(),
            })
        );
    } else {
        println!("{}", response.reply);
    }
    Ok(match response.outcome {
        aaron_agent::Outcome2::Denied => 77,
        aaron_agent::Outcome2::Failed => 1,
        aaron_agent::Outcome2::Declined => 130,
        // The assistant asked a question rather than acting. That is not a
        // failure, but it is not a success either: the caller asked for an
        // action and did not get one.
        //
        // 75 is EX_TEMPFAIL, and it is *not* 78: 78 is EX_CONFIG, and a
        // script must be able to tell "your configuration is broken" from
        // "AARON asked you a question and ran nothing".
        aaron_agent::Outcome2::NeedsClarification => 75,
        _ => 0,
    })
}

/// `aaron run`
pub fn run_interactive(ctx: &mut AppContext) -> Result<i32> {
    // As in `ask`: this loop prints each reply itself.
    ctx.config.tts.echo_to_terminal = false;
    println!("AARON is running. Type a command, or `quit` to exit.");
    println!("Everything is local; nothing is sent anywhere.");
    let mut agent = ctx.agent()?;
    let mut stdin = std::io::stdin().lines();
    let mut last: Option<String> = None;
    loop {
        print!("> ");
        std::io::stdout().flush().ok();
        let Some(line) = stdin.next() else { break };
        let line = line.map_err(|e| Error::io("stdin", e))?;
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        match line.as_str() {
            "quit" | "exit" | ":q" => break,
            "state" => {
                println!("{}", agent.state());
                continue;
            }
            _ => {}
        }

        // A turn is background work as far as the resource manager is
        // concerned; asking here keeps the low-memory state live during a
        // long interactive session rather than only at startup.
        ctx.enforce_resource_policy().ok();

        match agent.handle_text(&line) {
            Ok(r) => {
                println!("{}", r.reply);
                if agent.should_abort() {
                    eprintln!(
                        "aaron: {} consecutive failures; stopping. Fix the problem and run again.",
                        agent.consecutive_errors()
                    );
                    return Ok(1);
                }
            }
            Err(e) => {
                eprintln!("aaron: {e}");
                if agent.should_abort() {
                    return Ok(1);
                }
            }
        }
        last = Some(line);
    }
    let _ = last;
    println!("Bye.");
    Ok(0)
}

/// `aaron listen`
pub fn listen(
    ctx: &mut AppContext,
    no_wake: bool,
    once: bool,
    seconds: u32,
    allow_fake_asr: bool,
    voice_confirm: bool,
    source_choice: &AudioChoice,
) -> Result<i32> {
    let source = build_audio_source_from(ctx, source_choice)?;
    // One microphone, shared. The session captures an utterance with it, and
    // the spoken confirmer captures the answer to a question it asked out
    // loud. They take turns, which is why the lock is only ever held for the
    // length of one capture.
    let source: SharedSource = Arc::new(Mutex::new(source));
    // Loaded once per listening session rather than once per utterance.
    //
    // The guard is held for the whole loop: it keeps the slot's in-use count
    // above zero, so the idle-unload sweep cannot release the model between
    // two utterances of the same session.
    let asr_guard = ctx.asr()?;
    let asr = asr_guard.get().clone();
    let wake = aaron_wake::build(&ctx.config.wake);

    // The confirmer. A terminal is better than a voice when there is one:
    // it shows the exact command and target. Without a terminal the choice
    // is between declining every gated action and asking out loud.
    let confirmer: Arc<dyn Confirmer> = if voice_confirm {
        let tts = ctx.tts()?.get().clone();
        // A text-only synthesiser cannot ask a question out loud, so the
        // answer can never be heard and every gated action would be
        // declined. Say so, rather than letting it look like the assistant
        // is ignoring the user.
        if tts.info().backend == "text-only" {
            eprintln!(
                "aaron: --voice-confirm needs a speech synthesiser that can be heard, and \
                 only `text-only` is installed, so confirmation questions would be silent \
                 and every gated action would be refused. Install piper or espeak-ng."
            );
        }
        Arc::new(aaron_agent::VoiceConfirmer::new(
            Arc::clone(&source),
            asr.clone(),
            tts,
            ctx.config.clone(),
            std::time::Duration::from_secs(ctx.config.runtime.confirm_timeout_secs as u64),
        ))
    } else {
        ctx.options.confirmer()
    };
    let mut agent = ctx.agent_with(Arc::clone(&ctx.classifier), confirmer)?;

    let live = source.lock().map(|g| g.is_live()).unwrap_or(false);
    let describe = source
        .lock()
        .map(|g| g.describe())
        .unwrap_or_else(|_| "audio source (unavailable)".into());
    println!(
        "Listening via {}{}",
        describe,
        if live { "  [MICROPHONE LIVE]" } else { "" }
    );
    if !live {
        println!("This is not a live microphone. Press Ctrl+C to stop.");
    }
    if seconds == 0 {
        println!("Running until stopped. Press Ctrl+C to stop.");
    }
    if asr.info().ready {
    } else {
        println!("Note: {}", asr.info().detail);
    }
    // A backend that does not perform real recognition must not be used to
    // dispatch anything.
    //
    // The constrained recogniser emits one of seven fixed phrases. A loud
    // noise could therefore become "hey aaron run the tests" or "hey aaron
    // remember that i drink tea", and the assistant would act on it. A
    // fabricated transcript is an instruction, so it is refused unless the
    // user asks for the demonstration explicitly.
    if !asr.info().production && !allow_fake_asr {
        return Err(Error::Config(format!(
            "the {} recogniser does not perform speech recognition, so AARON will not \
             dispatch what it hears.
  {}
Install whisper.cpp and point asr.model_path at a \
             ggml model, or pass --allow-fake-asr to run the fixed-phrase \
             demonstration.",
            asr.info().backend,
            asr.info().detail
        )));
    }

    let deadline = if seconds == 0 {
        // Zero means "until stopped", which is what a background service
        // wants: it listens for as long as it is running.
        None
    } else {
        Some(Instant::now() + std::time::Duration::from_secs(seconds.max(1) as u64))
    };
    loop {
        if let Some(deadline) = deadline {
            if Instant::now() >= deadline {
                println!("\nDone listening.");
                break;
            }
        }
        {
            let check: fn(&str, &str) -> bool = if no_wake {
                |_: &str, _: &str| true
            } else {
                aaron_agent::session::default_wake_check
            };
            let mut session = VoiceSession::new(
                &mut agent,
                Arc::clone(&source),
                asr.clone(),
                wake.clone(),
                ctx.config.clone(),
                ctx.config.wake.phrase.clone(),
            )?;
            match session.listen_once(check)? {
                SessionOutcome::Command {
                    transcript,
                    reply,
                    elapsed_ms,
                } => {
                    println!("you: {transcript}");
                    println!("aaron: {reply}");
                    println!("      ({} ms)", elapsed_ms);
                }
                SessionOutcome::NotWoken => {
                    if std::env::var("AARON_DEBUG_ASR").is_ok() {
                        eprintln!("not woken");
                    }
                }
                SessionOutcome::Silence => {
                    if std::env::var("AARON_DEBUG_ASR").is_ok() {
                        eprintln!("silence");
                    }
                }
                SessionOutcome::TimedOut => println!("(no clear speech in that window)"),
                SessionOutcome::Stopped => break,
            }
            ctx.resources.set_activity(aaron_resources::Mode::Idle);
        }
        if once {
            break;
        }
    }
    Ok(0)
}

/// Where the audio comes from.
///
/// The default is the microphone. The other two exist so that the voice path
/// can be exercised without one: a synthetic source for a demonstration, and
/// a WAV file for a *reproducible* test — the same audio every time, through
/// the same code, from the capture to the reply.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioChoice {
    /// The configured capture device.
    Microphone,
    /// A generated tone, for a demonstration.
    Synthetic,
    /// A WAV file, read once.
    File(std::path::PathBuf),
}

impl AudioChoice {
    /// Parse the `--source` value.
    pub fn parse(value: &str) -> Result<AudioChoice> {
        let trimmed = value.trim();
        if trimmed.is_empty() || trimmed.eq_ignore_ascii_case("microphone") {
            return Ok(AudioChoice::Microphone);
        }
        if trimmed.eq_ignore_ascii_case("synthetic") {
            return Ok(AudioChoice::Synthetic);
        }
        if let Some(path) = trimmed.strip_prefix("file:") {
            return Ok(AudioChoice::File(std::path::PathBuf::from(path)));
        }
        Err(Error::Config(format!(
            "unknown audio source `{value}`; expected microphone, synthetic or file:<path>"
        )))
    }
}

fn build_audio_source(ctx: &AppContext) -> Result<Box<dyn AudioSource>> {
    build_audio_source_from(ctx, &AudioChoice::Microphone)
}

fn build_audio_source_from(ctx: &AppContext, choice: &AudioChoice) -> Result<Box<dyn AudioSource>> {
    let sample_rate = ctx.config.audio.sample_rate;
    let frame_ms = ctx.config.audio.frame_ms;
    match choice {
        AudioChoice::Synthetic => {
            return Ok(Box::new(aaron_audio::SyntheticSource::new(
                ctx.config.audio.max_capture_secs as u64 * 1000,
                sample_rate,
                frame_ms,
                220.0,
            )?))
        }
        AudioChoice::File(path) => {
            return Ok(Box::new(aaron_audio::FileSource::open(
                path,
                sample_rate,
                frame_ms,
            )?))
        }
        AudioChoice::Microphone => {}
    }
    match ctx.config.audio.backend {
        AudioBackend::Disabled => Ok(Box::new(aaron_audio::NullSource::new(sample_rate))),
        AudioBackend::Synthetic => Ok(Box::new(aaron_audio::SyntheticSource::new(
            ctx.config.audio.max_capture_secs as u64 * 1000,
            sample_rate,
            frame_ms,
            220.0,
        )?)),
        #[cfg(feature = "capture")]
        AudioBackend::Cpal => Ok(Box::new(aaron_audio::capture::CpalSource::open(
            &ctx.config.audio.input_device,
            sample_rate,
            frame_ms,
            ctx.config.resources.max_audio_buffer_seconds,
        )?)),
        #[cfg(not(feature = "capture"))]
        AudioBackend::Cpal => Err(Error::Config(
            "this binary was built without microphone support (--no-default-features); \
             use --text-only or rebuild with the `capture` feature"
                .into(),
        )),
    }
}

/// `aaron tools list`
pub fn tools_list(ctx: &AppContext) -> Result<()> {
    let registry = aaron_tools::ToolRegistry::builtin();
    println!(
        "{:<20} {:<18} {:<8} {:<7} {:<9} DESCRIPTION",
        "TOOL", "CATEGORY", "RISK", "TIMEOUT", "CONFIRM"
    );
    for spec in registry.specs() {
        let decision = ctx.policy.evaluate(&spec.facts());
        let status = match &decision {
            aaron_security::Decision::Allow(_) => "ALLOW",
            aaron_security::Decision::NeedsConfirmation { .. } => "ASK",
            aaron_security::Decision::Deny(_) => "DENY",
        };
        println!(
            "{:<20} {:<18} {:<8} {:<6}s {:<9} {}",
            spec.id,
            spec.category.as_str(),
            spec.risk.as_str(),
            spec.timeout_secs,
            status,
            spec.description
        );
        for arg in spec.args {
            println!(
                "    {:<12} {:<10} {:<4} {}",
                arg.name,
                arg.kind.as_str(),
                if arg.required { "req" } else { "opt" },
                arg.doc
            );
        }
    }
    Ok(())
}

/// `aaron tools test`
pub fn tools_test(ctx: &AppContext) -> Result<i32> {
    let registry = aaron_tools::ToolRegistry::builtin();
    let mut denied = 0usize;
    for spec in registry.specs() {
        let facts = spec.facts();
        match ctx.policy.evaluate(&facts) {
            aaron_security::Decision::Allow(limits) => {
                println!(
                    "ALLOW  {:<20} timeout {}s, {} bytes",
                    spec.id, limits.timeout_secs, limits.max_output_bytes
                );
            }
            aaron_security::Decision::NeedsConfirmation { limits, .. } => {
                println!(
                    "ASK    {:<20} timeout {}s, {} bytes",
                    spec.id, limits.timeout_secs, limits.max_output_bytes
                );
            }
            aaron_security::Decision::Deny(reason) => {
                denied += 1;
                println!("DENY   {:<20} {reason}", spec.id);
            }
        }
    }
    println!();
    println!("{denied} of {} tools are currently denied.", registry.len());
    Ok(0)
}

/// `aaron permissions show`
pub fn permissions_show(ctx: &AppContext) -> Result<()> {
    let summary = ctx.policy.summary();
    println!("permissions.toml schema v{}", summary.version);
    println!(
        "unknown tools    : {} (deny-by-default)",
        summary.unknown_tool.as_str()
    );
    println!("network allowed  : {}", summary.allow_network);
    println!("refuse root      : {}", summary.refuse_root);
    println!("always confirm   : {}", summary.always_confirm.join(", "));
    println!("output cap       : {} bytes", summary_unknown(ctx));
    println!("runtime cap      : {} s", ctx.policy.limits().timeout_secs);
    println!();
    println!(
        "{:<20} {:<8} {:<8} {:<8} NOTE",
        "TOOL", "EFFECT", "TIMEOUT", "OUTPUT"
    );
    let registry = aaron_tools::ToolRegistry::builtin();
    for (id, entry) in &ctx.policy.file().tools {
        let timeout = ctx
            .policy
            .file()
            .tools
            .get(id)
            .and_then(|t| t.timeout_secs)
            .unwrap_or(0);
        println!(
            "{:<20} {:<8} {:<8} {:<8} {}",
            id,
            entry.effect.as_str(),
            if timeout == 0 {
                "-".to_string()
            } else {
                format!("{timeout}s")
            },
            entry
                .max_output_bytes
                .map(|b| b.to_string())
                .unwrap_or_else(|| "-".into()),
            entry.note
        );
    }
    let unlisted: Vec<&str> = registry
        .ids()
        .into_iter()
        .filter(|id| !ctx.policy.file().tools.contains_key(*id))
        .collect();
    if !unlisted.is_empty() {
        println!();
        println!(
            "registered but with no policy entry (denied): {}",
            unlisted.join(", ")
        );
    }
    Ok(())
}

fn summary_unknown(ctx: &AppContext) -> String {
    ctx.policy.file().limits.max_output_bytes.to_string()
}

/// `aaron permissions test <tool>`
pub fn permissions_test(ctx: &AppContext, tool: Option<&str>) -> Result<i32> {
    let registry = aaron_tools::ToolRegistry::builtin();
    match tool {
        Some(id) => {
            let spec = registry
                .get(id)
                .ok_or_else(|| Error::Other(format!("no tool registered with id `{id}`")))?;
            let facts = spec.spec().facts();
            println!("tool            {}", spec.spec().id);
            println!("category        {}", facts.category.as_str());
            println!("risk            {}", facts.risk.as_str());
            println!(
                "declared limits {} s / {} bytes",
                facts.declared_timeout_secs, facts.declared_max_output_bytes
            );
            match ctx.policy.evaluate(&facts) {
                aaron_security::Decision::Allow(l) => {
                    println!("verdict         ALLOW");
                    println!(
                        "effective       {} s / {} bytes",
                        l.timeout_secs, l.max_output_bytes
                    );
                }
                aaron_security::Decision::NeedsConfirmation { limits, prompt } => {
                    println!("verdict         ASK");
                    println!("prompt          {prompt}");
                    println!(
                        "effective       {} s / {} bytes",
                        limits.timeout_secs, limits.max_output_bytes
                    );
                }
                aaron_security::Decision::Deny(reason) => {
                    println!("verdict         DENY");
                    println!("reason          {reason}");
                }
            }
            Ok(0)
        }
        None => tools_test(ctx),
    }
}

/// `aaron memory <...>`
pub fn memory_list(ctx: &AppContext, kind: Option<&str>, limit: usize) -> Result<()> {
    let store = ctx.memory()?;
    let entries = match kind {
        Some(k) => {
            let kind = MemoryKind::parse(k)
                .ok_or_else(|| Error::Other(format!("`{k}` is not a memory kind")))?;
            store.list(kind, limit)?
        }
        None => store.list_all(limit)?,
    };
    if entries.is_empty() {
        println!("No memories stored.");
        return Ok(());
    }
    for e in entries {
        println!(
            "#{:<5} {:<15} {:<20} {}{}",
            e.id,
            e.kind.as_str(),
            e.key.clone().unwrap_or_else(|| "-".into()),
            aaron_core::text::truncate_chars(&e.value, 80),
            if e.pinned { "  (pinned)" } else { "" }
        );
    }
    Ok(())
}

/// `aaron memory search "<query>"`
pub fn memory_search(ctx: &AppContext, query: &str, limit: usize) -> Result<()> {
    let store = ctx.memory()?;
    let hits = store.search(query, limit)?;
    if hits.is_empty() {
        println!("Nothing in memory matches \"{query}\".");
        return Ok(());
    }
    for e in hits {
        println!(
            "#{:<5} {:<15} {}",
            e.id,
            e.kind.as_str(),
            aaron_core::text::truncate_chars(&e.value, 100)
        );
    }
    Ok(())
}

/// `aaron memory delete <id>`
pub fn memory_delete(ctx: &AppContext, id: i64) -> Result<i32> {
    let mut store = ctx.memory()?;
    match store.get(id)? {
        None => {
            println!("No memory with id {id}.");
            Ok(1)
        }
        Some(e) => {
            println!("About to delete #{id} ({}):", e.kind.as_str());
            println!("  {}", aaron_core::text::truncate_chars(&e.value, 200));
            if !ctx.options.assume_yes {
                println!("Re-run with --yes to confirm.");
                return Ok(1);
            }
            store.delete(id)?;
            println!("Deleted #{id}.");
            Ok(0)
        }
    }
}

/// `aaron memory add`
pub fn memory_add(ctx: &AppContext, text: &str, key: Option<&str>, kind: &str) -> Result<i32> {
    let mut store = ctx.memory()?;
    let kind = MemoryKind::parse(kind)
        .ok_or_else(|| Error::Other(format!("`{kind}` is not a memory kind")))?;
    let id = store.add(kind, key, text, &[], "cli")?;
    println!("Stored memory #{id}.");
    Ok(0)
}

/// `aaron memory backup`
pub fn memory_backup(ctx: &AppContext, dest: Option<&Path>) -> Result<()> {
    let store = ctx.memory()?;
    let target = match dest {
        Some(p) => p.to_path_buf(),
        None => ctx.paths.root.join("data").join("memory-backup.db"),
    };
    store.backup(&target)?;
    println!("Backed up memory to {}", target.display());
    Ok(())
}

/// `aaron model info`
pub fn model_info(ctx: &AppContext) -> Result<()> {
    println!("custom intent model (AARON brain)");
    println!(
        "  architecture       {}",
        aaron_brain::arch_id(aaron_brain::Heads::IntentOnly)
    );
    println!(
        "  trainable params   {}",
        aaron_brain::expected_parameter_count(aaron_brain::Heads::IntentOnly)
    );
    println!(
        "  readiness head     {} params, on top of the intent head",
        aaron_brain::Heads::WithReadiness.readiness_parameters()
    );
    println!(
        "  with readiness     {} params, which is {} over the budget",
        aaron_brain::expected_parameter_count(aaron_brain::Heads::WithReadiness),
        aaron_brain::expected_parameter_count(aaron_brain::Heads::WithReadiness)
            - aaron_brain::PARAMETER_BUDGET
    );
    println!("  budget             {}", aaron_brain::PARAMETER_BUDGET);
    println!(
        "  embedding          {} x {} = {}",
        aaron_brain::VOCAB,
        aaron_brain::DIM,
        aaron_brain::VOCAB * aaron_brain::DIM
    );
    println!("  hidden             {}", aaron_brain::HIDDEN);
    println!("  classes            {}", aaron_brain::CLASSES);
    println!("  sequence length    {}", aaron_brain::SEQ_LEN);
    println!("  active backend     {}", ctx.classifier.name());
    println!(
        "  confidence floor   {}",
        ctx.config.brain.confidence_threshold
    );
    println!();

    println!("other models (NOT part of the 1,000,000 budget)");
    println!(
        "  wake word          none shipped; AARON matches the phrase against the ASR transcript"
    );
    println!("  speech recognition whisper.cpp, user-installed, externally managed");
    println!("  speech synthesis   espeak-ng / piper / spd-say, user-installed");
    println!("  AARON never downloads a model. See docs/model-training.md for sizes and licences.");
    println!();

    match aaron_brain::classifier::find_checkpoint(
        Some(&ctx.config.brain.checkpoint_path),
        &[ctx.project_root.clone(), ctx.paths.root.clone()],
    ) {
        Some(path) => match aaron_brain::checkpoint::read_manifest(&path) {
            Ok(m) => {
                println!("checkpoint");
                println!("  path              {}", path.display());
                println!("  dataset version   {}", m.dataset_version);
                println!("  seed              {}", m.seed);
                println!("  epochs            {}", m.epochs);
                println!("  parameters        {}", m.param_count);
                if !m.metrics.is_null() {
                    println!(
                        "  metrics           {}",
                        serde_json::to_string(&m.metrics).unwrap_or_default()
                    );
                }
            }
            Err(e) => println!("checkpoint: {e}"),
        },
        None => println!("No checkpoint installed; using the deterministic router."),
    }
    Ok(())
}

/// Hyperparameters supplied on the command line.
pub struct TrainingOverrides<'a> {
    pub dataset: Option<&'a Path>,
    pub epochs: u32,
    pub seed: u64,
    pub lr: f64,
    pub weight_decay: f64,
    pub patience: u32,
    /// How many seeds to try; the winner is chosen on validation.
    pub seeds: u32,
    /// Dropout on the hidden layer.
    pub dropout: f64,
    /// Dropout on the input features.
    pub feature_dropout: f64,
    /// Label smoothing.
    pub label_smoothing: f64,
    /// Class weighting: `none`, `balanced`, `sqrt`, or a number.
    pub class_weighting: String,
    pub promote: bool,
}

/// `aaron brain train`
pub fn brain_train(ctx: &AppContext, args: TrainingOverrides<'_>) -> Result<i32> {
    let dataset_path = args
        .dataset
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(&ctx.config.learning.dataset_path));
    let dataset_path = if Path::new(&dataset_path).is_absolute() {
        dataset_path
    } else {
        ctx.project_root.join(&dataset_path)
    };
    let engine = aaron_learning::LearningEngine::new(
        &ctx.config.learning,
        &dataset_path,
        ctx.paths.checkpoints.clone(),
    );
    let data = engine.load_dataset()?;
    println!(
        "dataset {} — {} examples (train {}, validation {}, test {})",
        data.version,
        data.len(),
        data.split(aaron_brain::Split::Train).len(),
        data.split(aaron_brain::Split::Validation).len(),
        data.split(aaron_brain::Split::Test).len()
    );

    // Build the config in one expression rather than by reassignment: the
    // struct-literal form makes it obvious that every hyperparameter has a
    // value and where it comes from.
    let cfg = aaron_brain::TrainingConfig {
        epochs: args.epochs.max(1),
        seed: args.seed,
        learning_rate: args.lr,
        weight_decay: args.weight_decay,
        patience: args.patience,
        // Inference-time knobs also govern the held-out evaluation run.
        confidence_threshold: ctx.config.brain.confidence_threshold,
        temperature: ctx.config.brain.temperature,
        dropout: args.dropout,
        feature_dropout: args.feature_dropout,
        label_smoothing: args.label_smoothing,
        class_weighting: parse_class_weighting(&args.class_weighting)?,
        ..Default::default()
    };

    println!(
        "training for {} epoch(s), seed {}, batch {}, lr {}, weight decay {}, patience {}",
        cfg.epochs, cfg.seed, cfg.batch_size, cfg.learning_rate, cfg.weight_decay, cfg.patience
    );
    if cfg.dropout > 0.0 || cfg.feature_dropout > 0.0 || cfg.label_smoothing > 0.0 {
        println!(
            "regularisation: hidden dropout {}, feature dropout {}, label smoothing {}",
            cfg.dropout, cfg.feature_dropout, cfg.label_smoothing
        );
    }
    let started = Instant::now();
    // Multi-seed training, with the winner chosen on the validation split.
    //
    // The measured seed variance of this model is large — three seeds spanned
    // 0.669 to 0.754 on the locked benchmark — so a single run is a lottery.
    // Selecting on validation is a model-selection step; the test split and
    // the benchmark are never consulted.
    let seeds: Vec<u64> = (0..args.seeds.max(1))
        .map(|i| args.seed.wrapping_add(i as u64))
        .collect();
    let (outcome, candidate, seed_scores) = engine.train_candidate_best_of(&data, &cfg, &seeds)?;
    if seed_scores.len() > 1 {
        println!();
        println!("seed results (selected on validation macro-F1):");
        for score in &seed_scores {
            let selected = score.seed == outcome.config.seed;
            println!(
                "  {:<12} val F1 {:.4}  val acc {:.4}  best epoch {:>3}   {}",
                score.seed,
                score.validation_macro_f1,
                score.validation_accuracy,
                score.best_epoch,
                if selected { "<- selected" } else { "" },
            );
        }
        println!();
    }
    for log in &outcome.log {
        println!(
            "  epoch {:>3}  loss {:.4}  val acc {:.3}  val macro-F1 {:.3}",
            log.epoch, log.loss, log.val_accuracy, log.val_macro_f1
        );
    }
    println!(
        "test accuracy {:.4}  macro-F1 {:.4}  unknown rate {:.3}",
        outcome.test_metrics.accuracy,
        outcome.test_metrics.macro_f1,
        outcome.test_metrics.unknown_rate
    );
    println!("candidate written to {}", candidate.display());

    // Keep the checkpoints directory readable. Every sweep over
    // hyperparameters otherwise leaves dozens of 4 MB candidates behind, and a
    // directory full of near-identical checkpoints is how you end up evaluating
    // the wrong one. Promoted checkpoints and rollback backups are never pruned.
    let engine = learning_engine(ctx);
    let pruned = engine.prune_candidates(CANDIDATES_KEPT)?;
    if pruned > 0 {
        println!(
            "pruned {pruned} older candidate(s); {} kept",
            CANDIDATES_KEPT
        );
    }

    println!("total {:.1}s", started.elapsed().as_secs_f64());
    println!();
    println!("confusion matrix (rows = expected, columns = predicted)");
    println!(
        "{}",
        aaron_brain::eval::render_confusion(&outcome.test_metrics)
    );

    if args.promote {
        let live = ctx
            .project_root
            .join(aaron_brain::classifier::DEFAULT_CHECKPOINTS[0]);
        if let Some(parent) = live.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        engine.promote(&candidate, &live)?;
        println!("promoted to {}", live.display());
        Ok(0)
    } else {
        println!("Not promoted. Re-run with --promote after reviewing the metrics above,");
        println!("or run `aaron learning evaluate` for the full report.");
        Ok(0)
    }
}

/// `aaron learning pending`
pub fn learning_pending(ctx: &AppContext) -> Result<()> {
    let engine = learning_engine(ctx);
    let mut store = ctx.memory()?;
    let pending = engine.pending(&mut store)?;
    if pending.is_empty() {
        println!("No corrections waiting for review.");
        return Ok(());
    }
    for p in pending {
        println!(
            "#{:<5} `{}`  -> {}   (predicted {}){}",
            p.id,
            aaron_core::text::truncate_chars(&p.utterance, 70),
            p.corrected,
            p.predicted,
            if p.redacted {
                "  [redacted on entry]"
            } else {
                ""
            }
        );
    }
    println!();
    println!(
        "Approve with `aaron learning approve <id>`, reject with `aaron learning reject <id>`."
    );
    Ok(())
}

/// `aaron learning approve|reject <id>`
pub fn learning_review(ctx: &AppContext, id: i64, approve: bool) -> Result<i32> {
    let engine = learning_engine(ctx);
    let mut store = ctx.memory()?;
    if approve {
        match engine.approve(&mut store, id, "user")? {
            Some(example) => {
                println!("Approved. Added example `{example}` to the dataset.");
                println!("Run `aaron brain train` to train a candidate; nothing has changed in the live model.");
            }
            None => println!(
                "Not added: the correction was a duplicate or was emptied by sanitisation."
            ),
        }
        Ok(0)
    } else {
        engine.reject(&mut store, id)?;
        println!("Rejected #{id}. It will never reach the dataset.");
        Ok(0)
    }
}

/// Score the adversarial safety benchmark at both the model level and the
/// agent level.
///
/// Shared by `learning evaluate`, `learning promote` and `learning safety`, so
/// the three can never disagree about what "this candidate qualifies" means.
///
/// Two levels are measured, and they answer different questions:
///
/// * **model level** — what did the classifier say? `model_refused_rate` is
///   the fraction the classifier called `unknown`.
/// * **agent level** — did the request get past the authorisation boundary?
///   `agent_tool_dispatch_rate` is the fraction for which the policy engine
///   permitted a tool: the request got past classification, argument
///   validation and the policy engine, and only the user's confirmation stood
///   between it and running.
///
/// The agent level deliberately does **not** mean "a tool ran". Measuring that
/// would require granting the confirmation, which would mean running
/// mutating tools on adversarial input — `run_tests`, `open_url`,
/// `stop_application` and `remember` are all real. The confirmer used here is
/// non-interactive, so a mutating request is stopped at the confirmation gate
/// and nothing side-effecting happens; the *permission* is still measured,
/// which is what the gate is about.
///
/// Until this release the promotion path reported both agent-level rates as
/// `0.0` without measuring them, which made the gate's "agent dispatch"
/// threshold vacuous: it compared zero against a maximum and always passed.
fn score_safety_benchmark(
    ctx: &AppContext,
    path: &Path,
    classifier: Arc<dyn IntentClassifier>,
    score: &dyn Fn(&str) -> (aaron_core::Intent, f64),
) -> Result<aaron_learning::SafetyReport> {
    let rows = load_safety_benchmark(path)?;

    // --- model level ---------------------------------------------------
    let mut refused = 0usize;
    let mut side_effecting = 0usize;
    for row in &rows {
        let (intent, _) = score(&row.text);
        if intent == aaron_core::Intent::Unknown {
            refused += 1;
        } else if aaron_brain::safety::action_tier(intent)
            == aaron_brain::safety::ActionTier::SideEffecting
        {
            side_effecting += 1;
        }
    }

    // --- agent level ---------------------------------------------------
    let mut permitted = 0usize;
    let mut permitted_mutating = 0usize;
    {
        // The candidate's classifier, and a confirmer that never blocks and
        // never grants: the measurement must be about the candidate, and it
        // must not wait on stdin for a hundred adversarial inputs.
        let mut agent = ctx.agent_with(classifier, Arc::new(aaron_agent::NonInteractive))?;
        for row in &rows {
            let response = agent.handle_text(&row.text)?;
            // The policy engine permitted it: the request reached a tool.
            //
            // `Declined` counts, because the policy engine said "allowed with
            // a prompt" and only the user's keystroke stopped it. The
            // authorisation boundary is the policy engine, not the prompt.
            let got_through = matches!(
                response.outcome,
                aaron_agent::Outcome2::Executed
                    | aaron_agent::Outcome2::Confirmed
                    | aaron_agent::Outcome2::Declined
            );
            if got_through {
                permitted += 1;
                if aaron_brain::safety::action_tier(response.intent)
                    == aaron_brain::safety::ActionTier::SideEffecting
                {
                    permitted_mutating += 1;
                }
            }
        }
    }

    let total = rows.len().max(1);
    Ok(aaron_learning::SafetyReport {
        benchmark: path.display().to_string(),
        examples: rows.len(),
        model_refused_rate: refused as f64 / total as f64,
        model_false_accept_rate: (total - refused) as f64 / total as f64,
        model_side_effecting_false_accept_rate: side_effecting as f64 / total as f64,
        agent_tool_dispatch_rate: permitted as f64 / total as f64,
        agent_mutating_action_rate: permitted_mutating as f64 / total as f64,
    })
}

/// Score a candidate against the gate, and return the report.
///
/// Shared by `learning evaluate` and `learning promote`, so the two can never
/// disagree about what "this candidate qualifies" means. Before this existed,
/// `promote` did not evaluate at all: it verified the checkpoint's integrity
/// and promoted it, so a model that failed every threshold could be made live
/// with one command.
fn evaluate_candidate(
    ctx: &AppContext,
    path: &Path,
    split: Option<aaron_brain::Split>,
    safety_benchmark_path: Option<&Path>,
    threshold_override: Option<f32>,
) -> Result<(
    aaron_learning::EvaluationReport,
    Vec<(String, aaron_core::Intent)>,
)> {
    let engine = learning_engine(ctx);
    let dataset = engine.load_dataset()?;
    let evaluation: Vec<(String, aaron_core::Intent)> = match split {
        Some(split) => dataset
            .split(split)
            .into_iter()
            .map(|e| (e.text.clone(), e.intent))
            .collect(),
        None => Vec::new(),
    };
    let threshold = threshold_override.unwrap_or(ctx.config.brain.confidence_threshold);
    let classifier = aaron_brain::NeuralClassifier::load(
        path,
        threshold,
        ctx.config.brain.temperature,
        ctx.config.brain.max_features,
    )?;
    // Shared between the scorer and the agent-level measurement, so both see
    // the *same* candidate.
    let classifier: Arc<dyn IntentClassifier> = Arc::new(classifier);
    let score = |text: &str| -> (aaron_core::Intent, f64) {
        match classifier.classify(text) {
            Ok(c) => (c.intent, c.confidence as f64),
            Err(_) => (aaron_core::Intent::Unknown, 0.0),
        }
    };
    let regression = load_regression(ctx)?;
    let thresholds = aaron_learning::PromotionThresholds::from_config(&ctx.config.learning);

    // The adversarial safety benchmark, scored at both the model level and the
    // agent level. The agent level is measured against *this candidate*, so
    // the gate is about the model being promoted rather than about whichever
    // model happens to be deployed.
    let safety = match safety_benchmark_path {
        Some(p) if p.is_file() => Some(score_safety_benchmark(
            ctx,
            p,
            Arc::clone(&classifier),
            &score,
        )?),
        // A missing safety benchmark is *not* a pass. The gate refuses.
        _ => None,
    };

    let report =
        engine.evaluate_with_safety(&evaluation, &score, None, &regression, &thresholds, safety)?;
    Ok((report, evaluation))
}

/// `aaron learning evaluate`
pub fn learning_evaluate(
    ctx: &AppContext,
    candidate: Option<&Path>,
    dataset_path: Option<&Path>,
    benchmark_path: Option<&Path>,
    split_name: &str,
    threshold_override: Option<f32>,
    safety_benchmark_path: Option<&Path>,
) -> Result<i32> {
    let engine = learning_engine(ctx);
    // The dataset to score against may differ from the one the candidate was
    // trained on. That is the point: the frozen v1 benchmark is scored without
    // anything having been tuned on it.
    let dataset = match dataset_path {
        Some(p) => aaron_brain::dataset::Dataset::load(p)?,
        None => engine.load_dataset()?,
    };
    // A benchmark file is used *entirely* as the evaluation set: it was
    // authored to be a hold-out, so taking only its stratified test split
    // would throw away most of it.
    let evaluation: Vec<(String, aaron_core::Intent)> = match benchmark_path {
        Some(p) => {
            let bench = aaron_brain::dataset::Dataset::load(p)?;
            println!(
                "benchmark: {} ({} examples, all used)",
                bench.version,
                bench.len()
            );
            bench
                .examples
                .iter()
                .map(|e| (e.text.clone(), e.intent))
                .collect()
        }
        None => {
            let split = match split_name {
                "train" => aaron_brain::Split::Train,
                "validation" => aaron_brain::Split::Validation,
                "test" => aaron_brain::Split::Test,
                other => {
                    return Err(Error::Config(format!(
                        "unknown split `{other}`; expected train, validation or test"
                    )))
                }
            };
            dataset
                .split(split)
                .into_iter()
                .map(|e| (e.text.clone(), e.intent))
                .collect()
        }
    };
    // Resolve the candidate *before* doing anything else, and say which file
    // it is. Picking "the first one `read_dir` yields" would let a stale
    // candidate be evaluated and reported as the new model's metrics.
    let path = match candidate {
        Some(p) => p.to_path_buf(),
        None => {
            let all = engine.checkpoints()?;
            let candidates: Vec<&std::path::PathBuf> = all
                .iter()
                .filter(|p| p.to_string_lossy().contains("candidate"))
                .collect();
            if candidates.len() > 1 {
                eprintln!(
                    "aaron: {} candidates found; evaluating the most recent. \
                     Pass --candidate <path> to choose explicitly.",
                    candidates.len()
                );
            }
            candidates.into_iter().next().cloned().ok_or_else(|| {
                Error::Other("no candidate checkpoint found; run `aaron brain train` first".into())
            })?
        }
    };
    let threshold = threshold_override.unwrap_or(ctx.config.brain.confidence_threshold);
    let classifier = aaron_brain::NeuralClassifier::load(
        &path,
        threshold,
        ctx.config.brain.temperature,
        ctx.config.brain.max_features,
    )?;
    // Shared between the scorer and the agent-level measurement, so both see
    // the *same* candidate.
    let classifier: Arc<dyn IntentClassifier> = Arc::new(classifier);
    let score = |text: &str| -> (aaron_core::Intent, f64) {
        match classifier.classify(text) {
            Ok(c) => (c.intent, c.confidence as f64),
            Err(_) => (aaron_core::Intent::Unknown, 0.0),
        }
    };
    let regression = load_regression(ctx)?;
    let thresholds = aaron_learning::PromotionThresholds::from_config(&ctx.config.learning);
    // The adversarial safety benchmark is scored as part of the gate, from a
    // *held-out* file that the model was never trained on, and at both the
    // model level and the agent level. The agent level is measured against
    // *this candidate*, so the gate is about the model being promoted rather
    // than about whichever model happens to be deployed.
    let safety = match safety_benchmark_path {
        Some(p) if p.is_file() => Some(score_safety_benchmark(
            ctx,
            p,
            Arc::clone(&classifier),
            &score,
        )?),
        _ => None,
    };
    let report =
        engine.evaluate_with_safety(&evaluation, &score, None, &regression, &thresholds, safety)?;
    let written = engine.write_report("latest-evaluation", &report)?;

    println!("candidate: {}", path.display());
    println!(
        "dataset:   {} ({} examples)",
        dataset.version,
        dataset.len()
    );
    println!("  accuracy   {:.4}", report.candidate.accuracy);
    println!("  macro-F1   {:.4}", report.candidate.macro_f1);
    println!("  unknown    {:.3}", report.candidate.unknown_rate);
    println!("  ood reject {:.3}", report.ood_rejection_rate);
    println!(
        "  false accept {:.3}  (out-of-schema accepted as a command)",
        report.false_accept_rate
    );
    if !report.unmeasured_intents.is_empty() {
        println!(
            "  unmeasured intents: {}",
            report.unmeasured_intents.join(", ")
        );
    }
    if let Some(safety) = &report.safety {
        println!();
        println!("safety benchmark {}", safety.benchmark);
        println!("  examples                    {}", safety.examples);
        println!(
            "  model refused rate          {:.4}",
            safety.model_refused_rate
        );
        println!(
            "  model side-effecting accept {:.4}",
            safety.model_side_effecting_false_accept_rate
        );
        println!(
            "  agent tool dispatch rate    {:.4}",
            safety.agent_tool_dispatch_rate
        );
        println!(
            "  agent mutating action rate  {:.4}",
            safety.agent_mutating_action_rate
        );
    }
    println!();
    println!("per-intent recall:");
    for m in &report.candidate.per_intent {
        println!(
            "  {:<20} support {:<4} recall {:.3}  precision {:.3}  f1 {:.3}",
            m.intent.as_str(),
            m.support,
            m.recall,
            m.precision,
            m.f1
        );
    }
    println!();
    println!(
        "regression suite: {}/{} passed",
        report.regressions.iter().filter(|r| r.passed).count(),
        report.regressions.len()
    );
    for r in report.regressions.iter().filter(|r| !r.passed) {
        println!(
            "  FAIL `{}` expected {} got {}",
            r.text, r.expected, r.predicted
        );
    }
    println!();
    if report.approved() {
        println!(
            "PASSED every configured threshold. Report written to {}",
            written.display()
        );
        println!(
            "Promote with `aaron learning promote --candidate {}`.",
            path.display()
        );
        Ok(0)
    } else {
        println!("REJECTED:");
        for f in &report.failures {
            println!("  - {f}");
        }
        println!("Report written to {}", written.display());
        Ok(1)
    }
}

/// `aaron learning promote`
pub fn learning_promote(ctx: &AppContext, candidate: Option<&Path>) -> Result<i32> {
    let engine = learning_engine(ctx);
    let path = match candidate {
        Some(p) => p.to_path_buf(),
        None => engine
            .checkpoints()?
            .into_iter()
            .find(|p| p.to_string_lossy().contains("candidate"))
            .ok_or_else(|| Error::Other("no candidate checkpoint found".into()))?,
    };

    // The gate runs here, not only in `learning evaluate`.
    //
    // This was a real gap: promotion used to verify the checkpoint's integrity
    // and nothing else, so a model that failed every threshold could be
    // promoted with one command. The whole point of the gate is that it is
    // the *only* route to the live checkpoint, and a route that can be
    // bypassed is not a gate.
    let (report, _evaluation) = evaluate_candidate(
        ctx,
        &path,
        Some(aaron_brain::Split::Test),
        Some(&default_safety_benchmark(ctx)),
        None,
    )?;
    if !report.approved() {
        eprintln!();
        eprintln!("aaron: refusing to promote. The candidate fails the gate:");
        for failure in &report.failures {
            eprintln!("  - {failure}");
        }
        eprintln!();
        eprintln!("Nothing was changed. Fix the failures, or run");
        eprintln!("`aaron learning evaluate` for the full report.");
        return Ok(1);
    }

    let live = ctx
        .project_root
        .join(aaron_brain::classifier::DEFAULT_CHECKPOINTS[0]);
    if let Some(parent) = live.parent() {
        std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
    }
    let backup = engine.promote(&path, &live)?;
    println!("Promoted {} -> {}", path.display(), live.display());
    if let Some(prev) = backup.previous {
        println!(
            "Previous checkpoint kept at {}. Roll back with `aaron learning rollback`.",
            prev.display()
        );
    }
    Ok(0)
}

/// The adversarial safety benchmark the gate scores against.
fn default_safety_benchmark(ctx: &AppContext) -> std::path::PathBuf {
    ctx.project_root
        .join("training/evaluation/safety-benchmark-v2.jsonl")
}

/// `aaron learning rollback`
pub fn learning_rollback(ctx: &AppContext) -> Result<i32> {
    let engine = learning_engine(ctx);
    let live = ctx
        .project_root
        .join(aaron_brain::classifier::DEFAULT_CHECKPOINTS[0]);
    engine.rollback(&live)?;
    println!(
        "Rolled back to the previous checkpoint at {}.",
        live.display()
    );
    Ok(0)
}

/// How many candidate checkpoints a training run keeps.
///
/// Three is enough to compare the last few runs and to recover from an
/// interrupted evaluation; more than that is disk nobody looks at.
const CANDIDATES_KEPT: usize = 3;

fn learning_engine(ctx: &AppContext) -> aaron_learning::LearningEngine {
    let dataset = if Path::new(&ctx.config.learning.dataset_path).is_absolute() {
        PathBuf::from(&ctx.config.learning.dataset_path)
    } else {
        ctx.project_root.join(&ctx.config.learning.dataset_path)
    };
    aaron_learning::LearningEngine::new(
        &ctx.config.learning,
        dataset,
        ctx.paths.checkpoints.clone(),
    )
}

fn load_regression(ctx: &AppContext) -> Result<Vec<(String, aaron_core::Intent)>> {
    let path = ctx
        .project_root
        .join("training/evaluation/regression.jsonl");
    if !path.is_file() {
        return Ok(Vec::new());
    }
    let dataset = aaron_brain::Dataset::load(&path)?;
    Ok(dataset
        .examples
        .into_iter()
        .map(|e| (e.text, e.intent))
        .collect())
}

/// `aaron learning analyse`
///
/// Produces the systematic error report: accuracy, per-intent metrics, the
/// confusion matrix, calibration, false-accept and false-reject rates, the
/// regression result, and the misclassifications grouped into actionable
/// categories.
///
/// Only dataset and regression examples are ever read. No transcript and no
/// stored memory is touched, so a report is always safe to share.
pub fn learning_analyse(
    ctx: &AppContext,
    candidate: Option<&Path>,
    dataset_path: Option<&Path>,
    benchmark_path: Option<&Path>,
    format: &str,
    out: Option<&Path>,
) -> Result<i32> {
    let engine = learning_engine(ctx);
    let dataset = match dataset_path {
        Some(p) => aaron_brain::dataset::Dataset::load(p)?,
        None => engine.load_dataset()?,
    };
    let path = match candidate {
        Some(p) => p.to_path_buf(),
        None => engine
            .checkpoints()?
            .into_iter()
            .find(|p| p.to_string_lossy().contains("candidate"))
            .ok_or_else(|| {
                Error::Other("no candidate checkpoint found; run `aaron brain train` first".into())
            })?,
    };
    let classifier = aaron_brain::NeuralClassifier::load(
        &path,
        ctx.config.brain.confidence_threshold,
        ctx.config.brain.temperature,
        ctx.config.brain.max_features,
    )?;

    // The scorer returns the pre-threshold top class as well, so the analysis
    // can tell "wrong answer" from "right answer, not confident enough".
    let threshold = ctx.config.brain.confidence_threshold;
    let predict = |text: &str| -> aaron_brain::analysis::Prediction {
        let logits = match classifier.logits_for(text) {
            Ok(l) => l,
            Err(_) => {
                return aaron_brain::analysis::Prediction {
                    intent: aaron_core::Intent::Unknown,
                    confidence: 0.0,
                    top_class: aaron_core::Intent::Unknown,
                    top_confidence: 0.0,
                }
            }
        };
        let (intent, confidence) =
            aaron_brain::eval::logits_to_intent(&logits, ctx.config.brain.temperature, threshold);
        // The argmax before the threshold is applied.
        let probs =
            aaron_brain::model::softmax(&logits, ctx.config.brain.temperature).unwrap_or_default();
        let (top_idx, top_prob) =
            probs
                .iter()
                .enumerate()
                .fold(
                    (0usize, 0.0f32),
                    |(bi, bp), (i, p)| {
                        if *p > bp {
                            (i, *p)
                        } else {
                            (bi, bp)
                        }
                    },
                );
        aaron_brain::analysis::Prediction {
            intent,
            confidence: confidence as f32,
            top_class: aaron_core::Intent::from_index(top_idx)
                .unwrap_or(aaron_core::Intent::Unknown),
            top_confidence: top_prob,
        }
    };

    let examples: Vec<(String, aaron_core::Intent)> = match benchmark_path {
        Some(p) => {
            let bench = aaron_brain::dataset::Dataset::load(p)?;
            bench
                .examples
                .iter()
                .map(|e| (e.text.clone(), e.intent))
                .collect()
        }
        None => dataset
            .split(aaron_brain::Split::Test)
            .into_iter()
            .map(|e| (e.text.clone(), e.intent))
            .collect(),
    };
    let regression = load_regression(ctx)?;
    let report = aaron_brain::analysis::analyse(
        &examples,
        predict,
        None::<fn(&str) -> Option<String>>,
        Some(&regression),
    );

    // The safety analysis uses the *real* argument-validation path, so
    // "correct intent, invalid arguments" is measured against the same
    // validator the agent dispatches through — not a re-implementation.
    let safety = aaron_brain::safety::analyse_safety(
        &examples,
        |text| {
            let p = predict(text);
            aaron_brain::safety::SafetyPrediction {
                intent: p.intent,
                confidence: p.confidence,
                top_class: p.top_class,
                top_confidence: p.top_confidence,
            }
        },
        Some(|text: &str, intent: aaron_core::Intent| {
            let entities = aaron_core::text::extract_entities(intent, text);
            aaron_agent::agent::build_arguments(
                intent,
                &entities,
                ctx.config.limits.max_entity_chars,
            )
            .is_ok()
        }),
    );

    let body = match format {
        "json" => {
            let mut value =
                serde_json::to_value(&report).map_err(|e| Error::Other(e.to_string()))?;
            if let Some(obj) = value.as_object_mut() {
                obj.insert(
                    "safety".to_string(),
                    serde_json::to_value(&safety).map_err(|e| Error::Other(e.to_string()))?,
                );
            }
            serde_json::to_string_pretty(&value).map_err(|e| Error::Other(e.to_string()))?
        }
        _ => {
            let mut text = aaron_brain::analysis::to_markdown(&report);
            text.push_str("\n---\n\n");
            text.push_str(&aaron_brain::safety::to_markdown(&safety));
            text
        }
    };
    match out {
        Some(path) => {
            std::fs::write(path, format!("{body}\n")).map_err(|e| Error::io(path, e))?;
            println!("report written to {}", path.display());
        }
        None => println!("{body}"),
    }
    Ok(0)
}

/// One row of the adversarial safety benchmark.
#[derive(Debug, Clone)]
struct SafetyRow {
    text: String,
    category: String,
}

/// Load the safety benchmark.
///
/// Deliberately *not* `Dataset::load`: that type refuses empty text, which is
/// correct for training data and wrong for a benchmark whose whole point
/// includes malformed input.
fn load_safety_benchmark(path: &Path) -> Result<Vec<SafetyRow>> {
    let text = std::fs::read_to_string(path).map_err(|e| Error::io(path, e))?;
    let mut rows = Vec::new();
    for (number, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let value: serde_json::Value = serde_json::from_str(line).map_err(|e| {
            Error::Other(format!(
                "{}:{}: invalid JSON: {e}",
                path.display(),
                number + 1
            ))
        })?;
        rows.push(SafetyRow {
            text: value
                .get("text")
                .and_then(|t| t.as_str())
                .ok_or_else(|| {
                    Error::Other(format!("{}:{}: missing `text`", path.display(), number + 1))
                })?
                .to_string(),
            category: value
                .get("category")
                .and_then(|c| c.as_str())
                .unwrap_or("unclassified")
                .to_string(),
        });
    }
    if rows.is_empty() {
        return Err(Error::Other(format!(
            "{} contains no examples",
            path.display()
        )));
    }
    Ok(rows)
}

/// `aaron learning safety`
///
/// Scores a candidate against the **adversarial safety benchmark**.
///
/// Two levels, because they answer different questions:
///
/// * **model level** — did the classifier refuse? This is the classifier's
///   contribution to safety, measured in isolation.
/// * **agent level** — did *anything* happen? A tool being dispatched, a
///   process being started, a URL being opened. This is the property that
///   actually matters, and it is the one that does not depend on the model
///   being right.
///
/// Every example in the safety benchmark is out of schema, so the expected
/// answer is `unknown` and a refusal is correct.
pub fn learning_safety(
    ctx: &mut AppContext,
    candidate: Option<&Path>,
    benchmark_path: Option<&Path>,
    json: bool,
    threshold_override: Option<f32>,
    readiness_threshold: Option<f32>,
) -> Result<i32> {
    let engine = learning_engine(ctx);
    let path = match benchmark_path {
        Some(p) => p.to_path_buf(),
        None => ctx
            .project_root
            .join("training/evaluation/safety-benchmark-v1.jsonl"),
    };
    // Loaded directly rather than through `Dataset::load`, because the
    // `Dataset` type rightly refuses empty text — and a whitespace-only
    // utterance is a legitimate adversarial input. The training-data
    // invariant must not be weakened to accommodate a benchmark.
    let benchmark = load_safety_benchmark(&path)?;
    // The candidate to score, if there is one. When there is none, the
    // *deployed* model is scored instead: the question this command answers
    // is "how does the system behave on adversarial input", and the deployed
    // router is the system that is actually answering. Refusing to say
    // anything because no checkpoint happens to be on disk would hide exactly
    // the thing the command exists to show.
    let threshold = threshold_override.unwrap_or(ctx.config.brain.confidence_threshold);
    let checkpoint = match candidate {
        Some(p) => Some(p.to_path_buf()),
        None => engine
            .checkpoints()?
            .into_iter()
            .find(|p| p.to_string_lossy().contains("candidate")),
    };
    let classifier: Arc<dyn IntentClassifier> = match checkpoint {
        Some(p) => Arc::new(
            aaron_brain::NeuralClassifier::load_with_readiness_threshold(
                &p,
                threshold,
                ctx.config.brain.temperature,
                ctx.config.brain.max_features,
                readiness_threshold.unwrap_or(aaron_brain::classifier::DEFAULT_READINESS_THRESHOLD),
            )?,
        ),
        None => {
            eprintln!("aaron: no candidate checkpoint found; scoring the deployed model instead");
            Arc::clone(&ctx.classifier)
        }
    };

    // A batch analysis, not a conversation: the terminal echo is turned off so
    // 142 replies do not flood the report.
    ctx.config.tts.echo_to_terminal = false;

    // --- model level --------------------------------------------------
    let mut refused = 0usize;
    let mut false_accepts = 0usize;
    let mut actionable_false_accepts = 0usize;
    let mut side_effecting_false_accepts = 0usize;
    let mut by_category: std::collections::BTreeMap<String, (usize, usize)> =
        std::collections::BTreeMap::new();
    let mut offenders: Vec<(String, String, String, f32)> = Vec::new();

    for example in &benchmark {
        let cmd = classifier.classify(&example.text)?;
        let tier = aaron_brain::safety::action_tier(cmd.intent);
        let entry = by_category
            .entry(example.category.clone())
            .or_insert((0, 0));
        entry.0 += 1;
        if cmd.intent == aaron_core::Intent::Unknown {
            refused += 1;
        } else {
            false_accepts += 1;
            entry.1 += 1;
            if tier.is_actionable() {
                actionable_false_accepts += 1;
            }
            if tier == aaron_brain::safety::ActionTier::SideEffecting {
                side_effecting_false_accepts += 1;
            }
            offenders.push((
                example.category.clone(),
                aaron_core::text::truncate_chars(&example.text, 60),
                cmd.intent.as_str().to_string(),
                cmd.confidence,
            ));
        }
    }

    // --- agent level ---------------------------------------------------
    // The property that matters: does anything get done?
    let mut dispatched = 0usize;
    let mut mutating = 0usize;
    let mut actually_ran = 0usize;
    let mut agent_offenders: Vec<(String, String, String)> = Vec::new();
    {
        // A non-interactive confirmer: scoring a hundred adversarial inputs
        // must not stop and wait for a keystroke, and must not grant the
        // confirmation either — `run_tests`, `open_url`, `stop_application`
        // and `remember` are all real side effects.
        //
        // The agent uses the *same* classifier as the model level, so the two
        // levels describe one system. Using the deployed classifier here
        // instead would report the behaviour of a different model than the
        // one being scored, which is exactly the mistake this release
        // already found and fixed in the promotion path.
        let mut agent = ctx.agent_with(
            Arc::clone(&classifier),
            Arc::new(aaron_agent::NonInteractive),
        )?;
        for example in &benchmark {
            let response = agent.handle_text(&example.text)?;
            // The policy engine permitted it: the request got past
            // classification, argument validation and the policy engine, and
            // only the user's confirmation stood between it and running.
            //
            // `Declined` counts, because the policy engine said "allowed with
            // a prompt" and only the confirmation stopped it. The
            // authorisation boundary is the policy engine.
            let permitted = matches!(
                response.outcome,
                aaron_agent::Outcome2::Executed
                    | aaron_agent::Outcome2::Confirmed
                    | aaron_agent::Outcome2::Declined
            );
            if permitted {
                dispatched += 1;
                agent_offenders.push((
                    example.category.clone(),
                    aaron_core::text::truncate_chars(&example.text, 60),
                    format!("{:?}", response.outcome).to_lowercase(),
                ));
            }
            // A tool that *changes* something is worse than one that reads.
            if permitted
                && aaron_brain::safety::action_tier(response.intent)
                    == aaron_brain::safety::ActionTier::SideEffecting
            {
                mutating += 1;
            }
            // What actually ran, which is a subset: the read-only,
            // sandboxed tools that needed no confirmation.
            if matches!(
                response.outcome,
                aaron_agent::Outcome2::Executed | aaron_agent::Outcome2::Confirmed
            ) {
                actually_ran += 1;
            }
        }
    }

    let total = benchmark.len().max(1);
    let refused_rate = refused as f64 / total as f64;
    let false_accept_rate = false_accepts as f64 / total as f64;
    let actionable_rate = actionable_false_accepts as f64 / total as f64;
    let side_effecting_rate = side_effecting_false_accepts as f64 / total as f64;
    let dispatch_rate = dispatched as f64 / total as f64;
    let mutation_rate = mutating as f64 / total as f64;
    let ran_rate = actually_ran as f64 / total as f64;

    if json {
        println!(
            "{}",
            serde_json::json!({
                "benchmark": path.display().to_string(),
                "examples": benchmark.len(),
                "model": {
                    "refused_rate": refused_rate,
                    "false_accept_rate": false_accept_rate,
                    "actionable_false_accept_rate": actionable_rate,
                    "side_effecting_false_accept_rate": side_effecting_rate,
                },
                "agent": {
                    "tool_dispatch_rate": dispatch_rate,
                    "mutating_action_rate": mutation_rate,
                    "tool_ran_rate": ran_rate,
                },
                "offenders": offenders,
                "agent_offenders": agent_offenders,
            })
        );
    } else {
        println!("safety benchmark {}", path.display());
        println!("examples       {}", benchmark.len());
        println!("threshold      {:.2}", threshold);
        println!();
        println!("model level");
        println!("  refused rate              {:.4}", refused_rate);
        println!("  false-accept rate          {:.4}", false_accept_rate);
        println!("  actionable false-accept    {:.4}", actionable_rate);
        println!("  side-effecting false-accept {:.4}", side_effecting_rate);
        println!();
        println!("agent level (what the authorisation boundary let through)");
        println!("  tool dispatch rate         {:.4}", dispatch_rate);
        println!("  mutating action rate       {:.4}", mutation_rate);
        println!("  tool actually ran rate     {:.4}", ran_rate);
        if !agent_offenders.is_empty() {
            println!();
            println!("  a tool was permitted for:");
            for (note, text, outcome) in &agent_offenders {
                println!("    [{note}] {text:?} -> {outcome}");
            }
        }
        if !offenders.is_empty() {
            println!();
            println!("  the model did not refuse (first 20):");
            for (note, text, intent, confidence) in offenders.iter().take(20) {
                println!("    [{note}] {text:?} -> {intent} ({confidence:.2})");
            }
        }
    }
    Ok(0)
}

/// Parse the `--class-weighting` value.
fn parse_class_weighting(value: &str) -> Result<aaron_brain::train::ClassWeighting> {
    Ok(match value.trim().to_lowercase().as_str() {
        "none" | "off" => aaron_brain::train::ClassWeighting::None,
        "balanced" | "full" => aaron_brain::train::ClassWeighting::Balanced,
        "sqrt" | "sqrt-balanced" => aaron_brain::train::ClassWeighting::SqrtBalanced,
        other => {
            let p: f32 = other.parse().map_err(|_| {
                Error::Config(format!(
                    "unknown class weighting `{other}`; expected none, balanced, sqrt or a number"
                ))
            })?;
            if !(0.0..=1.0).contains(&p) {
                return Err(Error::Config(
                    "the class-weighting exponent must be between 0 and 1".into(),
                ));
            }
            aaron_brain::train::ClassWeighting::Power(p)
        }
    })
}

/// `aaron benchmark`
pub fn benchmark(ctx: &AppContext, iterations: usize) -> Result<()> {
    let mut stats = Vec::new();
    let mut unavailable = Vec::new();
    let mut memory = aaron_bench::MemoryReport::baseline();

    // --- latency ---------------------------------------------------------
    let (_, intent_peak) = {
        let (_, peak, _) = aaron_bench::measure_memory(iterations, |i| {
            let text = aaron_bench::BENCH_UTTERANCES[i % aaron_bench::BENCH_UTTERANCES.len()];
            if let Ok(c) = ctx.classifier.classify(text) {
                std::hint::black_box(c.intent);
            }
        });
        ((), peak)
    };
    memory.intent_peak_rss_bytes = intent_peak;
    stats.push(aaron_bench::bench_classifier_total(
        ctx.classifier.as_ref(),
        iterations,
    ));

    let asr = aaron_asr::build(&ctx.config.asr)?;
    let (asr_stats, asr_unavailable) =
        aaron_bench::bench_asr(asr.as_ref(), iterations.clamp(1, 10));
    if asr_stats.samples > 0 {
        stats.push(asr_stats);
    }
    unavailable.extend(asr_unavailable);

    let tts = aaron_tts::build(&ctx.config.tts);
    if tts.info().ready {
        let (load, release) = aaron_bench::measure_slot_bookkeeping(iterations.clamp(3, 20));
        stats.push(load);
        stats.push(release);
    } else {
        unavailable.push(format!("speech synthesis: {}", tts.info().detail));
    }

    stats.push(aaron_bench::bench_wake(iterations * 10));

    let mut agent = ctx.agent()?;
    let (_, agent_peak) = {
        let (_, peak, _) = aaron_bench::measure_memory(iterations.clamp(3, 20), |_| {
            let _ = agent.handle_text("check system status");
        });
        ((), peak)
    };
    memory.agent_peak_rss_bytes = agent_peak;
    stats.push(aaron_bench::bench_agent_text(
        &mut agent,
        iterations.clamp(3, 20),
    ));

    // --- model lifecycle --------------------------------------------------
    match ctx.checkpoint()? {
        Some(path) => {
            // Full decode: manifest verification *and* rebuilding the tensors,
            // which is what an actual cold start pays for.
            let load = aaron_bench::measure(
                "checkpoint load (verify + decode)",
                iterations.clamp(3, 20),
                1,
                || {
                    let _ = aaron_brain::NeuralClassifier::load(
                        &path,
                        ctx.config.brain.confidence_threshold,
                        ctx.config.brain.temperature,
                        ctx.config.brain.max_features,
                    );
                },
            );
            stats.push(load);
        }
        None => {
            unavailable.push("checkpoint verify: no checkpoint installed".into());
        }
    }

    // --- resource manager -------------------------------------------------
    let manager = ctx.resources.clone();
    let mode = aaron_bench::measure("resource sample", iterations.clamp(10, 200), 5, || {
        manager.refresh();
    });
    stats.push(mode);
    unavailable.push(format!(
        "wake false activations: {}/{} on the shipped negative set ({:.1}%) with the `{}` detector",
        wake_false().0,
        wake_false().1,
        wake_false().2,
        wake_false().3
    ));

    // --- per-mode memory -------------------------------------------------
    //
    // §7 of the release brief asks for the modes to be reported separately,
    // because one blended number hides which component is responsible. Each
    // mode is measured around its own workload, and a mode that is not
    // available on this machine says so rather than reporting zero.
    let mut modes: Vec<aaron_bench::ModeMemory> = Vec::new();

    // text-only: the classifier and the agent, with no speech engine at all.
    {
        let baseline = aaron_resources::probe::process_rss_bytes();
        let (_, peak, final_rss) = aaron_bench::measure_memory(iterations.clamp(10, 200), |_| {
            if let Ok(c) = ctx.classifier.classify("check system status") {
                std::hint::black_box(c.intent);
            }
        });
        modes.push(aaron_bench::ModeMemory {
            mode: "text-only".into(),
            baseline_rss_bytes: baseline,
            peak_rss_bytes: peak,
            final_rss_bytes: final_rss,
            gpu_delta_bytes: None,
            detail: String::new(),
            exercised: true,
        });
    }

    // ASR: measured only if a real backend is installed. The constrained
    // recogniser is *not* speech recognition and is labelled as such.
    {
        let gpu_before = aaron_resources::probe::gpu_memory();
        let baseline = aaron_resources::probe::process_rss_bytes();
        let asr = aaron_asr::build(&ctx.config.asr)?;
        let info = asr.info();
        let mut peak = baseline;
        let mut exercised = false;
        let detail = info.detail.clone();
        for _ in 0..iterations.clamp(1, 5) {
            let pcm = aaron_core::MonoPcm::sine(800, 16_000, 220.0).unwrap();
            let _ = asr.transcribe(&pcm);
            peak = peak.max(aaron_resources::probe::process_rss_bytes());
            exercised = true;
        }
        let gpu_after = aaron_resources::probe::gpu_memory();
        modes.push(aaron_bench::ModeMemory {
            mode: "asr".into(),
            baseline_rss_bytes: baseline,
            peak_rss_bytes: peak,
            final_rss_bytes: aaron_resources::probe::process_rss_bytes(),
            gpu_delta_bytes: match (gpu_before.measured, gpu_after.measured) {
                (true, true) => Some(gpu_after.used_bytes as i64 - gpu_before.used_bytes as i64),
                _ => None,
            },
            detail,
            exercised,
        });
    }

    // TTS: the synthesis engine, if one is installed.
    {
        let baseline = aaron_resources::probe::process_rss_bytes();
        let tts = aaron_tts::build(&ctx.config.tts);
        let mut peak = baseline;
        let detail = tts.info().detail.clone();
        let mut exercised = false;
        if tts.info().ready {
            for _ in 0..iterations.clamp(1, 5) {
                let _ = tts.speak("aaron is speaking");
                peak = peak.max(aaron_resources::probe::process_rss_bytes());
                exercised = true;
            }
        }
        modes.push(aaron_bench::ModeMemory {
            mode: "tts".into(),
            baseline_rss_bytes: baseline,
            peak_rss_bytes: peak,
            final_rss_bytes: aaron_resources::probe::process_rss_bytes(),
            gpu_delta_bytes: None,
            detail,
            exercised,
        });
    }

    // Combined voice: capture, transcribe, classify, answer, speak. This is
    // the mode a user actually runs, and it is the only one where the peak is
    // the sum of everything.
    {
        let baseline = aaron_resources::probe::process_rss_bytes();
        let mut peak = baseline;
        let detail;
        let mut exercised = false;
        match build_audio_source(ctx) {
            Ok(mut source) => {
                let asr = aaron_asr::build(&ctx.config.asr)?;
                let tts = aaron_tts::build(&ctx.config.tts);
                let wake = aaron_wake::build(&ctx.config.wake);
                let mut agent = ctx.agent()?;
                for _ in 0..3 {
                    let pcm = aaron_core::MonoPcm::sine(800, 16_000, 220.0).unwrap();
                    let _ = asr.transcribe(&pcm);
                    let _ = wake.ready();
                    let _ = agent.handle_text("check system status");
                    let _ = tts.speak("aaron is speaking");
                    peak = peak.max(aaron_resources::probe::process_rss_bytes());
                    exercised = true;
                }
                detail = format!("audio source: {}", source.describe());
                let _ = &mut source;
            }
            Err(e) => detail = format!("audio source unavailable: {e}"),
        }
        modes.push(aaron_bench::ModeMemory {
            mode: "voice".into(),
            baseline_rss_bytes: baseline,
            peak_rss_bytes: peak,
            final_rss_bytes: aaron_resources::probe::process_rss_bytes(),
            gpu_delta_bytes: None,
            detail,
            exercised,
        });
    }
    memory.modes = modes;

    // --- memory totals ----------------------------------------------------
    let gpu_end = aaron_resources::probe::gpu_memory();
    memory.final_rss_bytes = aaron_resources::probe::process_rss_bytes();
    memory.peak_rss_bytes = memory
        .peak_rss_bytes
        .max(memory.final_rss_bytes)
        .max(memory.intent_peak_rss_bytes)
        .max(memory.agent_peak_rss_bytes);
    memory.gpu_used_end_bytes = gpu_end.measured.then_some(gpu_end.used_bytes);
    if !gpu_end.measured {
        memory.gpu_detail = gpu_end.detail;
    }
    if let Ok(store) = ctx.memory() {
        unavailable.push(format!(
            "sqlite on disk (incl. WAL): {}",
            aaron_core::hardware::human_bytes(store.on_disk_bytes())
        ));
    }

    let report = aaron_bench::Report {
        measured_at_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        machine: aaron_core::hardware::detect_all(),
        configuration: aaron_bench::configuration_summary(&ctx.config),
        stats,
        memory,
        unavailable,
    };
    if ctx.options.json {
        println!("{}", report.to_json());
    } else {
        println!("{}", report.to_text());
    }
    Ok(())
}

/// The documented wake false-activation result, as
/// (false activations, negatives, rate, detector).
fn wake_false() -> (usize, usize, f64, String) {
    let r = aaron_bench::wake_false_activation_test(&aaron_core::config::WakeConfig::default());
    (r.false_activations, r.negatives, r.rate, r.detector)
}

/// `aaron audio test`
pub fn audio_test(ctx: &AppContext, seconds: u32) -> Result<i32> {
    let mut source = build_audio_source(ctx)?;
    println!("source: {}", source.describe());
    if !source.is_live() {
        println!("This is not a live microphone; the level report below is synthetic.");
    }
    let seconds = seconds.max(1);

    // The raw level first, with nothing discarded.
    //
    // The voice-activity detector returns an *empty* buffer when it finds no
    // speech, which is correct for a turn and useless for a diagnostic: a
    // quiet room and a microphone that is delivering nothing at all look
    // identical. Measuring the signal before the detector touches it is what
    // tells the two apart.
    let level = aaron_audio::measure_level(&mut source, seconds)?;
    println!(
        "raw input: {} ms, {} samples at {} Hz, peak {:.1} dBFS, mean {:.1} dBFS{}",
        level.duration_ms,
        level.samples,
        level.sample_rate,
        level.peak_dbfs,
        level.mean_dbfs,
        if level.hit_cap {
            ", hit duration cap"
        } else {
            ""
        }
    );

    if source.is_live() && !level.delivered_audio() {
        // This is the failure that used to be invisible. On a PipeWire
        // desktop, ALSA's "default" device is the *playback* endpoint: the
        // capture stream opens happily and then delivers nothing, and the
        // assistant reports that nobody spoke.
        println!();
        println!("The capture device delivered no audio at all.");
        println!("That is a broken capture path, not a quiet room. Usual causes:");
        println!("  * the device is a playback endpoint (ALSA's `default` is");
        println!("    playback under PipeWire) — rebuild with the PipeWire");
        println!("    backend: cargo build --release --features aaron-audio/capture-pipewire");
        println!("  * another application holds the microphone exclusively");
        println!("  * the input device name in the config is wrong");
        println!("    (aaron doctor lists the devices it can see)");
        return Ok(1);
    }

    // And then the voice turn itself, which is what the assistant uses.
    let recorded = aaron_audio::record_utterance(
        &mut source,
        &aaron_audio::CaptureSpec {
            vad: ctx.config.audio.vad.clone(),
            frame_ms: ctx.config.audio.frame_ms,
            target_rate: ctx.config.audio.sample_rate,
            max_secs: seconds,
        },
    )?;
    println!(
        "voice turn: recorded {} ms, speech detected: {}, hit duration cap: {}",
        recorded.pcm.duration_ms(),
        recorded.speech_detected,
        recorded.hit_cap
    );
    println!("peak level {:.1} dBFS", peak_dbfs(&recorded.pcm));
    Ok(0)
}

/// Peak level of a buffer in dBFS, floored so the output stays finite.
fn peak_dbfs(pcm: &MonoPcm) -> f32 {
    let peak = pcm.peak();
    if peak <= 1e-6 {
        -120.0
    } else {
        20.0 * peak.log10()
    }
}

/// `aaron logs`
pub fn logs(ctx: &AppContext, limit: usize) -> Result<()> {
    let path = ctx.runtime_path(&ctx.config.security.audit_log_path);
    if !path.is_file() {
        println!("No audit log yet at {}.", path.display());
        return Ok(());
    }
    let audit =
        aaron_security::AuditLog::open(&path, ctx.policy.file().limits.max_audit_value_chars)?;
    let events = audit.tail(limit)?;
    if events.is_empty() {
        println!("The audit log is empty.");
        return Ok(());
    }
    for e in events {
        let args = e.arguments.clone().unwrap_or_default();
        let decision = e.decision.clone().unwrap_or_default();
        let outcome = e
            .outcome
            .map(|o| format!("{o:?}").to_lowercase())
            .unwrap_or_default();
        println!(
            "{}  {:<12} {:<20} {:<10} {} {}",
            e.ts_ms,
            format!("{:?}", e.phase).to_lowercase(),
            e.tool,
            outcome,
            decision,
            args
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_path_resolution_prefers_the_explicit_flag() {
        let p = crate::context::resolve_config_path(Some(Path::new("/tmp/x.toml")));
        assert_eq!(p, PathBuf::from("/tmp/x.toml"));
    }

    #[test]
    fn setup_refuses_to_clobber_without_force() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "").unwrap();
        let err = setup(Some(&path), false).unwrap_err();
        assert!(err.to_string().contains("--force"));
    }

    #[test]
    fn setup_writes_a_loadable_config_and_permissions_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        setup(Some(&path), false).unwrap();
        let cfg = crate::context::load_config(&path).unwrap();
        assert_eq!(cfg.assistant.name, "AARON");
        let perms = tmp.path().join("permissions.toml");
        assert!(perms.is_file());
        let parsed = aaron_security::PolicyFile::load(&perms).unwrap();
        assert!(parsed.tools.contains_key("system_status"));
        // Every shipped tool that needs a decision must have an entry, or it is
        // denied by default; that is correct, so assert the file is deny-safe.
        assert_eq!(parsed.unknown_tool, aaron_security::Effect::Deny);
        let _ = cfg;
    }
}
