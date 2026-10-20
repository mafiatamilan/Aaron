//! Application context: everything a command needs, loaded once.
//!
//! Loading is deliberately strict and ordered, so that a misconfigured install
//! fails at startup with one clear message rather than halfway through a
//! conversation:
//!
//! 1. configuration (validated),
//! 2. runtime directories,
//! 3. the path sandbox,
//! 4. the memory database (migrated),
//! 5. the policy file (deny-by-default),
//! 6. the tool registry,
//! 7. the intent classifier (neural if a checkpoint exists, else deterministic).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use aaron_agent::{Agent, AgentDeps, AlwaysDecline, AutoConfirm, Confirmer, NonInteractive};
use aaron_core::config::Config;
use aaron_core::error::{Error, Result};
use aaron_core::paths::RuntimePaths;
use aaron_core::IntentClassifier;
use aaron_memory::{MemoryStore, RetentionPolicy};
use aaron_security::{AuditLog, PathSandbox, PolicyEngine, PolicyFile};
use aaron_tools::{ToolContext, ToolRegistry};

/// Options that affect how a command runs, not what it does.
#[derive(Debug, Clone, Default)]
pub struct RunOptions {
    /// Approve every confirmation (`--yes`).
    pub assume_yes: bool,
    /// Decline every confirmation (`--no`).
    pub assume_no: bool,
    /// Do not attach the audit log.
    pub no_audit: bool,
    /// Force the deterministic router.
    pub text_only: bool,
    /// Print machine-readable JSON where a command supports it.
    pub json: bool,
}

impl RunOptions {
    /// The confirmer these options select.
    pub fn confirmer(&self) -> Arc<dyn Confirmer> {
        if self.assume_yes {
            Arc::new(AutoConfirm)
        } else if self.assume_no {
            Arc::new(AlwaysDecline)
        } else if std::io::IsTerminal::is_terminal(&std::io::stdin()) {
            Arc::new(StdinConfirmer::default())
        } else {
            Arc::new(NonInteractive)
        }
    }
}

/// Reads a yes/no answer from the terminal.
pub struct StdinConfirmer {
    /// How long to wait before auto-cancelling.
    pub timeout: std::time::Duration,
}

impl Default for StdinConfirmer {
    fn default() -> Self {
        StdinConfirmer {
            timeout: std::time::Duration::from_secs(60),
        }
    }
}

impl Confirmer for StdinConfirmer {
    fn confirm(&self, prompt: &str, timeout: std::time::Duration) -> Result<bool> {
        use std::io::Write;
        let deadline = std::time::Instant::now() + timeout.min(self.timeout);
        eprint!("{prompt}\nProceed? [y/N] ");
        std::io::stderr().flush().ok();

        // Read on a worker thread so a hung terminal cannot block forever.
        // `Stdin` is `Send`, so the handle can be opened inside the thread
        // rather than borrowing a lock from this one.
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut buf = String::new();
            let _ = std::io::stdin().read_line(&mut buf);
            let _ = tx.send(buf);
        });
        match rx.recv_timeout(deadline.saturating_duration_since(std::time::Instant::now())) {
            Ok(text) => {
                let line = text;
                let answer = line.trim().to_lowercase();
                Ok(answer == "y" || answer == "yes")
            }
            Err(_) => {
                eprintln!("\n[no answer within the confirmation timeout; treating it as no]");
                Ok(false)
            }
        }
    }
}

/// The loaded application.
pub struct AppContext {
    /// Central resource manager: budgets, mode and the low-memory state machine.
    pub resources: aaron_resources::ResourceManager,
    pub config: Config,
    pub config_path: PathBuf,
    pub project_root: PathBuf,
    pub paths: RuntimePaths,
    pub sandbox: Arc<PathSandbox>,
    pub memory: Arc<Mutex<MemoryStore>>,
    pub policy: PolicyEngine,
    pub registry: ToolRegistry,
    pub audit: AuditLog,
    pub classifier: Arc<dyn IntentClassifier>,
    pub options: RunOptions,
    /// Lazily loaded optional speech engines, released when idle.
    ///
    /// A `ModelSlot` rather than a plain `Option`, because the lifetime
    /// property that matters is that a model is never released while a caller
    /// holds it — and that is enforced by the guard type, not by convention.
    /// See docs/resources.md.
    asr_slot: aaron_resources::ModelSlot<
        Arc<dyn aaron_core::SpeechRecognizer>,
        aaron_resources::BoxedLoader<Arc<dyn aaron_core::SpeechRecognizer>>,
    >,
    tts_slot: aaron_resources::ModelSlot<
        Arc<dyn aaron_core::SpeechSynthesizer>,
        aaron_resources::BoxedLoader<Arc<dyn aaron_core::SpeechSynthesizer>>,
    >,
    /// The retained synthesiser lease. Holding it here is what stops the
    /// unload sweep from releasing a model that every `Agent` shares.
    tts_lease: std::sync::OnceLock<aaron_resources::TtsLease>,
}

/// Where configuration and data live, given the CLI's `--config` flag.
pub fn resolve_config_path(explicit: Option<&Path>) -> PathBuf {
    if let Some(p) = explicit {
        return p.to_path_buf();
    }
    if let Ok(p) = std::env::var("AARON_CONFIG") {
        return PathBuf::from(p);
    }
    // A source checkout keeps its config next to the binary-independent
    // `config/` directory; an installed binary uses the user's config dir.
    if let Some(root) =
        aaron_core::paths::project_root(&std::env::current_dir().unwrap_or_default())
    {
        let candidate = root.join("config").join("default.toml");
        if candidate.is_file() {
            return candidate;
        }
    }
    let xdg = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")));
    xdg.unwrap_or_else(|| PathBuf::from("."))
        .join("aaron")
        .join("config.toml")
}

/// Load a context.
pub fn load(config_path: Option<&Path>, options: RunOptions) -> Result<AppContext> {
    let config_path = resolve_config_path(config_path);
    let config = load_config(&config_path)?;

    let project_root =
        aaron_core::paths::project_root(&std::env::current_dir().unwrap_or_default())
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

    let paths = RuntimePaths::discover();
    paths.ensure()?;

    // Sandbox roots come from config, resolved against the project root and
    // with `~` expanded.
    let roots: Vec<PathBuf> = config
        .security
        .allowed_roots
        .iter()
        .map(|r| Config::resolve(&project_root, r))
        .collect();
    let sandbox = Arc::new(if roots.is_empty() {
        PathSandbox::deny_all()
    } else {
        PathSandbox::new(roots)?
    });

    let memory_path = if config.memory.database_path == ":memory:" {
        PathBuf::from(":memory:")
    } else {
        Config::resolve(&paths.root, &config.memory.database_path)
    };
    let retention = RetentionPolicy {
        retention_days: config.memory.retention_days,
        max_entries: config.memory.max_entries,
    };
    // Measure the machine and resolve the budgets before anything large is
    // allocated, so the first action is already inside the budget.
    let resources = aaron_resources::ResourceManager::new(config.resources.clone());

    let storage_limits = aaron_resources_storage_limits(&config);
    let mut store = MemoryStore::open(&memory_path, retention)?.with_max_value_chars(
        config
            .memory
            .max_value_chars
            .min(storage_limits.max_value_bytes / 4),
    );
    store.apply_limits(storage_limits)?;
    let memory = Arc::new(Mutex::new(store));

    // Relative paths in the configuration are resolved against the
    // configuration file's own directory, not the working directory. A user who
    // runs `aaron --config /etc/aaron/config.toml` from anywhere must get the
    // same permissions file every time.
    let config_dir = config_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    let permissions_path = Config::resolve(&config_dir, &config.security.permissions_file);
    let policy_file = PolicyFile::load(&permissions_path)?;
    let policy = PolicyEngine::new(policy_file, config.security.always_confirm.clone());

    let audit = if options.no_audit {
        AuditLog::disabled()
    } else {
        AuditLog::open(
            Config::resolve(&paths.root, &config.security.audit_log_path),
            policy.file().limits.max_audit_value_chars,
        )?
    };

    let registry = ToolRegistry::builtin();
    let classifier = build_classifier(&config, &project_root, options.text_only)?;

    // The speech engines are *not* built here. On a 4 GB card the desktop needs
    // most of the VRAM, and AARON has no business holding a voice model when it
    // is not transcribing or speaking. Each slot loads on first use and
    // releases itself once it has been idle for the configured window.
    let mut asr_config = config.asr.clone();
    // The wake phrase is the word the recogniser most needs to get right,
    // and a recogniser that mishears it does not wake the assistant. So the
    // wake phrase is the default hotword, unless the user configured their
    // own list.
    if asr_config.hotwords.trim().is_empty() {
        let phrase = config.wake.phrase.trim();
        if !phrase.is_empty() {
            asr_config.hotwords = phrase.to_string();
        }
    }
    let tts_config = config.tts.clone();
    let asr_slot = resources.model_slot("asr", move || aaron_asr::build(&asr_config));
    let tts_slot = resources.model_slot("tts", move || {
        Ok(aaron_tts::build(&tts_config) as Arc<dyn aaron_core::SpeechSynthesizer>)
    });

    Ok(AppContext {
        resources,
        config,
        config_path,
        project_root,
        paths,
        sandbox,
        memory,
        policy,
        registry,
        audit,
        classifier,
        options,
        asr_slot,
        tts_slot,
        tts_lease: std::sync::OnceLock::new(),
    })
}

/// Load the configuration, falling back to built-in defaults when the file is
/// absent. A *malformed* file is still an error.
pub fn load_config(path: &Path) -> Result<Config> {
    if path.is_file() {
        let mut config = Config::load(path)?;
        if !config.security.allowed_roots.is_empty() {
            // Resolve relative roots against the config file's directory so the
            // meaning does not depend on the working directory. Absolute roots
            // and `~` paths are untouched.
            let base = path
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));
            config.security.allowed_roots = config
                .security
                .allowed_roots
                .iter()
                .map(|r| Config::resolve(&base, r).display().to_string())
                .collect();
        }
        Ok(config)
    } else {
        tracing::debug!(path = %path.display(), "no config file; using built-in defaults");
        Ok(Config::default())
    }
}

fn build_classifier(
    config: &Config,
    project_root: &Path,
    text_only: bool,
) -> Result<Arc<dyn IntentClassifier>> {
    let backend = if text_only {
        aaron_core::config::BrainBackend::Heuristic
    } else {
        config.brain.backend
    };
    let roots = [project_root.to_path_buf(), config_path_root(config)];
    aaron_brain::classifier::build(
        backend,
        &config.brain.checkpoint_path,
        config.brain.confidence_threshold,
        config.brain.temperature,
        config.brain.max_features,
        &roots,
    )
}

fn config_path_root(_config: &Config) -> PathBuf {
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

impl AppContext {
    /// Build an agent from this context.
    pub fn agent(&self) -> Result<Agent> {
        self.agent_with(Arc::clone(&self.classifier), self.options.confirmer())
    }

    /// Build an agent that uses a different classifier and a different
    /// confirmer, but this context's policy engine, allow-list, sandbox,
    /// registry and audit log.
    ///
    /// The promotion path needs this: the agent-level safety measurement
    /// must be about the *candidate* checkpoint, not about whichever model
    /// happens to be deployed, and it must not block on a confirmation
    /// prompt while scoring a hundred adversarial inputs.
    pub fn agent_with(
        &self,
        classifier: Arc<dyn IntentClassifier>,
        confirmer: Arc<dyn Confirmer>,
    ) -> Result<Agent> {
        let ctx = ToolContext::new(
            Arc::clone(&self.sandbox),
            Arc::clone(&self.memory),
            Arc::new(self.config.clone()),
            self.project_root.clone(),
        );
        Ok(Agent::new(AgentDeps {
            classifier,
            registry: self.registry.clone(),
            policy: PolicyEngine::new(
                self.policy.file().clone(),
                self.config.security.always_confirm.clone(),
            ),
            audit: self.audit.clone(),
            context: ctx,
            memory: Arc::clone(&self.memory),
            tts: self.tts()?.get().clone(),
            confirmer,
            config: self.config.clone(),
        }))
    }

    /// Lock the memory database, mapping a poisoned mutex to a real error.
    pub fn memory(&self) -> Result<std::sync::MutexGuard<'_, MemoryStore>> {
        self.memory
            .lock()
            .map_err(|_| Error::Storage("memory lock poisoned".into()))
    }

    /// Number of ASR/TTS models currently resident.
    ///
    /// Both optional engines are loaded on demand and released when idle, so
    /// this is usually zero. `aaron resources limits` reports it so the
    /// lifecycle is visible rather than implied.
    pub fn model_slots_loaded(&self) -> usize {
        usize::from(self.asr_slot.is_loaded()) + usize::from(self.tts_slot.is_loaded())
    }

    /// The speech recogniser, loading it on first use.
    ///
    /// The returned guard keeps the model alive for as long as the caller needs
    /// it, so the idle-unload sweep can never pull it out from under a request
    /// in flight. That property is the reason this returns a guard rather than
    /// a bare reference.
    pub fn asr(&self) -> Result<aaron_resources::AsrGuard<'_>> {
        self.asr_slot.acquire()
    }

    /// The speech synthesiser, loading it on first use.
    ///
    /// This is a *lease*, and it is retained by the context for as long as the
    /// process runs. The synthesiser is owned by `Agent` for the whole session;
    /// reloading a Piper voice on every turn would cost far more than the
    /// memory it holds.
    pub fn tts(&self) -> Result<&aaron_resources::TtsLease> {
        if let Some(existing) = self.tts_lease.get() {
            return Ok(existing);
        }
        let lease = self.tts_slot.lease()?;
        // If another thread won the race its lease is the one retained; ours is
        // dropped, which un-pins without affecting the model the winner holds.
        Ok(self.tts_lease.get_or_init(|| lease))
    }

    /// Release any optional model that has been idle long enough.
    ///
    /// Returns how many were actually released, so `aaron resources status`
    /// can report the lifecycle rather than assuming it.
    pub fn release_idle_models(&self) -> usize {
        usize::from(self.asr_slot.release_if_idle()) + usize::from(self.tts_slot.release_if_idle())
    }

    /// Apply the current resource policy: release optional models when the
    /// machine is short on memory, and refuse to start background work.
    ///
    /// This is the hook that turns the low-memory state machine from a report
    /// into a behaviour. It is deliberately conservative — it only releases a
    /// model that is *not* in use, and it never touches the database.
    pub fn enforce_resource_policy(&self) -> Result<usize> {
        self.resources.refresh();
        self.resources.admits(true)?;
        if self.resources.mode() == aaron_resources::Mode::LowMemory {
            // Safe by construction: `release_if_idle` refuses while a caller
            // holds the model or a lease pins it.
            return Ok(self.release_idle_models());
        }
        Ok(0)
    }

    /// Load/unload counters for the optional engines.
    pub fn model_slot_stats(&self) -> (aaron_resources::SlotStats, aaron_resources::SlotStats) {
        (self.asr_slot.stats(), self.tts_slot.stats())
    }

    /// The intent checkpoint, if one is installed.
    pub fn checkpoint(&self) -> Result<Option<std::path::PathBuf>> {
        Ok(aaron_brain::classifier::find_checkpoint(
            Some(&self.config.brain.checkpoint_path),
            &[self.project_root.clone(), self.paths.root.clone()],
        ))
    }

    /// Resolve a configured path against the runtime root.
    pub fn runtime_path(&self, raw: &str) -> PathBuf {
        Config::resolve(&self.paths.root, raw)
    }
}

/// Translate the `[resources]` config into storage limits.
fn aaron_resources_storage_limits(config: &Config) -> aaron_memory::StorageLimits {
    aaron_memory::StorageLimits {
        cache_kib: config.resources.memory.sqlite_cache_kib,
        max_retrieved: config.resources.memory.max_retrieved_memories,
        max_value_bytes: config.resources.memory.max_memory_content_bytes,
    }
}

/// Print an error the way the CLI should: one line for humans, an exit code for
/// scripts.
pub fn report_error(err: &Error) -> i32 {
    eprintln!("aaron: {err}");
    for hint in remediation(err) {
        eprintln!("  hint: {hint}");
    }
    err.kind().exit_code()
}

fn remediation(err: &Error) -> Vec<String> {
    match err {
        Error::Config(_) => vec!["run `aaron setup` to write a fresh configuration".into()],
        Error::MissingDependency(_) => {
            vec!["run `aaron doctor` for what is missing and how to install it".into()]
        }
        Error::Model(_) => vec![
            "run `aaron brain train` to build a checkpoint, or set brain.backend = \"heuristic\""
                .into(),
        ],
        Error::PermissionDenied(_) => {
            vec!["run `aaron permissions show` to see the active rules".into()]
        }
        Error::PathRejected(_) => {
            vec!["add the directory to security.allowed_roots in your config".into()]
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relative_paths_resolve_against_the_config_file_not_the_cwd() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg = tmp.path().join("etc").join("aaron.toml");
        std::fs::create_dir_all(cfg.parent().unwrap()).unwrap();
        std::fs::write(
            tmp.path().join("etc").join("permissions.toml"),
            "version = 1\n",
        )
        .unwrap();
        let mut base = Config::default();
        base.security.permissions_file = "permissions.toml".into();
        std::fs::write(&cfg, base.to_toml().unwrap()).unwrap();

        let root = tmp.path().join("elsewhere");
        std::fs::create_dir_all(&root).unwrap();
        let previous = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();
        let loaded = load(Some(&cfg), RunOptions::default());
        std::env::set_current_dir(previous).unwrap();
        let ctx = loaded.expect("config beside its permissions file should load");
        assert!(ctx.policy.file().tools.is_empty());
    }

    #[test]
    fn default_config_loads_when_no_file_exists() {
        let cfg = load_config(Path::new("/definitely/not/here/config.toml")).unwrap();
        assert_eq!(cfg.assistant.name, "AARON");
    }

    #[test]
    fn a_malformed_config_is_an_error_not_a_silent_default() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.toml");
        std::fs::write(&path, "[assistant]\nnot_a_field = true\n").unwrap();
        let err = load_config(&path).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn confirmer_selection_follows_the_flags() {
        let mut o = RunOptions::default();
        assert!(
            o.confirmer().is_interactive() || !std::io::IsTerminal::is_terminal(&std::io::stdin())
        );
        o.assume_yes = true;
        assert!(o
            .confirmer()
            .confirm("x", std::time::Duration::from_millis(1))
            .unwrap());
        o.assume_yes = false;
        o.assume_no = true;
        assert!(!o
            .confirmer()
            .confirm("x", std::time::Duration::from_millis(1))
            .unwrap());
    }

    #[test]
    fn the_resource_manager_is_built_from_the_config() {
        let tmp = tempfile::tempdir().unwrap();
        let cfg_path = tmp.path().join("config.toml");
        let mut cfg = Config::default();
        cfg.memory.database_path = tmp.path().join("m.db").display().to_string();
        cfg.security.allowed_roots = Vec::new();
        std::fs::write(&cfg_path, cfg.to_toml().unwrap()).unwrap();

        let ctx = load(Some(&cfg_path), RunOptions::default()).unwrap();
        assert_eq!(
            ctx.resources.config().reserve_system_ram_mb,
            cfg.resources.reserve_system_ram_mb
        );
        assert_eq!(
            ctx.memory().unwrap().limits().cache_kib,
            cfg.resources.memory.sqlite_cache_kib
        );
    }

    #[test]
    fn storage_limits_come_from_the_resource_budget() {
        let mut cfg = Config::default();
        cfg.resources.memory.sqlite_cache_kib = 777;
        cfg.resources.memory.max_retrieved_memories = 3;
        cfg.resources.memory.max_memory_content_bytes = 999;
        let l = aaron_resources_storage_limits(&cfg);
        assert_eq!(
            l,
            aaron_memory::StorageLimits {
                cache_kib: 777,
                max_retrieved: 3,
                max_value_bytes: 999
            }
        );
    }

    #[test]
    fn errors_carry_actionable_hints() {
        assert!(!remediation(&Error::Config("x".into())).is_empty());
        assert!(!remediation(&Error::PermissionDenied("x".into())).is_empty());
        assert!(remediation(&Error::Other("x".into())).is_empty());
    }

    #[test]
    fn exit_codes_distinguish_failure_classes() {
        assert_eq!(Error::Cancelled.kind().exit_code(), 130);
        assert_eq!(Error::PermissionDenied("x".into()).kind().exit_code(), 77);
        assert_eq!(Error::Config("x".into()).kind().exit_code(), 78);
    }
}
