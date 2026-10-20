//! AARON — an offline-first voice assistant for Linux, written in Rust.
//!
//! The binary starts by loading configuration, the path sandbox, the memory
//! database, the policy file and the tool registry, in that order, and reports
//! a single clear error if any of them is wrong. Nothing is guessed and nothing
//! is downloaded.

mod commands;
mod context;
mod dataset_cmd;
mod resources_cmd;

use std::path::PathBuf;
use std::process::ExitCode;

use aaron_core::error::Error;
use clap::{Parser, Subcommand};

use commands::TrainingOverrides;
use context::RunOptions;

/// Offline-first, voice-controlled personal assistant for Linux.
#[derive(Debug, Parser)]
#[command(
    name = "aaron",
    version,
    about = "AARON — an offline-first personal assistant",
    long_about = "AARON listens for a wake phrase, classifies a short intent set with a \
custom 1,000,000-parameter model, checks every proposed action against a deny-by-default \
policy, and executes approved tools locally. Nothing is sent anywhere.",
    propagate_version = true
)]
struct Cli {
    /// Path to the configuration file.
    #[arg(long, short, global = true, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Approve every confirmation prompt.
    #[arg(long, short = 'y', global = true)]
    yes: bool,

    /// Decline every confirmation prompt.
    #[arg(long, short = 'n', global = true, conflicts_with = "yes")]
    no: bool,

    /// Never open the microphone; use the deterministic intent router.
    #[arg(long, global = true)]
    text_only: bool,

    /// Do not write the audit log for this run.
    #[arg(long, global = true)]
    no_audit: bool,

    /// Emit machine-readable JSON where supported.
    #[arg(long, global = true)]
    json: bool,

    /// Override the log level (error, warn, info, debug, trace).
    #[arg(long, global = true, value_name = "LEVEL")]
    log_level: Option<String>,

    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Write a starter configuration and permissions file.
    Setup {
        /// Overwrite existing files.
        #[arg(long)]
        force: bool,
    },

    /// Report what works on this machine and what is missing.
    Doctor,

    /// Start the interactive assistant.
    Run,

    /// Listen for one utterance (or until `--seconds` elapses).
    Listen {
        /// Skip wake-phrase detection and dispatch whatever is heard.
        #[arg(long)]
        no_wake: bool,
        /// Handle a single utterance and exit.
        #[arg(long)]
        once: bool,
        /// Stop listening after this many seconds.
        #[arg(long, default_value_t = 30)]
        seconds: u32,
        /// Where the audio comes from: `microphone` (default), `synthetic`,
        /// or `file:<path>` for a reproducible run through the same pipeline.
        #[arg(long, default_value = "microphone")]
        source: String,
        /// Allow the fixed-phrase demonstration recogniser.
        ///
        /// The constrained recogniser is a test double: it emits one of seven
        /// canned commands and does no acoustic analysis. AARON refuses to
        /// dispatch what it hears, because a fabricated transcript is an
        /// instruction.
        #[arg(long)]
        allow_fake_asr: bool,
    },

    /// Handle one typed command and exit.
    Ask {
        /// The utterance.
        text: String,
    },

    /// Inspect the tool registry.
    Tools {
        #[command(subcommand)]
        action: ToolsAction,
    },

    /// Inspect the active policy.
    Permissions {
        #[command(subcommand)]
        action: PermissionsAction,
    },

    /// Inspect and edit long-term memory.
    Memory {
        #[command(subcommand)]
        action: MemoryAction,
    },

    /// Review corrections, train, evaluate, promote and roll back.
    Learning {
        #[command(subcommand)]
        action: LearningAction,
    },

    /// Train the custom intent model from a dataset.
    Brain {
        #[command(subcommand)]
        action: BrainAction,
    },

    /// Describe the installed models and their parameter counts.
    Model,

    /// Measure latency and resource use on this machine.
    Benchmark {
        /// Iterations per measurement.
        #[arg(long, default_value_t = 50)]
        iterations: usize,
    },

    /// Show recent audit records.
    Logs {
        /// How many records to show.
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },

    /// Check the microphone without starting the assistant.
    Audio {
        #[command(subcommand)]
        action: AudioAction,
    },

    /// Inspect the training dataset: schema, duplicates, leakage, coverage.
    Dataset {
        #[command(subcommand)]
        action: DatasetAction,
    },

    /// Inspect resource budgets, check stability and prove memory does not leak.
    Resources {
        #[command(subcommand)]
        action: ResourcesAction,
    },
}

#[derive(Debug, Subcommand)]
enum DatasetAction {
    /// Validate the dataset and report every problem found.
    Check {
        /// Dataset to inspect. Defaults to the configured dataset.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Emit JSON instead of text.
        #[arg(long)]
        json: bool,
    },
    /// Report the per-intent and per-split coverage of the dataset.
    Report {
        /// Dataset to inspect. Defaults to the configured dataset.
        #[arg(long)]
        dataset: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum ResourcesAction {
    /// Measured memory, GPU and operating mode right now.
    Status,
    /// Print every configured budget.
    Limits,
    /// Run the repeatable memory workloads and report stability.
    Stress {
        /// Iterations per workload.
        #[arg(long, default_value_t = 20_000)]
        iterations: usize,
    },
    /// Interact continuously for a while and check that RSS does not grow.
    Soak {
        /// Duration in seconds.
        #[arg(long, default_value_t = 30)]
        seconds: u32,
    },
    /// Report the installed checkpoint's size and integrity.
    Checkpoint {
        /// Path to the checkpoint. Defaults to the standard location.
        #[arg(long)]
        path: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum ToolsAction {
    /// List every registered tool and its current policy verdict.
    List,
    /// Summarise the policy verdict for every tool.
    Test,
}

#[derive(Debug, Subcommand)]
enum PermissionsAction {
    /// Show the effective policy.
    Show,
    /// Show the verdict for one tool, or for all of them.
    Test {
        /// Tool id. Omit to test every tool.
        tool: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
enum MemoryAction {
    /// List stored memories.
    List {
        /// Filter by kind (fact, preference, session_context, summary).
        #[arg(long)]
        kind: Option<String>,
        /// Maximum rows.
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Search memory by keyword.
    Search {
        /// Query text.
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// Delete one memory by id.
    Delete {
        /// Memory id.
        id: i64,
    },
    /// Store a memory directly.
    Add {
        /// The text to store.
        text: String,
        #[arg(long)]
        key: Option<String>,
        #[arg(long, default_value = "fact")]
        kind: String,
    },
    /// Copy the database to a safe file.
    Backup {
        /// Destination path.
        #[arg(long)]
        dest: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum LearningAction {
    /// Corrections waiting for review.
    Pending,
    /// Approve a correction and add it to the dataset.
    Approve {
        /// Correction id from `aaron learning pending`.
        id: i64,
    },
    /// Reject a correction permanently.
    Reject {
        /// Correction id from `aaron learning pending`.
        id: i64,
    },
    /// Evaluate a candidate checkpoint against the thresholds.
    Evaluate {
        /// Candidate to evaluate. Defaults to the newest candidate.
        #[arg(long)]
        candidate: Option<PathBuf>,
        /// Dataset whose held-out split is used for scoring.
        ///
        /// This exists so a candidate trained on one dataset can be scored
        /// against a *different* dataset's hold-out — in particular the frozen
        /// v1 benchmark, which nothing is tuned against. Scoring a candidate
        /// on the split it was tuned for is not evidence of anything.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// A locked benchmark file, used entirely as the evaluation set.
        #[arg(long)]
        benchmark: Option<PathBuf>,
        /// Which split of the dataset to score against.
        ///
        /// Hyperparameters are selected on `validation`, never on `test`.
        /// Scoring the train split is useful only as a sanity check that the
        /// model actually learned something.
        #[arg(long, default_value = "test")]
        split: String,
        /// Override the confidence threshold.
        ///
        /// The threshold is an *inference-time* setting, so a sweep does not
        /// need to retrain: the same checkpoint is re-scored at each value.
        #[arg(long)]
        threshold: Option<f32>,
        /// The adversarial safety benchmark to gate against.
        #[arg(long)]
        safety_benchmark: Option<PathBuf>,
    },
    /// Score a candidate against the adversarial safety benchmark.
    Safety {
        /// Candidate to score. Defaults to the newest candidate.
        #[arg(long)]
        candidate: Option<PathBuf>,
        /// Safety benchmark to use.
        #[arg(long)]
        benchmark: Option<PathBuf>,
        /// Emit JSON instead of text.
        #[arg(long)]
        json: bool,
        /// Override the confidence threshold.
        #[arg(long)]
        threshold: Option<f32>,
        /// Override the readiness threshold.
        ///
        /// The probability above which the readiness head says a request is
        /// not specific enough to act on. It is a trade-off: a lower value
        /// catches more under-specified requests and refuses more good ones.
        #[arg(long)]
        readiness_threshold: Option<f32>,
    },
    /// Produce the systematic error-analysis report.
    Analyse {
        /// Candidate to analyse. Defaults to the newest candidate.
        #[arg(long)]
        candidate: Option<PathBuf>,
        /// Dataset whose held-out split is analysed.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// A locked benchmark file, used entirely as the evaluation set.
        #[arg(long)]
        benchmark: Option<PathBuf>,
        /// Output format.
        #[arg(long, default_value = "markdown")]
        format: String,
        /// Write to this file instead of stdout.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Promote an evaluated candidate to the live model.
    Promote {
        /// Candidate to promote. Defaults to the newest candidate.
        #[arg(long)]
        candidate: Option<PathBuf>,
    },
    /// Restore the previous checkpoint.
    Rollback,
}

#[derive(Debug, Subcommand)]
enum BrainAction {
    /// Train a candidate checkpoint from the labelled dataset.
    Train {
        /// Dataset path. Defaults to the configured dataset.
        #[arg(long)]
        dataset: Option<PathBuf>,
        /// Epochs.
        #[arg(long, default_value_t = 100)]
        epochs: u32,
        /// Seed; the same seed reproduces the same weights.
        #[arg(long, default_value_t = 2690307351)]
        seed: u64,
        /// Learning rate.
        #[arg(long, default_value_t = 1e-2)]
        lr: f64,
        /// AdamW weight decay. Raise this to fight overfitting on small data.
        #[arg(long, default_value_t = 1e-2)]
        weight_decay: f64,
        /// How many seeds to try. The winner is chosen on validation.
        ///
        /// More than one by default because the measured seed variance of this
        /// model is large — three seeds spanned 0.669 to 0.754 on the locked
        /// benchmark — so a single run is a lottery. The selection uses the
        /// validation split only.
        #[arg(long, default_value_t = 3)]
        seeds: u32,
        /// Dropout probability on the hidden layer. Adds no parameters.
        #[arg(long, default_value_t = 0.0)]
        dropout: f64,
        /// Dropout probability on the input features. Adds no parameters.
        #[arg(long, default_value_t = 0.0)]
        feature_dropout: f64,
        /// Label smoothing, in [0, 1). Adds no parameters.
        #[arg(long, default_value_t = 0.0)]
        label_smoothing: f64,
        /// Class weighting: none, balanced, sqrt or an exponent.
        ///
        /// `sqrt` is the middle ground: `w_c = (mean / n_c) ^ 0.5`.
        #[arg(long, default_value = "none")]
        class_weighting: String,
        /// Stop after this many epochs without a validation improvement.
        #[arg(long, default_value_t = 20)]
        patience: u32,
        /// Promote immediately instead of leaving the candidate on disk.
        #[arg(long)]
        promote: bool,
    },
}

#[derive(Debug, Subcommand)]
enum AudioAction {
    /// Capture audio and report levels, without running any command.
    Test {
        /// Seconds to listen for.
        #[arg(long, default_value_t = 5)]
        seconds: u32,
    },
    /// List capture devices.
    Devices,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli) {
        Ok(code) => ExitCode::from(code as u8),
        Err(err) => {
            let code = context::report_error(&err);
            ExitCode::from(code as u8)
        }
    }
}

fn run(cli: Cli) -> Result<i32, Error> {
    let options = RunOptions {
        assume_yes: cli.yes,
        assume_no: cli.no,
        no_audit: cli.no_audit,
        text_only: cli.text_only,
        json: cli.json,
    };

    // `setup` must work before a configuration exists.
    if let Command::Setup { force } = &cli.command {
        return commands::setup(cli.config.as_deref(), *force).map(|_| 0);
    }

    let mut ctx = context::load(cli.config.as_deref(), options)?;
    // Machine-readable output means machine-readable *only*: the reply is
    // carried inside the JSON document, so echoing it separately would make the
    // stream unparseable.
    if cli.json {
        ctx.config.tts.echo_to_terminal = false;
    }

    // Logging comes after config so `--log-level` can override it.
    let mut logging = ctx.config.logging.clone();
    if let Some(level) = &cli.log_level {
        logging.level = level.clone();
    }
    let filter = aaron_core::logging::init(&logging)?;
    if ctx.config.logging.file.is_empty() && cli.log_level.is_none() {
        tracing::debug!(filter = %filter, "logging initialised");
    }

    match cli.command {
        Command::Setup { .. } => unreachable!("handled above"),
        Command::Doctor => commands::doctor(&ctx).map(|_| 0),
        Command::Run => commands::run_interactive(&mut ctx),
        Command::Listen {
            no_wake,
            once,
            seconds,
            source,
            allow_fake_asr,
        } => {
            let choice = commands::AudioChoice::parse(&source)?;
            commands::listen(&mut ctx, no_wake, once, seconds, allow_fake_asr, &choice)
        }
        Command::Ask { text } => commands::ask(&mut ctx, &text),
        Command::Tools { action } => match action {
            ToolsAction::List => commands::tools_list(&ctx).map(|_| 0),
            ToolsAction::Test => commands::tools_test(&ctx),
        },
        Command::Permissions { action } => match action {
            PermissionsAction::Show => commands::permissions_show(&ctx).map(|_| 0),
            PermissionsAction::Test { tool } => commands::permissions_test(&ctx, tool.as_deref()),
        },
        Command::Memory { action } => match action {
            MemoryAction::List { kind, limit } => {
                commands::memory_list(&ctx, kind.as_deref(), limit).map(|_| 0)
            }
            MemoryAction::Search { query, limit } => {
                commands::memory_search(&ctx, &query, limit).map(|_| 0)
            }
            MemoryAction::Delete { id } => commands::memory_delete(&ctx, id),
            MemoryAction::Add { text, key, kind } => {
                commands::memory_add(&ctx, &text, key.as_deref(), &kind)
            }
            MemoryAction::Backup { dest } => {
                commands::memory_backup(&ctx, dest.as_deref()).map(|_| 0)
            }
        },
        Command::Learning { action } => match action {
            LearningAction::Pending => commands::learning_pending(&ctx).map(|_| 0),
            LearningAction::Approve { id } => commands::learning_review(&ctx, id, true),
            LearningAction::Reject { id } => commands::learning_review(&ctx, id, false),
            LearningAction::Evaluate {
                candidate,
                dataset,
                benchmark,
                split,
                threshold,
                safety_benchmark,
            } => commands::learning_evaluate(
                &ctx,
                candidate.as_deref(),
                dataset.as_deref(),
                benchmark.as_deref(),
                &split,
                threshold,
                safety_benchmark.as_deref(),
            ),
            LearningAction::Safety {
                candidate,
                benchmark,
                json,
                threshold,
                readiness_threshold,
            } => commands::learning_safety(
                &mut ctx,
                candidate.as_deref(),
                benchmark.as_deref(),
                json,
                threshold,
                readiness_threshold,
            ),
            LearningAction::Analyse {
                candidate,
                dataset,
                benchmark,
                format,
                out,
            } => commands::learning_analyse(
                &ctx,
                candidate.as_deref(),
                dataset.as_deref(),
                benchmark.as_deref(),
                &format,
                out.as_deref(),
            ),
            LearningAction::Promote { candidate } => {
                commands::learning_promote(&ctx, candidate.as_deref())
            }
            LearningAction::Rollback => commands::learning_rollback(&ctx),
        },
        Command::Brain { action } => match action {
            BrainAction::Train {
                dataset,
                epochs,
                seed,
                lr,
                weight_decay,
                patience,
                seeds,
                dropout,
                feature_dropout,
                label_smoothing,
                class_weighting,
                promote,
            } => commands::brain_train(
                &ctx,
                TrainingOverrides {
                    dataset: dataset.as_deref(),
                    epochs,
                    seed,
                    lr,
                    weight_decay,
                    patience,
                    seeds,
                    dropout,
                    feature_dropout,
                    label_smoothing,
                    class_weighting,
                    promote,
                },
            ),
        },
        Command::Model => commands::model_info(&ctx).map(|_| 0),
        Command::Benchmark { iterations } => {
            commands::benchmark(&ctx, iterations.clamp(1, 10_000)).map(|_| 0)
        }
        Command::Logs { limit } => commands::logs(&ctx, limit).map(|_| 0),
        Command::Dataset { action } => match action {
            DatasetAction::Check { dataset, json } => {
                dataset_cmd::check(&ctx, dataset.as_deref(), json)
            }
            DatasetAction::Report { dataset } => {
                dataset_cmd::report(&ctx, dataset.as_deref()).map(|_| 0)
            }
        },
        Command::Resources { action } => match action {
            ResourcesAction::Status => resources_cmd::status(&ctx, cli.json).map(|_| 0),
            ResourcesAction::Limits => resources_cmd::limits(&ctx).map(|_| 0),
            ResourcesAction::Stress { iterations } => {
                resources_cmd::stress(&ctx, iterations.clamp(100, 1_000_000), cli.json)
            }
            ResourcesAction::Soak { seconds } => {
                resources_cmd::soak(&ctx, seconds.clamp(1, 86_400))
            }
            ResourcesAction::Checkpoint { path } => {
                resources_cmd::checkpoint(&ctx, path.as_deref()).map(|_| 0)
            }
        },
        Command::Audio { action } => match action {
            AudioAction::Test { seconds } => commands::audio_test(&ctx, seconds),
            AudioAction::Devices => {
                for d in aaron_audio::describe_devices() {
                    println!("{d}");
                }
                Ok(0)
            }
        },
    }
}
