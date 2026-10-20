//! Logging setup.
//!
//! AARON never logs raw audio, full file contents or memory values at any level.
//! Callers that want to record something sensitive must pass it through
//! [`crate::text::redact_secrets`] and [`crate::text::truncate_chars`] first;
//! `aaron-audit` does this automatically.

use std::fs::OpenOptions;
use std::path::Path;

use tracing_subscriber::EnvFilter;

use crate::config::LoggingConfig;
use crate::error::{Error, Result};

/// Initialise the global tracing subscriber.
///
/// Idempotent: calling this twice in one process is not an error, which keeps
/// integration tests simple. Returns the effective filter string so callers can
/// report it.
pub fn init(cfg: &LoggingConfig) -> Result<String> {
    // `RUST_LOG` always wins, so a developer can debug without editing config.
    let filter_spec =
        std::env::var("RUST_LOG").unwrap_or_else(|_| format!("aaron={},{}", cfg.level, cfg.level));

    match cfg.format.as_str() {
        "text" => {
            let file_raw = cfg.file.clone();
            let subscriber = tracing_subscriber::fmt()
                .with_env_filter(EnvFilter::new(&filter_spec))
                .with_writer(move || TeeWriter::open(&file_raw))
                .finish();
            let _ = tracing::subscriber::set_global_default(subscriber);
        }
        "json" => {
            // JSON always goes to stderr: a structured stream is meant to be
            // piped, and writing it to the same file as the human log would
            // interleave two formats.
            let subscriber = tracing_subscriber::fmt()
                .json()
                .with_env_filter(EnvFilter::new(&filter_spec))
                .with_writer(std::io::stderr)
                .finish();
            let _ = tracing::subscriber::set_global_default(subscriber);
        }
        other => {
            return Err(Error::Config(format!(
                "logging.format `{other}` is invalid"
            )))
        }
    }
    Ok(filter_spec)
}

fn open_log_file(raw: &str) -> std::io::Result<std::fs::File> {
    let path = Path::new(raw);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    OpenOptions::new().create(true).append(true).open(path)
}

/// Writes every log line to stderr and, when configured, appends it to a file.
///
/// If the file cannot be opened the writer degrades to stderr-only rather than
/// losing the log entirely; `aaron doctor` reports the file as unwritable.
struct TeeWriter {
    file: Option<std::fs::File>,
}

impl TeeWriter {
    fn open(raw: &str) -> Self {
        let file = if raw.is_empty() {
            None
        } else {
            open_log_file(raw).ok()
        };
        TeeWriter { file }
    }
}

impl std::io::Write for TeeWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let stderr_result = std::io::stderr().write(buf);
        if let Some(file) = self.file.as_mut() {
            // A failing log file must never take the assistant down.
            let _ = file.write_all(buf);
        }
        stderr_result
    }

    fn flush(&mut self) -> std::io::Result<()> {
        std::io::stderr().flush()?;
        if let Some(file) = self.file.as_mut() {
            let _ = file.flush();
        }
        Ok(())
    }
}

/// Rotate a log file when it grows past `max_bytes`, keeping one `.1` backup.
///
/// Called from the CLI entry point, not from the logger itself, so that
/// rotation never races with a write in progress.
pub fn rotate_if_needed(path: &Path, max_bytes: u64) -> Result<bool> {
    let meta = match std::fs::metadata(path) {
        Ok(m) => m,
        Err(_) => return Ok(false),
    };
    if meta.len() < max_bytes {
        return Ok(false);
    }
    let backup = path.with_extension("log.1");
    let _ = std::fs::remove_file(&backup);
    std::fs::rename(path, &backup).map_err(|e| Error::io(path, e))?;
    Ok(true)
}
