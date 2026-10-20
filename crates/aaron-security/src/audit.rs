//! Append-only, privacy-conscious audit log.
//!
//! Every proposed action produces up to three records: what was requested, what
//! was authorised, and what actually happened. Values are redacted and
//! truncated before they are written, so the audit trail can be shipped or
//! inspected without leaking the content of the user's files or memory.
//!
//! The log is JSONL: one object per line, append-only, never rewritten.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use aaron_core::error::{Error, Result};
use aaron_core::text::{redact_secrets, truncate_chars};
use serde::{Deserialize, Serialize};

/// Stage of the audit trail for a single action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// The agent proposed an action.
    Requested,
    /// The policy engine produced a verdict.
    Authorized,
    /// The tool ran (or failed) and this is the actual outcome.
    Result,
    /// The request was refused.
    Denied,
}

/// Outcome recorded in a [`Phase::Result`] record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    Ok,
    Error,
    Cancelled,
    TimedOut,
    Truncated,
}

/// One audit record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuditEvent {
    /// Unix milliseconds.
    pub ts_ms: u64,
    /// Random per-process session id, so records can be grouped.
    pub session: String,
    pub phase: Phase,
    /// Tool id, or `"-"` for agent-level events.
    pub tool: String,
    /// Intent that led here, when known.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent: Option<String>,
    /// Redacted, truncated summary of the arguments.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
    /// Redacted, truncated policy verdict.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decision: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outcome: Option<Outcome>,
    /// Bytes produced, before truncation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_bytes: Option<usize>,
    /// Whether the log truncates the value it stored.
    #[serde(default)]
    pub redacted: bool,
    #[serde(default)]
    pub latency_ms: Option<u64>,
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// A short random session identifier derived from process id and start time.
///
/// Not a security token; it only has to be unique enough to group records from
/// one run of AARON.
pub fn session_id() -> String {
    let pid = std::process::id();
    let start = now_ms();
    let digest = aaron_core::text::fnv1a64(format!("{pid}-{start}").as_bytes());
    format!("{start:x}{:x}", digest & 0xffff_ffff)
}

/// Append-only JSONL audit writer.
///
/// Cloning shares the same file handle and lock, so all components of a running
/// agent write to one serialised stream.
#[derive(Clone)]
pub struct AuditLog {
    inner: Arc<AuditInner>,
}

struct AuditInner {
    path: PathBuf,
    file: Mutex<Option<std::fs::File>>,
    session: String,
    max_value_chars: usize,
    enabled: bool,
}

impl AuditLog {
    /// Open (creating parents) the audit file. An empty path disables auditing
    /// and every method becomes a no-op that still returns `Ok`.
    pub fn open(path: impl AsRef<Path>, max_value_chars: usize) -> Result<AuditLog> {
        let path = path.as_ref().to_path_buf();
        let enabled = !path.as_os_str().is_empty();
        let file = if enabled {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
            Some(
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path)
                    .map_err(|e| Error::io(&path, e))?,
            )
        } else {
            None
        };
        Ok(AuditLog {
            inner: Arc::new(AuditInner {
                path,
                file: Mutex::new(file),
                session: session_id(),
                max_value_chars,
                enabled,
            }),
        })
    }

    /// A log that discards everything. Used by tests and by `--no-audit`.
    pub fn disabled() -> AuditLog {
        AuditLog {
            inner: Arc::new(AuditInner {
                path: PathBuf::new(),
                file: Mutex::new(None),
                session: session_id(),
                max_value_chars: 160,
                enabled: false,
            }),
        }
    }

    /// Path the log writes to, or an empty path when disabled.
    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    fn sanitise(&self, value: &str) -> (String, bool) {
        let redacted = redact_secrets(value);
        let clipped = truncate_chars(&redacted, self.inner.max_value_chars);
        let changed = clipped != value;
        (clipped, changed)
    }

    /// Append one record.
    pub fn record(&self, mut event: AuditEvent) -> Result<()> {
        if !self.inner.enabled {
            return Ok(());
        }
        event.ts_ms = now_ms();
        event.session = self.inner.session.clone();
        let line = serde_json::to_string(&event).map_err(|e| Error::Storage(e.to_string()))?;
        let mut guard = self
            .inner
            .file
            .lock()
            .map_err(|_| Error::Storage("audit log mutex poisoned".into()))?;
        if let Some(file) = guard.as_mut() {
            writeln!(file, "{line}").map_err(|e| Error::io(&self.inner.path, e))?;
            file.flush().map_err(|e| Error::io(&self.inner.path, e))?;
        }
        Ok(())
    }

    /// Record that an action was proposed.
    pub fn requested(&self, tool: &str, intent: Option<&str>, arguments: &str) -> Result<()> {
        let (args, redacted) = self.sanitise(arguments);
        self.record(AuditEvent {
            ts_ms: 0,
            session: String::new(),
            phase: Phase::Requested,
            tool: tool.into(),
            intent: intent.map(|s| s.to_string()),
            arguments: Some(args),
            decision: None,
            outcome: None,
            output_bytes: None,
            redacted,
            latency_ms: None,
        })
    }

    /// Record the policy verdict.
    pub fn authorized(&self, tool: &str, decision: &str) -> Result<()> {
        let (dec, redacted) = self.sanitise(decision);
        self.record(AuditEvent {
            ts_ms: 0,
            session: String::new(),
            phase: Phase::Authorized,
            tool: tool.into(),
            intent: None,
            arguments: None,
            decision: Some(dec),
            outcome: None,
            output_bytes: None,
            redacted,
            latency_ms: None,
        })
    }

    /// Record a refusal.
    pub fn denied(&self, tool: &str, reason: &str) -> Result<()> {
        let (reason, redacted) = self.sanitise(reason);
        self.record(AuditEvent {
            ts_ms: 0,
            session: String::new(),
            phase: Phase::Denied,
            tool: tool.into(),
            intent: None,
            arguments: None,
            decision: Some(reason),
            outcome: None,
            output_bytes: None,
            redacted,
            latency_ms: None,
        })
    }

    /// Record the real outcome of an executed tool.
    pub fn result(
        &self,
        tool: &str,
        outcome: Outcome,
        output_bytes: usize,
        latency_ms: u64,
    ) -> Result<()> {
        self.record(AuditEvent {
            ts_ms: 0,
            session: String::new(),
            phase: Phase::Result,
            tool: tool.into(),
            intent: None,
            arguments: None,
            decision: None,
            outcome: Some(outcome),
            output_bytes: Some(output_bytes),
            redacted: false,
            latency_ms: Some(latency_ms),
        })
    }

    /// Read the most recent `limit` records, newest last.
    pub fn tail(&self, limit: usize) -> Result<Vec<AuditEvent>> {
        if !self.inner.enabled {
            return Ok(Vec::new());
        }
        let text = std::fs::read_to_string(&self.inner.path)
            .map_err(|e| Error::io(&self.inner.path, e))?;
        let mut events: Vec<AuditEvent> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| serde_json::from_str::<AuditEvent>(l).ok())
            .collect();
        if events.len() > limit {
            events.drain(..events.len() - limit);
        }
        Ok(events)
    }
}

impl std::fmt::Debug for AuditLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuditLog")
            .field("path", &self.inner.path)
            .field("enabled", &self.inner.enabled)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::policy::{Effect, LimitPolicy, PolicyFile};

    #[test]
    fn writes_requested_authorized_result_triples() {
        let tmp = tempfile::tempdir().unwrap();
        let log = AuditLog::open(tmp.path().join("audit.jsonl"), 200).unwrap();
        log.requested("read_file", Some("read_file"), "path=Cargo.toml")
            .unwrap();
        log.authorized("read_file", "allow").unwrap();
        log.result("read_file", Outcome::Ok, 1234, 7).unwrap();

        let events = log.tail(10).unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].phase, Phase::Requested);
        assert_eq!(events[1].phase, Phase::Authorized);
        assert_eq!(events[2].phase, Phase::Result);
        assert_eq!(events[2].output_bytes, Some(1234));
        assert_eq!(events[2].latency_ms, Some(7));
        assert!(events.iter().all(|e| !e.session.is_empty()));
    }

    #[test]
    fn secrets_are_redacted_before_writing() {
        let tmp = tempfile::tempdir().unwrap();
        let log = AuditLog::open(tmp.path().join("audit.jsonl"), 500).unwrap();
        log.requested(
            "open_url",
            Some("open_url"),
            "url=https://x/y?token=SUPERSECRET",
        )
        .unwrap();
        let raw = std::fs::read_to_string(tmp.path().join("audit.jsonl")).unwrap();
        assert!(
            !raw.contains("SUPERSECRET"),
            "secret leaked into audit log: {raw}"
        );
        assert!(raw.contains("[REDACTED]"));
        assert!(log.tail(1).unwrap()[0].redacted);
    }

    #[test]
    fn long_values_are_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let log = AuditLog::open(tmp.path().join("audit.jsonl"), 20).unwrap();
        log.requested("read_file", None, &"x".repeat(500)).unwrap();
        let ev = log.tail(1).unwrap().remove(0);
        assert!(ev.arguments.unwrap().chars().count() <= 20);
        assert!(ev.redacted);
    }

    #[test]
    fn tail_returns_newest_last_and_respects_limit() {
        let tmp = tempfile::tempdir().unwrap();
        let log = AuditLog::open(tmp.path().join("audit.jsonl"), 200).unwrap();
        for i in 0..10 {
            log.requested("t", None, &format!("n={i}")).unwrap();
        }
        let tail = log.tail(3).unwrap();
        assert_eq!(tail.len(), 3);
        assert!(tail[2].arguments.as_deref().unwrap().ends_with("n=9"));
    }

    #[test]
    fn disabled_log_writes_nothing_and_does_not_fail() {
        let log = AuditLog::disabled();
        log.requested("x", None, "y").unwrap();
        assert!(log.tail(5).unwrap().is_empty());
        assert_eq!(log.path().as_os_str(), "");
    }

    #[test]
    fn summary_reports_effect_counts() {
        let policy = PolicyFile {
            version: 1,
            unknown_tool: Effect::Deny,
            limits: LimitPolicy::default(),
            tools: Default::default(),
        };
        let engine = crate::policy::PolicyEngine::new(policy, Vec::new());
        let s = engine.summary();
        assert_eq!(s.total_entries, 0);
        assert_eq!(s.unknown_tool, Effect::Deny);
        assert!(s.refuse_root);
    }
}
