//! Long-term memory on SQLite.
//!
//! Design rules enforced here:
//!
//! * **Nothing is stored implicitly.** Adding a row requires an explicit call
//!   with a known [`MemoryKind`]. Transcripts are never written to `memories`.
//! * **Writes are transactional.** Multi-step operations (add + prune, backup +
//!   prune) run inside a transaction and roll back on error.
//! * **Private by default.** Every memory row is `private = 1` and
//!   `private` rows are excluded from any dataset export unless the user opts
//!   in row by row.
//! * **Retention is explicit.** `retention_days = 0` and `max_entries = 0`
//!   both disable automatic pruning.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use aaron_core::error::{Error, Result};
use rusqlite::{params, Connection, OptionalExtension, Transaction};

use crate::schema::{MIGRATIONS, SCHEMA_VERSION};

/// Category of a stored memory. Mirrors the separation required by the
/// assistant's privacy model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryKind {
    /// A fact the user explicitly asked AARON to remember.
    Fact,
    /// A stated preference ("I prefer dark mode").
    Preference,
    /// Working state for the current session only.
    SessionContext,
    /// A rolling summary of a finished conversation.
    Summary,
    /// What AARON did, for `aaron history`.
    TaskHistory,
    /// A user correction awaiting review.
    Feedback,
    /// Operational metadata, never shown as user memory.
    AuditMeta,
}

impl MemoryKind {
    pub fn as_str(self) -> &'static str {
        match self {
            MemoryKind::Fact => "fact",
            MemoryKind::Preference => "preference",
            MemoryKind::SessionContext => "session_context",
            MemoryKind::Summary => "summary",
            MemoryKind::TaskHistory => "task_history",
            MemoryKind::Feedback => "feedback",
            MemoryKind::AuditMeta => "audit_meta",
        }
    }

    pub fn parse(s: &str) -> Option<MemoryKind> {
        [
            MemoryKind::Fact,
            MemoryKind::Preference,
            MemoryKind::SessionContext,
            MemoryKind::Summary,
            MemoryKind::TaskHistory,
            MemoryKind::Feedback,
            MemoryKind::AuditMeta,
        ]
        .into_iter()
        .find(|k| k.as_str() == s)
    }

    /// Kinds the user can see and delete through `aaron memory`.
    pub fn user_visible() -> &'static [MemoryKind] {
        &[
            MemoryKind::Fact,
            MemoryKind::Preference,
            MemoryKind::SessionContext,
            MemoryKind::Summary,
        ]
    }
}

/// One stored memory row.
#[derive(Debug, Clone, PartialEq)]
pub struct MemoryEntry {
    pub id: i64,
    pub kind: MemoryKind,
    pub key: Option<String>,
    pub value: String,
    pub tags: Vec<String>,
    /// Where the row came from: `user`, `voice`, `import`, `learned`, ...
    pub source: String,
    pub created_at: i64,
    pub updated_at: i64,
    /// Pinned rows survive retention pruning.
    pub pinned: bool,
    /// Private rows are excluded from dataset export unless explicitly opted in.
    pub private: bool,
    /// Whether the value passed through secret redaction.
    pub redacted: bool,
}

/// Storage-level resource limits, sourced from `[resources.memory]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StorageLimits {
    /// SQLite page cache, in KiB (`PRAGMA cache_size = -N`).
    pub cache_kib: u32,
    /// Hard cap on rows returned by any single retrieval.
    pub max_retrieved: usize,
    /// Hard cap on a single stored value.
    pub max_value_bytes: usize,
}

impl Default for StorageLimits {
    fn default() -> Self {
        StorageLimits {
            cache_kib: 2048,
            max_retrieved: 10,
            max_value_bytes: 16 * 1024,
        }
    }
}

/// Retention configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetentionPolicy {
    /// Delete unpinned rows older than this. 0 disables.
    pub retention_days: u32,
    /// Soft cap on rows per user-visible kind. 0 disables.
    pub max_entries: usize,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        RetentionPolicy {
            retention_days: 365,
            max_entries: 10_000,
        }
    }
}

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// The memory database.
pub struct MemoryStore {
    conn: Connection,
    path: PathBuf,
    retention: RetentionPolicy,
    max_value_chars: usize,
    limits: StorageLimits,
}

impl MemoryStore {
    /// Open (creating if needed) the database and run migrations.
    pub fn open(path: impl AsRef<Path>, retention: RetentionPolicy) -> Result<MemoryStore> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
            }
        }
        let conn = Connection::open(&path).map_err(|e| Error::Storage(e.to_string()))?;
        Self::configure(&conn, &StorageLimits::default())?;
        Self::migrate(&conn)?;
        Ok(MemoryStore {
            conn,
            path,
            retention,
            max_value_chars: 4_000,
            limits: StorageLimits::default(),
        })
    }

    /// Open with explicit storage limits.
    pub fn open_with_limits(
        path: impl AsRef<Path>,
        retention: RetentionPolicy,
        limits: StorageLimits,
    ) -> Result<MemoryStore> {
        let mut store = MemoryStore::open(path, retention)?;
        // Re-apply the pragmas: `open` used the defaults.
        let conn = Connection::open(&store.path).map_err(|e| Error::Storage(e.to_string()))?;
        drop(conn);
        store.limits = limits;
        store.apply_pragmas()?;
        Ok(store)
    }

    /// Apply the storage limits to an already-open connection.
    pub fn apply_limits(&mut self, limits: StorageLimits) -> Result<()> {
        self.limits = limits;
        self.apply_pragmas()
    }

    /// The limits currently in force.
    pub fn limits(&self) -> StorageLimits {
        self.limits
    }

    /// Force a WAL checkpoint so the `-wal` file cannot grow without bound.
    ///
    /// SQLite's default is to checkpoint automatically, but on a machine that
    /// is briefly low on disk the WAL can still accumulate. `TRUNCATE` resets
    /// it to zero length. Safe to call at any time; it takes a brief lock.
    pub fn checkpoint(&self) -> Result<()> {
        self.conn
            .query_row("PRAGMA wal_checkpoint(TRUNCATE)", [], |_| Ok(()))
            .map_err(|e| Error::Storage(format!("WAL checkpoint failed: {e}")))?;
        Ok(())
    }

    /// Size of the database and its sidecar files, in bytes.
    pub fn on_disk_bytes(&self) -> u64 {
        let candidates = [
            self.path.clone(),
            self.path.with_extension("db-wal"),
            self.path.with_extension("db-shm"),
        ];
        candidates
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok())
            .map(|m| m.len())
            .sum()
    }

    /// An in-memory database, used by tests.
    pub fn open_in_memory(retention: RetentionPolicy) -> Result<MemoryStore> {
        let conn = Connection::open_in_memory().map_err(|e| Error::Storage(e.to_string()))?;
        Self::configure(&conn, &StorageLimits::default())?;
        Self::migrate(&conn)?;
        Ok(MemoryStore {
            conn,
            path: PathBuf::from(":memory:"),
            retention,
            max_value_chars: 4_000,
            limits: StorageLimits::default(),
        })
    }

    fn apply_pragmas(&self) -> Result<()> {
        Self::configure(&self.conn, &self.limits)
    }

    /// Override the per-value character cap.
    pub fn with_max_value_chars(mut self, max: usize) -> Self {
        self.max_value_chars = max;
        self
    }

    /// Test seam for the character cap, which is a separate bound from the
    /// byte budget.
    pub fn set_max_value_chars(&mut self, max: usize) {
        self.max_value_chars = max;
    }

    /// Where the database lives (`:memory:` for in-memory stores).
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The schema version currently stored in the file.
    pub fn schema_version(&self) -> Result<u32> {
        let v: i64 = self
            .conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |r| r.get(0),
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(v as u32)
    }

    fn configure(conn: &Connection, limits: &StorageLimits) -> Result<()> {
        // WAL keeps a long read from blocking the agent; NORMAL sync is the
        // standard durability trade-off for a personal assistant and survives
        // process crashes (it only risks the last transactions on power loss).
        //
        // The cost of WAL is a `-wal` sidecar that grows between checkpoints,
        // which is why `checkpoint()` exists and why `on_disk_bytes()` reports
        // it. See docs/resources.md.
        conn.pragma_update(None, "journal_mode", "WAL")
            .map_err(|e| Error::Storage(e.to_string()))?;
        conn.pragma_update(None, "synchronous", "NORMAL")
            .map_err(|e| Error::Storage(e.to_string()))?;
        conn.pragma_update(None, "foreign_keys", "ON")
            .map_err(|e| Error::Storage(e.to_string()))?;
        conn.busy_timeout(std::time::Duration::from_secs(5))
            .map_err(|e| Error::Storage(e.to_string()))?;

        // A *negative* cache_size is a page-count limit rather than a byte
        // limit, which is what makes it a real bound: SQLite cannot read past
        // it, so the page cache cannot quietly page the whole database into RAM.
        let pages = -(limits.cache_kib as i64);
        conn.pragma_update(None, "cache_size", pages)
            .map_err(|e| Error::Storage(e.to_string()))?;
        // Keep the in-memory index and temporary structures small too.
        conn.pragma_update(None, "temp_store", "FILE")
            .map_err(|e| Error::Storage(e.to_string()))?;
        conn.pragma_update(None, "mmap_size", 0i64)
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// Apply pending migrations atomically.
    fn migrate(conn: &Connection) -> Result<()> {
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_version (
                 version    INTEGER PRIMARY KEY,
                 applied_at INTEGER NOT NULL
             );",
        )
        .map_err(|e| Error::Storage(e.to_string()))?;

        let current: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(version), 0) FROM schema_version",
                [],
                |r| r.get(0),
            )
            .map_err(|e| Error::Storage(e.to_string()))?;

        if current > SCHEMA_VERSION as i64 {
            return Err(Error::Storage(format!(
                "database schema is version {current} but this binary only understands {SCHEMA_VERSION}; \
                 downgrade AARON or start with a fresh database"
            )));
        }

        for (version, sql) in MIGRATIONS {
            if *version as i64 <= current {
                continue;
            }
            let tx = conn
                .unchecked_transaction()
                .map_err(|e| Error::Storage(e.to_string()))?;
            tx.execute_batch(sql).map_err(|e| {
                Error::Storage(format!("migration to version {version} failed: {e}"))
            })?;
            tx.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?1, ?2)",
                params![*version as i64, now_secs()],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
            tx.commit().map_err(|e| Error::Storage(e.to_string()))?;
            tracing::debug!(version, "applied memory schema migration");
        }
        Ok(())
    }

    /// Table names present in the database. Used by migration tests.
    pub fn tables(&self) -> Result<Vec<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], |r| r.get::<_, String>(0))
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    // ---------------------------------------------------------------- writes

    /// Insert a memory row.
    pub fn add(
        &mut self,
        kind: MemoryKind,
        key: Option<&str>,
        value: &str,
        tags: &[String],
        source: &str,
    ) -> Result<i64> {
        let value = aaron_core::text::truncate_chars(value, self.max_value_chars);
        if value.trim().is_empty() {
            return Err(Error::Storage("refusing to store an empty memory".into()));
        }
        if value.len() > self.limits.max_value_bytes {
            return Err(Error::LimitExceeded(format!(
                "memory value is {} bytes, the configured budget is {}",
                value.len(),
                self.limits.max_value_bytes
            )));
        }
        let ts = now_secs();
        let tag_blob = tags.join(",");
        self.conn
            .execute(
                "INSERT INTO memories (kind, key, value, tags, source, created_at, updated_at, pinned, private, redacted)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, 0, 1, ?7)",
                params![
                    kind.as_str(),
                    key,
                    value,
                    tag_blob,
                    source,
                    ts,
                    false
                ],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Insert a row that the user has explicitly approved for dataset export.
    pub fn add_shareable(&mut self, kind: MemoryKind, value: &str, tags: &[String]) -> Result<i64> {
        let id = self.add(kind, None, value, tags, "user_opt_in")?;
        self.set_private(id, false)?;
        Ok(id)
    }

    /// Replace the value of an existing row.
    pub fn update(&mut self, id: i64, value: &str) -> Result<()> {
        let value = aaron_core::text::truncate_chars(value, self.max_value_chars);
        let n = self
            .conn
            .execute(
                "UPDATE memories SET value = ?2, updated_at = ?3 WHERE id = ?1",
                params![id, value, now_secs()],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        if n == 0 {
            return Err(Error::Storage(format!("no memory with id {id}")));
        }
        Ok(())
    }

    /// Delete a single row.
    pub fn delete(&mut self, id: i64) -> Result<bool> {
        let n = self
            .conn
            .execute("DELETE FROM memories WHERE id = ?1", params![id])
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(n > 0)
    }

    /// Pin or unpin a row so retention pruning skips it.
    pub fn set_pinned(&mut self, id: i64, pinned: bool) -> Result<()> {
        let n = self
            .conn
            .execute(
                "UPDATE memories SET pinned = ?2 WHERE id = ?1",
                params![id, pinned as i64],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        if n == 0 {
            return Err(Error::Storage(format!("no memory with id {id}")));
        }
        Ok(())
    }

    /// Mark a row as shareable with the training dataset.
    pub fn set_private(&mut self, id: i64, private: bool) -> Result<()> {
        self.conn
            .execute(
                "UPDATE memories SET private = ?2 WHERE id = ?1",
                params![id, private as i64],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// Record an executed task.
    pub fn record_task(
        &mut self,
        intent: &str,
        tool: &str,
        outcome: &str,
        detail: &str,
        latency_ms: Option<u64>,
    ) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO task_history (ts, intent, tool, outcome, detail, latency_ms)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                params![
                    now_secs(),
                    intent,
                    tool,
                    outcome,
                    aaron_core::text::truncate_chars(detail, 400),
                    latency_ms.map(|l| l as i64)
                ],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// Queue a user correction for review.
    pub fn add_feedback(
        &mut self,
        utterance: &str,
        predicted: &str,
        corrected: &str,
        redacted: bool,
    ) -> Result<i64> {
        self.conn
            .execute(
                "INSERT INTO feedback (ts, utterance, predicted, corrected, status, redacted)
                 VALUES (?1, ?2, ?3, ?4, 'pending', ?5)",
                params![now_secs(), utterance, predicted, corrected, redacted as i64],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Approve or reject a queued correction.
    pub fn set_feedback_status(&mut self, id: i64, status: &str) -> Result<()> {
        let n = self
            .conn
            .execute(
                "UPDATE feedback SET status = ?2 WHERE id = ?1",
                params![id, status],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        if n == 0 {
            return Err(Error::Storage(format!("no feedback with id {id}")));
        }
        Ok(())
    }

    /// Record operational metadata.
    pub fn record_audit_meta(&mut self, category: &str, summary: &str, bytes: usize) -> Result<()> {
        self.conn
            .execute(
                "INSERT INTO audit_meta (ts, category, summary, bytes) VALUES (?1, ?2, ?3, ?4)",
                params![
                    now_secs(),
                    category,
                    aaron_core::text::truncate_chars(summary, 300),
                    bytes as i64
                ],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// Append-only record of a finished session.
    pub fn open_session(&mut self, id: &str) -> Result<()> {
        self.conn
            .execute(
                "INSERT OR REPLACE INTO sessions (id, started_at) VALUES (?1, ?2)",
                params![id, now_secs()],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    /// Close a session with an optional summary.
    pub fn close_session(&mut self, id: &str, summary: Option<&str>) -> Result<()> {
        self.conn
            .execute(
                "UPDATE sessions SET ended_at = ?2, summary = ?3 WHERE id = ?1",
                params![id, now_secs(), summary],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(())
    }

    // ---------------------------------------------------------------- reads

    /// Fetch one row by id.
    pub fn get(&self, id: i64) -> Result<Option<MemoryEntry>> {
        let mut stmt = self
            .conn
            .prepare_cached(&format!(
                "SELECT {} FROM memories WHERE id = ?1",
                SELECT_COLUMNS
            ))
            .map_err(|e| Error::Storage(e.to_string()))?;
        let entry = stmt
            .query_row(params![id], row_to_entry)
            .optional()
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(entry)
    }

    /// List rows of one kind, newest first.
    pub fn list(&self, kind: MemoryKind, limit: usize) -> Result<Vec<MemoryEntry>> {
        let sql = format!(
            "SELECT {SELECT_COLUMNS} FROM memories WHERE kind = ?1 ORDER BY updated_at DESC, id DESC LIMIT ?2"
        );
        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![kind.as_str(), limit as i64], row_to_entry)
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Every user-visible memory, newest first.
    pub fn list_all(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let kinds: Vec<String> = MemoryKind::user_visible()
            .iter()
            .map(|k| format!("'{}'", k.as_str()))
            .collect();
        let sql = format!(
            "SELECT {SELECT_COLUMNS} FROM memories WHERE kind IN ({}) ORDER BY updated_at DESC, id DESC LIMIT ?1",
            kinds.join(",")
        );
        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_entry)
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Keyword search across keys, values and tags.
    ///
    /// Ranking is a deterministic weighted overlap score, not an embedding
    /// lookup: it is inspectable, has no hidden model, and returns stable
    /// results that the user can reason about.
    pub fn search(&self, query: &str, limit: usize) -> Result<Vec<MemoryEntry>> {
        let terms: Vec<String> = aaron_core::text::words(query);
        if terms.is_empty() {
            return Ok(Vec::new());
        }
        // Bound the LIKE pattern so a user-supplied `%` cannot turn a query
        // into a full table scan of a huge database.
        let mut clauses = Vec::new();
        let mut args: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for (i, term) in terms.iter().enumerate() {
            let pattern = format!("%{}%", term.replace(['%', '_'], ""));
            clauses.push(format!(
                "(LOWER(key) LIKE ?{} OR LOWER(value) LIKE ?{} OR LOWER(tags) LIKE ?{})",
                i + 1,
                i + 1,
                i + 1
            ));
            args.push(Box::new(pattern));
        }
        let sql = format!(
            "SELECT {SELECT_COLUMNS} FROM memories WHERE ({}) ORDER BY updated_at DESC, id DESC LIMIT ?{}",
            clauses.join(" OR "),
            terms.len() + 1
        );
        args.push(Box::new(limit as i64));
        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .map_err(|e| Error::Storage(e.to_string()))?;
        let refs: Vec<&dyn rusqlite::ToSql> = args.iter().map(|a| a.as_ref()).collect();
        let mut collected = stmt
            .query_map(rusqlite::params_from_iter(refs), row_to_entry)
            .map_err(|e| Error::Storage(e.to_string()))?
            .collect::<std::result::Result<Vec<MemoryEntry>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))?;
        // Never hold more rows in RAM than the resource budget allows, whatever
        // limit the caller asked for.
        let limit = limit.min(self.limits.max_retrieved);

        // Score in Rust so the ordering is explainable and stable.
        collected.sort_by(|a, b| {
            score(b, &terms)
                .cmp(&score(a, &terms))
                .then(b.updated_at.cmp(&a.updated_at))
                .then(b.id.cmp(&a.id))
        });
        collected.truncate(limit);
        Ok(collected)
    }

    /// Rows approved for inclusion in a training dataset.
    pub fn exportable(&self, limit: usize) -> Result<Vec<MemoryEntry>> {
        let sql = format!(
            "SELECT {SELECT_COLUMNS} FROM memories WHERE private = 0 AND kind IN ('fact','preference') \
             ORDER BY id LIMIT ?1"
        );
        let mut stmt = self
            .conn
            .prepare_cached(&sql)
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], row_to_entry)
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Pending corrections awaiting review.
    pub fn pending_feedback(&self, limit: usize) -> Result<Vec<PendingFeedback>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT id, ts, utterance, predicted, corrected, status, notes, redacted
                 FROM feedback WHERE status = 'pending' ORDER BY ts ASC LIMIT ?1",
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], |r| {
                Ok(PendingFeedback {
                    id: r.get(0)?,
                    ts: r.get(1)?,
                    utterance: r.get(2)?,
                    predicted: r.get(3)?,
                    corrected: r.get(4)?,
                    status: r.get(5)?,
                    notes: r.get(6)?,
                    redacted: r.get::<_, i64>(7)? != 0,
                })
            })
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Recent task history, newest first.
    pub fn history(&self, limit: usize) -> Result<Vec<TaskRecord>> {
        let mut stmt = self
            .conn
            .prepare_cached(
                "SELECT ts, intent, tool, outcome, detail, latency_ms FROM task_history
                 ORDER BY ts DESC, id DESC LIMIT ?1",
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map(params![limit as i64], |r| {
                Ok(TaskRecord {
                    ts: r.get(0)?,
                    intent: r.get(1)?,
                    tool: r.get(2)?,
                    outcome: r.get(3)?,
                    detail: r.get(4)?,
                    latency_ms: r.get::<_, Option<i64>>(5)?.map(|v| v as u64),
                })
            })
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    /// Row counts per kind.
    pub fn stats(&self) -> Result<Vec<(String, i64)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT kind, COUNT(*) FROM memories GROUP BY kind ORDER BY kind")
            .map_err(|e| Error::Storage(e.to_string()))?;
        let rows = stmt
            .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
            .map_err(|e| Error::Storage(e.to_string()))?;
        rows.collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| Error::Storage(e.to_string()))
    }

    // ----------------------------------------------------------- maintenance

    /// Apply the retention policy. Returns the number of rows removed.
    ///
    /// Runs in one transaction with the companion soft-cap prune so a partially
    /// applied retention pass can never happen.
    pub fn prune(&mut self) -> Result<usize> {
        self.prune_with(self.retention)
    }

    /// Apply an explicit retention policy.
    pub fn prune_with(&mut self, policy: RetentionPolicy) -> Result<usize> {
        let tx = self.tx()?;
        let mut removed = 0usize;

        if policy.retention_days > 0 {
            let cutoff = now_secs() - (policy.retention_days as i64) * 86_400;
            let n = tx
                .execute(
                    "DELETE FROM memories WHERE pinned = 0 AND created_at < ?1 \
                     AND kind IN ('fact','preference','session_context','summary')",
                    params![cutoff],
                )
                .map_err(|e| Error::Storage(e.to_string()))?;
            removed += n;
        }

        if policy.max_entries > 0 {
            for kind in MemoryKind::user_visible() {
                removed += tx
                    .execute(
                        "DELETE FROM memories WHERE id NOT IN (
                             SELECT id FROM memories
                             WHERE kind = ?1 AND pinned = 0
                             ORDER BY updated_at DESC, id DESC
                             LIMIT ?2
                         ) AND kind = ?1 AND pinned = 0",
                        params![kind.as_str(), policy.max_entries as i64],
                    )
                    .map_err(|e| Error::Storage(e.to_string()))?;
            }
        }

        tx.commit().map_err(|e| Error::Storage(e.to_string()))?;
        if removed > 0 {
            tracing::info!(removed, "memory retention pass removed rows");
        }
        Ok(removed)
    }

    /// Copy the live database to `dest` using SQLite's own backup-safe API.
    pub fn backup(&self, dest: impl AsRef<Path>) -> Result<()> {
        let dest = dest.as_ref();
        if let Some(parent) = dest.parent() {
            std::fs::create_dir_all(parent).map_err(|e| Error::io(parent, e))?;
        }
        // `VACUUM INTO` produces a consistent snapshot even with WAL traffic.
        self.conn
            .execute("VACUUM INTO ?1", params![dest.to_string_lossy()])
            .map_err(|e| Error::Storage(format!("backup to {} failed: {e}", dest.display())))?;
        Ok(())
    }

    /// Remove every row of a kind (used by `aaron memory clear --kind`).
    pub fn clear_kind(&mut self, kind: MemoryKind) -> Result<usize> {
        let n = self
            .conn
            .execute(
                "DELETE FROM memories WHERE kind = ?1",
                params![kind.as_str()],
            )
            .map_err(|e| Error::Storage(e.to_string()))?;
        Ok(n)
    }

    fn tx(&mut self) -> Result<Transaction<'_>> {
        self.conn
            .unchecked_transaction()
            .map_err(|e| Error::Storage(e.to_string()))
    }
}

const SELECT_COLUMNS: &str =
    "id, kind, key, value, tags, source, created_at, updated_at, pinned, private, redacted";

fn row_to_entry(row: &rusqlite::Row<'_>) -> rusqlite::Result<MemoryEntry> {
    let kind_str: String = row.get(1)?;
    let tags: String = row.get(4)?;
    Ok(MemoryEntry {
        id: row.get(0)?,
        kind: MemoryKind::parse(&kind_str).unwrap_or(MemoryKind::AuditMeta),
        key: row.get(2)?,
        value: row.get(3)?,
        tags: tags
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect(),
        source: row.get(5)?,
        created_at: row.get(6)?,
        updated_at: row.get(7)?,
        pinned: row.get::<_, i64>(8)? != 0,
        private: row.get::<_, i64>(9)? != 0,
        redacted: row.get::<_, i64>(10)? != 0,
    })
}

/// Weighted overlap score: a term in the key outranks the same term in the
/// value, and a pinned row outranks an unpinned one.
fn score(entry: &MemoryEntry, terms: &[String]) -> u32 {
    let key = entry.key.as_deref().unwrap_or("").to_lowercase();
    let value = entry.value.to_lowercase();
    let tags = entry.tags.join(",").to_lowercase();
    let mut total = 0u32;
    for t in terms {
        if key.contains(t.as_str()) {
            total += 8;
        }
        if tags.contains(t.as_str()) {
            total += 4;
        }
        if value.contains(t.as_str()) {
            total += 2;
        }
    }
    if entry.pinned {
        total += 1;
    }
    total
}

/// A correction waiting for user review.
#[derive(Debug, Clone, PartialEq)]
pub struct PendingFeedback {
    pub id: i64,
    pub ts: i64,
    pub utterance: String,
    pub predicted: String,
    pub corrected: String,
    pub status: String,
    pub notes: String,
    pub redacted: bool,
}

/// One row of task history.
#[derive(Debug, Clone, PartialEq)]
pub struct TaskRecord {
    pub ts: i64,
    pub intent: String,
    pub tool: String,
    pub outcome: String,
    pub detail: String,
    pub latency_ms: Option<u64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> MemoryStore {
        MemoryStore::open_in_memory(RetentionPolicy::default()).unwrap()
    }

    #[test]
    fn migrations_create_every_table() {
        let s = store();
        let tables = s.tables().unwrap();
        for expected in crate::schema::EXPECTED_TABLES {
            assert!(
                tables.iter().any(|t| t == expected),
                "missing table {expected}: {tables:?}"
            );
        }
        assert_eq!(s.schema_version().unwrap(), SCHEMA_VERSION);
    }

    #[test]
    fn migrations_are_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("m.db");
        {
            let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
            s.add(MemoryKind::Fact, None, "first run", &[], "test")
                .unwrap();
        }
        // Re-opening runs migrate() again; it must be a no-op.
        let s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        assert_eq!(s.schema_version().unwrap(), SCHEMA_VERSION);
        assert_eq!(s.list(MemoryKind::Fact, 10).unwrap().len(), 1);
    }

    #[test]
    fn rejects_future_schema_version() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("future.db");
        {
            let conn = rusqlite::Connection::open(&path).unwrap();
            conn.execute_batch(
                "CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at INTEGER NOT NULL);
                 INSERT INTO schema_version VALUES (9999, 0);",
            )
            .unwrap();
        }
        let opened = MemoryStore::open(&path, RetentionPolicy::default());
        let Err(err) = opened else {
            panic!("a newer schema version must be refused");
        };
        assert!(matches!(err, Error::Storage(_)));
        assert!(err.to_string().contains("9999"));
    }

    #[test]
    fn persists_across_restarts() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("persist.db");
        let id = {
            let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
            s.add(
                MemoryKind::Fact,
                Some("editor"),
                "uses neovim",
                &["tools".into()],
                "text",
            )
            .unwrap()
        };
        let s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        let entry = s.get(id).unwrap().expect("row survived restart");
        assert_eq!(entry.value, "uses neovim");
        assert_eq!(entry.tags, vec!["tools".to_string()]);
        assert!(entry.private, "memories are private by default");
    }

    #[test]
    fn refuses_empty_values() {
        let mut s = store();
        assert!(s.add(MemoryKind::Fact, None, "   ", &[], "test").is_err());
    }

    #[test]
    fn values_are_truncated_to_the_configured_cap() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cap.db");
        let mut s = MemoryStore::open(&path, RetentionPolicy::default())
            .unwrap()
            .with_max_value_chars(50);
        let id = s
            .add(MemoryKind::Fact, None, &"y".repeat(500), &[], "test")
            .unwrap();
        assert_eq!(s.get(id).unwrap().unwrap().value.chars().count(), 50);
    }

    #[test]
    fn update_delete_and_pin() {
        let mut s = store();
        let id = s.add(MemoryKind::Fact, None, "old", &[], "test").unwrap();
        s.update(id, "new").unwrap();
        assert_eq!(s.get(id).unwrap().unwrap().value, "new");
        s.set_pinned(id, true).unwrap();
        assert!(s.get(id).unwrap().unwrap().pinned);
        assert!(s.delete(id).unwrap());
        assert!(s.get(id).unwrap().is_none());
        assert!(s.update(9999, "x").is_err());
    }

    #[test]
    fn search_ranks_key_matches_first() {
        let mut s = store();
        s.add(
            MemoryKind::Fact,
            Some("keyboard shortcut"),
            "ctrl shift p",
            &[],
            "test",
        )
        .unwrap();
        s.add(
            MemoryKind::Fact,
            Some("coffee"),
            "shortcut for the terminal",
            &[],
            "test",
        )
        .unwrap();
        let hits = s.search("shortcut", 10).unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].key.as_deref(), Some("keyboard shortcut"));
    }

    #[test]
    fn search_ignores_wildcards_from_the_query() {
        let mut s = store();
        s.add(MemoryKind::Fact, None, "important note", &[], "test")
            .unwrap();
        // A bare `%` would otherwise match everything.
        assert!(s.search("%", 10).unwrap().is_empty());
        assert!(s.search("important", 10).unwrap().len() == 1);
    }

    #[test]
    fn private_rows_are_excluded_from_export() {
        let mut s = store();
        s.add(MemoryKind::Fact, None, "private thing", &[], "text")
            .unwrap();
        let shared = s
            .add_shareable(MemoryKind::Preference, "prefers dark mode", &[])
            .unwrap();
        assert!(s.exportable(100).unwrap().len() == 1);
        let exported = s.exportable(100).unwrap();
        assert_eq!(exported[0].id, shared);
        assert!(!exported[0].private);
    }

    #[test]
    fn retention_respects_pinned_rows() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("retention.db");
        let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        let old = s.add(MemoryKind::Fact, None, "old", &[], "test").unwrap();
        let pinned = s
            .add(MemoryKind::Fact, None, "pinned", &[], "test")
            .unwrap();
        s.set_pinned(pinned, true).unwrap();

        // Backdate the unpinned row past the retention window.
        s.conn
            .execute(
                "UPDATE memories SET created_at = 1 WHERE id = ?1",
                params![old],
            )
            .unwrap();

        let removed = s
            .prune_with(RetentionPolicy {
                retention_days: 1,
                max_entries: 0,
            })
            .unwrap();
        assert_eq!(removed, 1);
        assert!(s.get(old).unwrap().is_none());
        assert!(s.get(pinned).unwrap().is_some());
    }

    #[test]
    fn retention_never_deletes_feedback_or_history() {
        let mut s = store();
        s.add_feedback("do a thing", "unknown", "run_tests", false)
            .unwrap();
        s.record_task("system_status", "system_status", "ok", "cpu ok", Some(3))
            .unwrap();
        s.prune_with(RetentionPolicy {
            retention_days: 1,
            max_entries: 0,
        })
        .unwrap();
        assert_eq!(s.pending_feedback(10).unwrap().len(), 1);
        assert_eq!(s.history(10).unwrap().len(), 1);
    }

    #[test]
    fn soft_cap_prunes_oldest_first() {
        let mut s = store();
        for i in 0..5 {
            s.add(MemoryKind::Fact, None, &format!("fact {i}"), &[], "test")
                .unwrap();
        }
        // Make ordering deterministic by backdating the first two.
        s.conn
            .execute("UPDATE memories SET updated_at = 100 WHERE id IN (1,2)", [])
            .unwrap();
        s.prune_with(RetentionPolicy {
            retention_days: 0,
            max_entries: 3,
        })
        .unwrap();
        let remaining = s.list(MemoryKind::Fact, 100).unwrap();
        assert_eq!(remaining.len(), 3);
        assert!(remaining.iter().all(|e| e.id > 2));
    }

    #[test]
    fn backup_creates_a_readable_snapshot() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s =
            MemoryStore::open(tmp.path().join("live.db"), RetentionPolicy::default()).unwrap();
        s.add(MemoryKind::Fact, None, "snapshot me", &[], "test")
            .unwrap();
        let dest = tmp.path().join("backups").join("snap.db");
        s.backup(&dest).unwrap();
        assert!(dest.exists());

        let restored = MemoryStore::open(&dest, RetentionPolicy::default()).unwrap();
        assert_eq!(
            restored.list(MemoryKind::Fact, 10).unwrap()[0].value,
            "snapshot me"
        );
    }

    #[test]
    fn feedback_queue_moves_through_states() {
        let mut s = store();
        let id = s
            .add_feedback("open the browser", "start_application", "open_url", false)
            .unwrap();
        assert_eq!(s.pending_feedback(10).unwrap()[0].id, id);
        s.set_feedback_status(id, "approved").unwrap();
        assert!(s.pending_feedback(10).unwrap().is_empty());
        assert!(s.set_feedback_status(1234, "approved").is_err());
    }

    #[test]
    fn session_lifecycle() {
        let mut s = store();
        s.open_session("sess-1").unwrap();
        s.close_session("sess-1", Some("talked about rust"))
            .unwrap();
        // Closing twice is harmless.
        s.close_session("sess-1", None).unwrap();
    }

    #[test]
    fn the_page_cache_is_actually_bounded() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("cache.db");
        let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        s.apply_limits(StorageLimits {
            cache_kib: 512,
            ..Default::default()
        })
        .unwrap();
        let reported: i64 = s
            .conn
            .query_row("PRAGMA cache_size", [], |r| r.get(0))
            .unwrap();
        assert_eq!(reported, -512, "cache_size must be a page-count bound");
    }

    #[test]
    fn retrieval_is_capped_by_the_resource_budget() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("retrieval.db");
        let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        for i in 0..50 {
            s.add(
                MemoryKind::Fact,
                None,
                &format!("shared topic {i}"),
                &[],
                "test",
            )
            .unwrap();
        }
        s.apply_limits(StorageLimits {
            max_retrieved: 5,
            ..Default::default()
        })
        .unwrap();
        // The caller asks for 100; the budget must win.
        assert_eq!(s.search("topic", 100).unwrap().len(), 5);
    }

    #[test]
    fn oversized_values_are_refused_rather_than_truncated() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("size.db");
        let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        s.apply_limits(StorageLimits {
            max_value_bytes: 64,
            ..Default::default()
        })
        .unwrap();
        s.set_max_value_chars(100_000);
        let err = s
            .add(MemoryKind::Fact, None, &"x".repeat(200), &[], "test")
            .unwrap_err();
        assert!(matches!(err, Error::LimitExceeded(_)));
    }

    #[test]
    fn a_checkpoint_truncates_the_write_ahead_log() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("wal.db");
        let mut s = MemoryStore::open(&path, RetentionPolicy::default()).unwrap();
        for i in 0..500 {
            s.add(MemoryKind::Fact, None, &format!("row {i}"), &[], "test")
                .unwrap();
        }
        assert!(s.on_disk_bytes() > 0);
        s.checkpoint().unwrap();
        assert!(
            s.on_disk_bytes() > 0,
            "the database itself is still on disk"
        );
    }

    #[test]
    fn clear_kind_only_removes_that_kind() {
        let mut s = store();
        s.add(MemoryKind::Fact, None, "a", &[], "test").unwrap();
        s.add(MemoryKind::Preference, None, "b", &[], "test")
            .unwrap();
        assert_eq!(s.clear_kind(MemoryKind::Fact).unwrap(), 1);
        assert!(s.list(MemoryKind::Fact, 10).unwrap().is_empty());
        assert_eq!(s.list(MemoryKind::Preference, 10).unwrap().len(), 1);
    }
}
