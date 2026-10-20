//! Database schema and forward-only migrations.
//!
//! Migrations are an ordered list of SQL statements tagged with a version. They
//! run inside a single transaction together with the version bump, so an
//! interrupted upgrade can never leave a half-migrated database behind.
//!
//! Migrations are never edited once released. Changing one requires appending a
//! new one, which is what makes an old database on a new binary safe.

/// Current schema version. Bump when appending to [`MIGRATIONS`].
pub const SCHEMA_VERSION: u32 = 2;

/// `(version, sql)` pairs applied in ascending order.
pub const MIGRATIONS: &[(u32, &str)] = &[
    (
        1,
        r#"
CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT    NOT NULL,
    key         TEXT,
    value       TEXT    NOT NULL,
    tags        TEXT    NOT NULL DEFAULT '',
    source      TEXT    NOT NULL DEFAULT 'user',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL,
    pinned      INTEGER NOT NULL DEFAULT 0,
    private     INTEGER NOT NULL DEFAULT 1,
    redacted    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_memories_kind  ON memories(kind, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_key   ON memories(key);

CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started_at  INTEGER NOT NULL,
    ended_at    INTEGER,
    summary     TEXT
);

CREATE TABLE IF NOT EXISTS task_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    intent      TEXT    NOT NULL,
    tool        TEXT    NOT NULL,
    outcome     TEXT    NOT NULL,
    detail      TEXT    NOT NULL DEFAULT '',
    latency_ms  INTEGER
);
CREATE INDEX IF NOT EXISTS idx_task_history_ts ON task_history(ts DESC);

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            INTEGER NOT NULL,
    utterance     TEXT    NOT NULL,
    predicted     TEXT    NOT NULL,
    corrected     TEXT    NOT NULL,
    status        TEXT    NOT NULL DEFAULT 'pending',
    notes         TEXT    NOT NULL DEFAULT '',
    redacted      INTEGER NOT NULL DEFAULT 0,
    dataset_ver   TEXT
);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback(status, ts DESC);
"#,
    ),
    (
        2,
        r#"
-- Explicit audit metadata, kept apart from user-facing memory so that
-- retention cleanup can never touch it and `aaron memory purge` cannot
-- silently erase the record of what happened.
CREATE TABLE IF NOT EXISTS audit_meta (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          INTEGER NOT NULL,
    category    TEXT    NOT NULL,
    summary     TEXT    NOT NULL,
    bytes       INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_audit_meta_ts ON audit_meta(ts DESC);
"#,
    ),
];

/// Current user-defined table names, used by tests to assert migrations ran.
pub const EXPECTED_TABLES: &[&str] = &[
    "memories",
    "sessions",
    "task_history",
    "feedback",
    "audit_meta",
    "schema_version",
];
