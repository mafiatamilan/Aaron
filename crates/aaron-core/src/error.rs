//! Error types shared across every AARON crate.
//!
//! Every fallible boundary in AARON returns an [`Error`]. The variants are
//! deliberately explicit: the assistant must be able to tell the user *why*
//! something failed (missing model, denied by policy, device disappeared)
//! rather than reporting a generic failure.

use std::fmt;
use std::path::PathBuf;

/// Result alias used throughout AARON.
pub type Result<T> = std::result::Result<T, Error>;

/// Central error type for AARON.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Configuration file missing, unparsable or semantically invalid.
    #[error("configuration error: {0}")]
    Config(String),

    /// A required external component is not installed (binary, shared library).
    /// The message carries its own remediation hint so callers never have to
    /// guess what to install.
    #[error("missing dependency: {0}")]
    MissingDependency(String),

    /// A model file is absent or failed checksum verification.
    #[error("model error: {0}")]
    Model(String),

    /// An operation was refused by the policy engine.
    #[error("permission denied: {0}")]
    PermissionDenied(String),

    /// The user declined a confirmation prompt.
    #[error("cancelled by user")]
    Cancelled,

    /// A guard (timeout, output cap, loop breaker) fired.
    #[error("{0}")]
    LimitExceeded(String),

    /// Audio capture failed (no device, device in use, unsupported format).
    #[error("audio error: {0}")]
    Audio(String),

    /// Filesystem access was rejected by the sandbox (traversal, symlink escape).
    #[error("path rejected: {0}")]
    PathRejected(String),

    /// Persistence layer failure.
    #[error("storage error: {0}")]
    Storage(String),

    /// Speech recognition failure.
    #[error("speech recognition error: {0}")]
    Asr(String),

    /// Speech synthesis failure.
    #[error("speech synthesis error: {0}")]
    Tts(String),

    /// Model inference / training failure.
    #[error("brain error: {0}")]
    Brain(String),

    /// I/O failure with the underlying path preserved for the user.
    #[error("i/o error at {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// Anything else, with a human readable message.
    #[error("{0}")]
    Other(String),
}

impl Error {
    /// Attach a remediation hint to [`Error::MissingDependency`].
    pub fn missing(what: impl Into<String>) -> Self {
        Error::MissingDependency(what.into())
    }

    /// Attach a remediation hint to [`Error::MissingDependency`].
    pub fn missing_with_hint(what: impl Into<String>, hint: impl Into<String>) -> Self {
        Error::MissingDependency(format!("{} ({})", what.into(), hint.into()))
    }

    /// Wrap a [`std::io::Error`] together with the path that produced it.
    pub fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }

    /// Stable machine-readable discriminant, used by the CLI to pick exit codes
    /// and by tests to assert on failure classes without string matching.
    pub fn kind(&self) -> ErrorKind {
        match self {
            Error::Config(_) => ErrorKind::Config,
            Error::MissingDependency(_) => ErrorKind::MissingDependency,
            Error::Model(_) => ErrorKind::Model,
            Error::PermissionDenied(_) => ErrorKind::PermissionDenied,
            Error::Cancelled => ErrorKind::Cancelled,
            Error::LimitExceeded(_) => ErrorKind::LimitExceeded,
            Error::Audio(_) => ErrorKind::Audio,
            Error::PathRejected(_) => ErrorKind::PathRejected,
            Error::Storage(_) => ErrorKind::Storage,
            Error::Asr(_) => ErrorKind::Asr,
            Error::Tts(_) => ErrorKind::Tts,
            Error::Brain(_) => ErrorKind::Brain,
            Error::Io { .. } => ErrorKind::Io,
            Error::Other(_) => ErrorKind::Other,
        }
    }
}

/// Machine-readable error classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    Config,
    MissingDependency,
    Model,
    PermissionDenied,
    Cancelled,
    LimitExceeded,
    Audio,
    PathRejected,
    Storage,
    Asr,
    Tts,
    Brain,
    Io,
    Other,
}

impl ErrorKind {
    /// Short snake_case identifier, printed by `aaron logs --format json` and by
    /// the CLI's machine-readable error output.
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::Config => "config",
            ErrorKind::MissingDependency => "missing_dependency",
            ErrorKind::Model => "model",
            ErrorKind::PermissionDenied => "permission_denied",
            ErrorKind::Cancelled => "cancelled",
            ErrorKind::LimitExceeded => "limit_exceeded",
            ErrorKind::Audio => "audio",
            ErrorKind::PathRejected => "path_rejected",
            ErrorKind::Storage => "storage",
            ErrorKind::Asr => "asr",
            ErrorKind::Tts => "tts",
            ErrorKind::Brain => "brain",
            ErrorKind::Io => "io",
            ErrorKind::Other => "other",
        }
    }

    /// Process exit code suggested for this error class.
    pub fn exit_code(self) -> i32 {
        match self {
            ErrorKind::PermissionDenied => 77,
            ErrorKind::Cancelled => 130,
            ErrorKind::MissingDependency => 69,
            ErrorKind::Config => 78,
            _ => 1,
        }
    }
}

impl fmt::Display for ErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}
