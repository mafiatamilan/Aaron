//! SQLite long-term memory for AARON.
//!
//! AARON remembers very little on purpose. Facts enter the database only when
//! the user says "remember ..." and confirms, or through an explicit
//! `aaron memory add`. Transcripts are never promoted automatically.

pub mod schema;
pub mod store;

pub use schema::{MIGRATIONS, SCHEMA_VERSION};
pub use store::{
    MemoryEntry, MemoryKind, MemoryStore, PendingFeedback, RetentionPolicy, StorageLimits,
    TaskRecord,
};
