//! Security boundary for AARON: policy, sandbox and audit trail.
//!
//! Nothing in this crate executes anything. It answers one question — "is this
//! action allowed, and with what limits?" — and records the answer. The
//! executor in `aaron-tools` consumes the [`crate::policy::ExecutionLimits`]
//! it returns and is required to honour them.

pub mod audit;
pub mod policy;
pub mod sandbox;

pub use audit::{AuditEvent, AuditLog, Outcome, Phase};
pub use policy::{
    Decision, Effect, ExecutionLimits, LimitPolicy, PermissionCategory, PolicyEngine, PolicyFile,
    PolicySummary, RiskLevel, ToolFacts, ToolPolicy,
};
pub use sandbox::{
    canonicalize_lenient, expand_home, lexical_normalize, running_as_root, PathSandbox,
};
