//! The AARON tool registry.
//!
//! AARON has exactly one way to affect the world: a named tool from this
//! registry, run with arguments validated against a static spec, under limits
//! produced by the policy engine. There is no shell tool, no "run this command"
//! tool, and no path from a model prediction straight to a process.

pub mod builtin;
pub mod proc;
pub mod registry;
pub mod spec;

pub use proc::{CommandOutput, ExecOptions, ExitKind};
pub use registry::{Tool, ToolContext, ToolOutput, ToolRegistry};
pub use spec::{ArgKind, ArgSpec, ToolArgs, ToolSpec};
