//! The built-in tool set.
//!
//! Every tool here is read-only except `remember`, `run_tests`,
//! `open_url`, `start_application` and `stop_application`, each of which either
//! declares `requires_confirmation` or is gated by the policy file. The
//! destructive tools AARON deliberately does *not* have (delete a file, install
//! a package, change a firewall) are absent on purpose and would be refused by
//! the policy engine even if someone added them.

pub mod files;
pub mod memory;
pub mod proc_tools;
pub mod system;

use std::sync::Arc;

use crate::registry::Tool;

/// Every registered tool, in a stable order.
pub fn all() -> Vec<Arc<dyn Tool>> {
    vec![
        Arc::new(system::SystemStatusTool),
        Arc::new(files::SearchFilesTool),
        Arc::new(files::ReadFileTool),
        Arc::new(proc_tools::GitStatusTool),
        Arc::new(proc_tools::RunTestsTool),
        Arc::new(proc_tools::OpenUrlTool),
        Arc::new(proc_tools::StartApplicationTool),
        Arc::new(proc_tools::StopApplicationTool),
        Arc::new(memory::RememberTool),
        Arc::new(memory::RecallTool),
        Arc::new(proc_tools::CancelTool),
    ]
}
