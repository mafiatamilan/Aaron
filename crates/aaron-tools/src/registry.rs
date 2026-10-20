//! The tool registry.
//!
//! A tool is a trait object with a static [`ToolSpec`]. The registry is the
//! *only* way the agent can reach the outside world, and the agent reaches it
//! only after the policy engine has approved a call.
//!
//! Two properties are enforced structurally rather than by convention:
//!
//! * Tools are looked up by a stable `&'static str` id. There is no dynamic
//!   dispatch on a model-supplied string that is not in this map.
//! * `Tool::execute` is only reachable through [`ToolRegistry::execute`], which
//!   re-validates arguments. A tool that wants looser validation than the spec
//!   declares has to change the spec, which changes what CI tests cover.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use aaron_core::config::Config;
use aaron_core::error::{Error, Result};
use aaron_memory::MemoryStore;
use aaron_security::PathSandbox;
use serde::Serialize;
use serde_json::Value;

use crate::spec::{ToolArgs, ToolSpec};

/// Everything a tool is allowed to reach.
pub struct ToolContext {
    /// Filesystem roots.
    pub sandbox: Arc<PathSandbox>,
    /// Long-term memory.
    pub memory: Arc<Mutex<MemoryStore>>,
    /// Effective configuration.
    pub config: Arc<Config>,
    /// Repository root, used as the default working directory for git and test
    /// commands.
    pub project_root: std::path::PathBuf,
}

impl ToolContext {
    /// Build a context. `sandbox` and `memory` are shared, not moved.
    pub fn new(
        sandbox: Arc<PathSandbox>,
        memory: Arc<Mutex<MemoryStore>>,
        config: Arc<Config>,
        project_root: impl Into<std::path::PathBuf>,
    ) -> Arc<Self> {
        Arc::new(ToolContext {
            sandbox,
            memory,
            config,
            project_root: project_root.into(),
        })
    }

    /// A minimal context for tests: empty sandbox, in-memory database.
    pub fn for_tests(project_root: impl Into<std::path::PathBuf>) -> Arc<Self> {
        Arc::new(ToolContext {
            sandbox: Arc::new(PathSandbox::deny_all()),
            memory: Arc::new(Mutex::new(
                MemoryStore::open_in_memory(Default::default()).expect("in-memory db"),
            )),
            config: Arc::new(Config::default()),
            project_root: project_root.into(),
        })
    }
}

/// What a tool returns.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ToolOutput {
    /// Human-readable result, already truncated to the allowed output size.
    pub text: String,
    /// Whether the text was cut short.
    pub truncated: bool,
    /// Untruncated size in bytes, so the audit log can record the true cost.
    pub bytes: usize,
    /// Optional structured payload for the agent to summarise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl ToolOutput {
    pub fn text(text: impl Into<String>) -> ToolOutput {
        let text = text.into();
        let bytes = text.len();
        ToolOutput {
            text,
            truncated: false,
            bytes,
            data: None,
        }
    }

    pub fn with_data(mut self, data: Value) -> Self {
        self.data = Some(data);
        self
    }

    /// Truncate to `max` characters, recording that it happened.
    pub fn capped(mut self, max: usize) -> Self {
        if self.text.chars().count() > max {
            self.text = aaron_core::text::truncate_chars(&self.text, max);
            self.truncated = true;
        }
        self
    }
}

/// One executable tool.
pub trait Tool: Send + Sync {
    /// Static declaration. The `'static` lifetime is what makes the spec
    /// addressable from the registry without borrowing the tool itself.
    fn spec(&self) -> &'static ToolSpec;

    /// Run the tool. Arguments have already been validated against the spec.
    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput>;
}

/// All tools AARON knows about.
#[derive(Default, Clone)]
pub struct ToolRegistry {
    tools: BTreeMap<&'static str, Arc<dyn Tool>>,
}

impl ToolRegistry {
    /// An empty registry.
    pub fn empty() -> ToolRegistry {
        ToolRegistry::default()
    }

    /// Register a tool, replacing any previous tool with the same id.
    pub fn register(&mut self, tool: Arc<dyn Tool>) {
        self.tools.insert(tool.spec().id, tool);
    }

    /// Look up a tool.
    pub fn get(&self, id: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(id).cloned()
    }

    /// Whether a tool id is registered.
    pub fn contains(&self, id: &str) -> bool {
        self.tools.contains_key(id)
    }

    /// All registered ids, sorted.
    pub fn ids(&self) -> Vec<&'static str> {
        self.tools.keys().copied().collect()
    }

    /// All specs, sorted by id.
    pub fn specs(&self) -> Vec<&'static ToolSpec> {
        let mut specs: Vec<&'static ToolSpec> = self.tools.values().map(|t| t.spec()).collect();
        specs.sort_by_key(|s| s.id);
        specs
    }

    /// Length of the registry.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// Look up, validate and run a tool.
    ///
    /// This is the single entry point. Validation happens here so that a caller
    /// cannot bypass it by holding an `Arc<dyn Tool>` directly — and so that the
    /// error message always names the tool the user asked for.
    pub fn execute(&self, id: &str, raw_args: &Value, ctx: &ToolContext) -> Result<ToolOutput> {
        let tool = self
            .get(id)
            .ok_or_else(|| Error::Other(format!("no tool registered with id `{id}`")))?;
        let args = ToolArgs::validate(tool.spec(), raw_args)?;
        tool.execute(ctx, &args)
    }

    /// Build the default registry.
    pub fn builtin() -> ToolRegistry {
        let mut r = ToolRegistry::empty();
        for tool in crate::builtin::all() {
            r.register(tool);
        }
        r
    }
}

impl std::fmt::Debug for ToolRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolRegistry")
            .field("ids", &self.ids())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtin_registry_contains_every_documented_tool() {
        let r = ToolRegistry::builtin();
        for id in [
            "system_status",
            "search_files",
            "read_file",
            "git_status",
            "run_tests",
            "open_url",
            "start_application",
            "stop_application",
            "remember",
            "recall",
            "cancel",
        ] {
            assert!(r.contains(id), "missing built-in tool `{id}`");
        }
    }

    #[test]
    fn registry_ids_are_sorted_and_unique() {
        let r = ToolRegistry::builtin();
        let ids = r.ids();
        let mut sorted = ids.clone();
        sorted.sort();
        assert_eq!(ids, sorted);
        let mut dedup = sorted.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), sorted.len());
    }

    #[test]
    fn no_tool_declares_a_free_form_command_argument() {
        // A shell-escaping argument would defeat the whole security model.
        for spec in ToolRegistry::builtin().specs() {
            for arg in spec.args {
                assert_ne!(
                    arg.name, "command",
                    "tool `{}` declares a `command` argument",
                    spec.id
                );
                assert_ne!(
                    arg.name, "cmd",
                    "tool `{}` declares a `cmd` argument",
                    spec.id
                );
                assert_ne!(
                    arg.name, "shell",
                    "tool `{}` declares a `shell` argument",
                    spec.id
                );
            }
        }
    }

    #[test]
    fn every_tool_declares_a_category_risk_and_timeout() {
        for spec in ToolRegistry::builtin().specs() {
            assert!(spec.timeout_secs > 0, "{} has no timeout", spec.id);
            assert!(spec.max_output_bytes > 0, "{} has no output cap", spec.id);
            assert!(!spec.description.is_empty());
            assert!(!spec.id.is_empty());
        }
    }

    #[test]
    fn unknown_tool_id_is_an_error_not_a_panic() {
        let r = ToolRegistry::builtin();
        let ctx = ToolContext::for_tests("/tmp");
        let err = r.execute("nope", &Value::Null, &ctx).unwrap_err();
        assert!(err.to_string().contains("no tool registered"));
    }

    #[test]
    fn execute_validates_arguments_before_running() {
        let r = ToolRegistry::builtin();
        let ctx = ToolContext::for_tests("/tmp");
        // `read_file` requires a `path`.
        let err = r
            .execute("read_file", &serde_json::json!({"nope": 1}), &ctx)
            .unwrap_err();
        assert!(err.to_string().contains("read_file"));
    }
}
