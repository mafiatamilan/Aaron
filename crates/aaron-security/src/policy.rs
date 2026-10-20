//! Deny-by-default policy engine.
//!
//! The model is deliberately small and boring:
//!
//! * Every registered tool has a policy entry in `config/permissions.toml`.
//! * A tool with **no** entry is denied. There is no implicit allow, ever.
//! * An entry says `allow`, `ask` or `deny`, and can tighten the tool's own
//!   declared timeout and output cap but never loosen the global caps.
//! * Certain high-risk *categories* are denied outright regardless of the
//!   per-tool entry, so a mistake in the TOML cannot hand the model `sudo`.
//!
//! The engine never sees model internals. It receives a tool id, a permission
//! category and a risk level supplied by the *tool registry*, not by the
//! classifier, so a misclassification cannot escalate privileges.

use std::collections::BTreeMap;
use std::path::Path;

use aaron_core::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// What a tool is allowed to touch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionCategory {
    /// Observes process/system state and returns it.
    SystemRead,
    /// Reads files inside the sandbox.
    FilesystemRead,
    /// Writes or deletes files.
    FilesystemWrite,
    /// Spawns a process from the fixed tool registry.
    ProcessExec,
    /// Performs outbound network activity.
    Network,
    /// Writes to long-term memory.
    MemoryWrite,
    /// Changes system configuration or privileges. Always denied.
    SystemChange,
}

impl PermissionCategory {
    pub fn as_str(self) -> &'static str {
        match self {
            PermissionCategory::SystemRead => "system_read",
            PermissionCategory::FilesystemRead => "filesystem_read",
            PermissionCategory::FilesystemWrite => "filesystem_write",
            PermissionCategory::ProcessExec => "process_exec",
            PermissionCategory::Network => "network",
            PermissionCategory::MemoryWrite => "memory_write",
            PermissionCategory::SystemChange => "system_change",
        }
    }
}

/// How much damage a mistake could do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

impl RiskLevel {
    pub fn as_str(self) -> &'static str {
        match self {
            RiskLevel::Low => "low",
            RiskLevel::Medium => "medium",
            RiskLevel::High => "high",
            RiskLevel::Critical => "critical",
        }
    }
}

/// The effect a policy entry grants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Run without asking.
    Allow,
    /// Run only after the user confirms.
    Ask,
    /// Never run.
    Deny,
}

impl Effect {
    pub fn as_str(self) -> &'static str {
        match self {
            Effect::Allow => "allow",
            Effect::Ask => "ask",
            Effect::Deny => "deny",
        }
    }
}

/// Global caps that no tool entry may exceed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LimitPolicy {
    pub max_runtime_secs: u32,
    pub max_output_bytes: usize,
    /// Truncation limit for the value stored in an audit record.
    pub max_audit_value_chars: usize,
    /// Refuse everything if the process is running as uid 0.
    pub refuse_root: bool,
    /// Master switch for tools in the `network` category.
    pub allow_network: bool,
}

impl Default for LimitPolicy {
    fn default() -> Self {
        Self {
            max_runtime_secs: 30,
            max_output_bytes: 64 * 1024,
            max_audit_value_chars: 160,
            refuse_root: true,
            allow_network: false,
        }
    }
}

/// Per-tool policy entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ToolPolicy {
    pub effect: Effect,
    /// Optional tighter timeout. `None` uses the global cap.
    pub timeout_secs: Option<u32>,
    /// Optional tighter output cap. `None` uses the global cap.
    pub max_output_bytes: Option<usize>,
    /// Human-readable note shown by `aaron permissions show`.
    pub note: String,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        ToolPolicy {
            effect: Effect::Deny,
            timeout_secs: None,
            max_output_bytes: None,
            note: String::new(),
        }
    }
}

/// The parsed `permissions.toml`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PolicyFile {
    /// Schema version. Bumping it forces a review of new keys.
    pub version: u32,
    /// Effect applied to a tool with no entry. Must be `deny` in any sane
    /// config; the loader rejects `allow` here.
    pub unknown_tool: Effect,
    pub limits: LimitPolicy,
    pub tools: BTreeMap<String, ToolPolicy>,
}

impl Default for PolicyFile {
    fn default() -> Self {
        PolicyFile {
            version: 1,
            unknown_tool: Effect::Deny,
            limits: LimitPolicy::default(),
            tools: BTreeMap::new(),
        }
    }
}

impl PolicyFile {
    /// Parse a policy file from TOML text.
    pub fn from_toml(text: &str) -> Result<PolicyFile> {
        let policy: PolicyFile = toml::from_str(text).map_err(|e| Error::Config(e.to_string()))?;
        if policy.unknown_tool != Effect::Deny {
            return Err(Error::Config(
                "permissions.unknown_tool must be `deny`; AARON is deny-by-default".into(),
            ));
        }
        if policy.limits.max_runtime_secs == 0 {
            return Err(Error::Config(
                "permissions.limits.max_runtime_secs must be > 0".into(),
            ));
        }
        if policy.limits.max_output_bytes == 0 {
            return Err(Error::Config(
                "permissions.limits.max_output_bytes must be > 0".into(),
            ));
        }
        Ok(policy)
    }

    /// Read a policy file from disk, falling back to deny-everything when it is
    /// missing so that a deleted config cannot silently open the assistant up.
    pub fn load(path: impl AsRef<Path>) -> Result<PolicyFile> {
        let path = path.as_ref();
        match std::fs::read_to_string(path) {
            Ok(text) => PolicyFile::from_toml(&text),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                tracing::warn!(path = %path.display(), "permissions file missing; denying every tool");
                Ok(PolicyFile::default())
            }
            Err(e) => Err(Error::io(path, e)),
        }
    }
}

/// Facts about a tool, supplied by the registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolFacts {
    pub id: String,
    pub category: PermissionCategory,
    pub risk: RiskLevel,
    /// The tool's own declared defaults; the policy may only tighten them.
    pub declared_timeout_secs: u32,
    pub declared_max_output_bytes: usize,
    /// The tool declares that it always needs confirmation regardless of policy.
    pub requires_confirmation: bool,
}

/// The verdict for one proposed action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    /// Run it.
    Allow(ExecutionLimits),
    /// Ask the user first; the display string must show the exact action.
    NeedsConfirmation {
        limits: ExecutionLimits,
        prompt: String,
    },
    /// Refuse. The string is shown to the user verbatim.
    Deny(String),
}

impl Decision {
    pub fn is_allowed(&self) -> bool {
        matches!(
            self,
            Decision::Allow(_) | Decision::NeedsConfirmation { .. }
        )
    }
}

/// Hard limits handed to the tool executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionLimits {
    pub timeout_secs: u32,
    pub max_output_bytes: usize,
}

/// The runtime policy engine.
#[derive(Debug, Clone)]
pub struct PolicyEngine {
    file: PolicyFile,
    /// Tools whose category is never permitted, regardless of configuration.
    always_denied_categories: Vec<PermissionCategory>,
    /// Tool ids that require confirmation even if the TOML says `allow`.
    always_confirm: Vec<String>,
}

impl PolicyEngine {
    /// Construct an engine from a parsed policy file.
    pub fn new(file: PolicyFile, always_confirm: Vec<String>) -> Self {
        PolicyEngine {
            file,
            // `SystemChange` is hard-coded off. It exists in the enum so the
            // vocabulary is complete and so a future tool cannot accidentally
            // pick a "not listed means fine" category.
            always_denied_categories: vec![PermissionCategory::SystemChange],
            always_confirm,
        }
    }

    /// The effective global limits.
    pub fn limits(&self) -> ExecutionLimits {
        ExecutionLimits {
            timeout_secs: self.file.limits.max_runtime_secs,
            max_output_bytes: self.file.limits.max_output_bytes,
        }
    }

    /// The parsed file, for `aaron permissions show`.
    pub fn file(&self) -> &PolicyFile {
        &self.file
    }

    /// Decide whether `facts` may run.
    pub fn evaluate(&self, facts: &ToolFacts) -> Decision {
        // 1. Root refusal comes first: nothing runs as uid 0, ever.
        if self.file.limits.refuse_root && super::sandbox::running_as_root() {
            return Decision::Deny(
                "AARON refuses to execute tools as root. Re-run as your normal user.".into(),
            );
        }

        // 2. Hard-denied categories.
        if self.always_denied_categories.contains(&facts.category) {
            return Decision::Deny(format!(
                "tool `{}` is in the permanently denied category `{}`",
                facts.id,
                facts.category.as_str()
            ));
        }

        // 3. Network master switch.
        if facts.category == PermissionCategory::Network && !self.file.limits.allow_network {
            return Decision::Deny(
                "network access is disabled (permissions.limits.allow_network = false)".into(),
            );
        }

        // 4. Look up the entry; absent means deny.
        let Some(entry) = self.file.tools.get(&facts.id) else {
            // Name the fix. A refusal that does not say what to do about it
            // is a dead end: the user is told the tool is off, and not told
            // that the file to edit is sitting next to the message.
            return Decision::Deny(format!(
                "tool `{}` has no entry in permissions.toml; deny-by-default applies. \
                 To allow it, add a [tools.{}] section with effect = \"ask\" to your \
                 permissions file.",
                facts.id, facts.id
            ));
        };

        let limits = ExecutionLimits {
            timeout_secs: self
                .file
                .tools
                .get(&facts.id)
                .and_then(|e| e.timeout_secs)
                .unwrap_or(self.file.limits.max_runtime_secs)
                .min(self.file.limits.max_runtime_secs)
                .min(facts.declared_timeout_secs.max(1)),
            max_output_bytes: entry
                .max_output_bytes
                .unwrap_or(self.file.limits.max_output_bytes)
                .min(self.file.limits.max_output_bytes)
                .min(facts.declared_max_output_bytes.max(1)),
        };

        match entry.effect {
            Effect::Deny => Decision::Deny(format!(
                "tool `{}` is denied by permissions.toml{}",
                facts.id,
                if entry.note.is_empty() {
                    String::new()
                } else {
                    format!(" ({})", entry.note)
                }
            )),
            Effect::Allow => {
                let needs_confirm = facts.requires_confirmation
                    || self.always_confirm.iter().any(|t| t == &facts.id);
                if needs_confirm {
                    Decision::NeedsConfirmation {
                        limits,
                        prompt: format!("Run `{}` ({})?", facts.id, facts.risk.as_str()),
                    }
                } else {
                    Decision::Allow(limits)
                }
            }
            Effect::Ask => Decision::NeedsConfirmation {
                limits,
                prompt: format!("Run `{}` ({})?", facts.id, facts.risk.as_str()),
            },
        }
    }

    /// A short machine-readable summary for `aaron permissions test`.
    pub fn summary(&self) -> PolicySummary {
        let allow = self
            .file
            .tools
            .iter()
            .filter(|(_, e)| e.effect == Effect::Allow)
            .count();
        let ask = self
            .file
            .tools
            .iter()
            .filter(|(_, e)| e.effect == Effect::Ask)
            .count();
        let deny = self.file.tools.len() - allow - ask;
        PolicySummary {
            version: self.file.version,
            total_entries: self.file.tools.len(),
            allow,
            ask,
            deny,
            unknown_tool: self.file.unknown_tool,
            allow_network: self.file.limits.allow_network,
            refuse_root: self.file.limits.refuse_root,
            always_confirm: self.always_confirm.clone(),
        }
    }
}

/// Aggregated policy state for the CLI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct PolicySummary {
    pub version: u32,
    pub total_entries: usize,
    pub allow: usize,
    pub ask: usize,
    pub deny: usize,
    pub unknown_tool: Effect,
    pub allow_network: bool,
    pub refuse_root: bool,
    pub always_confirm: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts(id: &str, category: PermissionCategory, risk: RiskLevel) -> ToolFacts {
        ToolFacts {
            id: id.into(),
            category,
            risk,
            declared_timeout_secs: 10,
            declared_max_output_bytes: 1024,
            requires_confirmation: false,
        }
    }

    fn engine_with(toml: &str) -> PolicyEngine {
        PolicyEngine::new(PolicyFile::from_toml(toml).unwrap(), Vec::new())
    }

    #[test]
    fn unknown_tools_are_denied() {
        let e = engine_with("version = 1\n");
        let d = e.evaluate(&facts(
            "rm_rf",
            PermissionCategory::FilesystemWrite,
            RiskLevel::High,
        ));
        assert!(matches!(d, Decision::Deny(_)));
        assert!(d.to_string_hint().contains("deny-by-default"));
    }

    #[test]
    fn explicit_deny_wins_over_allow_listing_elsewhere() {
        let e = engine_with(
            r#"
version = 1
[tools.read_file]
effect = "deny"
note = "user does not want file reads"
"#,
        );
        let d = e.evaluate(&facts(
            "read_file",
            PermissionCategory::FilesystemRead,
            RiskLevel::Low,
        ));
        assert!(
            matches!(d, Decision::Deny(ref m) if m.contains("user does not want file reads")),
            "got {d:?}"
        );
    }

    #[test]
    fn network_category_is_checked_before_the_tool_entry() {
        let e = engine_with(
            r#"
version = 1
[tools.open_url]
effect = "allow"
note = "user does not want browser launches"
"#,
        );
        // `allow_network` is false, so the network category is refused even
        // though the tool entry says allow.
        assert!(matches!(
            e.evaluate(&facts("open_url", PermissionCategory::Network, RiskLevel::Medium)),
            Decision::Deny(ref m) if m.contains("network access is disabled")
        ));
    }

    #[test]
    fn system_change_category_is_always_denied() {
        let e = engine_with(
            r#"
version = 1
[tools.install_package]
effect = "allow"
"#,
        );
        let d = e.evaluate(&facts(
            "install_package",
            PermissionCategory::SystemChange,
            RiskLevel::Critical,
        ));
        assert!(matches!(d, Decision::Deny(ref m) if m.contains("permanently denied")));
    }

    #[test]
    fn network_is_denied_unless_explicitly_enabled() {
        let e = engine_with(
            r#"
version = 1
[limits]
allow_network = false
[tools.open_url]
effect = "allow"
"#,
        );
        assert!(matches!(
            e.evaluate(&facts(
                "open_url",
                PermissionCategory::Network,
                RiskLevel::Medium
            )),
            Decision::Deny(_)
        ));
    }

    #[test]
    fn policy_cannot_loosen_the_global_timeout() {
        let e = engine_with(
            r#"
version = 1
[limits]
max_runtime_secs = 5
[tools.run_tests]
effect = "allow"
timeout_secs = 600
"#,
        );
        match e.evaluate(&facts(
            "run_tests",
            PermissionCategory::ProcessExec,
            RiskLevel::Medium,
        )) {
            Decision::Allow(l) => assert_eq!(l.timeout_secs, 5),
            other => panic!("expected allow, got {other:?}"),
        }
    }

    #[test]
    fn ask_requires_confirmation() {
        let e = engine_with(
            r#"
version = 1
[tools.run_tests]
effect = "ask"
"#,
        );
        assert!(matches!(
            e.evaluate(&facts(
                "run_tests",
                PermissionCategory::ProcessExec,
                RiskLevel::Medium
            )),
            Decision::NeedsConfirmation { .. }
        ));
    }

    #[test]
    fn always_confirm_list_overrides_allow() {
        let e = PolicyEngine::new(
            PolicyFile::from_toml(
                r#"
version = 1
[tools.read_file]
effect = "allow"
"#,
            )
            .unwrap(),
            vec!["read_file".into()],
        );
        assert!(matches!(
            e.evaluate(&facts(
                "read_file",
                PermissionCategory::FilesystemRead,
                RiskLevel::Low
            )),
            Decision::NeedsConfirmation { .. }
        ));
    }

    #[test]
    fn unknown_tool_allow_is_rejected_at_parse_time() {
        let err = PolicyFile::from_toml("version = 1\nunknown_tool = \"allow\"\n").unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn missing_file_denies_everything() {
        let p = PolicyFile::load("/definitely/not/a/policy/file.toml").unwrap();
        assert!(p.tools.is_empty());
        assert_eq!(p.unknown_tool, Effect::Deny);
    }

    // Small helper so the first test can assert on the message.
    impl Decision {
        fn to_string_hint(&self) -> String {
            match self {
                Decision::Deny(m) => m.clone(),
                _ => String::new(),
            }
        }
    }
}
