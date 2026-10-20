//! Tool declarations and argument validation.
//!
//! A tool is data plus a validated argument contract. The registry holds the
//! data; validation happens here, *before* the policy engine sees the call, so
//! a malformed or oversized argument can never reach an executor.
//!
//! There is deliberately no "free-form command" argument anywhere in this
//! module. Nothing in AARON accepts a string and runs it.

use std::collections::BTreeMap;

use aaron_core::error::{Error, Result};
use aaron_security::{PermissionCategory, RiskLevel};
use serde_json::{Map, Value};

/// Expected type of one tool argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArgKind {
    /// Free text, length bounded.
    Text,
    /// A count / integer.
    Count,
    /// A filesystem path, resolved through the sandbox by the tool.
    Path,
    /// An absolute URL with an `http`/`https` scheme.
    Url,
    /// A key that must exist in a fixed allow-list (applications, test suites).
    Choice,
}

impl ArgKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ArgKind::Text => "text",
            ArgKind::Count => "count",
            ArgKind::Path => "path",
            ArgKind::Url => "url",
            ArgKind::Choice => "choice",
        }
    }
}

/// One declared argument.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArgSpec {
    pub name: &'static str,
    pub kind: ArgKind,
    pub required: bool,
    /// Maximum accepted length in characters.
    pub max_len: usize,
    /// Allowed values for [`ArgKind::Choice`].
    pub choices: &'static [&'static str],
    /// Shown by `aaron tools list`.
    pub doc: &'static str,
}

impl ArgSpec {
    pub const fn text(
        name: &'static str,
        required: bool,
        max_len: usize,
        doc: &'static str,
    ) -> Self {
        ArgSpec {
            name,
            kind: ArgKind::Text,
            required,
            max_len,
            choices: &[],
            doc,
        }
    }

    pub const fn path(
        name: &'static str,
        required: bool,
        max_len: usize,
        doc: &'static str,
    ) -> Self {
        ArgSpec {
            name,
            kind: ArgKind::Path,
            required,
            max_len,
            choices: &[],
            doc,
        }
    }

    pub const fn url(
        name: &'static str,
        required: bool,
        max_len: usize,
        doc: &'static str,
    ) -> Self {
        ArgSpec {
            name,
            kind: ArgKind::Url,
            required,
            max_len,
            choices: &[],
            doc,
        }
    }

    pub const fn count(name: &'static str, required: bool, doc: &'static str) -> Self {
        ArgSpec {
            name,
            kind: ArgKind::Count,
            required,
            max_len: 10,
            choices: &[],
            doc,
        }
    }

    pub const fn choice(
        name: &'static str,
        required: bool,
        choices: &'static [&'static str],
        doc: &'static str,
    ) -> Self {
        ArgSpec {
            name,
            kind: ArgKind::Choice,
            required,
            max_len: 64,
            choices,
            doc,
        }
    }
}

/// Static description of a tool. Everything the policy engine and the CLI need
/// is here; nothing about a tool is discoverable only from its source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolSpec {
    /// Stable identifier. This string is the policy key.
    pub id: &'static str,
    /// One-line description shown to the user and the model prompt.
    pub description: &'static str,
    pub category: PermissionCategory,
    pub risk: RiskLevel,
    /// Tool-declared timeout; the policy may only lower it.
    pub timeout_secs: u32,
    /// Tool-declared output cap; the policy may only lower it.
    pub max_output_bytes: usize,
    /// The tool always asks, whatever the policy says.
    pub requires_confirmation: bool,
    pub args: &'static [ArgSpec],
}

impl ToolSpec {
    /// Facts for the policy engine.
    pub fn facts(&self) -> aaron_security::ToolFacts {
        aaron_security::ToolFacts {
            id: self.id.to_string(),
            category: self.category,
            risk: self.risk,
            declared_timeout_secs: self.timeout_secs,
            declared_max_output_bytes: self.max_output_bytes,
            requires_confirmation: self.requires_confirmation,
        }
    }

    /// A one-line summary including every declared argument.
    pub fn usage(&self) -> String {
        let args: Vec<String> = self
            .args
            .iter()
            .map(|a| {
                let base = format!("{}:{}", a.name, a.kind.as_str());
                if a.required {
                    base
                } else {
                    format!("{base}?")
                }
            })
            .collect();
        format!("{}({}) — {}", self.id, args.join(", "), self.description)
    }
}

/// Validated arguments for one tool invocation.
///
/// Construction is the only way to obtain one, and it enforces every
/// [`ArgSpec`]: unknown keys are rejected (so a typo cannot silently do
/// nothing), missing required keys are rejected, and every value is length
/// bounded.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolArgs {
    values: BTreeMap<String, String>,
    /// Keys that were supplied but are not declared. Always empty after
    /// validation; kept for the audit trail of a rejected call.
    rejected_keys: Vec<String>,
}

impl ToolArgs {
    /// Validate `raw` against `spec`.
    pub fn validate(spec: &ToolSpec, raw: &Value) -> Result<ToolArgs> {
        let obj = match raw {
            Value::Object(map) => map,
            Value::Null => {
                let map = Map::new();
                return Self::from_map(spec, &map);
            }
            other => {
                return Err(Error::Other(format!(
                    "tool `{}` expects an object of arguments, got {}",
                    spec.id,
                    type_name(other)
                )))
            }
        };
        Self::from_map(spec, obj)
    }

    fn from_map(spec: &ToolSpec, map: &Map<String, Value>) -> Result<ToolArgs> {
        let mut values = BTreeMap::new();
        let mut rejected = Vec::new();

        for (key, value) in map {
            let Some(arg) = spec.args.iter().find(|a| a.name == key.as_str()) else {
                rejected.push(key.clone());
                continue;
            };
            let text = match value {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => {
                    return Err(Error::Other(format!(
                        "argument `{}` of tool `{}` must not be null",
                        key, spec.id
                    )))
                }
                other => {
                    return Err(Error::Other(format!(
                        "argument `{}` of tool `{}` must be a scalar, got {}",
                        key,
                        spec.id,
                        type_name(other)
                    )))
                }
            };
            values.insert(key.clone(), validate_one(spec, arg, &text)?);
        }

        if !rejected.is_empty() {
            return Err(Error::Other(format!(
                "tool `{}` received undeclared argument(s): {}. Allowed: {}",
                spec.id,
                rejected.join(", "),
                spec.args
                    .iter()
                    .map(|a| a.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }

        for arg in spec.args.iter().filter(|a| a.required) {
            if !values.contains_key(arg.name) {
                return Err(Error::Other(format!(
                    "tool `{}` requires argument `{}` ({})",
                    spec.id, arg.name, arg.doc
                )));
            }
        }

        Ok(ToolArgs {
            values,
            rejected_keys: rejected,
        })
    }

    /// Build arguments directly from string pairs, for CLI invocation.
    pub fn from_pairs(spec: &ToolSpec, pairs: &[(&str, &str)]) -> Result<ToolArgs> {
        let mut map = Map::new();
        for (k, v) in pairs {
            map.insert((*k).to_string(), Value::String((*v).to_string()));
        }
        ToolArgs::from_map(spec, &map)
    }

    /// Build an argument set from values that have already been validated.
    ///
    /// Used by tests that need to reach a tool's own defence-in-depth check
    /// (which normally runs after spec validation has already passed) and by
    /// callers replaying a previously captured, validated call.
    pub fn from_validated<I, K, V>(pairs: I) -> ToolArgs
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<String>,
        V: Into<String>,
    {
        ToolArgs {
            values: pairs
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
            rejected_keys: Vec::new(),
        }
    }

    /// Fetch a validated string argument.
    pub fn get(&self, name: &str) -> Option<&str> {
        self.values.get(name).map(|s| s.as_str())
    }

    /// Fetch a validated count argument.
    pub fn get_count(&self, name: &str) -> Option<u32> {
        self.values.get(name).and_then(|s| s.parse().ok())
    }

    /// Every validated argument, for audit and display.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.values.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Undeclared keys seen during validation. Always empty on success.
    pub fn rejected_keys(&self) -> &[String] {
        &self.rejected_keys
    }
}

fn validate_one(spec: &ToolSpec, arg: &ArgSpec, text: &str) -> Result<String> {
    if text.chars().count() > arg.max_len {
        return Err(Error::LimitExceeded(format!(
            "argument `{}` of tool `{}` is {} chars, limit is {}",
            arg.name,
            spec.id,
            text.chars().count(),
            arg.max_len
        )));
    }
    match arg.kind {
        ArgKind::Text => {
            if text.contains('\0') {
                return Err(Error::Other(format!(
                    "argument `{}` contains a NUL byte",
                    arg.name
                )));
            }
            Ok(text.to_string())
        }
        ArgKind::Count => {
            let n: u32 = text.trim().parse().map_err(|_| {
                Error::Other(format!(
                    "argument `{}` must be a positive integer",
                    arg.name
                ))
            })?;
            if n == 0 {
                return Err(Error::Other(format!("argument `{}` must be > 0", arg.name)));
            }
            Ok(n.to_string())
        }
        ArgKind::Path => {
            aaron_security::PathSandbox::check_argument_safety(text)?;
            Ok(text.to_string())
        }
        ArgKind::Url => {
            if !(text.starts_with("http://") || text.starts_with("https://")) {
                return Err(Error::LimitExceeded(format!(
                    "argument `{}` must be an http:// or https:// URL",
                    arg.name
                )));
            }
            if text.len() > arg.max_len {
                return Err(Error::LimitExceeded("URL too long".into()));
            }
            // Embedded credentials are refused. The scheme check above would
            // accept `https://user:token@example.com/`, and handing that to a
            // browser writes the password into history, bookmarks and
            // `~/.mozilla` in plain text. A user who needs authenticated URLs
            // should let their browser handle them, not say them out loud.
            if let Some(authority) = url_authority(text) {
                if authority.contains('@') {
                    return Err(Error::PermissionDenied(format!(
                        "argument `{}` is a URL with embedded credentials; \
                         remove the `user:password@` part",
                        arg.name
                    )));
                }
            }
            Ok(text.to_string())
        }
        ArgKind::Choice => {
            if arg.choices.is_empty() || arg.choices.contains(&text) {
                Ok(text.to_string())
            } else {
                Err(Error::PermissionDenied(format!(
                    "`{text}` is not an allowed value for `{}`. Allowed: {}",
                    arg.name,
                    arg.choices.join(", ")
                )))
            }
        }
    }
}

/// The authority component of an absolute URL, or `None` if there is none.
///
/// Stops at the first `/`, `?` or `#` after `://`, which is where the host
/// ends by RFC 3986.
fn url_authority(url: &str) -> Option<&str> {
    let after_scheme = url.split_once("://")?.1;
    let end = after_scheme
        .find(['/', '?', '#'])
        .unwrap_or(after_scheme.len());
    Some(&after_scheme[..end])
}

fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const APPS: &[&str] = &["firefox", "code"];
    const SPEC: ToolSpec = ToolSpec {
        id: "start_application",
        description: "Launch a registered application",
        category: PermissionCategory::ProcessExec,
        risk: RiskLevel::Medium,
        timeout_secs: 10,
        max_output_bytes: 4096,
        requires_confirmation: true,
        args: &[
            ArgSpec::text("application", true, 64, "registered application name"),
            ArgSpec::count("wait_secs", false, "how long to wait"),
            ArgSpec::choice("flavour", false, APPS, "must be in the allow-list"),
        ],
    };

    #[test]
    fn accepts_valid_arguments() {
        let args = ToolArgs::validate(
            &SPEC,
            &json!({"application": "firefox", "wait_secs": 3, "flavour": "code"}),
        )
        .unwrap();
        assert_eq!(args.get("application"), Some("firefox"));
        assert_eq!(args.get_count("wait_secs"), Some(3));
    }

    #[test]
    fn rejects_unknown_arguments() {
        let err = ToolArgs::validate(&SPEC, &json!({"application": "firefox", "cmd": "rm -rf /"}))
            .unwrap_err();
        assert!(err.to_string().contains("undeclared argument"));
    }

    #[test]
    fn rejects_missing_required_arguments() {
        let err = ToolArgs::validate(&SPEC, &json!({})).unwrap_err();
        assert!(err.to_string().contains("requires argument `application`"));
    }

    #[test]
    fn rejects_oversized_arguments() {
        let long = "x".repeat(200);
        let err = ToolArgs::validate(&SPEC, &json!({"application": long})).unwrap_err();
        assert!(matches!(err, Error::LimitExceeded(_)));
    }

    #[test]
    fn rejects_non_integer_counts() {
        let err = ToolArgs::validate(
            &SPEC,
            &json!({"application": "firefox", "wait_secs": "soon"}),
        )
        .unwrap_err();
        assert!(err.to_string().contains("positive integer"));
    }

    #[test]
    fn rejects_values_outside_a_choice_allow_list() {
        let err = ToolArgs::validate(&SPEC, &json!({"application": "firefox", "flavour": "rm"}))
            .unwrap_err();
        assert!(matches!(err, Error::PermissionDenied(_)));
    }

    #[test]
    fn rejects_non_http_urls() {
        const URL_SPEC: ToolSpec = ToolSpec {
            id: "open_url",
            description: "open a url",
            category: PermissionCategory::Network,
            risk: RiskLevel::Medium,
            timeout_secs: 5,
            max_output_bytes: 1024,
            requires_confirmation: true,
            args: &[ArgSpec::url("url", true, 2048, "target url")],
        };
        for bad in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "ftp://x/y",
            "not a url",
        ] {
            assert!(
                ToolArgs::validate(&URL_SPEC, &json!({"url": bad})).is_err(),
                "`{bad}` should have been rejected"
            );
        }
        assert!(ToolArgs::validate(&URL_SPEC, &json!({"url": "https://example.com"})).is_ok());
    }

    #[test]
    fn rejects_nested_and_array_values() {
        let err = ToolArgs::validate(&SPEC, &json!({"application": ["a", "b"]})).unwrap_err();
        assert!(err.to_string().contains("must be a scalar"));
        let err = ToolArgs::validate(&SPEC, &json!(["a"])).unwrap_err();
        assert!(err.to_string().contains("expects an object"));
    }

    #[test]
    fn null_arguments_are_rejected() {
        let err = ToolArgs::validate(&SPEC, &json!({"application": null})).unwrap_err();
        assert!(err.to_string().contains("must not be null"));
    }

    #[test]
    fn path_arguments_reject_unsafe_shapes() {
        const P_SPEC: ToolSpec = ToolSpec {
            id: "read_file",
            description: "read a file",
            category: PermissionCategory::FilesystemRead,
            risk: RiskLevel::Low,
            timeout_secs: 5,
            max_output_bytes: 4096,
            requires_confirmation: false,
            args: &[ArgSpec::path("path", true, 4096, "file to read")],
        };
        assert!(
            ToolArgs::validate(&P_SPEC, &json!({"path": "../etc/passwd"})).is_ok(),
            "traversal is caught by the sandbox, not by shape validation"
        );
        assert!(ToolArgs::validate(&P_SPEC, &json!({"path": "a\0b"})).is_err());
        assert!(ToolArgs::validate(&P_SPEC, &json!({"path": "-rf"})).is_err());
    }

    #[test]
    fn usage_lists_declared_arguments() {
        let u = SPEC.usage();
        assert!(u.contains("start_application"));
        assert!(u.contains("application:text"));
        assert!(u.contains("wait_secs:count?"));
    }
}
