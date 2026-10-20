//! Memory tools: `remember` and `recall`.
//!
//! `remember` writes a fact the user explicitly asked AARON to store, and
//! requires confirmation unless the user has configured otherwise. `recall` is
//! a keyword search over the same store.
//!
//! Neither tool reads a transcript. Nothing is written to memory unless a
//! `remember` command reaches this tool.

use aaron_core::error::{Error, Result};
use aaron_memory::MemoryKind;
use serde_json::json;

use crate::registry::{Tool, ToolContext, ToolOutput};
use crate::spec::{ArgSpec, ToolArgs, ToolSpec};
use aaron_security::{PermissionCategory, RiskLevel};

pub const REMEMBER_SPEC: ToolSpec = ToolSpec {
    id: "remember",
    description: "Store a user-approved fact in long-term memory",
    category: PermissionCategory::MemoryWrite,
    risk: RiskLevel::Medium,
    timeout_secs: 5,
    max_output_bytes: 2048,
    requires_confirmation: true,
    args: &[
        ArgSpec::text("text", true, 2000, "the fact to remember"),
        ArgSpec::text("key", false, 64, "optional short label"),
        ArgSpec::text("tags", false, 128, "comma-separated tags"),
        ArgSpec::text("kind", false, 16, "`fact` or `preference`"),
    ],
};

pub const RECALL_SPEC: ToolSpec = ToolSpec {
    id: "recall",
    description: "Search long-term memory",
    category: PermissionCategory::SystemRead,
    risk: RiskLevel::Low,
    timeout_secs: 5,
    max_output_bytes: 8192,
    requires_confirmation: false,
    args: &[
        ArgSpec::text("query", true, 200, "what to look for"),
        ArgSpec::count("limit", false, "maximum number of results"),
    ],
};

#[derive(Debug)]
pub struct RememberTool;

impl Tool for RememberTool {
    fn spec(&self) -> &'static ToolSpec {
        &REMEMBER_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let text = args
            .get("text")
            .ok_or_else(|| Error::Other("remember requires `text`".into()))?;
        if text.trim().is_empty() {
            return Err(Error::Other(
                "refusing to remember an empty statement".into(),
            ));
        }
        let key = args.get("key").filter(|k| !k.trim().is_empty());
        let kind = match args.get("kind").unwrap_or("fact") {
            "preference" => MemoryKind::Preference,
            "fact" => MemoryKind::Fact,
            other => {
                return Err(Error::Other(format!(
                    "`{other}` is not a memory kind AARON can store; use `fact` or `preference`"
                )))
            }
        };

        let tags: Vec<String> =
            aaron_core::text::extract_tags(args.get("tags").unwrap_or("").trim_matches(','))
                .into_iter()
                .collect();
        // The free-text form is also scanned for `#tags`.
        let mut tags = tags;
        for t in aaron_core::text::extract_tags(text) {
            if !tags.contains(&t) {
                tags.push(t);
            }
        }

        let guard = ctx
            .memory
            .lock()
            .map_err(|_| Error::Storage("memory lock poisoned".into()))?;
        let mut memory = guard;
        let id = memory.add(kind, key, text, &tags, "user")?;
        memory.prune()?;

        Ok(ToolOutput::text(format!(
            "Remembered (id {id}): {}",
            aaron_core::text::truncate_chars(text, 120)
        ))
        .with_data(json!({"id": id, "kind": kind.as_str(), "tags": tags})))
    }
}

#[derive(Debug)]
pub struct RecallTool;

impl Tool for RecallTool {
    fn spec(&self) -> &'static ToolSpec {
        &RECALL_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let query = args
            .get("query")
            .ok_or_else(|| Error::Other("recall requires a `query`".into()))?;
        let limit = args.get_count("limit").unwrap_or(10).clamp(1, 100) as usize;

        let guard = ctx
            .memory
            .lock()
            .map_err(|_| Error::Storage("memory lock poisoned".into()))?;
        let hits = guard.search(query, limit)?;

        let text = if hits.is_empty() {
            format!("Nothing in memory matches \"{query}\".")
        } else {
            hits.iter()
                .map(|h| {
                    let label = h.key.clone().unwrap_or_else(|| format!("#{}", h.id));
                    let tags = if h.tags.is_empty() {
                        String::new()
                    } else {
                        format!(" [{}]", h.tags.join(", "))
                    };
                    let pinned = if h.pinned { " (pinned)" } else { "" };
                    format!(
                        "{label}: {}{tags}{pinned}",
                        aaron_core::text::truncate_chars(&h.value, 160)
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };

        Ok(
            ToolOutput::text(text).with_data(json!({"hits": hits.iter().map(|h| json!({
                "id": h.id, "kind": h.kind.as_str(), "key": h.key, "value": h.value,
                "tags": h.tags, "pinned": h.pinned, "source": h.source,
            })).collect::<Vec<_>>() })),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ToolContext;
    use aaron_memory::MemoryStore;
    use aaron_security::PathSandbox;
    use std::sync::{Arc, Mutex};

    fn ctx() -> Arc<ToolContext> {
        Arc::new(ToolContext {
            sandbox: Arc::new(PathSandbox::deny_all()),
            memory: Arc::new(Mutex::new(
                MemoryStore::open_in_memory(Default::default()).unwrap(),
            )),
            config: Arc::new(aaron_core::Config::default()),
            project_root: std::path::PathBuf::from("/tmp"),
        })
    }

    fn args(spec: &ToolSpec, pairs: &[(&str, &str)]) -> ToolArgs {
        ToolArgs::from_pairs(spec, pairs).unwrap()
    }

    #[test]
    fn remember_then_recall_round_trips() {
        let c = ctx();
        RememberTool
            .execute(
                &c,
                &args(
                    &REMEMBER_SPEC,
                    &[
                        ("text", "My favourite editor is neovim"),
                        ("key", "editor"),
                        ("tags", "#tools"),
                    ],
                ),
            )
            .unwrap();
        let out = RecallTool
            .execute(&c, &args(&RECALL_SPEC, &[("query", "editor")]))
            .unwrap();
        assert!(out.text.contains("neovim"));
        assert!(out.text.contains("tools"));
    }

    #[test]
    fn remember_refuses_empty_and_unknown_kinds() {
        let c = ctx();
        assert!(RememberTool
            .execute(&c, &args(&REMEMBER_SPEC, &[("text", "   ")]))
            .is_err());
        let err = RememberTool
            .execute(
                &c,
                &args(&REMEMBER_SPEC, &[("text", "x"), ("kind", "secrets")]),
            )
            .unwrap_err();
        assert!(err.to_string().contains("not a memory kind"));
    }

    #[test]
    fn recall_reports_an_empty_store_honestly() {
        let c = ctx();
        let out = RecallTool
            .execute(&c, &args(&RECALL_SPEC, &[("query", "nothing")]))
            .unwrap();
        assert!(out.text.contains("Nothing in memory"));
    }

    #[test]
    fn memories_are_private_by_default() {
        let c = ctx();
        RememberTool
            .execute(&c, &args(&REMEMBER_SPEC, &[("text", "secret plan")]))
            .unwrap();
        let store = c.memory.lock().unwrap();
        assert!(store.exportable(10).unwrap().is_empty());
        assert_eq!(store.list(MemoryKind::Fact, 10).unwrap().len(), 1);
    }
}
