//! Filesystem tools: `search_files` and `read_file`.
//!
//! Both go through the [`aaron_security::PathSandbox`]. Neither can be pointed
//! outside `security.allowed_roots`, and neither follows a symlink out of a
//! root. `search_files` refuses to descend into symlinked directories at all,
//! which removes the possibility of a symlink loop.

use aaron_core::error::{Error, Result};
use serde_json::json;
use walkdir::WalkDir;

use crate::registry::{Tool, ToolContext, ToolOutput};
use crate::spec::{ArgSpec, ToolArgs, ToolSpec};
use aaron_security::{PermissionCategory, RiskLevel};

pub const SEARCH_SPEC: ToolSpec = ToolSpec {
    id: "search_files",
    description: "Find files by name inside the configured roots",
    category: PermissionCategory::FilesystemRead,
    risk: RiskLevel::Low,
    timeout_secs: 15,
    max_output_bytes: 16 * 1024,
    requires_confirmation: false,
    args: &[
        ArgSpec::text("pattern", true, 128, "file name or glob such as `*.rs`"),
        ArgSpec::path(
            "root",
            false,
            4096,
            "directory to search; defaults to the first allowed root",
        ),
    ],
};

pub const READ_SPEC: ToolSpec = ToolSpec {
    id: "read_file",
    description: "Read the beginning of a text file",
    category: PermissionCategory::FilesystemRead,
    risk: RiskLevel::Low,
    timeout_secs: 10,
    max_output_bytes: 64 * 1024,
    requires_confirmation: true,
    args: &[
        ArgSpec::path("path", true, 4096, "file to read"),
        ArgSpec::count("lines", false, "how many lines to return (default 40)"),
    ],
};

/// Convert a user-supplied name or glob into a matcher.
///
/// The pattern is never handed to a shell or to `glob(3)`. It is parsed here:
/// `*` matches within a path segment, `?` matches one character, and anything
/// else must match literally. Characters that only mean something to a regex
/// are therefore always literal.
fn glob_matcher(pattern: &str) -> impl Fn(&str) -> bool + '_ {
    move |name: &str| glob_match_inner(pattern.as_bytes(), name.as_bytes())
}

fn glob_match_inner(pattern: &[u8], name: &[u8]) -> bool {
    // Iterative backtracking: linear in practice and immune to catastrophic
    // backtracking, unlike a translated regex.
    let (mut p, mut n) = (0usize, 0usize);
    let (mut star, mut mark) = (usize::MAX, 0usize);
    while n < name.len() {
        if p < pattern.len() && (pattern[p] == b'?' || pattern[p] == name[n]) {
            p += 1;
            n += 1;
        } else if p < pattern.len() && pattern[p] == b'*' {
            star = p;
            mark = n;
            p += 1;
        } else if star != usize::MAX {
            p = star + 1;
            mark += 1;
            n = mark;
        } else {
            return false;
        }
    }
    while p < pattern.len() && pattern[p] == b'*' {
        p += 1;
    }
    p == pattern.len()
}

/// Whether a file looks like text, so `read_file` never dumps a binary blob.
fn looks_like_text(path: &std::path::Path) -> Result<bool> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).map_err(|e| Error::io(path, e))?;
    let mut buf = [0u8; 4096];
    let n = f.read(&mut buf).map_err(|e| Error::io(path, e))?;
    if n == 0 {
        return Ok(true);
    }
    let slice = &buf[..n];
    // A NUL byte is the conventional "this is binary" signal.
    if slice.contains(&0) {
        return Ok(false);
    }
    let printable = slice
        .iter()
        .filter(|b| (**b >= 0x20 && **b < 0x7f) || **b == b'\n' || **b == b'\r' || **b == b'\t')
        .count();
    Ok(printable * 100 / slice.len() >= 85)
}

#[derive(Debug)]
pub struct SearchFilesTool;

impl Tool for SearchFilesTool {
    fn spec(&self) -> &'static ToolSpec {
        &SEARCH_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let pattern = args
            .get("pattern")
            .ok_or_else(|| Error::Other("search_files requires a `pattern`".into()))?;
        let root = ctx.sandbox.resolve_search_root(args.get("root"))?;
        let max_results = ctx.config.limits.max_search_results;
        let matcher = glob_matcher(pattern);

        let mut results: Vec<String> = Vec::new();
        let mut skipped_symlinks = 0usize;
        let mut truncated = false;

        for entry in WalkDir::new(&root)
            .follow_links(false)
            .max_depth(12)
            .into_iter()
            .filter_entry(|e| {
                let is_symlink = e.file_type().is_symlink();
                if is_symlink {
                    skipped_symlinks += 1;
                    return false;
                }
                ctx.sandbox.should_descend(e.path(), false)
            })
        {
            let Ok(entry) = entry else { continue };
            if !entry.file_type().is_file() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if !matcher(&name) {
                continue;
            }
            // Strip control characters from a hostile filename so it can be
            // printed to a terminal without doing terminal things.
            let safe: String = aaron_core::text::truncate_chars(&name, 120)
                .chars()
                .filter(|c| !c.is_control())
                .collect();
            if safe.is_empty() {
                continue;
            }
            let rel = entry
                .path()
                .strip_prefix(&root)
                .unwrap_or(entry.path())
                .display()
                .to_string();
            results.push(rel);
            if results.len() >= max_results {
                truncated = true;
                break;
            }
        }

        let mut text = if results.is_empty() {
            format!("No files matching `{pattern}` under {}", root.display())
        } else {
            let mut t = results.join("\n");
            if truncated {
                t.push_str(&format!(
                    "\n\n(showing the first {max_results} matches; narrow the pattern for more)"
                ));
            }
            t
        };
        if skipped_symlinks > 0 {
            text.push_str(&format!(
                "\n\n({skipped_symlinks} symlink(s) skipped; AARON never follows them during search)"
            ));
        }

        Ok(ToolOutput {
            bytes: text.len(),
            text,
            truncated,
            data: Some(json!({
                "root": root.display().to_string(),
                "pattern": pattern,
                "matches": results.len(),
                "skipped_symlinks": skipped_symlinks,
            })),
        }
        .capped(SEARCH_SPEC.max_output_bytes))
    }
}

#[derive(Debug)]
pub struct ReadFileTool;

impl Tool for ReadFileTool {
    fn spec(&self) -> &'static ToolSpec {
        &READ_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let raw = args
            .get("path")
            .ok_or_else(|| Error::Other("read_file requires a `path`".into()))?;
        let lines = args.get_count("lines").unwrap_or(40).clamp(1, 5_000) as usize;

        let path = ctx.sandbox.resolve_read(raw)?;
        if !looks_like_text(&path)? {
            return Err(Error::Other(format!(
                "`{}` looks like a binary file; AARON will not print binary contents",
                path.display()
            )));
        }

        let contents = std::fs::read(&path).map_err(|e| Error::io(&path, e))?;
        let byte_cap = ctx
            .config
            .limits
            .max_read_bytes
            .min(READ_SPEC.max_output_bytes);
        let truncated_bytes = contents.len() > byte_cap;
        let slice = &contents[..contents.len().min(byte_cap)];

        let text = String::from_utf8_lossy(slice);
        let mut out: String = text.lines().take(lines).collect::<Vec<_>>().join("\n");
        let total_lines = text.lines().count();
        if total_lines > lines {
            out.push_str(&format!("\n\n({} of {total_lines} lines shown)", lines));
        }
        if truncated_bytes {
            out.push_str("\n\n(file truncated by the configured read limit)");
        }

        Ok(ToolOutput {
            bytes: contents.len(),
            text: out,
            truncated: truncated_bytes || total_lines > lines,
            data: Some(json!({
                "path": path.display().to_string(),
                "lines_available": total_lines,
                "lines_returned": lines,
                "bytes_total": contents.len(),
            })),
        })
    }
}

/// Shared helper for tests and for `aaron tools list`.
pub fn glob_examples() -> Vec<(&'static str, &'static str, bool)> {
    vec![
        ("*.rs", "main.rs", true),
        ("*.rs", "main.py", false),
        ("README*", "README.md", true),
        ("a?c.txt", "abc.txt", true),
        ("a?c.txt", "abbc.txt", false),
        ("exact", "exact", true),
        ("exact", "exactly", false),
        ("*", "*", true),
        ("", "anything", false),
        ("*a*b*", "xxaxxbxx", true),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use aaron_memory::MemoryStore;
    use aaron_security::PathSandbox;
    use std::fs;
    use std::sync::Arc;

    fn ctx_with_root(root: &std::path::Path) -> Arc<ToolContext> {
        ctx_with(root, aaron_core::Config::default())
    }

    fn ctx_with(root: &std::path::Path, config: aaron_core::Config) -> Arc<ToolContext> {
        Arc::new(ToolContext {
            sandbox: Arc::new(PathSandbox::new([root]).unwrap()),
            memory: Arc::new(std::sync::Mutex::new(
                MemoryStore::open_in_memory(Default::default()).unwrap(),
            )),
            config: Arc::new(config),
            project_root: root.to_path_buf(),
        })
    }

    #[test]
    fn glob_matching_behaves_like_a_shell_glob_not_a_regex() {
        for (pattern, name, expected) in glob_examples() {
            assert_eq!(
                glob_match_inner(pattern.as_bytes(), name.as_bytes()),
                expected,
                "pattern `{pattern}` against `{name}`"
            );
        }
    }

    #[test]
    fn regex_metacharacters_are_literal_in_patterns() {
        // `.` must not match any character, and `+`/`(` must not be special.
        assert!(!glob_match_inner(b"file.txt", b"fileXtxt"));
        assert!(glob_match_inner(b"file.txt", b"file.txt"));
        assert!(glob_match_inner(b"a+b", b"a+b"));
        assert!(!glob_match_inner(b"a+b", b"aab"));
    }

    #[test]
    fn search_finds_matching_files_inside_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        fs::create_dir_all(tmp.path().join("src")).unwrap();
        fs::write(tmp.path().join("src/main.rs"), "fn main(){}").unwrap();
        fs::write(tmp.path().join("src/lib.rs"), "").unwrap();
        fs::write(tmp.path().join("src/readme.md"), "").unwrap();

        let ctx = ctx_with_root(tmp.path());
        let out = SearchFilesTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&SEARCH_SPEC, &[("pattern", "*.rs")]).unwrap(),
            )
            .unwrap();
        assert!(out.text.contains("main.rs"));
        assert!(out.text.contains("lib.rs"));
        assert!(!out.text.contains("readme.md"));
    }

    #[test]
    fn search_respects_the_result_cap() {
        let tmp = tempfile::tempdir().unwrap();
        for i in 0..30 {
            fs::write(tmp.path().join(format!("f{i}.txt")), "").unwrap();
        }
        let mut config = aaron_core::Config::default();
        config.limits.max_search_results = 5;
        let ctx = ctx_with(tmp.path(), config);

        let out = SearchFilesTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&SEARCH_SPEC, &[("pattern", "*.txt")]).unwrap(),
            )
            .unwrap();
        assert!(out.text.contains("showing the first 5 matches"));
        assert!(out.truncated);
    }

    #[test]
    fn search_refuses_to_escape_the_sandbox() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(tmp.path().join("secret.txt"), "").unwrap();

        let ctx = ctx_with_root(&root);
        let out = SearchFilesTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&SEARCH_SPEC, &[("pattern", "*.txt"), ("root", "..")])
                    .unwrap(),
            )
            .unwrap_err();
        assert!(matches!(out, aaron_core::Error::PathRejected(_)));
    }

    #[test]
    fn search_does_not_follow_symlinks_out_of_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(tmp.path().join("outside.txt"), "").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(tmp.path().join("outside.txt"), root.join("link.txt")).unwrap();

        let ctx = ctx_with_root(&root);
        let out = SearchFilesTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&SEARCH_SPEC, &[("pattern", "*.txt")]).unwrap(),
            )
            .unwrap();
        assert!(!out.text.contains("link.txt"));
        assert!(out.text.contains("symlink"));
    }

    #[test]
    fn read_returns_the_requested_number_of_lines() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("a.txt"), "1\n2\n3\n4\n5\n").unwrap();
        let ctx = ctx_with_root(tmp.path());
        let out = ReadFileTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&READ_SPEC, &[("path", "a.txt"), ("lines", "2")]).unwrap(),
            )
            .unwrap();
        assert!(out.text.starts_with("1\n2"));
        assert!(out.text.contains("2 of 5 lines shown"));
    }

    #[test]
    fn read_rejects_traversal_and_symlink_escapes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(tmp.path().join("secret.txt"), "classified").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(tmp.path().join("secret.txt"), root.join("link.txt")).unwrap();

        let ctx = ctx_with_root(&root);
        for attempt in ["../secret.txt", "link.txt"] {
            let err = ReadFileTool
                .execute(
                    &ctx,
                    &ToolArgs::from_pairs(&READ_SPEC, &[("path", attempt)]).unwrap(),
                )
                .unwrap_err();
            assert!(
                matches!(err, aaron_core::Error::PathRejected(_)),
                "`{attempt}` produced {err}"
            );
        }
    }

    #[test]
    fn read_refuses_binary_files() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("blob.bin"), [0u8, 1, 2, 3, 0, 255]).unwrap();
        let ctx = ctx_with_root(tmp.path());
        let err = ReadFileTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&READ_SPEC, &[("path", "blob.bin")]).unwrap(),
            )
            .unwrap_err();
        assert!(err.to_string().contains("binary"));
    }

    #[test]
    fn read_reports_when_no_roots_are_configured() {
        let ctx = ToolContext::for_tests("/tmp");
        let err = ReadFileTool
            .execute(
                &ctx,
                &ToolArgs::from_pairs(&READ_SPEC, &[("path", "/etc/hostname")]).unwrap(),
            )
            .unwrap_err();
        assert!(matches!(err, aaron_core::Error::PermissionDenied(_)));
    }
}
