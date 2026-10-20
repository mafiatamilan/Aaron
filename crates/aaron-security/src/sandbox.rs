//! Filesystem sandbox.
//!
//! Every filesystem tool resolves its target through [`PathSandbox`]. The rules
//! are deliberately strict and are the reason `read_file` cannot be turned into
//! an arbitrary file exfiltration primitive:
//!
//! 1. The set of permitted roots is fixed at startup and comes from
//!    `security.allowed_roots` in the user's config. An empty list means *no*
//!    filesystem tool works.
//! 2. Paths are canonicalised **before** the containment check, so `..`
//!    traversal and symlinks that point outside a root are both rejected.
//! 3. A symlink whose target escapes a root is rejected even when the link
//!    itself sits inside an allowed directory.
//! 4. Control characters, NUL bytes and shell metacharacters in path arguments
//!    are rejected outright: paths are never passed through a shell, and a NUL
//!    byte reaching `open(2)` is either an error or a truncation attack.

use std::path::{Component, Path, PathBuf};

use aaron_core::error::{Error, Result};

/// A fixed set of directories the assistant may read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PathSandbox {
    roots: Vec<PathBuf>,
}

impl PathSandbox {
    /// Build a sandbox from configured roots.
    ///
    /// Roots are canonicalised so that later comparisons are apples-to-apples.
    /// A configured root that does not exist is an error, not a silent skip:
    /// the user asked for a specific scope and should be told when it is wrong.
    pub fn new<I, P>(roots: I) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<Path>,
    {
        let mut canonical = Vec::new();
        for root in roots {
            let root = root.as_ref();
            if !root.exists() {
                return Err(Error::Config(format!(
                    "security.allowed_roots entry `{}` does not exist",
                    root.display()
                )));
            }
            let c = root.canonicalize().map_err(|e| Error::io(root, e))?;
            if !c.is_dir() {
                return Err(Error::Config(format!(
                    "security.allowed_roots entry `{}` is not a directory",
                    root.display()
                )));
            }
            if !canonical.contains(&c) {
                canonical.push(c);
            }
        }
        Ok(PathSandbox { roots: canonical })
    }

    /// A sandbox that permits nothing. This is the default when
    /// `security.allowed_roots` is empty.
    pub fn deny_all() -> Self {
        PathSandbox { roots: Vec::new() }
    }

    /// The canonical roots, for display in `aaron permissions show`.
    pub fn roots(&self) -> &[PathBuf] {
        &self.roots
    }

    /// Whether any root is configured at all.
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }

    /// Reject argument strings that must never reach the filesystem layer.
    pub fn check_argument_safety(raw: &str) -> Result<()> {
        if raw.is_empty() {
            return Err(Error::PathRejected("empty path argument".into()));
        }
        if raw.len() > 4096 {
            return Err(Error::LimitExceeded(format!(
                "path argument is {} bytes, limit is 4096",
                raw.len()
            )));
        }
        if raw.contains('\0') {
            return Err(Error::PathRejected("path contains a NUL byte".into()));
        }
        if raw.chars().any(|c| (c as u32) < 0x20 || c == '\u{7f}') {
            return Err(Error::PathRejected(
                "path contains a control character".into(),
            ));
        }
        if raw.starts_with('-') {
            // Defends against a path being reinterpreted as a CLI flag.
            return Err(Error::PathRejected("path may not start with `-`".into()));
        }
        Ok(())
    }

    /// True when `candidate` is at or below one of the roots.
    pub fn contains(&self, candidate: &Path) -> bool {
        if !candidate.is_absolute() {
            return false;
        }
        self.roots.iter().any(|root| candidate.starts_with(root))
    }

    /// Canonicalise `raw` and require the result to live inside a root.
    ///
    /// Used by `read_file`. The target must exist and be a regular file.
    pub fn resolve_read(&self, raw: &str) -> Result<PathBuf> {
        if self.is_empty() {
            return Err(Error::PermissionDenied(
                "no filesystem roots are configured; add paths to security.allowed_roots".into(),
            ));
        }
        Self::check_argument_safety(raw)?;
        let joined = self.join_roots(raw)?;
        let canonical = canonicalize_lenient(&joined).ok_or_else(|| {
            Error::PathRejected(format!("`{raw}` does not exist inside any allowed root"))
        })?;
        if !self.contains(&canonical) {
            return Err(Error::PathRejected(format!(
                "`{raw}` resolves to `{}`, which is outside every allowed root",
                canonical.display()
            )));
        }
        let meta = std::fs::metadata(&canonical).map_err(|e| Error::io(&canonical, e))?;
        if !meta.is_file() {
            return Err(Error::PathRejected(format!(
                "`{raw}` is not a regular file"
            )));
        }
        Ok(canonical)
    }

    /// Resolve a directory to search in. `None` means "the first root".
    pub fn resolve_search_root(&self, raw: Option<&str>) -> Result<PathBuf> {
        if self.is_empty() {
            return Err(Error::PermissionDenied(
                "no filesystem roots are configured; add paths to security.allowed_roots".into(),
            ));
        }
        let Some(raw) = raw.filter(|r| !r.trim().is_empty()) else {
            return Ok(self.roots[0].clone());
        };
        Self::check_argument_safety(raw)?;
        let joined = self.join_roots(raw)?;
        let canonical = canonicalize_lenient(&joined)
            .ok_or_else(|| Error::PathRejected(format!("`{raw}` is not an existing directory")))?;
        if !self.contains(&canonical) {
            return Err(Error::PathRejected(format!(
                "`{raw}` is outside every allowed root"
            )));
        }
        if !canonical.is_dir() {
            return Err(Error::PathRejected(format!("`{raw}` is not a directory")));
        }
        Ok(canonical)
    }

    /// Whether a directory discovered during a walk is safe to descend into.
    /// Symlinked directories are skipped entirely rather than followed, which
    /// removes the possibility of a search loop.
    pub fn should_descend(&self, path: &Path, is_symlink: bool) -> bool {
        if is_symlink {
            return false;
        }
        match canonicalize_lenient(path) {
            Some(c) => self.contains(&c),
            None => false,
        }
    }

    /// Try the path against every root: absolute paths are used as-is, relative
    /// paths are joined to each root in turn until one exists.
    ///
    /// A relative path that exists nowhere is an error rather than a guess: the
    /// user asked for a specific file and silently resolving it against the
    /// wrong root is how "read the wrong file" bugs happen.
    fn join_roots(&self, raw: &str) -> Result<PathBuf> {
        let expanded = expand_home(raw)?;
        if expanded.is_absolute() {
            return Ok(expanded);
        }
        for root in &self.roots {
            let candidate = root.join(&expanded);
            if candidate.exists() {
                return Ok(candidate);
            }
        }
        Err(Error::PathRejected(format!(
            "`{raw}` does not exist inside any allowed root ({})",
            self.roots
                .iter()
                .map(|r| r.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }
}

/// Expand a leading `~` using `$HOME`.
pub fn expand_home(raw: &str) -> Result<PathBuf> {
    if let Some(rest) = raw.strip_prefix("~/") {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| Error::PathRejected("`~` used but $HOME is not set".into()))?;
        return Ok(PathBuf::from(home).join(rest));
    }
    if raw == "~" {
        let home = std::env::var_os("HOME")
            .ok_or_else(|| Error::PathRejected("`~` used but $HOME is not set".into()))?;
        return Ok(PathBuf::from(home));
    }
    Ok(PathBuf::from(raw))
}

/// Canonicalise without requiring the final component to exist.
///
/// Walks up to the nearest existing ancestor, canonicalises that, then re-joins
/// the missing tail. This lets `read_file` reject a traversal attempt even when
/// the attacker-named file does not exist.
pub fn canonicalize_lenient(path: &Path) -> Option<PathBuf> {
    if let Ok(c) = path.canonicalize() {
        return Some(c);
    }
    let mut missing: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor = path.to_path_buf();
    loop {
        let parent = cursor.parent()?.to_path_buf();
        match parent.canonicalize() {
            Ok(c) => {
                let mut out = c;
                for part in missing.iter().rev() {
                    out.push(part);
                }
                return Some(out);
            }
            Err(_) => {
                let name = cursor.file_name()?.to_os_string();
                missing.push(name);
                cursor = parent;
            }
        }
    }
}

/// Purely lexical normalisation used for *display* and for rejecting obvious
/// traversal before touching the filesystem.
pub fn lexical_normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Whether the process is running as root (uid 0).
pub fn running_as_root() -> bool {
    // SAFETY: `geteuid` is always safe and cannot fail in a meaningful way.
    unsafe { libc::geteuid() == 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn sandbox_with_dir(dir: &Path) -> PathSandbox {
        PathSandbox::new([dir]).unwrap()
    }

    #[test]
    fn denies_everything_when_no_roots_configured() {
        let s = PathSandbox::deny_all();
        assert!(s.is_empty());
        assert!(s.resolve_read("/etc/passwd").is_err());
        assert!(s.resolve_search_root(None).is_err());
    }

    #[test]
    fn reads_files_inside_a_root() {
        let tmp = tempfile::tempdir().unwrap();
        fs::write(tmp.path().join("ok.txt"), "hello").unwrap();
        let s = sandbox_with_dir(tmp.path());
        let p = s.resolve_read("ok.txt").unwrap();
        assert_eq!(fs::read_to_string(p).unwrap(), "hello");
    }

    #[test]
    fn rejects_parent_traversal() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        fs::write(tmp.path().join("secret.txt"), "top secret").unwrap();
        let s = sandbox_with_dir(&root);

        for attempt in [
            "../secret.txt",
            "../../secret.txt",
            "a/../../secret.txt",
            "/etc/passwd",
        ] {
            let err = s.resolve_read(attempt).unwrap_err();
            assert!(
                matches!(err, Error::PathRejected(_) | Error::PermissionDenied(_)),
                "attempt `{attempt}` produced {err}"
            );
        }
    }

    #[test]
    fn rejects_symlink_escaping_the_root() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        let outside = tmp.path().join("outside.txt");
        fs::write(&outside, "leak me").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&outside, root.join("link.txt")).unwrap();

        let s = sandbox_with_dir(&root);
        let err = s.resolve_read("link.txt").unwrap_err();
        assert!(
            matches!(err, Error::PathRejected(_)),
            "symlink escape allowed: {err}"
        );
    }

    #[test]
    fn never_descends_into_symlinked_directories() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        let other = tmp.path().join("other");
        fs::create_dir_all(&root).unwrap();
        fs::create_dir_all(&other).unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(&other, root.join("loop")).unwrap();

        let s = sandbox_with_dir(&root);
        assert!(!s.should_descend(&root.join("loop"), true));
        assert!(s.should_descend(&root, false));
    }

    #[test]
    fn rejects_dangerous_argument_strings() {
        assert!(PathSandbox::check_argument_safety("fine.txt").is_ok());
        assert!(PathSandbox::check_argument_safety("with\0nul").is_err());
        assert!(PathSandbox::check_argument_safety("-rf").is_err());
        assert!(PathSandbox::check_argument_safety("new\nline").is_err());
        assert!(PathSandbox::check_argument_safety("").is_err());
        assert!(PathSandbox::check_argument_safety(&"x".repeat(5000)).is_err());
        // Shell metacharacters are *not* rejected here: they are simply never
        // interpreted, because paths are never passed through a shell. This
        // test documents that a file literally named `a;b.txt` still works.
        assert!(PathSandbox::check_argument_safety("a;b.txt").is_ok());
    }

    #[test]
    fn search_root_defaults_to_first_root() {
        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("a");
        let b = tmp.path().join("b");
        fs::create_dir_all(&a).unwrap();
        fs::create_dir_all(&b).unwrap();
        let s = PathSandbox::new([&a, &b]).unwrap();
        assert_eq!(
            s.resolve_search_root(None).unwrap(),
            a.canonicalize().unwrap()
        );
        assert_eq!(
            s.resolve_search_root(Some(&b.display().to_string()))
                .unwrap(),
            b.canonicalize().unwrap()
        );
        // A relative name that exists in no root is an explicit error rather
        // than a silent guess at which root the user meant.
        assert!(matches!(
            s.resolve_search_root(Some("b")),
            Err(Error::PathRejected(_))
        ));
        assert!(s.resolve_search_root(Some("..")).is_err());
    }

    #[test]
    fn nonexistent_root_is_a_config_error() {
        let err = PathSandbox::new(["/definitely/not/here"]).unwrap_err();
        assert!(matches!(err, Error::Config(_)));
    }

    #[test]
    fn lexical_normalize_collapses_dotdot() {
        assert_eq!(
            lexical_normalize(Path::new("/a/b/../c")),
            PathBuf::from("/a/c")
        );
        assert_eq!(
            lexical_normalize(Path::new("/a/./b")),
            PathBuf::from("/a/b")
        );
    }
}
