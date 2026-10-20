//! Runtime path resolution.
//!
//! AARON keeps everything inside one runtime directory so the whole install can
//! be inspected, backed up or deleted from a single place.
//!
//! Resolution order for the runtime root:
//! 1. `$AARON_HOME` if set,
//! 2. `$XDG_STATE_HOME/aaron` if set,
//! 3. `~/.local/state/aaron`.

use std::path::{Path, PathBuf};

use crate::error::{Error, Result};

/// Well-known locations inside the runtime directory.
#[derive(Debug, Clone)]
pub struct RuntimePaths {
    pub root: PathBuf,
    pub data: PathBuf,
    pub logs: PathBuf,
    pub audit: PathBuf,
    pub memory_db: PathBuf,
    pub checkpoints: PathBuf,
}

impl RuntimePaths {
    /// Build the path set rooted at `root`, creating directories on demand in
    /// [`RuntimePaths::ensure`].
    pub fn new(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        RuntimePaths {
            data: root.join("data"),
            logs: root.join("logs"),
            audit: root.join("data").join("audit"),
            memory_db: root.join("data").join("memory.db"),
            checkpoints: root.join("data").join("checkpoints"),
            root,
        }
    }

    /// Derive the runtime root from the environment.
    pub fn discover() -> Self {
        if let Some(explicit) = std::env::var_os("AARON_HOME") {
            return RuntimePaths::new(explicit);
        }
        if let Some(state) = std::env::var_os("XDG_STATE_HOME") {
            return RuntimePaths::new(PathBuf::from(state).join("aaron"));
        }
        let home = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."));
        RuntimePaths::new(home.join(".local/state/aaron"))
    }

    /// Create the directories AARON writes to.
    pub fn ensure(&self) -> Result<()> {
        for dir in [
            &self.root,
            &self.data,
            &self.logs,
            &self.audit,
            &self.checkpoints,
        ] {
            std::fs::create_dir_all(dir).map_err(|e| Error::io(dir, e))?;
        }
        Ok(())
    }
}

/// Locate the repository root by walking up from `start` looking for a
/// `Cargo.toml` that declares the AARON workspace.
///
/// Installed binaries fall back to `$AARON_HOME` so that they never depend on a
/// source checkout.
pub fn project_root(start: &Path) -> Option<PathBuf> {
    let mut dir = start;
    loop {
        let manifest = dir.join("Cargo.toml");
        if manifest.is_file() {
            if let Ok(text) = std::fs::read_to_string(&manifest) {
                if text.contains("aaron-core") || text.contains("[workspace]") {
                    return Some(dir.to_path_buf());
                }
            }
        }
        dir = dir.parent()?;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_paths_nest_under_root() {
        let p = RuntimePaths::new("/tmp/aaron-test-root");
        assert!(p.memory_db.starts_with(&p.root));
        assert!(p.audit.starts_with(&p.root));
        assert!(p.checkpoints.starts_with(&p.root));
    }

    #[test]
    fn discover_respects_explicit_env() {
        // SAFETY: single-threaded test body, restored before returning.
        let prev = std::env::var_os("AARON_HOME");
        std::env::set_var("AARON_HOME", "/tmp/aaron-explicit");
        let p = RuntimePaths::discover();
        assert_eq!(p.root, PathBuf::from("/tmp/aaron-explicit"));
        match prev {
            Some(v) => std::env::set_var("AARON_HOME", v),
            None => std::env::remove_var("AARON_HOME"),
        }
    }
}
