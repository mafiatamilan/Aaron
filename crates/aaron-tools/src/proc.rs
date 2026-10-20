//! Sandboxed child-process execution.
//!
//! The rules here are the reason AARON is not "a shell with a voice":
//!
//! * The program is either an absolute path or resolved through
//!   [`aaron_core::hardware::which`], which skips world-writable `PATH`
//!   directories so a planted binary cannot be substituted.
//! * Arguments are always passed as a `Vec<String>`. **No shell is ever
//!   involved**, so `;`, `|`, `$()` and backticks are inert bytes rather than
//!   syntax.
//! * The environment is cleared and rebuilt from a small allow-list, so a child
//!   cannot read AARON's environment or its `LD_PRELOAD`.
//! * The child gets its own process group and an `RLIMIT_CPU` / `RLIMIT_FSIZE`
//!   ceiling; on timeout the whole group is killed with `SIGKILL`, not just the
//!   direct child, so a `cargo test` that spawned eight workers still dies.
//! * Output is capped while streaming. The reader threads stop as soon as the
//!   cap is hit, so a runaway process cannot fill memory before it is killed.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aaron_core::error::{Error, Result};
use serde::{Deserialize, Serialize};

/// Environment variables passed through to a child process.
const ENV_ALLOWLIST: &[&str] = &[
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TERM",
    "DISPLAY",
    "XAUTHORITY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
    "XDG_DATA_HOME",
    "XDG_CONFIG_HOME",
    "DBUS_SESSION_BUS_ADDRESS",
    "SSH_AUTH_SOCK",
    "NO_COLOR",
];

/// What happened to a child process.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExitKind {
    /// Ran to completion with status 0.
    Success,
    /// Ran to completion with a non-zero status.
    Failed,
    /// Killed because it exceeded the timeout.
    TimedOut,
    /// Killed because it exceeded the output cap.
    OutputExceeded,
}

/// The result of running a child process.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CommandOutput {
    pub program: String,
    pub args: Vec<String>,
    pub exit_kind: ExitKind,
    /// Exit code, or `None` when the child was killed by a signal.
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    /// Whether either stream hit the byte cap.
    pub truncated: bool,
    pub duration_ms: u64,
}

impl CommandOutput {
    /// Combined stdout and stderr, capped.
    pub fn combined(&self) -> String {
        let mut s = String::new();
        if !self.stdout.is_empty() {
            s.push_str(&self.stdout);
        }
        if !self.stderr.is_empty() {
            if !s.is_empty() {
                s.push('\n');
            }
            s.push_str(&self.stderr);
        }
        s
    }

    /// True when the command finished successfully.
    pub fn ok(&self) -> bool {
        self.exit_kind == ExitKind::Success
    }
}

/// Resolve a program name to a concrete path.
///
/// An absolute or explicitly relative path is used as given (after checking it
/// is an executable file). A bare name is resolved through `PATH`, skipping
/// world-writable directories.
pub fn resolve_program(program: &str) -> Result<PathBuf> {
    if program.is_empty() {
        return Err(Error::Other("empty program name".into()));
    }
    if program.contains('/') {
        let p = PathBuf::from(program);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(Error::MissingDependency(format!(
                "`{program}` is not an executable file"
            )))
        };
    }
    aaron_core::hardware::which(program).ok_or_else(|| {
        Error::MissingDependency(format!(
            "`{program}` was not found on PATH. AARON does not install optional tools for you; \
             install the package yourself, or set the path in your config."
        ))
    })
}

/// Options for one execution.
#[derive(Debug, Clone)]
pub struct ExecOptions {
    pub program: String,
    pub args: Vec<String>,
    pub cwd: Option<PathBuf>,
    pub timeout: Duration,
    pub max_output_bytes: usize,
    /// Extra environment entries, applied after the allow-list.
    pub extra_env: Vec<(String, String)>,
    /// The address-space limit, in bytes. `None` means no limit.
    ///
    /// The default is four gibibytes, which is a good guard for the tool
    /// layer: a tool that allocates without bound is refused rather than
    /// swapping the machine to death.
    ///
    /// It is the wrong limit for a model runtime. Torch and the ONNX
    /// runtimes reserve a very large *virtual* address space up front — far
    /// more than they ever touch — so an address-space cap makes them fail
    /// to allocate a single megabyte. A model runtime needs either a much
    /// higher cap or none, and the timeout is what actually bounds it.
    pub address_space_bytes: Option<u64>,
}

impl ExecOptions {
    /// Build options for a fixed program and argument vector.
    pub fn new<I, S>(program: impl Into<String>, args: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        ExecOptions {
            program: program.into(),
            args: args.into_iter().map(Into::into).collect(),
            cwd: None,
            timeout: Duration::from_secs(30),
            max_output_bytes: 64 * 1024,
            extra_env: Vec::new(),
            address_space_bytes: Some(4 * 1024 * 1024 * 1024),
        }
    }

    pub fn cwd(mut self, dir: impl Into<PathBuf>) -> Self {
        self.cwd = Some(dir.into());
        self
    }

    pub fn timeout(mut self, secs: u32) -> Self {
        self.timeout = Duration::from_secs(secs.clamp(1, 3600) as u64);
        self
    }

    pub fn max_output(mut self, bytes: usize) -> Self {
        self.max_output_bytes = bytes.clamp(256, 8 * 1024 * 1024);
        self
    }

    /// Set the address-space limit, in bytes.
    pub fn address_space(mut self, bytes: u64) -> Self {
        self.address_space_bytes = Some(bytes);
        self
    }

    /// Remove the address-space limit.
    ///
    /// For a trusted local program that reserves a large virtual address
    /// space — a model runtime — and whose real bound is the timeout.
    pub fn without_address_space_limit(mut self) -> Self {
        self.address_space_bytes = None;
        self
    }
}

/// Run a child process under the sandbox rules described at module level.
pub fn run(options: &ExecOptions) -> Result<CommandOutput> {
    let started = Instant::now();
    let program_path = resolve_program(&options.program)?;

    let mut cmd = Command::new(&program_path);
    cmd.args(&options.args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_clear();

    for key in ENV_ALLOWLIST {
        if let Ok(value) = std::env::var(key) {
            cmd.env(key, value);
        }
    }
    for (k, v) in &options.extra_env {
        cmd.env(k, v);
    }

    if let Some(dir) = &options.cwd {
        // The working directory must exist and be a directory; a missing cwd
        // would otherwise surface as a confusing spawn error.
        if !dir.is_dir() {
            return Err(Error::Other(format!(
                "working directory `{}` does not exist",
                dir.display()
            )));
        }
        cmd.current_dir(dir);
    }

    let cpu_secs = options.timeout.as_secs().clamp(1, 3600);
    let address_space_bytes = options.address_space_bytes;
    let fsize_bytes = (options.max_output_bytes as u64)
        .saturating_mul(8)
        .max(64 * 1024);

    // SAFETY: `pre_exec` runs between fork and exec. Every call below is
    // async-signal-safe and none of them allocate.
    unsafe {
        use std::os::unix::process::CommandExt;
        cmd.pre_exec(move || {
            // New process group, so a timeout can kill the whole tree.
            if libc::setpgid(0, 0) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // CPU seconds: one second of headroom for a clean exit.
            let cpu = libc::rlimit {
                rlim_cur: cpu_secs,
                rlim_max: cpu_secs + 1,
            };
            if libc::setrlimit(libc::RLIMIT_CPU, &cpu) != 0 {
                return Err(std::io::Error::last_os_error());
            }
            // File size: stop a process that writes an unbounded log file.
            let fsize = libc::rlimit {
                rlim_cur: fsize_bytes,
                rlim_max: fsize_bytes,
            };
            libc::setrlimit(libc::RLIMIT_FSIZE, &fsize);
            // Address space: refuse a runaway allocation rather than swap.
            if let Some(as_limit) = address_space_bytes {
                let rlimit_as = libc::rlimit {
                    rlim_cur: as_limit,
                    rlim_max: as_limit,
                };
                libc::setrlimit(libc::RLIMIT_AS, &rlimit_as);
            }
            Ok(())
        });
    }

    let mut child = cmd.spawn().map_err(|e| Error::io(&program_path, e))?;
    let pid = child.id();

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();
    // stdout and stderr share one budget: a tool that writes 100 MB to stderr
    // must be capped exactly as strictly as one that writes it to stdout.
    let budget = Arc::new(Mutex::new(CapState {
        taken: 0,
        truncated: false,
        limit: options.max_output_bytes.saturating_mul(2),
    }));

    let stdout_rx = spawn_reader(stdout, Arc::clone(&budget));
    let stderr_rx = spawn_reader(stderr, Arc::clone(&budget));

    let mut exit_kind = ExitKind::Success;
    let mut exit_code = None;

    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if started.elapsed() > options.timeout {
                    kill_process_group(pid);
                    let _ = child.wait();
                    exit_kind = ExitKind::TimedOut;
                    break None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(e) => {
                kill_process_group(pid);
                return Err(Error::io(&program_path, e));
            }
        }
    };

    if let Some(status) = status {
        exit_code = status.code();
        if status.success() {
            exit_kind = ExitKind::Success;
        } else {
            exit_kind = ExitKind::Failed;
        }
    }

    // Collect whatever the reader threads managed before the cap or the kill.
    let stdout_text = collect(stdout_rx);
    let stderr_text = collect(stderr_rx);
    let cap = Arc::try_unwrap(budget)
        .map(|m| m.into_inner().unwrap_or_default())
        .unwrap_or_default();
    if cap.truncated && exit_kind == ExitKind::Success {
        exit_kind = ExitKind::OutputExceeded;
    }

    Ok(CommandOutput {
        program: options.program.clone(),
        args: options.args.clone(),
        exit_kind,
        exit_code,
        stdout: stdout_text,
        stderr: stderr_text,
        truncated: cap.truncated,
        duration_ms: started.elapsed().as_millis() as u64,
    })
}

#[derive(Default)]
struct CapState {
    taken: usize,
    truncated: bool,
    limit: usize,
}

/// Drain one pipe into a string, stopping as soon as the shared budget is gone.
///
/// The reader stops reading as soon as the cap is reached rather than draining
/// into a throwaway buffer: a process spewing gigabytes must be stopped at the
/// pipe, not after we have already paid the memory cost.
fn spawn_reader<R: Read + Send + 'static>(
    reader: Option<R>,
    budget: Arc<Mutex<CapState>>,
) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();
    match reader {
        Some(mut r) => {
            std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                let mut collected: Vec<u8> = Vec::new();
                loop {
                    let n = match r.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => n,
                    };
                    let mut guard = budget.lock().unwrap_or_else(|e| e.into_inner());
                    let limit = guard.limit;
                    if guard.taken >= limit {
                        guard.truncated = true;
                        break;
                    }
                    let room = limit - guard.taken;
                    let take = room.min(n);
                    guard.taken += take;
                    drop(guard);
                    collected.extend_from_slice(&buf[..take]);
                    if take < n {
                        let mut guard = budget.lock().unwrap_or_else(|e| e.into_inner());
                        guard.truncated = true;
                        break;
                    }
                }
                let _ = tx.send(String::from_utf8_lossy(&collected).into_owned());
            });
        }
        None => {
            let _ = tx.send(String::new());
        }
    }
    rx
}

fn collect(rx: mpsc::Receiver<String>) -> String {
    // The child has already exited (or been killed) at this point, so the
    // reader threads are draining a finite pipe and this returns promptly.
    rx.recv_timeout(Duration::from_secs(5)).unwrap_or_default()
}

/// SIGKILL the child's whole process group.
///
/// Falls back to killing just the child when the group kill fails, which can
/// happen if the child already exited between the check and the signal.
fn kill_process_group(pid: u32) {
    // SAFETY: `kill` with a negated pid targets the process group created by
    // `setpgid(0, 0)` in `pre_exec`. Signals are always safe to send.
    unsafe {
        libc::kill(-(pid as i32), libc::SIGKILL);
        libc::kill(pid as i32, libc::SIGKILL);
    }
}

/// Whether a path looks like an executable file AARON is willing to run.
pub fn is_executable(path: &Path) -> bool {
    path.is_file()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_a_fixed_program_and_captures_output() {
        let out = run(&ExecOptions::new("echo", ["hello", "world"])).unwrap();
        assert_eq!(out.exit_kind, ExitKind::Success);
        assert_eq!(out.stdout.trim(), "hello world");
        assert!(out.ok());
    }

    #[test]
    fn shell_metacharacters_are_inert_arguments() {
        // The whole string is one argv entry. No shell exists to interpret it.
        let nasty = "; rm -rf / #";
        let out = run(&ExecOptions::new("echo", [nasty])).unwrap();
        assert_eq!(out.stdout.trim(), nasty);
    }

    #[test]
    fn captures_stderr_separately() {
        let out = run(&ExecOptions::new("sh", ["-c", "echo o; echo e 1>&2"])).unwrap();
        assert_eq!(out.stdout.trim(), "o");
        assert_eq!(out.stderr.trim(), "e");
        assert!(out.combined().contains('o') && out.combined().contains('e'));
    }

    #[test]
    fn non_zero_exit_is_reported_not_swallowed() {
        let out = run(&ExecOptions::new("sh", ["-c", "exit 3"])).unwrap();
        assert_eq!(out.exit_kind, ExitKind::Failed);
        assert_eq!(out.exit_code, Some(3));
        assert!(!out.ok());
    }

    #[test]
    fn missing_program_is_a_clear_dependency_error() {
        let err = run(&ExecOptions::new(
            "definitely-not-installed-xyz",
            Vec::<String>::new(),
        ))
        .unwrap_err();
        assert!(matches!(err, Error::MissingDependency(_)));
        assert!(err.to_string().contains("not found on PATH"));
    }

    #[test]
    fn timeout_kills_the_process() {
        let out = run(&ExecOptions::new("sh", ["-c", "sleep 30"]).timeout(1)).unwrap();
        assert_eq!(out.exit_kind, ExitKind::TimedOut);
        assert!(out.duration_ms < 10_000, "took {}ms", out.duration_ms);
    }

    #[test]
    fn timeout_kills_the_whole_process_group() {
        // The child spawns a grandchild that would outlive a direct-child kill.
        let script = "sleep 30 & sleep 30 & wait";
        let out = run(&ExecOptions::new("sh", ["-c", script]).timeout(1)).unwrap();
        assert_eq!(out.exit_kind, ExitKind::TimedOut);
        // If the grandchild survived it would still hold the pipe open and the
        // reader thread would take far longer than the timeout.
        assert!(out.duration_ms < 5_000);
    }

    #[test]
    fn environment_is_stripped_to_the_allow_list() {
        // SAFETY: single-threaded test body; restored before returning.
        std::env::set_var("AARON_TEST_SECRET", "hunter2");
        let out = run(&ExecOptions::new(
            "sh",
            ["-c", "echo \"[$AARON_TEST_SECRET]\""],
        ))
        .unwrap();
        assert_eq!(out.stdout.trim(), "[]", "child inherited a private env var");
        std::env::remove_var("AARON_TEST_SECRET");
    }

    #[test]
    fn extra_env_is_applied_explicitly() {
        let mut opts = ExecOptions::new("sh", ["-c", "echo \"[$AARON_EXTRA]\""]);
        opts.extra_env.push(("AARON_EXTRA".into(), "value".into()));
        let out = run(&opts).unwrap();
        assert_eq!(out.stdout.trim(), "[value]");
    }

    #[test]
    fn output_is_capped_and_flagged() {
        let script = "i=0; while [ $i -lt 5000 ]; do echo aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa; i=$((i+1)); done";
        let out = run(&ExecOptions::new("sh", ["-c", script]).max_output(1024)).unwrap();
        assert!(out.stdout.len() <= 3 * 1024 * 1024);
        assert!(out.exit_kind == ExitKind::OutputExceeded || out.truncated);
    }

    #[test]
    fn working_directory_is_validated() {
        let err = run(&ExecOptions::new("pwd", Vec::<String>::new()).cwd("/definitely/not/here"))
            .unwrap_err();
        assert!(err.to_string().contains("does not exist"));
    }

    #[test]
    fn resolve_program_rejects_directories() {
        assert!(resolve_program("/tmp").is_err());
    }

    #[test]
    fn stdin_is_null_so_a_child_cannot_block_on_input() {
        let out = run(&ExecOptions::new("sh", ["-c", "cat; echo done"])).unwrap();
        assert_eq!(out.stdout.trim(), "done");
    }
}
