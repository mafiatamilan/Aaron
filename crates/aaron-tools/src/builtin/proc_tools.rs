//! Process-backed tools: `git_status`, `run_tests`, `open_url`,
//! `start_application` and `stop_application`.
//!
//! Every one of them resolves a *name* to a fixed program and a fixed argument
//! vector that came from the user's configuration file. None of them accepts a
//! command string, and none of them builds an argument by concatenating
//! model- or user-supplied text with a program path.

use std::time::Duration;

use aaron_core::config::{DEFAULT_STOP_SIGNAL, MAX_STOP_SIGNAL};
use aaron_core::error::{Error, Result};
use aaron_security::{PermissionCategory, RiskLevel};
use serde_json::json;

use crate::proc::{run, ExecOptions, ExitKind};
use crate::registry::{Tool, ToolContext, ToolOutput};
use crate::spec::{ArgSpec, ToolArgs, ToolSpec};

pub const GIT_SPEC: ToolSpec = ToolSpec {
    id: "git_status",
    description: "Show the working-tree status of a git repository",
    category: PermissionCategory::SystemRead,
    risk: RiskLevel::Low,
    timeout_secs: 15,
    max_output_bytes: 16 * 1024,
    requires_confirmation: false,
    args: &[ArgSpec::path(
        "repository",
        false,
        4096,
        "repository directory; defaults to the project root",
    )],
};

pub const TESTS_SPEC: ToolSpec = ToolSpec {
    id: "run_tests",
    description: "Run a registered test suite (a fixed command, never a shell string)",
    category: PermissionCategory::ProcessExec,
    risk: RiskLevel::Medium,
    timeout_secs: 300,
    max_output_bytes: 32 * 1024,
    requires_confirmation: true,
    args: &[
        ArgSpec::text("suite", false, 64, "registered suite name, e.g. `cargo`"),
        ArgSpec::path(
            "cwd",
            false,
            4096,
            "directory to run in; must be inside an allowed root",
        ),
    ],
};

pub const OPEN_URL_SPEC: ToolSpec = ToolSpec {
    id: "open_url",
    description: "Open an http(s) URL in the desktop browser",
    category: PermissionCategory::Network,
    risk: RiskLevel::Medium,
    timeout_secs: 15,
    max_output_bytes: 4096,
    requires_confirmation: true,
    args: &[ArgSpec::url("url", true, 2048, "http:// or https:// URL")],
};

pub const START_APP_SPEC: ToolSpec = ToolSpec {
    id: "start_application",
    description: "Launch an application from the configured allow-list",
    category: PermissionCategory::ProcessExec,
    risk: RiskLevel::Medium,
    timeout_secs: 20,
    max_output_bytes: 4096,
    requires_confirmation: true,
    args: &[
        ArgSpec::text("application", true, 64, "registered application name"),
        ArgSpec::text(
            "url",
            false,
            2048,
            "optional http(s) URL passed to a browser-style application",
        ),
    ],
};

pub const STOP_APP_SPEC: ToolSpec = ToolSpec {
    id: "stop_application",
    description: "Stop an application from the configured allow-list",
    category: PermissionCategory::ProcessExec,
    risk: RiskLevel::High,
    timeout_secs: 15,
    max_output_bytes: 4096,
    requires_confirmation: true,
    args: &[ArgSpec::text(
        "application",
        true,
        64,
        "registered application name",
    )],
};

pub const CANCEL_SPEC: ToolSpec = ToolSpec {
    id: "cancel",
    description: "Abandon the operation currently in flight",
    category: PermissionCategory::SystemRead,
    risk: RiskLevel::Low,
    timeout_secs: 1,
    max_output_bytes: 1024,
    requires_confirmation: false,
    args: &[],
};

/// Resolve a working directory: it must exist and be inside a configured root
/// (or be the project root, which `setup` adds to the allow-list).
fn resolve_cwd(
    ctx: &ToolContext,
    raw: Option<&str>,
    fallback: &std::path::Path,
) -> Result<std::path::PathBuf> {
    match raw {
        Some(p) => ctx.sandbox.resolve_search_root(Some(p)),
        None => {
            let canonical = aaron_security::canonicalize_lenient(fallback)
                .ok_or_else(|| Error::Other(format!("`{}` does not exist", fallback.display())))?;
            if ctx.sandbox.contains(&canonical) {
                Ok(canonical)
            } else {
                Err(Error::PermissionDenied(format!(
                    "`{}` is not inside an allowed root; add it to security.allowed_roots",
                    canonical.display()
                )))
            }
        }
    }
}

fn summarise(out: &crate::proc::CommandOutput, what: &str) -> String {
    let mut text = format!("{what} ({:?}, {} ms)", out.exit_kind, out.duration_ms);
    if out.truncated {
        text.push_str("\n\n[output truncated at the configured limit]");
    }
    let body = out.combined();
    let body = body.trim();
    if !body.is_empty() {
        text.push_str("\n\n");
        text.push_str(body);
    }
    text
}

#[derive(Debug)]
pub struct GitStatusTool;

impl Tool for GitStatusTool {
    fn spec(&self) -> &'static ToolSpec {
        &GIT_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let cwd = resolve_cwd(ctx, args.get("repository"), &ctx.project_root)?;
        let opts = ExecOptions::new(
            ctx.config.tools.git_binary.clone(),
            [
                "status",
                "--porcelain=v1",
                "--branch",
                "--untracked-files=normal",
            ],
        )
        .cwd(cwd)
        .timeout(GIT_SPEC.timeout_secs);

        let out = run(&opts)?;
        let text = if out.stdout.trim().is_empty() && out.ok() {
            "Working tree is clean.".to_string()
        } else {
            summarise(&out, "git status")
        };
        Ok(ToolOutput::text(text)
            .with_data(json!({
                "exit_kind": out.exit_kind,
                "exit_code": out.exit_code,
                "duration_ms": out.duration_ms,
            }))
            .capped(GIT_SPEC.max_output_bytes))
    }
}

#[derive(Debug)]
pub struct RunTestsTool;

impl Tool for RunTestsTool {
    fn spec(&self) -> &'static ToolSpec {
        &TESTS_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let suites = &ctx.config.tools.test_suites;
        if suites.is_empty() {
            return Err(Error::Other(
                "no test suites are registered; add them under [tools.test_suites]".into(),
            ));
        }
        // Default to the single suite when the user did not name one, so
        // "run the tests" works on a fresh install.
        let name = args
            .get("suite")
            .map(|s| s.to_string())
            .unwrap_or_else(|| suites.keys().next().cloned().unwrap_or_default());

        let suite = suites.get(&name).ok_or_else(|| {
            Error::PermissionDenied(format!(
                "`{name}` is not a registered test suite. Available: {}",
                suites.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;

        let base = resolve_cwd(ctx, args.get("cwd"), &ctx.project_root)?;
        let cwd = match &suite.cwd {
            Some(rel) => {
                let joined = base.join(rel);
                let canonical = aaron_security::canonicalize_lenient(&joined).ok_or_else(|| {
                    Error::Other(format!("suite cwd `{}` does not exist", joined.display()))
                })?;
                if !ctx.sandbox.contains(&canonical) {
                    return Err(Error::PathRejected(format!(
                        "suite cwd `{}` is outside every allowed root",
                        canonical.display()
                    )));
                }
                canonical
            }
            None => base,
        };

        let opts = ExecOptions::new(suite.program.clone(), suite.args.clone())
            .cwd(cwd)
            .timeout(TESTS_SPEC.timeout_secs)
            .max_output(TESTS_SPEC.max_output_bytes);

        let out = run(&opts)?;
        let verdict = match out.exit_kind {
            ExitKind::Success => "passed",
            ExitKind::Failed => "failed",
            ExitKind::TimedOut => "timed out",
            ExitKind::OutputExceeded => "produced too much output",
        };
        Ok(
            ToolOutput::text(summarise(&out, &format!("test suite `{name}` {verdict}")))
                .with_data(json!({
                    "suite": name,
                    "exit_kind": out.exit_kind,
                    "exit_code": out.exit_code,
                    "duration_ms": out.duration_ms,
                }))
                .capped(TESTS_SPEC.max_output_bytes),
        )
    }
}

#[derive(Debug)]
pub struct OpenUrlTool;

impl Tool for OpenUrlTool {
    fn spec(&self) -> &'static ToolSpec {
        &OPEN_URL_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let url = args
            .get("url")
            .ok_or_else(|| Error::Other("open_url requires a `url`".into()))?;
        let scheme = url
            .split_once("://")
            .map(|(s, _)| s.to_string())
            .unwrap_or_default();
        if !ctx.config.tools.allowed_url_schemes.contains(&scheme) {
            return Err(Error::PermissionDenied(format!(
                "scheme `{scheme}` is not allowed. Allowed: {}",
                ctx.config.tools.allowed_url_schemes.join(", ")
            )));
        }
        // Defensive re-validation: the URL arrived from a model prediction, so
        // it is checked again here rather than trusted.
        crate::spec::ToolArgs::validate(&OPEN_URL_SPEC, &json!({"url": url}))?;

        let opts = ExecOptions::new(ctx.config.tools.url_opener.clone(), [url])
            .timeout(OPEN_URL_SPEC.timeout_secs);
        let out = run(&opts)?;
        if out.ok() {
            Ok(
                ToolOutput::text(format!("Opened {url} in the default browser."))
                    .with_data(json!({"url": url})),
            )
        } else {
            Ok(ToolOutput::text(format!(
                "The URL opener exited with {:?}: {}",
                out.exit_kind,
                out.stderr.trim()
            )))
        }
    }
}

#[derive(Debug)]
pub struct StartApplicationTool;

impl Tool for StartApplicationTool {
    fn spec(&self) -> &'static ToolSpec {
        &START_APP_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let name = args
            .get("application")
            .ok_or_else(|| Error::Other("start_application requires an `application`".into()))?;
        let apps = &ctx.config.tools.applications;
        let app = apps.get(name).ok_or_else(|| {
            Error::PermissionDenied(format!(
                "`{name}` is not a registered application. Available: {}",
                apps.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;

        let mut argv: Vec<String> = app.args.clone();
        // A browser-style application may take a URL, and only a URL: it is
        // re-validated against the same rules as `open_url`.
        if let Some(url) = args.get("url") {
            crate::spec::ToolArgs::validate(&OPEN_URL_SPEC, &json!({"url": url}))?;
            argv.push(url.to_string());
        }

        let opts = ExecOptions::new(app.program.clone(), argv).timeout(START_APP_SPEC.timeout_secs);
        let out = run(&opts)?;

        // Desktop applications detach; a short-lived failure is still an error.
        Ok(
            ToolOutput::text(summarise(&out, &format!("start `{name}`")))
                .with_data(json!({"application": name, "exit_kind": out.exit_kind}))
                .capped(START_APP_SPEC.max_output_bytes),
        )
    }
}

/// Resolve the signal used to stop an application.
///
/// The value comes from the configuration file only — never from speech and
/// never from the model — and it is validated before it can reach the
/// command line. `None` means SIGTERM.
fn resolve_stop_signal(stop_signal: Option<i32>, program: &str) -> Result<i32> {
    match stop_signal {
        Some(signal) => {
            aaron_core::config::validate_stop_signal(signal).map_err(|_| {
                Error::Config(format!(
                    "invalid stop_signal {signal} for `{program}`; use 1..={MAX_STOP_SIGNAL}"
                ))
            })?;
            Ok(signal)
        }
        None => Ok(DEFAULT_STOP_SIGNAL),
    }
}

/// Build the fixed `pkill` argument vector that stops `program` with `signal`.
///
/// The signal travels as its own argv element, so there is no shell to
/// reinterpret it and no way for it to become an option.
///
/// `pkill -x` matches the exact executable name, so a partial name can never
/// take down an unrelated process.
fn stop_argv(program: &str, signal: i32) -> Vec<String> {
    let name = std::path::Path::new(program)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| program.to_string());
    vec![
        "--signal".to_string(),
        signal.to_string(),
        "-x".to_string(),
        name,
    ]
}

#[derive(Debug)]
pub struct StopApplicationTool;

impl Tool for StopApplicationTool {
    fn spec(&self) -> &'static ToolSpec {
        &STOP_APP_SPEC
    }

    fn execute(&self, ctx: &ToolContext, args: &ToolArgs) -> Result<ToolOutput> {
        let name = args
            .get("application")
            .ok_or_else(|| Error::Other("stop_application requires an `application`".into()))?;
        let apps = &ctx.config.tools.applications;
        let app = apps.get(name).ok_or_else(|| {
            Error::PermissionDenied(format!(
                "`{name}` is not a registered application. Available: {}",
                apps.keys().cloned().collect::<Vec<_>>().join(", ")
            ))
        })?;

        // `pkill -x <program>` matches the exact executable name, so a partial
        // name can never take down an unrelated process. The signal is the one
        // the configuration asked for, or SIGTERM when it did not ask.
        let signal = resolve_stop_signal(app.stop_signal, &app.program)?;
        let argv = stop_argv(&app.program, signal);

        let opts = ExecOptions::new("pkill", argv).timeout(STOP_APP_SPEC.timeout_secs);
        match run(&opts) {
            Ok(out) if out.ok() => Ok(ToolOutput::text(format!("Stopped `{name}`.")).with_data(
                json!({
                    "application": name,
                    "signal": signal,
                    "exit_kind": out.exit_kind,
                }),
            )),
            Ok(out) if out.exit_code == Some(1) => Ok(ToolOutput::text(format!(
                "`{name}` was not running (pkill exit 1 means no process matched)."
            ))
            .with_data(json!({
                "application": name,
                "signal": signal,
                "exit_kind": out.exit_kind,
            }))),
            Ok(out) => Ok(ToolOutput::text(summarise(
                &out,
                &format!("could not stop `{name}`"),
            ))),
            Err(Error::MissingDependency(_)) => Err(Error::missing_with_hint(
                "pkill (procps-ng)",
                "install the `procps-ng` package to enable stop_application",
            )),
            Err(e) => Err(e),
        }
    }
}

#[derive(Debug)]
pub struct CancelTool;

impl Tool for CancelTool {
    fn spec(&self) -> &'static ToolSpec {
        &CANCEL_SPEC
    }

    fn execute(&self, _ctx: &ToolContext, _args: &ToolArgs) -> Result<ToolOutput> {
        Ok(ToolOutput::text("Cancelled. Nothing else will run."))
    }
}

/// How long a confirmable action may wait for the user.
pub const CONFIRM_DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::ToolContext;
    use aaron_memory::MemoryStore;
    use aaron_security::PathSandbox;
    use std::fs;
    use std::sync::{Arc, Mutex};

    fn ctx(root: &std::path::Path, cfg: aaron_core::Config) -> Arc<ToolContext> {
        Arc::new(ToolContext {
            sandbox: Arc::new(PathSandbox::new([root]).unwrap()),
            memory: Arc::new(Mutex::new(
                MemoryStore::open_in_memory(Default::default()).unwrap(),
            )),
            config: Arc::new(cfg),
            project_root: root.to_path_buf(),
        })
    }

    fn args(spec: &ToolSpec, pairs: &[(&str, &str)]) -> ToolArgs {
        ToolArgs::from_pairs(spec, pairs).unwrap()
    }

    #[test]
    fn run_tests_refuses_an_unregistered_suite() {
        let tmp = tempfile::tempdir().unwrap();
        let c = ctx(tmp.path(), aaron_core::Config::default());
        let err = RunTestsTool
            .execute(&c, &args(&TESTS_SPEC, &[("suite", "rm -rf /")]))
            .unwrap_err();
        assert!(matches!(err, Error::PermissionDenied(_)));
        assert!(err.to_string().contains("cargo"));
    }

    #[test]
    fn run_tests_uses_the_fixed_argument_vector() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = aaron_core::Config::default();
        cfg.tools.test_suites.insert(
            "list".to_string(),
            aaron_core::config::TestSuiteSpec {
                program: "echo".into(),
                args: vec!["fixed-arg".into()],
                cwd: None,
            },
        );
        let c = ctx(tmp.path(), cfg);
        let out = RunTestsTool
            .execute(&c, &args(&TESTS_SPEC, &[("suite", "list")]))
            .unwrap();
        assert!(out.text.contains("fixed-arg"));
    }

    #[test]
    fn run_tests_refuses_a_cwd_outside_the_roots() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path().join("root");
        fs::create_dir_all(&root).unwrap();
        let c = ctx(&root, aaron_core::Config::default());
        let err = RunTestsTool
            .execute(&c, &args(&TESTS_SPEC, &[("suite", "cargo"), ("cwd", "..")]))
            .unwrap_err();
        assert!(matches!(
            err,
            Error::PathRejected(_) | Error::PermissionDenied(_)
        ));
    }

    #[test]
    fn open_url_rejects_a_disallowed_scheme() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = aaron_core::Config::default();
        cfg.tools.allowed_url_schemes = vec!["https".into()];
        let c = ctx(tmp.path(), cfg);
        // The spec validator already blocks this, so we exercise the tool with a
        // pre-built arg set to reach the tool's own check as well.
        let direct =
            ToolArgs::from_validated([("url".to_string(), "http://evil.example".to_string())]);
        let err = OpenUrlTool.execute(&c, &direct).unwrap_err();
        assert!(matches!(err, Error::PermissionDenied(_)));
    }

    #[test]
    fn start_application_refuses_an_unregistered_name() {
        let tmp = tempfile::tempdir().unwrap();
        let c = ctx(tmp.path(), aaron_core::Config::default());
        let err = StartApplicationTool
            .execute(
                &c,
                &args(
                    &START_APP_SPEC,
                    &[("application", "definitely-not-registered")],
                ),
            )
            .unwrap_err();
        assert!(matches!(err, Error::PermissionDenied(_)));
    }

    #[test]
    fn start_application_rejects_a_url_argument_that_is_not_http() {
        let tmp = tempfile::tempdir().unwrap();
        let c = ctx(tmp.path(), aaron_core::Config::default());
        let err = StartApplicationTool
            .execute(
                &c,
                &args(
                    &START_APP_SPEC,
                    &[("application", "browser"), ("url", "file:///etc/passwd")],
                ),
            )
            .unwrap_err();
        assert!(err.to_string().contains("http"));
    }

    #[test]
    fn stop_application_refuses_an_unregistered_name() {
        let tmp = tempfile::tempdir().unwrap();
        let c = ctx(tmp.path(), aaron_core::Config::default());
        let err = StopApplicationTool
            .execute(&c, &args(&STOP_APP_SPEC, &[("application", "rm")]))
            .unwrap_err();
        assert!(matches!(err, Error::PermissionDenied(_)));
    }

    #[test]
    fn stop_application_defaults_to_sigterm() {
        let signal = resolve_stop_signal(None, "xdg-open").unwrap();
        assert_eq!(signal, 15);
        assert_eq!(
            stop_argv("xdg-open", signal),
            vec!["--signal", "15", "-x", "xdg-open"]
        );
    }

    #[test]
    fn stop_application_uses_the_configured_stop_signal() {
        // The executable name is reduced to its basename, and the configured
        // signal travels as its own argv element.
        let signal = resolve_stop_signal(Some(9), "/usr/bin/firefox").unwrap();
        assert_eq!(signal, 9);
        assert_eq!(
            stop_argv("/usr/bin/firefox", signal),
            vec!["--signal", "9", "-x", "firefox"]
        );
    }

    #[test]
    fn stop_application_rejects_an_out_of_range_stop_signal() {
        for bad in [0, -1, 65, 999] {
            let err = resolve_stop_signal(Some(bad), "xdg-open").unwrap_err();
            assert!(matches!(err, Error::Config(_)), "{bad} should be rejected");
            assert!(err.to_string().contains("stop_signal"));
        }
    }

    #[test]
    fn stop_application_refuses_an_invalid_configured_signal_before_running() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = aaron_core::Config::default();
        cfg.tools.applications.insert(
            "broken".to_string(),
            aaron_core::config::ApplicationSpec {
                program: "xdg-open".into(),
                args: Vec::new(),
                stop_signal: Some(0),
            },
        );
        let c = ctx(tmp.path(), cfg);
        let err = StopApplicationTool
            .execute(&c, &args(&STOP_APP_SPEC, &[("application", "broken")]))
            .unwrap_err();
        assert!(matches!(err, Error::Config(_)));
        assert!(err.to_string().contains("stop_signal"));
    }

    #[test]
    fn stop_application_reports_the_signal_it_used() {
        let tmp = tempfile::tempdir().unwrap();
        let mut cfg = aaron_core::Config::default();
        // A name no process can have: the kernel truncates `comm` to 15
        // characters, so `pkill -x` can never match it and nothing is killed.
        cfg.tools.applications.insert(
            "aaron-test-nonexistent".to_string(),
            aaron_core::config::ApplicationSpec {
                program: "aaron-test-nonexistent".into(),
                args: Vec::new(),
                stop_signal: Some(9),
            },
        );
        let c = ctx(tmp.path(), cfg);
        match StopApplicationTool.execute(
            &c,
            &args(&STOP_APP_SPEC, &[("application", "aaron-test-nonexistent")]),
        ) {
            Ok(out) => {
                let data = out.data.expect("the payload carries the signal");
                assert_eq!(data["signal"], json!(9));
            }
            // No `pkill` on this machine: there is no payload to inspect.
            Err(Error::MissingDependency(_)) => {}
            Err(e) => panic!("unexpected error: {e}"),
        }
    }

    #[test]
    fn git_status_outside_a_repository_reports_honestly() {
        let tmp = tempfile::tempdir().unwrap();
        let c = ctx(tmp.path(), aaron_core::Config::default());
        let out = GitStatusTool.execute(&c, &ToolArgs::default()).unwrap();
        // Either git is missing (a clear message) or it says "not a repository".
        // Either way the tool must not pretend it succeeded.
        let lower = out.text.to_lowercase();
        assert!(
            lower.contains("not a git repository")
                || lower.contains("failed")
                || lower.contains("missing dependency"),
            "unexpected git_status output: {out:?}"
        );
    }

    #[test]
    fn cancel_is_a_no_op_with_a_truthful_message() {
        let out = CancelTool
            .execute(&ToolContext::for_tests("/tmp"), &ToolArgs::default())
            .unwrap();
        assert_eq!(out.text, "Cancelled. Nothing else will run.");
    }
}
