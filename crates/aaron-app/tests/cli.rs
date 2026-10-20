//! End-to-end CLI tests.
//!
//! These drive the real binary against a real temporary installation. They
//! assert on behaviour the user can see: exit codes, output text, and what
//! does *not* happen. Nothing here needs a microphone, a GPU or a model file —
//! that is the point of the text-only path existing at all.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Path to the compiled binary under test.
fn aaron() -> PathBuf {
    // `CARGO_BIN_EXE_<name>` is set by cargo for integration tests.
    PathBuf::from(env!("CARGO_BIN_EXE_aaron"))
}

/// A throwaway installation: config, permissions, data and audit paths.
struct Install {
    dir: tempfile::TempDir,
}

impl Install {
    fn new() -> Install {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("config.toml");
        std::fs::write(&cfg, default_config()).expect("write config");
        std::fs::write(
            dir.path().join("permissions.toml"),
            include_str!("../../../config/permissions.toml"),
        )
        .expect("write permissions");
        Install { dir }
    }

    fn path(&self) -> &Path {
        self.dir.path()
    }

    fn run(&self, args: &[&str]) -> Output {
        Command::new(aaron())
            .arg("--config")
            .arg(self.path().join("config.toml"))
            // Keep every run's state inside the throwaway directory.
            .env("AARON_HOME", self.path().join("state"))
            .env("XDG_STATE_HOME", self.path().join("state"))
            .env("HOME", self.path())
            .env("NO_COLOR", "1")
            .args(args)
            .output()
            .expect("aaron should be runnable")
    }
}

fn default_config() -> String {
    include_str!("../../../config/default.toml").to_string()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).to_string()
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).to_string()
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

#[test]
fn help_works_and_lists_the_real_commands() {
    let out = Command::new(aaron())
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(out.status.success());
    let text = stdout(&out);
    for expected in [
        "setup",
        "doctor",
        "run",
        "listen",
        "ask",
        "tools",
        "permissions",
        "memory",
        "learning",
        "brain",
        "model",
        "benchmark",
        "logs",
        "audio",
        "resources",
    ] {
        assert!(
            text.contains(expected),
            "`{expected}` missing from --help:\n{text}"
        );
    }
}

#[test]
fn doctor_runs_and_reports_honestly() {
    let i = Install::new();
    let out = i.run(&["doctor"]);
    // `doctor` reports problems but is not itself a failure.
    assert_eq!(
        code(&out),
        0,
        "doctor exited {}: {}",
        code(&out),
        stderr(&out)
    );
    let text = stdout(&out);
    assert!(text.contains("AARON"), "{text}");
    assert!(text.contains("hardware"));
    assert!(text.contains("policy"));
    assert!(text.contains("deny-by-default") || text.contains("denied"));
}

#[test]
fn ask_answers_system_status_without_any_model() {
    let i = Install::new();
    let out = i.run(&["--yes", "ask", "check system status"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("CPU"), "no CPU line in:\n{text}");
    assert!(text.contains("Memory"), "no memory line in:\n{text}");
    assert!(text.contains("GPU"), "no GPU line in:\n{text}");
}

#[test]
fn a_reply_is_printed_exactly_once() {
    let i = Install::new();
    let out = stdout(&i.run(&["ask", "check system status"]));
    let occurrences = out.matches("System status:").count();
    assert_eq!(
        occurrences, 1,
        "the reply was printed {occurrences} times; the agent's terminal echo and \
         the CLI are both writing:\n{out}"
    );
    // And the report must be complete: a scrubber that deletes AARON's own
    // output loses real data while looking like a security feature.
    for expected in ["System:", "CPU:", "Memory:", "GPU:", "Audio:", "Privilege:"] {
        assert!(out.contains(expected), "`{expected}` missing from:\n{out}");
    }
    assert!(
        !out.contains("looked like instructions"),
        "AARON scrubbed its own report:\n{out}"
    );
}

#[test]
fn a_rejected_tool_exits_with_the_permission_code() {
    let i = Install::new();
    let out = i.run(&["ask", "run the tests"]);
    assert_eq!(code(&out), 77, "{}", stdout(&out));
}

#[test]
fn ask_is_deterministic_without_a_checkpoint() {
    let i = Install::new();
    let first = stdout(&i.run(&["--yes", "ask", "check system status"]));
    let second = stdout(&i.run(&["--yes", "ask", "check system status"]));
    // The CPU line must be identical; the memory line legitimately varies.
    let cpu = |t: &str| {
        t.lines()
            .find(|l| l.starts_with("CPU"))
            .unwrap_or_default()
            .to_string()
    };
    assert_eq!(cpu(&first), cpu(&second));
}

#[test]
fn json_output_is_machine_readable() {
    let i = Install::new();
    let out = i.run(&["--json", "--yes", "ask", "check system status"]);
    let text = stdout(&out);
    let v: serde_json::Value = serde_json::from_str(text.trim())
        .unwrap_or_else(|e| panic!("output was not JSON ({e}):\n{text}"));
    assert_eq!(v["intent"], "system_status");
    assert_eq!(v["tool"], "system_status");
    assert_eq!(v["outcome"], "executed");
    assert!(v["confidence"].is_number());
    assert!(v["reply"].as_str().unwrap().contains("CPU"));
}

#[test]
fn greeting_and_help_do_not_run_any_tool() {
    let i = Install::new();
    let out = i.run(&["--json", "ask", "hello"]);
    let v: serde_json::Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(v["intent"], "greeting");
    assert!(v["tool"].is_null());

    let out = i.run(&["--json", "ask", "what can you do"]);
    let v: serde_json::Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(v["intent"], "help");
    assert!(v["tool"].is_null());
}

#[test]
fn a_denied_tool_exits_77_and_says_why() {
    let i = Install::new();
    // The shipped policy denies `run_tests`, which is not in permissions.toml.
    let out = i.run(&["ask", "run the tests"]);
    assert_eq!(
        code(&out),
        77,
        "expected the permission-denied exit code: {}",
        stdout(&out)
    );
    let text = stdout(&out);
    assert!(text.contains("I will not do that"), "{text}");
    assert!(text.contains("permissions.toml"), "{text}");
}

#[test]
fn filesystem_tools_are_refused_without_allowed_roots() {
    let i = Install::new();
    let out = i.run(&["--json", "--yes", "ask", "read the file /etc/hostname"]);
    let text = stdout(&out);
    let v: serde_json::Value =
        serde_json::from_str(text.trim()).unwrap_or_else(|e| panic!("not JSON: {e}\n{text}"));
    let outcome = v["outcome"].as_str().unwrap_or_default();
    assert!(
        matches!(outcome, "denied" | "declined" | "failed"),
        "an unrestricted read should not have succeeded: {v}"
    );
    assert!(!text.contains("localhost"), "file contents leaked: {text}");
}

#[test]
fn memory_round_trips_across_processes() {
    let i = Install::new();
    let add = i.run(&[
        "--yes",
        "memory",
        "add",
        "the deploy key is rotated quarterly",
    ]);
    assert_eq!(code(&add), 0, "{}", stderr(&add));

    // A completely separate process must see it: this is the persistence claim.
    let listed = i.run(&["memory", "list"]);
    assert!(
        stdout(&listed).contains("rotated quarterly"),
        "{}",
        stdout(&listed)
    );

    let searched = i.run(&["memory", "search", "rotated"]);
    assert!(stdout(&searched).contains("rotated quarterly"));
}

#[test]
fn memory_delete_refuses_without_confirmation() {
    let i = Install::new();
    i.run(&["--yes", "memory", "add", "delete me"]);
    let listed = stdout(&i.run(&["memory", "list"]));
    let id: String = listed
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().next())
        .map(|s| s.trim_start_matches('#').to_string())
        .expect("an id should be listed");
    assert!(
        id.chars().all(|c| c.is_ascii_digit()),
        "unexpected id `{id}`"
    );

    let refused = i.run(&["memory", "delete", &id]);
    assert_eq!(code(&refused), 1, "delete should need --yes");
    assert!(stdout(&refused).contains("--yes"));

    let deleted = i.run(&["--yes", "memory", "delete", &id]);
    assert_eq!(code(&deleted), 0, "{}", stderr(&deleted));
    assert!(!stdout(&i.run(&["memory", "list"])).contains("delete me"));
}

#[test]
fn tools_list_shows_the_policy_verdict_for_every_tool() {
    let i = Install::new();
    let out = i.run(&["tools", "list"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = stdout(&out);
    for tool in [
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
        assert!(text.contains(tool), "tool `{tool}` missing from tools list");
    }
    assert!(
        text.contains("DENY"),
        "run_tests is not in the shipped policy and must be denied"
    );
    assert!(text.contains("ALLOW"), "system_status is allowed");
}

#[test]
fn permissions_show_reports_the_deny_by_default_posture() {
    let i = Install::new();
    let out = i.run(&["permissions", "show"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("deny"), "{text}");
    assert!(
        text.contains("registered but with no policy entry"),
        "{text}"
    );
}

#[test]
fn permissions_test_explains_a_single_tool() {
    let i = Install::new();
    let out = i.run(&["permissions", "test", "system_status"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(stdout(&out).contains("verdict         ALLOW"));

    let denied = i.run(&["permissions", "test", "run_tests"]);
    assert!(stdout(&denied).contains("verdict         DENY"));

    let unknown = i.run(&["permissions", "test", "definitely-not-a-tool"]);
    assert_ne!(code(&unknown), 0);
}

#[test]
fn model_info_reports_the_parameter_count_and_separates_other_models() {
    let i = Install::new();
    let out = i.run(&["model"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("trainable params   1000000"), "{text}");
    assert!(
        text.contains("NOT part of the 1,000,000 budget"),
        "the report must separate AARON's model from optional speech models:\n{text}"
    );
    assert!(text.contains("never downloads"), "{text}");
}

#[test]
fn resources_status_and_limits_work() {
    let i = Install::new();
    let status = i.run(&["resources", "status"]);
    assert_eq!(code(&status), 0, "{}", stderr(&status));
    assert!(stdout(&status).contains("system memory"));

    let limits = i.run(&["resources", "limits"]);
    assert_eq!(code(&limits), 0, "{}", stderr(&limits));
    let text = stdout(&limits);
    assert!(text.contains("reserve_system_ram_mb"));
    assert!(text.contains("low_memory_recover_percent"));
    assert!(text.contains("sqlite_cache_kib"));
}

#[test]
fn resources_status_is_machine_readable() {
    let i = Install::new();
    let out = i.run(&["--json", "resources", "status"]);
    let text = stdout(&out);
    let v: serde_json::Value =
        serde_json::from_str(text.trim()).unwrap_or_else(|e| panic!("not JSON: {e}\n{text}"));
    assert!(v["mode"].is_string());
    assert!(v["budget"]["trigger_percent"].is_number());
    // Recovery must be strictly above the trigger; otherwise the config is wrong.
    assert!(
        v["budget"]["recover_percent"].as_u64().unwrap()
            > v["budget"]["trigger_percent"].as_u64().unwrap()
    );
    assert!(v["process_rss_bytes"].as_u64().unwrap() > 0);
}

#[test]
fn a_missing_checkpoint_says_what_to_do_not_just_that_it_is_missing() {
    let i = Install::new();
    let out = i.run(&["resources", "checkpoint"]);
    let text = stderr(&out);
    assert!(
        text.contains("aaron brain train"),
        "the error should say how to fix it: {text}"
    );
    assert!(
        !text.contains("No such file or directory"),
        "a raw i/o error tells the user nothing: {text}"
    );

    let explicit = i.run(&[
        "resources",
        "checkpoint",
        "--path",
        "/tmp/aaron-not-here.ckpt",
    ]);
    assert!(
        stderr(&explicit).contains("/tmp/aaron-not-here.ckpt"),
        "an explicit path should be named in the error: {}",
        stderr(&explicit)
    );
}

#[test]
fn setup_refuses_to_overwrite_and_does_not_redirect_you_to_itself() {
    let i = Install::new();
    let out = i.run(&["setup"]);
    assert_ne!(code(&out), 0);
    let text = stderr(&out);
    assert!(text.contains("--force"), "{text}");
    assert!(
        !text.contains("run `aaron setup`"),
        "the hint points at the command the user just ran: {text}"
    );
    // And the file is genuinely untouched.
    assert!(i.path().join("config.toml").is_file());
}

#[test]
fn logs_are_written_for_executed_actions() {
    let i = Install::new();
    i.run(&["--yes", "ask", "check system status"]);
    let out = i.run(&["logs", "--limit", "50"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("system_status"), "no audit records:\n{text}");
    assert!(
        text.contains("requested") && text.contains("authorized"),
        "{text}"
    );
}

#[test]
fn the_audit_log_can_be_turned_off() {
    let i = Install::new();
    i.run(&["--no-audit", "--yes", "ask", "check system status"]);
    let out = i.run(&["--no-audit", "logs"]);
    assert!(stdout(&out).contains("No audit log"), "{}", stdout(&out));
}

#[test]
fn a_malformed_config_is_a_clear_error_not_a_crash() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    std::fs::write(&cfg, "[assistant]\nnot_a_real_field = 1\n").unwrap();
    let out = Command::new(aaron())
        .arg("--config")
        .arg(&cfg)
        .arg("doctor")
        .output()
        .unwrap();
    assert_ne!(code(&out), 0);
    assert!(
        stderr(&out).contains("configuration error"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn an_unsafe_resource_budget_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    let base = include_str!("../../../config/default.toml").to_string();
    // Zero pending requests would disable request admission entirely.
    let broken = base.replace("max_pending_requests = 4", "max_pending_requests = 0");
    std::fs::write(&cfg, broken).unwrap();
    let out = Command::new(aaron())
        .arg("--config")
        .arg(&cfg)
        .arg("doctor")
        .output()
        .unwrap();
    assert_ne!(code(&out), 0);
    assert!(
        stderr(&out).contains("max_pending_requests"),
        "{}",
        stderr(&out)
    );
}

#[test]
fn broken_hysteresis_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("config.toml");
    let base = include_str!("../../../config/default.toml").to_string();
    let broken = base
        .replace(
            "low_memory_free_percent = 12",
            "low_memory_free_percent = 25",
        )
        .replace(
            "low_memory_recover_percent = 20",
            "low_memory_recover_percent = 25",
        );
    std::fs::write(&cfg, broken).unwrap();
    let out = Command::new(aaron())
        .arg("--config")
        .arg(&cfg)
        .arg("doctor")
        .output()
        .unwrap();
    assert_ne!(code(&out), 0);
    assert!(stderr(&out).contains("hysteresis"), "{}", stderr(&out));
}

#[test]
fn cancel_is_always_available() {
    let i = Install::new();
    let out = i.run(&["ask", "cancel"]);
    assert_eq!(code(&out), 0);
    assert!(stdout(&out).contains("Cancelled"));
}

#[test]
fn a_nonsense_utterance_does_not_run_anything() {
    let i = Install::new();
    let out = i.run(&[
        "--json",
        "ask",
        "the mitochondria is the powerhouse of the cell",
    ]);
    let v: serde_json::Value = serde_json::from_str(stdout(&out).trim()).unwrap();
    assert_eq!(v["intent"], "unknown");
    assert!(v["tool"].is_null());
    assert!(v["reply"].as_str().unwrap().contains("did not understand"));
}

#[test]
fn learning_starts_with_an_empty_queue() {
    let i = Install::new();
    let out = i.run(&["learning", "pending"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(stdout(&out).contains("No corrections waiting"));
}

#[test]
fn setup_writes_a_usable_installation() {
    let dir = tempfile::tempdir().unwrap();
    let cfg = dir.path().join("sub").join("config.toml");
    let out = Command::new(aaron())
        .arg("--config")
        .arg(&cfg)
        .arg("setup")
        .env("HOME", dir.path())
        .env("XDG_STATE_HOME", dir.path().join("state"))
        .output()
        .unwrap();
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(cfg.is_file(), "setup did not write the config");
    let perms = cfg.parent().unwrap().join("permissions.toml");
    assert!(perms.is_file(), "setup did not write permissions");

    // And the result must actually load.
    let doctor = Command::new(aaron())
        .arg("--config")
        .arg(&cfg)
        .arg("doctor")
        .env("HOME", dir.path())
        .env("XDG_STATE_HOME", dir.path().join("state"))
        .output()
        .unwrap();
    assert_eq!(code(&doctor), 0, "{}", stderr(&doctor));
}

#[test]
fn the_exit_codes_do_not_collide() {
    // A script has to be able to tell the outcomes apart. "your
    // configuration is broken" and "AARON asked you a question and ran
    // nothing" were both 78, which is indistinguishable.
    let config = aaron_core::config::Config::default();
    assert_eq!(
        aaron_core::error::ErrorKind::Config.exit_code(),
        78,
        "a configuration error is EX_CONFIG"
    );
    assert_eq!(
        aaron_core::error::ErrorKind::PermissionDenied.exit_code(),
        77,
        "a refusal is EX_NOPERM"
    );
    assert_eq!(
        aaron_core::error::ErrorKind::Cancelled.exit_code(),
        130,
        "a cancellation is 128 + SIGINT"
    );
    assert_eq!(
        aaron_core::error::ErrorKind::MissingDependency.exit_code(),
        69,
        "a missing dependency is EX_UNAVAILABLE"
    );
    let _ = config;
}

#[test]
fn the_shipped_configuration_is_machine_independent() {
    // The shipped configuration is used by CI, by the benchmark script, by
    // the installer and by anyone who clones the repository. It must not
    // name a path from the machine it was written on: a configured root that
    // does not exist is a hard configuration error, so one stale absolute
    // path stops AARON from starting at all.
    //
    // It used to name `/home/david/Documents`, and every machine but that
    // one failed to start.
    let shipped = include_str!("../../../config/default.toml");
    for line in shipped.lines() {
        let line = line.trim();
        if !line.starts_with("allowed_roots") {
            continue;
        }
        let value = line
            .split_once('=')
            .map(|(_, v)| v.trim())
            .unwrap_or_default();
        assert!(
            !value.contains("/home/") && !value.contains("/Users/") && !value.contains('~'),
            "the shipped configuration names a path from one machine: {line:?}"
        );
        assert!(
            !value.contains('"'),
            "the shipped configuration must not name an absolute path: {line:?}"
        );
    }
}
