# Permissions

`config/permissions.toml` decides what AARON may do. This document explains each
setting and the reasoning behind the shipped defaults.

The rule that matters: **AARON is deny-by-default.** A registered tool with no
entry in this file cannot run.

```console
$ aaron permissions show
permissions.toml schema v1
unknown tools    : deny (deny-by-default)
network allowed  : false
refuse root      : true
always confirm   : open_url, start_application, stop_application, run_tests, read_file
output cap       : 65536 bytes
runtime cap      : 30 s

TOOL                EFFECT    TIMEOUT  OUTPUT  NOTE
system_status       allow     8s       8192    reads /proc and optionally nvidia-smi
search_files        allow     15s      16384   walks the configured roots only
git_status          allow     15s      16384   runs `git status` with a fixed argv
recall              allow     5s       8192    reads your own memory
cancel              allow     -        -       no side effects
read_file           ask       10s      65536   shows the exact path before reading
```

---

## `effect`

| value | meaning |
|---|---|
| `allow` | run without asking |
| `ask` | run only after the user confirms, with the exact arguments shown |
| `deny` | never run |

`deny` on a tool the user has removed from their config is how you permanently
disable a capability without deleting code.

## `timeout_secs` and `max_output_bytes`

Per-tool ceilings. They can only **lower** the global cap, never raise it:

```toml
[limits]
max_runtime_secs = 30      # global
max_output_bytes = 65536   # global

[tools.run_tests]
effect = "allow"
timeout_secs = 9999        # clamped to 30
max_output_bytes = 999999  # clamped to 65536
```

`policy_cannot_loosen_the_global_limits` proves this.

## `note`

Free text, shown by `aaron permissions show`. Use it to record *why* a decision
was made. A policy file with notes is one you will still understand in six
months.

---

## Global limits

```toml
[limits]
max_runtime_secs = 30
max_output_bytes = 65536
max_audit_value_chars = 160
refuse_root = true
allow_network = false
```

* **`refuse_root`** — AARON refuses to execute tools as uid 0. Turning it off
  is allowed, but it is not something you should do.
* **`allow_network`** — the master switch for the `network` category. With it
  false, `open_url` is refused regardless of its own entry.
* **`max_audit_value_chars`** — audit values are truncated to this before
  being written, so a log you share cannot carry a page of file contents.

---

## Categories that are permanently denied

`SystemChange` is refused no matter what this file says. It exists in the
vocabulary so that a future tool cannot pick an "unlisted means fine" category,
and so that a typo in a tool id cannot accidentally land on something permitted.

AARON ships no tool in that category. Adding one — installing a package,
editing a firewall rule, changing a service — would require changing
`always_denied_categories` in code, deliberately, with review.

---

## Confirmation gates

Two independent sources:

1. **The tool's own spec.** A tool with `requires_confirmation: true` always
   asks: `read_file`, `run_tests`, `open_url`, `start_application`,
   `stop_application`, `remember`.
2. **`security.always_confirm`** in `config/default.toml`, which adds names
   without editing the policy file.

A non-interactive session (a pipe, a service, CI) **declines** rather than
assuming consent. `--yes` and `--no` are explicit user choices and count as
authorisation.

---

## What ships allowed, and why

The shipped file enables exactly the read-only tools and nothing else:

| tool | effect | reasoning |
|---|---|---|
| `system_status` | allow | reads `/proc`; changes nothing |
| `search_files` | allow | sandboxed; cannot leave the allowed roots |
| `git_status` | allow | fixed argv, read-only git subcommand |
| `recall` | allow | reads your own memory; the results are shown to you |
| `cancel` | allow | no side effects |
| `read_file` | ask | your files are private; show the path first |
| `remember` | ask | writes to durable storage |
| `run_tests` | *commented out* | executes a process |
| `open_url` | *commented out* | hands a URL to your browser |
| `start_application` | *commented out* | launches a program |
| `stop_application` | *commented out* | kills a process |

Uncommenting the last four is a deliberate act. The commented block in
`config/permissions.toml` carries the reasoning for each.

`aaron setup` copies this file verbatim. It does not enable anything extra.

---

## Auditing what you changed

```console
$ aaron permissions test run_tests
tool            run_tests
category        process_exec
risk            medium
declared limits 300 s / 32768 bytes
verdict         DENY
reason          tool `run_tests` has no entry in permissions.toml; deny-by-default applies
```

`aaron tools list` shows the verdict for every tool in one table, and
`aaron logs` shows what was requested, authorised and actually happened.

---

## Test it before you trust it

```bash
cargo test -p aaron-app --test security   # 26 adversarial tests
cargo test -p aaron-security              # 25 policy and sandbox unit tests
```

The suite covers: an empty policy denying everything, unlisted tools, limits
that try to raise the global caps, `unknown_tool = "allow"` being rejected at
parse time, the permanently-denied category, the network switch, the
confirmation gate in a non-interactive context, and the shipped file asserting
that `refuse_root` is still `true`.