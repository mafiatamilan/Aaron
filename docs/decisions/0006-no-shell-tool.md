# 0006 — No shell tool, ever

**Status:** accepted

## Context

A voice assistant that runs commands is one prompt-injection away from being a
remote shell.

## Decision

No tool accepts a command string. `run_tests` takes a *registered suite name*
that resolves to a fixed program and a fixed argument vector from the
configuration. `start_application` takes a name from an allow-list.

## Reasoning

* **Structural, not conventional.** `no_tool_accepts_a_free_form_command`
  walks every declared argument of every registered tool and fails if any looks
  command-shaped. A future contributor adding a `command` argument breaks CI.
* **No shell means no metacharacters.** `;`, `|`, `$()` and backticks are inert
  bytes in an argv entry.
* **The model cannot influence it.** The model chooses one of 14 intents; the
  tool registry holds `&'static str` ids; the policy engine takes facts from
  the registry rather than from the classifier.

## Consequences

* AARON cannot run a command the user did not first register.
* Some things that would be easy with a shell are awkward. They are awkward on
  purpose.
