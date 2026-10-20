# Test fixtures

Deterministic inputs shared by the integration and security suites, so they do
not depend on the host, the clock or any downloaded model.

| file | used by |
|---|---|
| `../integration/` | end-to-end CLI tests (live in `crates/aaron-app/tests/`) |
| `../security/` | adversarial tests (live in `crates/aaron-app/tests/security.rs`) |

Both suites build their own temporary trees with `tempfile` rather than relying
on fixtures on disk, because a fixture that can be modified between runs is not
a fixture. What lives here instead is the *description* of what those suites
cover and the property each one pins.

## Fixtures the suites rely on

| source | what it provides |
|---|---|
| `config/default.toml` | the shipped configuration; every suite loads it |
| `config/permissions.toml` | the shipped deny-by-default policy |
| `training/dataset/current.jsonl` | the labelled intent dataset |
| `training/evaluation/regression.jsonl` | 42 hard cases that must not regress |

## Deterministic synthetic signals

Audio tests never open a device. They generate:

* `MonoPcm::sine(500, 16_000, 220.0)` — a tone, for level and VAD tests;
* `MonoPcm::silence(1000, 16_000)` — silence, for the "nothing heard" path;
* `SyntheticSource` — alternating 300 ms tone / 200 ms silence, so the VAD sees
  a realistic utterance envelope.

## What is deliberately absent

No microphone, no GPU, no model file and no network. A test that needs any of
those is a test that will fail on someone else's machine, which makes it worse
than no test at all.

`aaron-asr` asserts on its own `Cargo.toml` that it depends on no HTTP client,
so "no network" is enforced rather than assumed.
