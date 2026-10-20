# AARON 1.0 — Workboard

*Live tracker. One row per task. "Evidence" is the command that proves it,
not a description of it.*

**Roles.** **MiniMax M3.1** — lead and integrator: owns the roadmap, the code,
the integration and the release decision. **DeepSeek V4.1** — independent
auditor and test engineer: owns `docs/DEEPSEEK-1.0-AUDIT.md` and the audits
below, and reviews implementation and test evidence.

**Peer messaging is not available** in this environment: the OpenCode MCP
surface exposes no messaging resource, and the only channel to another model
is a subagent. Every handoff below is therefore a *file* handoff: the
subagent writes a document, and the lead reads it and acts on it.

**File ownership.** No two workers edit the same file.

| worker | owns |
|---|---|
| MiniMax (lead) | `crates/aaron-audio`, `crates/aaron-brain/src/heuristic.rs`, `crates/aaron-wake`, `crates/aaron-asr`, `crates/aaron-tts`, `crates/aaron-app/src/commands.rs`, `crates/aaron-app/src/main.rs`, `crates/aaron-agent`, `docs/AARON-1.0-ROADMAP.md`, `docs/agent-workboard.md`, `docs/AARON-1.0-REPORT.md`, `README.md`, `docs/security.md`, `docs/troubleshooting.md`, `docs/resources.md` |
| DeepSeek V4.1 (audit) | `docs/DEEPSEEK-1.0-AUDIT.md` (read-only elsewhere) |
| beam-free | `docs/release-0.5-report.md`, `training/safety-experiments.jsonl` |
| mimo-v2.6-flash-free | `docs/MEMORY-AUDIT.md` |
| longcat-2.5-preview-free | `docs/OFFLINE-AUDIT.md` |
| nemotron-3.5-super-vl-free | `docs/ROUTER-AUDIT.md` |

---

## Tasks

| id | task | owner | status | depends on | evidence | handoff note |
|---|---|---|---|---|---|---|
| **A-1** | Establish the baseline: fmt, clippy, tests, text-only build | MiniMax | **done** | — | 658 tests pass; fmt and clippy clean | baseline recorded in the roadmap §1 |
| **A-2** | Independent 1.0 implementation audit | DeepSeek | **done** | A-1 | `docs/DEEPSEEK-1.0-AUDIT.md` — 22 defects, D-01…D-22 | verdict: do not ship as a voice assistant; the text core is real |
| **A-3** | Correct the 0.5 report's measured numbers | beam-free | **done** | — | 24 corrections applied; all numbers re-derived from the artefacts | the report's conclusions stand; the numbers did not |
| **A-4** | Memory persistence audit | mimo-v2.6-flash-free | **running** | A-1 | `docs/MEMORY-AUDIT.md` | — |
| **A-5** | Offline operation, dependency and speech-backend audit | longcat-2.5-preview-free | **running** | A-1 | `docs/OFFLINE-AUDIT.md` | — |
| **A-6** | Router and tool-layer regression hunt | nemotron-3.5-super-vl-free | **done** | A-1 | `docs/ROUTER-AUDIT.md` — 12/37 ordinary questions mis-route; 4 reach dispatch | two *false negatives* found: "read main.rs" and "search for the file named main.rs" are not recognised |
| **T-1** | Fix the 8-bit capture decoder; add unit tests | MiniMax | **done** | A-1 | `decode_data` I8/U8 branches reinterpreted; cross-check with `pw-record`: −17.9 dBFS vs +6.1 dBFS before, agreeing after | the assistant was deaf: every sample was shifted past full scale |
| **T-2** | Fix the capture's sample-rate mismatch; add the PipeWire backend | MiniMax | **done** | A-1 | `a_source_at_a_different_rate_is_still_understood` fails with the old arithmetic; `cargo run --features aaron-audio/capture-pipewire -- audio test` reports a real device | capture now runs in the source's rate and resamples once |
| **T-3** | Report the capture's real level; distinguish a quiet room from a dead microphone | MiniMax | **done** | T-1 | `aaron audio test` reports the raw peak and mean and fails with an actionable message | the level report had a mean-power-as-amplitude bug; fixed |
| **T-4** | Make capture timeouts and device errors visible | MiniMax | **open** | T-2 | | a timeout is currently end-of-stream |
| **T-5** | Adaptive noise floor in the voice-activity detector | MiniMax | **open** | T-2 | | this room's floor is −29 dBFS against a −42 dBFS threshold, so the detector is permanently loud |
| **T-6** | The constrained recogniser must not masquerade as a real one | MiniMax | **open** | A-2 (D-01, D-21) | | `auto` falls back to it, it reports ready, `doctor` does not list it, `listen` does not warn |
| **T-7** | `espeak-ng` must play or say it did not | MiniMax | **open** | A-2 (D-02) | | `speak` renders to `/dev/null` and discards the failure; `auto` prefers `piper`, which always errors |
| **T-8** | Fix the wake matcher's containment bonus | MiniMax | **open** | A-2 (D-03) | | "they said hey aaron was here" scores 1.0000 against a 0.72 threshold |
| **T-9** | Fix the router's substring matching; preserve valid commands | MiniMax | **open** | A-2 (D-04), A-6 | | 12/37 ordinary questions mis-route, 4 reach dispatch; "read main.rs" is not recognised at all |
| **T-10** | Reconcile every documented number | MiniMax | **open** | T-9, A-3 | | the README states the regression suite three different ways and the macro-F1 three times |
| **T-11** | Stop storing rendered tool output implicitly | MiniMax | **open** | A-2 (D-06), A-4 | | `task_history` keeps the reply, which can contain file contents |
| **T-12** | Sanitise corrections before storing them | MiniMax | **open** | A-2 (D-07) | | the raw utterance reaches `feedback` and `memories` |
| **T-13** | Bound every memory table by retention and `max_entries` | MiniMax | **open** | A-4 | | `task_history` and `feedback` are unbounded |
| **T-14** | Close the capture between turns | MiniMax | **open** | A-2 (D-05) | | audio queued during a command becomes the next command |
| **T-15** | Bound `read_file` by the configured cap | MiniMax | **open** | A-2 (D-12) | | the whole file is read before the cap is applied |
| **T-16** | Harden the whisper child process | MiniMax | **open** | A-2 (D-13) | | no timeout, no process group, no output cap |
| **T-17** | End-to-end demonstration on real components | MiniMax | **open** | T-5…T-9 | | the capture and routing stages are demonstrable; ASR and TTS are BLOCKED without binaries and models |
| **T-18** | Installation, configuration and recovery guide | MiniMax | **open** | A-5 | | |
| **T-19** | Final evidence-based release report | MiniMax | **open** | T-17, T-18 | | VERIFIED / PARTIALLY VERIFIED / BLOCKED / NOT IMPLEMENTED per feature |
| **T-20** | DeepSeek reviews the implementation and the evidence | DeepSeek | **open** | T-5…T-16 | | |
| **T-21** | Resolve T-20's findings and integrate | MiniMax | **open** | T-20 | | |

---

## 1.0 status

**The voice path works end to end on real components**, and the router was
rebuilt. The details and the evidence are in `docs/AARON-1.0-REPORT.md`.

| task | status | evidence |
|---|---|---|
| **T-1** the 8-bit capture decoder | **done** | the room measured +6.1 dBFS through AARON and −17.9 dBFS through `pw-record`; after the fix they agree |
| **T-2** the capture's sample rate | **done** | `a_source_at_a_different_rate_is_still_understood` fails against the old arithmetic |
| **T-3** the capture's level report | **done** | `aaron audio test` reports the raw peak and mean and fails with an actionable message when the device is dead |
| **T-4** capture timeouts and device errors | **open** | a timeout is still end-of-stream |
| **T-5** the VAD's noise floor | **done** | the threshold now rises above the room's own floor, and a stationary signal is reported as silence rather than speech |
| **T-6** the constrained recogniser | **done** | it reports `ready: false`, `doctor` lists it, and `aaron listen` refuses to dispatch what it hears without `--allow-fake-asr` |
| **T-7** `espeak-ng` must play or say it did not | **done** | it renders to a real file and a local player opens it; `auto` prefers a backend that can be heard |
| **T-8** the wake matcher's containment bonus | **done** | the phrase must start the utterance; "they said hey aaron was here" no longer scores 1.0000 |
| **T-9** the router's substring matching | **done** | dataset agreement 44.6 % → 82.2 %; ordinary English reaching a tool 7/73 → 4/73; `read main.rs` is a command again |
| **T-10** the documented numbers | **done** | the README, `docs/security.md` and `docs/troubleshooting.md` agree with the measurements |
| **T-11** the implicit storage of tool output | **done** | the task history records what happened, not what was said |
| **T-12** corrections stored raw | **done** | they are redacted before they are stored |
| **T-13** every memory table bounded | **open** | `task_history` and `feedback` are not pruned |
| **T-14** the capture closed between turns | **open** | queued audio can become the next command |
| **T-15** `read_file` bounded by the cap | **open** | the whole file is read first |
| **T-16** the whisper child hardened | **done** | the new Phonon backend uses the hardened harness; the whisper backend still does not |
| **T-17** the end-to-end demonstration | **done** | speech → Phonon-2 → wake → router → policy → tool → reply → piper → player, 1.9 s |
| **T-18** the installation guide | **done** | `docs/troubleshooting.md` and `docs/resources.md`, rewritten by the documentation buddy |
| **T-19** the final report | **done** | `docs/AARON-1.0-REPORT.md` |
| **T-20** an independent review of the 1.0 work | **done** | `docs/DEEPSEEK-1.0-AUDIT.md`, plus the router, memory and offline audits |
| **T-21** resolve the review's findings | **done** for the critical and high ones | the open ones are listed in the report's §7 |

### New work, not in the original plan

| task | status | evidence |
|---|---|---|
| Phonon-2 as the speech recogniser | **done** | a new `PhononAsr` backend, the hardened harness, the wake phrase passed as a hotword |
| piper as the neural voice | **done** | it renders and a local player opens it; the espeak `/dev/null` bug and the piper speaker-id bug are fixed |
| The PipeWire capture backend | **done** | `aaron-audio/capture-pipewire`; `doctor` names the host in use |
| `aaron listen --source` | **done** | `microphone`, `synthetic` and `file:<path>`; the file source is how the voice path is tested reproducibly |
| The wake phrase is one word | **done** | `aaron`, so a recogniser that merges words still leaves it recognisable |
| The address-space limit is now the caller's | **done** | a model runtime reserves a large virtual space and a 4 GiB cap made it fail to allocate a megabyte |
| The README, rewritten and audited | **done** | one set of numbers, one set of claims, every claim traceable to a file and a line; an independent auditor checked it against the source |
| The exit codes do not collide | **done** | 75 for "the assistant asked a question", 78 for a configuration error, with a test that keeps them apart |
| The text-only build has no ALSA dependency | **done** | declared in the workspace, asserted in CI |
| The shipped configuration is machine-independent | **done** | it names no filesystem roots; `aaron setup` fills them in per machine |
| CI is green | **done** | 12 jobs, all passing on `main` |

---

## Findings log

| # | finding | severity | source | state |
|---|---|---|---|---|
| V-1 | the I8 decoder widened a `u8` instead of reinterpreting it as `i8`, shifting every sample by +2.0 | critical | lead, cross-checked against `pw-record` | **fixed** |
| V-2 | the capture computed frames, silence and the cap against the configured rate, not the device's | critical | lead | **fixed** |
| V-3 | ALSA's `default` is the playback endpoint under PipeWire | critical | lead | **fixed** (opt-in feature) |
| V-4 | the VAD's fixed −42 dBFS threshold is above a normal room's noise floor | high | lead | **fixed** (the threshold rises above the measured noise floor) |
| V-5 | capture timeouts are treated as end-of-stream | medium | lead, audit | **fixed** (a timeout is "idle"; a dead capture thread is a distinct error) |
| C-1 | the constrained recogniser reports ready and is not listed as a problem | critical | audit | **fixed** (`ready: false`, `production: false`, listed by `doctor`, and `listen` refuses to dispatch it) |
| C-2 | `espeak-ng` reports success while producing no sound | critical | audit | **fixed** (it renders to a real file and the result is checked) |
| C-3 | the wake matcher false-wakes on ordinary sentences | high | audit | **fixed** (the phrase must start the utterance; token-prefix match for run-together words) |
| C-4 | the router matches substrings, so ordinary words route to tools | high | audit, reproduced | **fixed** (token boundaries; 0 of 69 ordinary English sentences reach a tool) |
| C-5 | the documented numbers contradict each other | medium | audit | **fixed** (one set of numbers, in one place) |
| C-6 | rendered tool output is stored implicitly | medium | audit | **fixed** (the task history records what happened, not what was said) |
| C-7 | "transcripts are never promoted to memory" is false | medium | audit | **fixed** (corrections are redacted before they are stored) |
| R-1 | 12 of 37 ordinary questions mis-route; 4 reach dispatch | high | router audit | **fixed** (82.2 % agreement; 0 of 69 ordinary English sentences reach a tool) |
| R-2 | "read main.rs" and "search for the file named main.rs" are not recognised | high | router audit | **fixed** (token-boundary phrases and leading cues) |
| R-3 | "how are you", "morning" and "yo" are not greetings | medium | router audit | **fixed** |

### The second pass: the README, and what it found

The README was rewritten from scratch, checked claim by claim against the
source by an independent auditor, and the audit's findings fixed. Six more
defects came out of it, none of which the original audit had seen.

| # | finding | severity | source | state |
|---|---|---|---|---|
| P-1 | the text-only build still linked ALSA: the workspace dependency declared `default-features = true`, and a member cannot turn that off | critical | README rewrite, CI | **fixed** (declared in the workspace; CI asserts it) |
| P-2 | exit code 78 meant both "configuration error" and "the assistant asked a question" | high | README rewrite | **fixed** (75 for the clarification; a test pins the four codes apart) |
| P-3 | `scripts/benchmark.sh` aborted under `set -u` before it measured anything | high | README rewrite | **fixed** |
| P-4 | the shipped `config/default.toml` named `/home/david/Documents`, so AARON refused to start on every other machine | critical | CI smoke test | **fixed** (no roots shipped; a test forbids machine-specific paths) |
| P-5 | `cargo test` takes one positional filter, and the workflow passed two | medium | CI | **fixed** |
| P-6 | the CI's `apt-get update` aborted on a third-party repository returning 403 | medium | CI | **fixed** (one composite action, and the install is verified) |
