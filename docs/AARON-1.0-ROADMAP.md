# AARON 1.0 — Roadmap

*Created 2026-10-20. Findings are evidence-based: every "confirmed" item below
was reproduced on this machine, with the command that reproduces it recorded.
Items marked "reported" come from an independent audit and were checked
against the code but not yet re-reproduced.*

**Status: complete.** The voice path works end to end on real components, the
router was rebuilt, and the false claims are fixed. The evidence is in
`docs/AARON-1.0-REPORT.md`; the task tracker is `docs/agent-workboard.md`.
What is still open is listed in the report's §7.

**Objective.** Finish AARON 1.0 as a *usable* assistant. The text, tool,
security and memory core is real. The voice path is not: on this machine the
assistant listened to a real microphone and reported that nobody spoke, for
three separate reasons, and it now has three fixes that are not yet finished
or tested end to end.

**Definition of done.** The documented core workflow works end to end on real
components; the deterministic router stays the production default; the
authorisation and argument-validation boundaries stay independently enforced;
memory behaves correctly across restarts; voice functionality is tested on a
real microphone or is explicitly marked BLOCKED with what is needed; the
release checks pass; and the final report distinguishes VERIFIED, PARTIALLY
VERIFIED, BLOCKED and NOT IMPLEMENTED.

---

## 1. Baseline

| check | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo test --workspace` | 658 passed, 0 failed |
| text-only build (`--no-default-features`) | builds |
| `models/brain/` | empty — the deterministic router is deployed |
| machine | Arch Linux, Ryzen 5 7535HS, 22.7 GiB RAM, RTX 2050 4 GB |
| audio server | PipeWire 1.6.9, two microphone sources present |
| speech binaries installed | **none** (no whisper, espeak-ng, piper, spd-say) |
| model files present | **none** (no ASR model, no TTS voice, no classifier checkpoint) |

**Verified as working:** the text CLI, the tool registry and its argument
contracts, the deny-by-default policy engine, the confirmation gate, the
filesystem sandbox, process execution hardening, the audit log, and the
intent model's training and promotion gate.

---

## 2. Confirmed defects, most severe first

### P0 — the voice path does not work

**V-1. The eight-bit decoder read signed samples as unsigned.**
`crates/aaron-audio/src/capture.rs`. `Data::bytes()` hands back raw bytes, so
`b` is a `u8`; the `SampleFormat::I8` branch widened it with `*b as f32`
instead of reinterpreting it as `i8`. Every sample was shifted by +2.0: the
signal sat past full scale, the level read **+6.1 dBFS** for a room that
PipeWire's own recorder measured at **−17.9 dBFS**, and the voice-activity
detector saw a permanently loud input and never found an utterance.
A PipeWire source commonly offers 8-bit audio, so this was the common case.
**Fixed**; the cross-check now agrees with `pw-record`.

**V-2. The capture mixed the device's sample rate with the configured one.**
`crates/aaron-audio/src/source.rs`. A microphone that offers only 48 kHz is
normal. The detector's frames, the trailing-silence requirement and the
duration cap were all computed against the configured 16 kHz, so every
duration was wrong by 3×: a 30 ms frame became 10 ms, the 700 ms of silence
became 233 ms, and the 12-second cap became 4. The symptom was the same
"nobody spoke" report, for a second and independent reason.
**Fixed**: the capture runs in the source's own rate and is resampled once at
the end. A regression test reproduces the old arithmetic and fails.

**V-3. The default input device on a PipeWire desktop is the *playback*
endpoint.** `crates/aaron-audio/src/capture.rs`. ALSA's `default` is the
playback device under PipeWire, so an ALSA capture stream opened happily and
delivered nothing at all. `cpal` has a PipeWire host and prefers it when the
feature is on.
**Fixed** by adding an opt-in `capture-pipewire` feature; verified on this
machine: the device is now `default_input` and real audio arrives.

**V-4. The voice-activity detector's fixed threshold does not survive a real
room.** `crates/aaron-audio/src/vad.rs`. The threshold is −42 dBFS; this
room's noise floor is about **−29 dBFS**, so every frame counts as speech,
the utterance never ends, and the capture is discarded. An energy VAD with a
fixed threshold is only correct in a silent room.
**Open.** Needs an adaptive noise floor, and a report of the measured floor
so the user can see why the assistant is deaf.

**V-5. The capture's failure modes are swallowed.** `crates/aaron-audio/src/source.rs`.
A receive timeout is mapped to an empty block, which the recorder treats as
end-of-stream, so a slow device truncates the utterance instead of waiting.
The CPAL error callback only logs.
**Open.**

### P0 — false claims

**C-1. A fresh install silently uses a fake recogniser and calls it ready.**
`crates/aaron-asr/src/constrained.rs`, `crates/aaron-app/src/commands.rs`.
`asr.backend = "auto"` falls back to `ConstrainedAsr`, which does no acoustic
analysis and emits one of seven fixed phrases that all begin "hey aaron". It
reports `ready: true`. `doctor` prints the caveat in the backend table but
does **not** list it under "Not ready yet", and `listen` never warns — so
any utterance the VAD accepts becomes a canned command.
**Open.**

**C-2. `espeak-ng` reports success while producing no sound.**
`crates/aaron-tts/src/lib.rs`. `speak` calls `render_wav(text, "/dev/null")`,
which returns `Ok(false)` because `/dev/null` is not a regular file, and the
`false` is discarded. `auto` also prefers `piper`, whose `render_wav` always
errors, over `spd-say`, which actually plays.
**Open.**

**C-3. The wake matcher fires on ordinary sentences.**
`crates/aaron-wake/src/lib.rs`. A containment bonus gives every phrase token
found *anywhere* in the utterance credit, so "they said hey aaron was here"
scores 1.0000 against a 0.72 threshold.
**Open.**

**C-4. The router matches substrings, so ordinary words route to tools.**
`crates/aaron-brain/src/heuristic.rs`. Reproduced on this machine:

| utterance | AARON did |
|---|---|
| `which file is the largest` | greeted it — "hi" inside "which" |
| `they told me the system is down` | greeted it — "hey" inside "they" |
| `the killer feature is broken` | `stop_application` |
| `http logs are noisy` | `open_url` |
| `launchpad will not start` | `start_application` |
| `do you remember my password` | `recall`, and executed it |

**Open.** This is the production router, so it is the highest-impact routing
defect in the project.

**C-5. Documented numbers contradict each other.** The README reports the
regression suite as 35/42, 32/42 and 42/42 in three places, and the locked
benchmark's macro-F1 as 0.729, 0.755 and 0.741. The test count is stated as
620 and 656. `docs/security.md` states the 0.3 refusal rate as 51.0 % and
48.04 %.
**Open.**

**C-6. "Nothing is stored implicitly" is false.** `crates/aaron-agent/src/agent.rs`,
`crates/aaron-memory/src/store.rs`. Every executed turn writes the rendered
reply — which can contain file contents — into `task_history` without
confirmation, and a raw correction is written verbatim into `feedback` and,
on approval, into `memories`.
**Reported** by the audit, confirmed by reading the code.

**C-7. The privacy claim in `docs/security.md` ("Transcripts are never
promoted to memory") is false** for the same reason as C-6.
**Reported.**

### P1 — real but smaller

| id | what | where |
|---|---|---|
| P-1 | capture stays open across turns and during tool execution, so queued audio becomes the next command; no barge-in | `crates/aaron-app/src/commands.rs`, `crates/aaron-agent/src/session.rs` |
| P-2 | `read_file` reads the whole file before applying the 64 KiB cap | `crates/aaron-tools/src/builtin/files.rs` |
| P-3 | `whisper-cli` is spawned with no timeout, no process group, no output cap | `crates/aaron-asr/src/whisper.rs` |
| P-4 | retention and `max_entries` do not bound `task_history` or `feedback`; `doctor`'s row count omits them | `crates/aaron-memory/src/store.rs` |
| P-5 | the low-memory release path is unreachable, and no command calls the hook | `crates/aaron-app/src/context.rs` |
| P-6 | TTS is leased for the process lifetime, so it is never idle-unloaded | `crates/aaron-app/src/context.rs` |
| P-7 | `sanitize` slices the original string at a byte offset found in a lowercased copy; Unicode case folding can change the length | `crates/aaron-brain/src/dataset.rs` |
| P-8 | `ArgKind::Choice` with an empty choice list accepts anything | `crates/aaron-tools/src/spec.rs` |
| P-9 | the free-form-command safety test inspects argument *names* only, not kinds or values | `crates/aaron-app/tests/security.rs` |

---

## 3. Priorities and acceptance criteria

### Wave 1 — the voice path works or says it does not

| id | task | acceptance |
|---|---|---|
| **T-1** | Finish the 8-bit decoder fix: add unit tests for the I8/U8 branches | `decode_data` maps 0x00→−1.0 and 0x80→0.0 for I8; 0x00→−1.0 and 0xFF→+1.0 for U8; the room measurement agrees with `pw-record` within 6 dB |
| **T-2** | Finish the sample-rate fix: the regression test exists; add the PipeWire feature to the release build | `cargo test -p aaron-audio` passes; `aaron audio test` on this machine reports a peak within 6 dB of `pw-record` |
| **T-3** | Make the capture failure modes visible | a device that times out is retried, not treated as end-of-stream; the CPAL error callback increments a counter that `aaron audio test` reports |
| **T-4** | Adaptive noise floor in the VAD | the detector tracks the room's floor and reports it; a −29 dBFS room no longer produces a permanently-loud input; a real utterance at −20 dBFS is still detected |
| **T-5** | Make `aaron audio test` a first-class diagnostic | it distinguishes a quiet room from a dead microphone, names the host in use, and says which feature to build for a PipeWire desktop |

### Wave 2 — no false claims

| id | task | acceptance |
|---|---|---|
| **T-6** | The constrained recogniser must not masquerade as one | it reports `ready: false` with a `production` flag; `doctor` lists it under "Not ready yet"; `listen` refuses to run with it unless `--allow-fake-asr` is passed, and says why |
| **T-7** | `espeak-ng` must either play or say it did not | `speak` returns an error when nothing was rendered; `auto` prefers a backend that works; `doctor` reports "would play" honestly |
| **T-8** | Fix the wake matcher's containment bonus | ordinary sentences containing "hey"/"aaron" score below the threshold; a negative corpus test covers at least 20 such sentences |
| **T-9** | Fix the router's substring matching | token-boundary matching; the six reproduced misroutes are fixed; a negative corpus test covers at least 40 ordinary questions and statements |
| **T-10** | Reconcile every documented number | one source of truth per number; the README, `docs/security.md` and `docs/troubleshooting.md` agree with each other and with the measurements |

### Wave 3 — behaviour and privacy

| id | task | acceptance |
|---|---|---|
| **T-11** | Stop storing user content implicitly | `task_history` keeps the intent, tool and outcome but not the rendered output; the privacy claim in `docs/security.md` becomes true |
| **T-12** | Sanitise corrections before they are stored | the raw utterance never reaches `feedback` or `memories` |
| **T-13** | Bound the memory tables | retention and `max_entries` apply to every table; `doctor` reports them all |
| **T-14** | Close the capture between turns | audio queued during a command is not consumed as the next command |
| **T-15** | Bound `read_file` | a file larger than the cap is read up to the cap, not in full |
| **T-16** | Harden the whisper child | timeout, process group, cleared environment, capped output — reusing the existing process harness |

### Wave 4 — the release

| id | task | acceptance |
|---|---|---|
| **T-17** | The end-to-end demonstration | a real microphone, a real utterance, a real tool, a real reply — or a precise BLOCKED note naming what is missing |
| **T-18** | The installation and recovery guide | one command per platform to a working voice assistant; the first failure is named with its fix |
| **T-19** | The final report | VERIFIED / PARTIALLY VERIFIED / BLOCKED / NOT IMPLEMENTED per feature, with the evidence for each |

---

## 4. Dependencies and risks

| risk | effect | mitigation |
|---|---|---|
| No speech binaries and no models are installed | the voice path cannot be demonstrated end to end on this machine | the *capture* and *routing* stages are demonstrable with real hardware; the ASR and TTS stages are BLOCKED with the exact install steps |
| The VAD's behaviour is room-dependent | a fix tuned to this room may not generalise | the noise floor is *measured and reported* rather than assumed, and the threshold is relative to it |
| The router is the production path | every change to it changes what ships | each fix is a targeted case with a test, not a rule rewrite; the 42-case suite is the regression gate |
| Fixing the router changes the measured model metrics | the README's numbers move | re-measure everything after the router changes and re-publish the numbers, not the old ones |
| The 0.5 classifier work is a distraction | it consumes effort that the assistant needs | the classifier is frozen for 1.0: no new training, no promotion, the deterministic router ships |

## 5. Explicitly out of scope for 1.0

* Any new classifier training, dataset generation or readiness-head work. The
  head stays rejected; the contract stays; the deterministic router ships.
* Lowering any promotion threshold or the one-million-parameter budget.
* A barge-in implementation beyond draining the capture between turns.
* A neural wake-word model. It is a separate download with its own licence
  and parameter count, and the transcript matcher is the honest default.
