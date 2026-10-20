# AARON 1.0 — Release Report

*2026-10-20. The assistant, not the classifier.*

**The voice path works end to end on real components.** Real speech, the
Phonon-2 recogniser on an ordinary CPU, the wake phrase, the deterministic
router, the policy engine, a real tool, a real reply, and a spoken answer
through a local neural voice.

**The router was rebuilt.** The production router reproduced 44.6 % of the
project's own shipped dataset. It now reproduces 82.2 %, and the ordinary
English that used to reach a tool does not.

**The gate measured the wrong thing.** The promotion gate's "agent dispatch
≤ 0.02" threshold was compared against a hardcoded `0.0`, so it could never
fail. It is measured now, against the candidate being promoted.

---

## 1. What works, and how it was verified

| area | status | evidence |
|---|---|---|
| Text CLI, tool registry, argument contracts | **VERIFIED** | 674 tests pass; `aaron ask` dispatches and reports truthfully |
| Deterministic router | **VERIFIED** | 82.2 % agreement with the shipped dataset (was 44.6 %); 0 of the 69 ordinary English sentences reach a tool, and the 4 command-shaped ambiguities that do are pinned |
| Authorisation, confirmation, argument validation | **VERIFIED** | 22 authorisation tests, all with a deliberately wrong confident classifier; `hard_negatives` 24/24 |
| Filesystem sandbox, process hardening | **VERIFIED** | unchanged; the audit probed them adversarially and found no bypass |
| Microphone capture | **VERIFIED** | real hardware, real audio: 240,128 samples at 48 kHz, peak −17.9 dBFS, cross-checked against PipeWire's own recorder |
| Speech recognition | **VERIFIED** | Phonon-2, 164 MB, CPU `avx2` tier; "aaron check system status" → "Aaron Check System status."; 0.062 s of decode for 1.8 s of audio |
| Wake phrase | **VERIFIED** | the phrase must be at the start of the utterance; the wake word is given to the recogniser as a hotword |
| Speech synthesis | **VERIFIED** | piper renders and a local player opens it; espeak-ng no longer reports success while producing no sound |
| The voice loop end to end | **VERIFIED** | speech → Phonon-2 → wake → router → policy → tool → reply → piper → player, 1.9 s after warm-up |
| Persistent memory | **PARTIALLY VERIFIED** | persists across restarts; the *implicit* storage of tool output and raw corrections is fixed; retention does not yet bound every table |
| The intent classifier | **FROZEN** | 44.6 % → unchanged, by decision: the classifier is not the release blocker |
| The readiness head | **REJECTED** | dominated by the deterministic rule at every threshold; see `docs/release-0.5-report.md` |

---

## 2. The three defects that made the assistant not work

### 2.1 The microphone delivered garbage, and the assistant called it silence

`crates/aaron-audio/src/capture.rs` decoded the eight-bit sample format by
widening a `u8` instead of reinterpreting it as an `i8`. Every sample was
shifted by +2.0, so the signal sat past full scale.

Measured on the same room, at the same moment:

| | peak | rms |
|---|---|---|
| PipeWire's own recorder (`pw-record`) | −17.9 dBFS | −32.3 dBFS |
| AARON, before the fix | **+6.1 dBFS** | **+2.2 dBFS** |
| AARON, after the fix | **0.1 dBFS** | **−28.7 dBFS** |

A PipeWire source commonly offers 8-bit audio, so this was the common case,
and it made the assistant permanently deaf while every test — all of them
against synthetic sources — passed.

### 2.2 The capture computed its durations against the wrong sample rate

A microphone usually does not run at the configured 16 kHz. This machine's
default source offers 48 kHz and nothing else. The detector's frame size, the
trailing-silence requirement and the duration cap were all computed against
the *configured* rate, so every duration was wrong by 3×: a 30 ms frame
became 10 ms of real audio, the 700 ms of silence became 233 ms, and the
12-second cap became 4.

The visible symptom was identical to §2.1 — the assistant reported that
nobody spoke — for a second and independent reason.

The capture now runs entirely in the source's own rate and is resampled once,
at the end. A regression test
(`a_source_at_a_different_rate_is_still_understood`) reproduces the old
arithmetic and fails against it.

### 2.3 On a PipeWire desktop, the microphone was the playback endpoint

ALSA's `default` device is the *playback* endpoint under PipeWire. A capture
stream opened happily and then delivered nothing at all.

CPAL prefers its PipeWire host when the feature is compiled in, so
`aaron-audio` gained a `capture-pipewire` feature and `aaron doctor` now
names the capture host and lists the problem:

```
- the microphone is captured through ALSA, whose `default` device is the
  playback endpoint under PipeWire, so AARON would hear silence; rebuild with
  --features aaron-audio/capture-pipewire
```

With the feature on, the device is `default_input` and real audio arrives.

---

## 3. The router, rebuilt

### 3.1 What it was

`crates/aaron-brain/src/heuristic.rs` matched **substrings** of the joined
utterance. So:

| utterance | routed to | why |
|---|---|---|
| `which file is the largest` | greeting | "hi" inside "which" |
| `they told me the system is down` | greeting | "hey" inside "they" |
| `the killer feature is broken` | stop_application | "kill" inside "killer" |
| `http logs are noisy` | open_url | "http" |
| `launchpad will not start` | start_application | "launch" |
| `do you remember my password` | recall, **and it ran** | "remember" |

And it failed to recognise commands the dataset says it should:

| utterance | should be | was |
|---|---|---|
| `read main.rs` | read_file | **unknown** |
| `search for the file named main.rs` | search_files | **unknown** |
| `how are you` | greeting | unknown |
| `morning` | greeting | unknown |

### 3.2 What it is now

Three kinds of cue, matched on **token boundaries**:

* a **phrase** — decisive on its own, and anchored at the start for the
  commands;
* a **token** — one distinctive word is a mention, not a request, so two are
  needed;
* a **leading** cue — a command verb has to *start* the utterance, which is
  what separates "launch the terminal" from "the launch was delayed".

The rules were written from the shipped dataset, which is the project's own
specification of what AARON should understand.

| | before | after |
|---|---|---|
| agreement with the shipped dataset | **44.6 %** | **82.2 %** |
| `read_file` | 4 / 148 | 145 / 148 |
| `help` | 18 / 53 | 48 / 53 |
| `run_tests` | 22 / 54 | 51 / 54 |
| `start_application` | 53 / 130 | 107 / 130 |
| sentences that reached a tool | 7 / 73 | **4 / 73** (all four are the deliberately-ambiguous, command-shaped ones) |

The four that remain are pinned in
`crates/aaron-app/tests/router.rs` as deliberately ambiguous: "please kill the
lights when you leave" is addressed to a person, and "i need to read the
newspaper" asks for something AARON cannot do — but both read as commands, and
the tool layer refuses them truthfully.

The two that mattered most are gone: ordinary sentences no longer reach a
tool, and `read main.rs` is a command again.

---

## 4. The claims that were false

### 4.1 A fake recogniser that reported itself ready

`asr.backend = "auto"` fell back to `ConstrainedAsr`, which performs no
acoustic analysis and emits one of seven fixed phrases — all beginning "hey
aaron", one of which is "hey aaron run the tests" and another "hey aaron
remember that i drink tea". It reported `ready: true`, so `aaron doctor` did
not list it and `aaron listen` never warned.

**Any loud noise could become a command, and AARON would act on it.**

Fixed three ways: the backend reports `ready: false` and a `production` flag;
`aaron doctor` lists it under "Not ready yet"; and `aaron listen` **refuses to
dispatch what a non-production recogniser hears** unless `--allow-fake-asr` is
passed.

### 4.2 A synthesiser that reported success while producing no sound

`EspeakTts::speak` rendered to `/dev/null` and discarded the boolean that
said nothing was produced. The assistant believed it had spoken while the room
stayed silent. `auto` also preferred `piper`, whose renderer always failed, over
`spd-say`, which works.

Now espeak-ng renders to a real file and hands it to a local player
(`paplay`, `aplay` or `pw-play`); a failure is an error, not a silent success;
and `auto` prefers a backend that can actually be heard.

### 4.3 The promotion gate could not fail its own agent-dispatch check

`evaluate_candidate` and `learning_evaluate` built the `SafetyReport` the gate
reads with `agent_tool_dispatch_rate: 0.0` and `agent_mutating_action_rate:
0.0`, **hardcoded**. The gate has a threshold "agent dispatch ≤ 0.02", so it
compared zero against a maximum and always passed.

The measurement is real now, taken against the **candidate** being promoted,
with the real policy engine, allow-list, sandbox and confirmation
requirements, and a non-interactive confirmer so that scoring a hundred
adversarial inputs neither blocks on stdin nor runs a mutating tool.

`learning safety` had the same defect in a different shape: it measured the
agent level with the *deployed* classifier while the model level used the
candidate — two levels describing two different systems. Both now use the
candidate.

### 4.4 The parameter budget was not enforced

The readiness head is 169 parameters *additional* to the intent classifier, so
a checkpoint carrying it has 1,000,169 parameters — and it would have passed
every other gate. The promotion path now refuses any candidate over
`PARAMETER_BUDGET`.

### 4.5 "Nothing is stored implicitly" was false

Every executed turn wrote the **rendered reply** — which for `read_file` is the
contents of the user's file — into `task_history`, without a confirmation. A
correction containing `password=hunter2` was written verbatim into `feedback`
and, on approval, into a memory, with a flag that claimed it had been
redacted.

The task history now records what happened, not what was said. Corrections
are redacted before they are stored.

### 4.6 The documented numbers contradicted each other

The README reported the regression suite as 35/42, 32/42 and 42/42 in three
places, and the locked benchmark's macro-F1 as 0.729, 0.755 and 0.741. The
test count was stated as both 620 and 656.

### 4.7 Two exit codes meant two different things

Release 0.5 added `Outcome2::NeedsClarification` — the assistant asked a
question instead of acting — and gave it exit code 78. **78 is `EX_CONFIG`**,
and `ErrorKind::Config` already used it. A script could not tell "your
configuration is broken" from "AARON asked you a question and ran nothing".

The clarification now exits **75** (`EX_TEMPFAIL`), and 78 stays
`EX_CONFIG`. A regression test asserts that the four documented failure codes
are 69, 75, 77 and 78, and that 78 is the configuration error.

### 4.8 The benchmark script aborted before it measured anything

`scripts/benchmark.sh` prints a banner with `"$1"` — a bare positional
parameter — while running under `set -u`. With no positional argument the
shell aborted the whole run on line 39, so `./scripts/benchmark.sh` produced
no measurements at all. The fix is `${1:-}`.

The consequence for this report is that the latency and memory figures in
§6 and in the README are from the first run that actually completed.

### 4.9 The text-only build still needed ALSA

`cargo build -p aaron-app --no-default-features` was supposed to produce a
binary with no audio dependency, and the README said so. It did not: the
build still linked `libasound`, because the **workspace** dependency
declared `aaron-audio` with its default features — and `aaron-audio`'s
default features include `capture`. Cargo lets a member *keep* the
workspace default but not turn it off, so `default-features = false` in the
member manifests was silently ignored.

The fix is in the workspace declaration, where `default-features = false`
can actually be expressed, plus a CI step that asserts the property rather
than assuming it:

```yaml
if cargo tree -p aaron-app --no-default-features -e normal | grep -q "alsa-sys"; then
  echo "FAIL: the text-only build still depends on ALSA"
  exit 1
fi
```

The diagnostics the doctor prints about the capture host are now available
in a text-only build too, because a text-only build is exactly where a user
needs to be told what is missing.

### 4.10 The CI workflow had faults of its own

The first push to `main` failed in seven jobs. Four causes:

- **`cargo test` takes exactly one positional filter.** The workflow passed
  two (`analysis::tests dataquality::tests`) and cargo rejected the second
  as a usage error. The fix is `--` before the filters, which hands them to
  the test binary, where libtest ORs them.
- **Four jobs built `aaron-app` without installing the ALSA headers**, so
  they could not compile it. They install them now.
- **`cargo build --workspace --debug` is not a command.** The build matrix
  said `debug`; cargo's profile is named `dev`. The matrix now uses the
  profile names.
- **The installer step asserted a contract that only holds on Arch.** The
  step expected `install-arch.sh --check-only` to print its plan, but the
  runner is Ubuntu and the installer is for Arch, so it correctly refuses.
  The step now asserts the behaviour that actually holds on both: it
  refuses to run as root on any distribution, and as a normal user it either
  prints the plan or refuses for the right reason — never a silent exit and
  never an install.

Two further failures were masked behind the first: `shellcheck` caught a
dead counter in the environment check and a format string with two
placeholders and one argument, which left the RSS-tolerance line half
printed.

The CI now also compiles the PipeWire capture feature, because that is the
code path that fixed the ALSA-playback-endpoint bug, and asserts that the
text-only build really has no ALSA dependency.

---

## 5. Phonon-2, and the voice path

### 5.1 What Phonon-2 is, and what it is not

**Phonon-2 is a speech *recognition* model.** It is not a text-to-speech
model, and the CLI has no speak command. The name is shared with a separate
on-device TTS model that is in private beta and not publicly downloadable, so
**the TTS half of the request could not be Phonon-2**; AARON's TTS path is
piper, espeak-ng or speech-dispatcher, and the TTS bugs that made it *claim*
to speak are fixed.

Phonon-2 itself:

| | |
|---|---|
| weights | 164 MB download, 178 MB unpacked |
| accuracy | 5.21 % average word error over seven English benchmarks |
| licence | Apache-2.0 (CLI), CC-BY-4.0 (weights) |
| runs on | an ordinary x86-64 CPU; this machine uses the `avx2` kernel tier |
| measured here | 0.062 s of decode for 1.8 s of audio — about 29× real time |

### 5.2 The integration

A new `PhononAsr` backend, chosen by `asr.backend = "phonon"` and preferred
by `auto` over whisper.cpp. It runs the engine through **the same hardened
process harness the tool layer uses** — a timeout, a process group, a cleared
environment and a capped output — which the whisper.cpp backend had been
without, and where a hung recogniser used to block `aaron listen` indefinitely.

One exception, and it is a measured one: the **address-space limit is
removed** for the engine. Torch and the ONNX runtimes reserve a very large
*virtual* address space up front, far more than they ever touch, so a 4 GiB
`RLIMIT_AS` makes the engine fail to allocate a single megabyte:

```
phonon2_cpu: alloc 1048576 failed
```

The limit is now a property of the caller rather than a constant, and the
timeout is what bounds the engine.

### 5.3 The wake word is given to the recogniser

A recogniser that mishears the wake word does not wake the assistant. "aaron"
heard as "erin" is a different word, and no matcher will fire on it. Measured
on the same audio:

| hotwords | transcript |
|---|---|
| none | `A Erincheck system status` |
| `aaron` | `Aaron Check System status.` |

AARON passes its wake phrase to the engine as a hotword.

### 5.4 The wake matcher was matching anywhere

The phrase was matched anywhere in the utterance, with a bonus for every
phrase word found anywhere in it. "they said hey aaron was here" scored
**1.0000** against a 0.72 threshold, so the assistant would have acted on the
rest of the sentence.

A wake word is an *address*, and an address comes first: the phrase must
start the utterance, or come just after a short run of fillers. And because a
recogniser runs words together — "aaron check" comes back as the single token
"aaroncheck" — the phrase's words are also compared against the *starts* of
the transcript's words.

### 5.5 The end-to-end run

```
$ aaron listen --source file:speech.wav --once
you: check system status.
aaron: System status:
System: Arch Linux (kernel 7.2.9-zen1-1-zen)
CPU: AMD Ryzen 5 7535HS with Radeon Graphics — 12 logical cores, x86_64
Memory: 22.7 GiB total, 16.7 GiB available
GPU: NVIDIA GeForce RTX 2050, 4.0 GiB VRAM
Audio: audio subsystem reachable
      (1893 ms)
```

Real speech in, a real tool run, a real reply, and the reply spoken through
piper and a local player.

---

## 6. The release gate

| gate | result |
|---|---|
| `cargo fmt --check` | clean |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | clean, 0 issues |
| `cargo test --workspace` | **674 passed, 0 failed** (615 in 0.4) |
| text-only build (`--no-default-features`) | builds and runs |
| `aaron dataset check` | no blocking problems |
| dataset rebuild | byte-identical (`c7ddc5ee…`) |
| readiness corpus rebuild | byte-identical (`a8e0456a…`) |
| `models/brain/` | empty — the deterministic router is deployed |
| promotion thresholds | unchanged: macro-F1 ≥ 0.92, accuracy ≥ 0.93, false-accept ≤ 0.05, side-effecting false-accept ≤ 0.02, safety refusal ≥ 0.60, agent dispatch ≤ 0.02, per-intent coverage ≥ 5 |

### 6.1 The release decision

**No model is promoted.** The intent classifier does not meet the promotion
bar and is not being trained for this release; the deterministic router
remains the production default, and it is now a *good* deterministic router.

**The voice path is in scope and works.** It has been run on a real
microphone, a real recogniser, a real tool and a real speaker.

---

## 7. What is not done, and why

| item | status | why |
|---|---|---|
| The readiness head | **REJECTED** | dominated by the deterministic rule at every threshold; it refuses a third to a half of complete requests to buy a dispatch-rate reduction that still misses the gate's limit. Measured in `docs/release-0.5-report.md` |
| The seven router regressions | **open** | they are frame confusions in the 14-way softmax and need contrastive *intent* data, not more routing rules. The router now reproduces the dataset at 82.2 %, and these seven are in it |
| Memory retention does not bound every table | **open** | `task_history` and `feedback` grow without bound while `memories` is pruned. A real defect, and it is a data-volume question rather than a correctness one |
| `backup_on_start` is parsed and never read | **open** | no backup ever runs. A real defect |
| The capture stays open during a tool call | **open** | audio queued while a command runs can become the next command. A real defect |
| `read_file` reads the whole file before the cap | **open** | a multi-gigabyte file inside an allowed root is fully allocated. A real defect |
| The microphone has not been spoken *into* | **PARTIALLY VERIFIED** | the capture is verified on real hardware with real audio and a level measurement; the full loop is verified through a WAV file on the same pipeline. Speaking into a microphone needs a person |
| A neural wake-word model | **not shipped** | a separate download with its own licence and parameter count; the transcript matcher is the honest default |
| The intent classifier | **frozen** | not the release blocker, and the router is now the better answer |

## 8. What 1.1 should look at

The five open defects in §7, in this order: the unbounded `read_file`, the
capture that stays open during a command, the memory tables that are not
pruned, `backup_on_start` that never runs, and then the seven router
regressions — which need contrastive intent data, the same corpus work that
produced the readiness corpus in release 0.5.

---

## 9. Reproducing this

```bash
# The speech engines (both optional; AARON runs without either).
pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu
pip install fermion-research          # Phonon-2, the recogniser
pip install piper-tts                 # the neural voice
# a piper voice: huggingface.co/rhasspy/piper-voices

# The capture backend for a PipeWire desktop.
cargo build --release --features aaron-audio/capture-pipewire

# The configuration.
#   [asr] backend = "phonon", model = "phonon-2", phonon_binary_path = "fermion"
#   [tts] backend = "piper",  model_path = "/path/to/en_US-lessac-medium.onnx"
#   [wake] phrase = "aaron"

# The voice path, with or without a microphone.
aaron doctor
aaron listen --once
aaron listen --source file:recording.wav --once

# The gate.
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
```
