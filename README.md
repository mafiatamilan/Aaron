# AARON

An offline-first, voice-controlled personal assistant for Linux, written in
Rust.

AARON listens for a wake word, works out what you said, checks every proposed
action against a **deny-by-default policy**, runs approved tools locally, and
speaks the answer back.

It is a *local* assistant. AARON's own code has no HTTP client anywhere in
its dependency graph, so it cannot open a network connection even by
accident. Nothing you say is uploaded, and there is no telemetry. (The speech
engine you install separately fetches its own model weights once — see
[Models and licences](#models-and-licences).)

```console
$ aaron ask "check system status"
System status:
System: Arch Linux (kernel 7.2.9-zen1-1-zen)
CPU: AMD Ryzen 5 7535HS with Radeon Graphics — 12 logical cores, x86_64
Memory: 22.7 GiB total, 16.2 GiB available
GPU: NVIDIA GeForce RTX 2050, 4.0 GiB VRAM, driver 615.78.08, CUDA runtime 13.4
Audio: audio subsystem reachable
Inference: cpu (CUDA backend not compiled in)
Privilege: normal user
```

(Verbatim, except that the free-memory figure changes with every boot.)

And, with a microphone and a speech engine installed:

```console
$ aaron listen
Listening via live microphone `Default Audio Device` at 16000 Hz (buffer budget 625 KiB, queue 333 frames)  [MICROPHONE LIVE]
you: check system status
aaron: System status:
System: Arch Linux (kernel 7.2.9-zen1-1-zen)
...
```

(The device name and sample rate are whatever your machine reports.)

---

## What works

| | |
|---|---|
| **Text commands**, end to end | works, with no downloads and no models |
| **System status, git status** | works |
| **File search and read** | works once you set `security.allowed_roots` |
| **Memory** (`remember` / `recall`), in SQLite | works, and survives a restart |
| **Deny-by-default policy** with confirmation gates | works |
| **Audit log** — append-only, redacted, truncated | works |
| **Microphone capture** | works; see [the microphone needs the right backend](#the-microphone-needs-the-right-backend) |
| **Voice activity detection** | works, and has no model |
| **Wake word** | works, by matching the recognised transcript |
| **Speech recognition** | works, via **Phonon-2** or whisper.cpp — both are yours to install |
| **Speech synthesis** | works, via piper, `espeak-ng` or `spd-say` — all yours to install |
| **The whole voice loop**, speech to speech | **partially verified** — run end to end on real components, but nobody has spoken into the microphone yet |
| **AARON's own 1,000,000-parameter intent model** | trains, but does not meet AARON's own promotion bar, so **nothing is deployed** and the **deterministic router is what runs** |

The last row is the important one, and it is not a failure of the model. It
is the design working: the model proposes, and the policy disposes. A bad
model and a good model are both safe here.

---

## Install

### From source (Arch Linux)

```bash
git clone https://github.com/mafiatamilan/Aaron.git && cd Aaron

./scripts/check-environment.sh     # reports what is and is not present
./scripts/install-arch.sh          # prints the package list, asks, then builds

sudo install -Dm755 target/release/aaron /usr/local/bin/aaron
aaron setup                        # writes ~/.config/aaron/{config,permissions}.toml
aaron doctor                       # what works on *this* machine
```

`install-arch.sh` lists every package before touching `pacman` and asks first.
It never uses an AUR helper, it refuses to start as root, it changes no system
configuration, and it calls `sudo` only for the individual `pacman` and
`install` commands.

### The voice stack (optional)

AARON is fully usable as a text assistant with none of this. With it, the
assistant can hear and speak:

```bash
# Speech recognition: Phonon-2, a 164 MB open model that runs on your CPU.
pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu
pip install fermion-research

# Speech synthesis: a neural voice.
pip install piper-tts
# and a voice from https://huggingface.co/rhasspy/piper-voices
#     en_US-lessac-medium.onnx is 63 MB
```

Then point the configuration at them:

```toml
[asr]
backend = "phonon"          # or "whisper", or "auto"
model = "phonon-2"
phonon_binary_path = "fermion"

[tts]
backend = "piper"            # or "espeak", "spd-say", "auto"
model_path = "/path/to/en_US-lessac-medium.onnx"

[wake]
phrase = "aaron"
```

`aaron doctor` then reports each engine as ready or not, and says why.

### Build without audio

```bash
cargo build --release -p aaron-app --no-default-features
```

A text-only binary with no ALSA dependency.

---

## The voice path

The whole chain is real, and it has been run on real components:

```
speech → Phonon-2 (CPU) → wake word → deterministic router → policy →
tool → reply → piper → a local player
```

Two things about it are worth knowing before you start.

### The microphone needs the right backend

On a PipeWire desktop, ALSA's `default` device is the **playback** endpoint:
a capture stream opens successfully and then delivers nothing at all, so the
assistant hears silence and says nobody spoke.

Build with the PipeWire backend — it needs `libpipewire-0.3` and `libspa-0.2`
headers:

```bash
cargo build --release --features aaron-audio/capture-pipewire
```

`aaron doctor` names the capture host in use, and lists this as a problem
when it is wrong.

### The wake word is given to the recogniser

A recogniser that mishears the wake word does not wake the assistant: "aaron"
heard as "erin" is a different word, and no matcher will fire on it. AARON
passes its wake phrase to the engine as a hotword.

The phrase is one word — `aaron` — so a recogniser that merges words still
leaves it recognisable.

### Trying it without a microphone

```bash
aaron listen --source file:recording.wav --once   # a WAV through the same pipeline
aaron listen --source synthetic --once            # a generated tone
aaron audio test                                 # is the microphone alive?
```

`aaron audio test` reports the raw input level, so you can tell a quiet room
from a dead microphone.

---

## How it decides to do something

```
  utterance
      │
      ▼
  ┌────────────────────┐
  │ classify           │  the deterministic router, or AARON's own model
  │ intent + entities  │  when a checkpoint is installed
  └─────────┬──────────┘
             ▼
  ┌────────────────────┐
  │ build arguments    │  every value length-bounded and type-checked
  └─────────┬──────────┘
             ▼
  ┌────────────────────┐
  │ POLICY CHECK       │  deny by default; a tool with no entry cannot run
  └─────────┬──────────┘
             ▼
  ┌────────────────────┐
  │ confirm?           │  shown with the exact command and target
  └─────────┬──────────┘
             ▼
  ┌────────────────────┐
  │ execute            │  fixed argv, sandboxed path, timeout, output cap
  └─────────┬──────────┘
             ▼
  ┌────────────────────┐
  │ audit + respond    │  from the *real* result, with untrusted text scrubbed
  └────────────────────┘
```

The model proposes; deterministic code disposes. It never selects a program,
never supplies a command line, and cannot skip the policy check.

The router is the production path, and it was rebuilt in release 1.0. It
reproduces **82.2 %** of this project's own shipped dataset — it used to
reproduce 44.6 %, because it matched *substrings*, so "hi" inside "which"
made a greeting and "do you remember my password" ran a recall. It now
matches on token boundaries, and **none of the 69 ordinary English sentences**
in the regression set reaches a tool any more.

---

## Safety

AARON is deny-by-default. A tool with no entry in `permissions.toml` cannot
run.

There is no shell tool, and no tool accepts a free-form command string.
Paths are canonicalised before the containment check, so `..` traversal and
symlink escapes are both rejected, and search never follows a symlink at all.
Child processes run with a scrubbed environment — the inherited variables are
replaced by a short allow-list, so a token in your shell environment is not
handed to a tool — with their own process group, CPU and file-size limits,
and are killed by group on timeout. Everything read
from a file, a process or memory is treated as untrusted data:
instruction-shaped lines are stripped before they can reach a response. The
audit log is append-only, redacted and truncated, and records what was
requested, what was authorised and what actually happened.

The boundary is tested **independently of the model**: the authorisation
suite drives a stub classifier that returns a chosen intent at confidence 1.0,
deliberately wrong for most inputs. A boundary that only held for a good
model would not be a boundary.

Details: [docs/security.md](docs/security.md). Adversarial coverage:
[crates/aaron-app/tests/security.rs](crates/aaron-app/tests/security.rs) and
[crates/aaron-app/tests/authorization.rs](crates/aaron-app/tests/authorization.rs).

---

## Memory and resources

AARON treats memory as a budget. Every queue and retained buffer has a
configured, validated bound; a model is never released while a caller holds
it; and low-memory mode degrades AARON's own behaviour and nothing else.

```console
$ aaron resources status
mode              idle
                  normal operation (idle)
sample age        0s

system memory
  total           22.7 GiB
  available       16.4 GiB
  free            72.3%  (low-memory trigger at 12%)
  usable budget   15.4 GiB   (1.0 GiB reserved)

gpu memory
  total           4.0 GiB
  used            40.0 MiB  (99.0% free)
  usable budget   3.2 GiB   (768.0 MiB reserved)

this process
  rss             6.4 MiB
  checkpoint      4.0 MiB derived (1000000 parameters) — estimated, not measured

nothing is degraded.
```

AARON never assumes the 4 GB card is available to it, never kills another
application to free memory, and never discards user-approved memories to
reduce RAM use. Full detail: [docs/resources.md](docs/resources.md).

---

## Learning from corrections

```console
$ aaron ask "launch pterm"          # misheard
I did not understand that (heuristic 0% confidence). Try `aaron help` for
examples, or correct me and I will remember it.
$ aaron learning pending            # (after you record a correction)
#3  `launch pterm`  -> start_application   (predicted unknown)
$ aaron learning approve 3          # sanitised, appended to the dataset
$ aaron brain train                 # trains a *candidate*
$ aaron learning evaluate           # held-out metrics + regression suite
$ aaron learning promote            # only if every threshold passed
```

Corrections never reach the model's weights on their own. Promotion is an
explicit command, it checks measured thresholds against a held-out split, a
regression suite and an adversarial benchmark, and the previous checkpoint is
kept for rollback.

**The candidate does not pass, and the gate refuses it.** On the locked
held-out benchmark it reaches accuracy **0.731** and macro-F1 **0.755**, and
it refuses **45.8 %** of the adversarial benchmark against a required 60 %.
On the side-effecting measure it admits **19.7 %** against a required 2 %.
So `aaron learning promote` **correctly refuses**, and AARON keeps using the
deterministic router.

The agent-level numbers are the ones that matter, and they are the reason the
model was never promoted: on the 142-input adversarial benchmark the agent
permitted a tool for **16.9 %** of the inputs and a tool actually ran for
**7.0 %**. The gate's limit is 2 %, so it misses it by more than eight times.
The authorisation boundary is what makes the system safe, and it is
independent of the model.

---

## Models and licences

### AARON's own model — exactly 1,000,000 parameters

| tensor | shape | parameters |
|---|---|---|
| `emb.weight` | 10240 × 84 | 860,160 |
| `ln1.weight` | 168 | 168 |
| `ln1.bias` | 168 | 168 |
| `fc1.weight` | 754 × 168 | 126,672 |
| `fc1.bias` | 754 | 754 |
| `ln2.weight` | 754 | 754 |
| `ln2.bias` | 754 | 754 |
| `fc2.weight` | 14 × 754 | 10,556 |
| `fc2.bias` | 14 | 14 |
| **total** | | **1,000,000** |

The pooled width is 168 rather than 84 because the model pools two statistics
over the utterance — the signed mean embedding and the per-dimension RMS — and
concatenates them. The hidden width is 754.

The count is enforced by `aaron-brain`'s `parameter_count_is_exactly_one_million`,
which walks the real architecture rather than reading a stored constant, and
`aaron model` prints it at runtime.

Architecture and the reasoning behind the dimensions:
[docs/model-training.md](docs/model-training.md).

### Everything else is separate

The 1,000,000 budget covers **AARON's intent model only**. It does not cover:

| Component | Model | Download | Licence | Installed by |
|---|---|---|---|---|
| Speech recognition | **Phonon-2**, via the `fermion-research` package | 164 MB (178 MB unpacked) | Apache-2.0 (CLI); CC-BY-4.0 (weights) | **you**; the engine fetches it on first use |
| Speech recognition | Whisper (ggml), via whisper.cpp | 39 M – 1.55 B *parameters*, depending on the size you pick | MIT (code); weights under their own terms | **you** |
| Speech synthesis | Piper voice (VITS, ONNX) | ~63 MB per voice (about 30 M parameters) | MIT (code), per-voice terms | **you** |
| Speech synthesis | espeak-ng (formant synthesiser, no model) | none | GPL-3.0 | `pacman -S espeak-ng` |
| Wake word | none shipped | none | — | — |

AARON **never downloads a model**. The one thing that does is the speech
*engine*, once, for its own weights, into `~/.cache/fermion/speech/` — which
is the engine's behaviour, not AARON's.

`constrained-grammar` appears in `aaron doctor` when no real recogniser is
installed. It is **not speech recognition**: it emits one of a fixed list of
phrases so the audio pipeline can be exercised with no model installed.
`aaron doctor` labels it as such, and `aaron listen` **refuses to dispatch
what it hears** — a fabricated transcript is an instruction, and AARON will
not act on one. `--allow-fake-asr` runs the demonstration anyway.

---

## Measurements

Release build, AMD Ryzen 5 7535HS / 22.7 GiB RAM / RTX 2050 4 GB, measured
once with `./scripts/benchmark.sh --release`. Latency and RSS move by a few
percent between runs on the same machine, so treat them as indicative rather
than exact. Re-run the script on your own hardware; do not quote these
numbers for a different machine.

| | |
|---|---|
| tests | **674 passing**, no hardware or models required |
| intent inference, 14 utterances | median **301 µs**, p95 351 µs (n = 200) |
| wake-phrase match | median **27 µs**, p95 36 µs (n = 2,000) |
| end-to-end text turn | median **59 µs**; the p95, tens of milliseconds, is one SQLite write and one denial check, not the classifier |
| idle process RSS | **8.6 MiB** baseline; **9.1 MiB** text-only; **14.7 MiB** peak in the combined voice path |
| memory stability | 5 workloads — 20,000 classifications, 2,000 database writes, 5,000 allocations, 5,000 audio buffers, 2,000 idle samples — every one `stable`, no unbounded growth |
| memory on disk | 253 KiB (SQLite, including the WAL) |
| GPU memory | 40 MiB → 40 MiB, i.e. **zero** attributable to AARON |
| wake false activations | **0 of 10** on the shipped negative set |
| router, shipped dataset | **82.2 %** agreement (was 44.6 %) |
| ordinary English reaching a tool | **0 of 69**; the 4 command-shaped ambiguities are pinned, and reach the tool on purpose |
| classifier, locked benchmark | accuracy **0.731**, macro-F1 **0.755** |
| locked benchmark false-accept | **0.320** |
| safety benchmark, agent permitted a tool | **0.169**; a tool actually ran **0.070** |
| regression suite | **32 / 42** |

The regression figure and the classifier figures are the **release-0.4
architecture, retrained** on the fixed hold-out — release 0.4's own checkpoint
cannot be re-scored, because it was trained on 12 of the rows the hold-out
pins. Release 0.5 found that release 0.4's hold-out fix had stopped at the
dataset builder: the **trainer** re-derived the splits and trained on those
12 rows. That is fixed, and every comparison now runs on a freshly trained
baseline over the frozen hold-out. The retrained baseline is statistically
indistinguishable from the 0.4 one (paired McNemar p = 1.00 on the locked
benchmark): the hold-out fix changed the *measurement*, not the model.

The most effective next step is harder data, not more of the easy kind: the
remaining errors are out-of-schema sentences the model confidently claims to
understand, and a confident false positive is what runs the wrong tool.

One figure quoted here is not from the benchmark script: the Phonon-2 decode
took **0.062 s for 1.8 s of audio**, about 29× real time, on this machine's
CPU `avx2` tier. `./scripts/benchmark.sh` does not measure recognition
latency, because it runs without a recogniser installed.

---

## What AARON is not

It is worth being clear about this, because an assistant that overstates
itself is worse than one that does less.

- **It is not a general-purpose LLM assistant.** The intent set is fourteen
  closed values, and the router is a keyword-and-verb matcher with an
  argument resolver. It will not write your essay.
- **Its own model does not qualify, so it does not ship.** AARON's 1,000,000-
  parameter classifier reaches macro-F1 0.755 against a bar of 0.92, and the
  promotion gate refuses it. The deterministic router is what runs.
- **There is no neural wake-word model.** The wake word is matched against
  the recognised transcript, which means it only works when a recogniser is
  installed, and a recogniser that mishears it does not wake the assistant.
- **It is single-user and local.** One SQLite database, one audit log, one
  machine.
- **Five things are still open**, and they are listed with reasons in
  [docs/AARON-1.0-REPORT.md](docs/AARON-1.0-REPORT.md) §7: `read_file` reads
  a whole file before applying the cap, the microphone stays open during a
  tool call, two memory tables are not pruned, `backup_on_start` never runs,
  and seven router regressions remain — frame confusions in the 14-way
  softmax, which need contrastive *intent* data rather than more rules.
- **The voice path was verified on one machine, and nobody has spoken into
  it.** The capture is verified on real hardware with real audio and a level
  measurement, and the whole loop is verified through a WAV file on the same
  pipeline. Speaking into a microphone needs a person.

---

## Commands

```
aaron setup                       write a starter config and permissions file
aaron doctor                      what works on this machine, and what is missing
aaron run                         interactive assistant
aaron listen [--no-wake] [--once] [--seconds N] [--source SRC] [--allow-fake-asr]
aaron ask "<utterance>"           one command, then exit
aaron tools list                  every tool and its current policy verdict
aaron tools test                  summarise the policy verdict for every tool
aaron permissions show            the effective policy
aaron permissions test [tool]     the verdict for one tool, or all of them
aaron memory list|search|add|delete|backup
aaron learning pending|approve|reject|evaluate|analyse|safety|promote|rollback
aaron brain train [--epochs N] [--lr F] [--seed N] [--promote]
aaron model                       parameter counts, licences, checkpoint metadata
aaron benchmark                   latency and memory, measured now
aaron logs                        recent audit records
aaron audio test|devices          microphone check without starting the assistant
aaron resources status|limits|stress|soak|checkpoint
aaron dataset check|report        # schema, duplicates, leakage, coverage
```

Global flags: `--config FILE`, `--yes`, `--no`, `--text-only`, `--no-audit`,
`--json`, `--log-level LEVEL`.

Exit codes: `0` success · `1` failure · `69` missing dependency · `75` the
assistant asked a question and ran nothing · `77` permission denied · `78`
configuration error · `130` cancelled.

---

## Repository layout

```
crates/
  aaron-core        types, config, errors, traits, text pipeline, hardware probes
  aaron-security    policy engine, path sandbox, audit log
  aaron-resources   budgets, low-memory state machine, model lifecycle, stress
  aaron-memory      SQLite store, migrations, retention
  aaron-tools       tool registry, argument validation, sandboxed processes
  aaron-brain       the 1M-parameter model, dataset, training, evaluation
  aaron-audio       capture, VAD, audio sources
  aaron-wake        wake-word detection
  aaron-asr         speech recognition backends
  aaron-tts         speech synthesis backends
  aaron-agent       state machine, guards, untrusted-content handling
  aaron-learning    correction queue, promotion, rollback
  aaron-bench       latency and memory benchmarks
  aaron-app         the CLI
```

---

## Documentation

**Start here**
* [docs/AARON-1.0-REPORT.md](docs/AARON-1.0-REPORT.md) — the 1.0 release: the voice path, the router, the false claims that were fixed
* [docs/architecture.md](docs/architecture.md) — how the pieces fit
* [docs/security.md](docs/security.md) — the threat model and its coverage
* [docs/troubleshooting.md](docs/troubleshooting.md) — what to do when something fails
* [docs/permissions.md](docs/permissions.md) — what each policy decision means
* [docs/resources.md](docs/resources.md) — memory budgets and low-memory behaviour
* [docs/model-training.md](docs/model-training.md) — architecture, dataset, training, evaluation
* [docs/dataset-schema.md](docs/dataset-schema.md) — the data format, the quality rules, and the readiness corpus

**The engineering record**
* [docs/AARON-1.0-ROADMAP.md](docs/AARON-1.0-ROADMAP.md) — the 1.0 plan, and what it found
* [docs/agent-workboard.md](docs/agent-workboard.md) — the task tracker
* [docs/DEEPSEEK-1.0-AUDIT.md](docs/DEEPSEEK-1.0-AUDIT.md) — an independent audit of the implementation
* [docs/ROUTER-AUDIT.md](docs/ROUTER-AUDIT.md) — the router, rule by rule
* [docs/MEMORY-AUDIT.md](docs/MEMORY-AUDIT.md) — the memory store, and where it leaks
* [docs/OFFLINE-AUDIT.md](docs/OFFLINE-AUDIT.md) — the offline claim, and the real dependency list
* [docs/release-0.5-report.md](docs/release-0.5-report.md) — the readiness contract, a machine-written corpus, and a measured negative result
* [docs/release-0.5-plan.md](docs/release-0.5-plan.md) — the reviewed plan, and what the review changed
* [docs/release-0.4-report.md](docs/release-0.4-report.md) — the hold-out fix, the paired statistics, and a measured negative result
* [docs/release-0.3-report.md](docs/release-0.3-report.md) — the safety work: metrics, benchmarks, experiments
* [docs/decisions/](docs/decisions/) — design decisions and why they were made

---

## Development

```bash
cargo test --workspace        # 674 tests, no hardware or models required
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
./scripts/benchmark.sh        # measure on this machine
```

Regenerate the shipped data:

```bash
python3 training/build_dataset_v2.py            # the intent dataset
python3 training/build_readiness_corpus.py      # the readiness corpus
python3 training/build_readiness_corpus_v2.py   # + the model-generated slices
python3 training/apply_readiness_audit.py       # + the blind audit
```

---

## Licence

MIT. See [LICENSE](LICENSE).

Third-party components keep their own licences; see
[models/manifest.toml](models/manifest.toml).
