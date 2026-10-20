# AARON

An offline-first, voice-controlled personal assistant for Arch Linux, written in
Rust.

AARON listens for a wake phrase, classifies what you said into a fixed set of
intents using a **custom 1,000,000-parameter model**, checks every proposed
action against a **deny-by-default policy**, and runs approved tools locally.

Nothing is uploaded. Nothing is downloaded. There is no telemetry and no code
path that opens a network socket.

```console
$ aaron ask "check system status"
System status:
System: Arch Linux (kernel 6.17.4-arch1-1)
CPU: AMD Ryzen 5 7535HS with Radeon Graphics — 12 logical cores, x86_64
Memory: 22.7 GiB total, 15.0 GiB available
GPU: NVIDIA GeForce RTX 2050, 4.0 GiB VRAM, driver 615.78.08, CUDA runtime 13.4
Audio: audio subsystem reachable
Inference: cpu (CUDA backend not compiled in)
Privilege: normal user
```

(Verbatim except for the kernel and free-memory figures, which change every
boot.)

---

## What works right now, on a fresh install

| Capability | Status |
|---|---|
| Text commands, end to end | **works**, no downloads, no models |
| System status, git status | **works** |
| File search and read, inside allowed roots | **works** once you set `allowed_roots` |
| Memory (`remember` / `recall`), persisted in SQLite | **works** |
| Deny-by-default policy with confirmation gates | **works** |
| Audit log | **works** |
| Audio capture (CPAL/ALSA/PipeWire) | **works** if a microphone exists |
| Voice activity detection | **works**, no model |
| Wake phrase | **works** by matching the ASR transcript; no neural wake-word model |
| Speech recognition | **works** via **Phonon-2** (the engine fetches the model on first use) or whisper.cpp; both are yours to install |
| Speech synthesis | **works** via piper, `espeak-ng` or `spd-say` if installed; otherwise text |
| The 1M-parameter intent model | **trains and deploys**, but does not yet meet AARON's own promotion bar (see below) |

Measured on the development machine, release build, AMD Ryzen 5 7535HS /
22.7 GiB RAM / RTX 2050 4 GB (`scripts/benchmark.sh`):

| | |
|---|---|
| tests | **620 passing**, no hardware or models required |
| intent inference | median **29 µs**, p95 37 µs per utterance (5,000 samples) |
| end-to-end text turn | median **27 µs** (p95 is dominated by one write path; see below) |
| wake-phrase match | median **60 µs** over 50,000 matches |
| idle process RSS | **9.6 MiB** (text-only); **14.9 MiB** peak in the combined voice path |
| memory stability | 5 workloads × 20,000 iterations, all `stable` (peak 9.7 MiB) |
| GPU memory | 39 MiB → 39 MiB, i.e. **zero** attributable to AARON |
| per-mode peak RSS | text-only 9.6, asr 9.6, tts 9.6, **voice 14.9 MiB** |
| classifier, locked benchmark | accuracy **0.723**, macro-F1 **0.729** |
| locked benchmark false-accept rate | **0.160** |
| regression suite | **35 / 42** — the 0.3 figure of 42/42 was memorisation |

The 0.3 figures were measured on a hold-out that was **not** held out: 41 of
the 42 regression cases were inside the training data, and there were 40
cross-split near-duplicate pairs. Release 0.4 fixed the hold-out regime, which
made every number *lower* and made the regression suite real. See
[docs/release-0.4-report.md](docs/release-0.4-report.md).

Release 0.5 found that the 0.4 hold-out fix had stopped at the builder: the
**trainer** re-derived the splits and was training on 12 of the 214 rows the
hold-out pinned. That is fixed, and every comparison now runs on a freshly
trained baseline over the frozen hold-out. On that baseline the locked
benchmark is accuracy **0.731**, macro-F1 **0.755**, false-accept **0.320**,
and the regression suite is **32 / 42** — statistically indistinguishable
from the 0.4 model (paired McNemar p = 1.00 on the locked benchmark). The
honest conclusion is that the hold-out fix changed the *measurement*, not the
model. See [docs/release-0.5-report.md](docs/release-0.5-report.md).

The `end-to-end text turn` p95 of ~91 ms is one SQLite write and one denial
check, not the classifier. It is reported rather than smoothed away.

Re-run `./scripts/benchmark.sh` on your own hardware. Do not quote these
numbers for a different machine.

---

## Install

### From source (Arch Linux)

```bash
git clone <this repository> aaron && cd aaron

./scripts/check-environment.sh     # reports what is and is not present
./scripts/install-arch.sh          # prints the package list, asks, then builds

sudo install -Dm755 target/release/aaron /usr/local/bin/aaron
aaron setup                        # writes ~/.config/aaron/{config,permissions}.toml
aaron doctor                       # what works on *this* machine
```

`install-arch.sh` lists every package before touching `pacman` and asks first.
It never uses an AUR helper, never runs as root, and never changes system
configuration. Speech models are deliberately *not* installed for you; see
[Models](#models).

### Build without audio

```bash
cargo build --release -p aaron-app --no-default-features
```

Gives a text-only binary with no ALSA dependency.

---

## Commands

```
aaron setup                       write a starter config and permissions file
aaron doctor                      what works on this machine, and what is missing
aaron run                         interactive assistant
aaron listen [--no-wake] [--once] [--seconds N]
aaron ask "<utterance>"           one command, then exit
aaron tools list                  every tool and its current policy verdict
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

Exit codes: `0` success, `77` permission denied, `130` cancelled, `69` missing
dependency, `78` configuration error, `1` other failure.

---

## How it decides to do something

```
  utterance
      │
      ▼
  ┌────────────────────┐
  │ classify           │  custom 1,000,000-parameter model, or the
  │ intent + entities  │  deterministic router when no checkpoint exists
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

---

## Models

### AARON's own model — exactly 1,000,000 parameters

| tensor | shape | parameters |
|---|---|---|
| `emb.weight` | 10240 × 84 | 860,160 |
| `ln1.weight` / `ln1.bias` | 168 | 336 |
| `fc1.weight` / `fc1.bias` | 754 × 168 / 754 | 127,426 |
| `ln2.weight` / `ln2.bias` | 754 | 1,508 |
| `fc2.weight` / `fc2.bias` | 14 × 754 / 14 | 10,570 |
| **total** | | **1,000,000** |

The count is enforced by `aaron-brain`'s `parameter_count_is_exactly_one_million`,
which walks the real architecture rather than reading a stored constant, and
`aaron model` prints it at runtime.

Architecture and the reasoning behind the dimensions:
[docs/model-training.md](docs/model-training.md).

### Everything else is separate

The 1,000,000 budget covers **AARON's intent model only**. It does not cover:

| Component | Model | Parameters | Licence | Installed by |
|---|---|---|---|---|
| Speech recognition | **Phonon-2**, via the `fermion-research` package | 164 MB download, 178 MB unpacked | Apache-2.0 (CLI); CC-BY-4.0 (weights) | **you**, and the engine fetches it on first use |
| Speech recognition | Whisper (ggml), via whisper.cpp | 39 M – 1.55 B, depending on the size you pick | MIT (code); weights under their own terms | **you** |
| Speech synthesis | Piper voice (VITS, ONNX) | ~63 M per voice | MIT (code), per-voice terms | **you** |
| Speech synthesis | espeak-ng (formant synthesiser, no model) | none | GPL-3.0 | `pacman -S espeak-ng` |
| Wake word | none shipped | none | — | — |

AARON **never downloads a model**. If a model file is missing, `aaron doctor`
says so and the assistant degrades to text.

`constrained-grammar` appears in `aaron doctor` when no real recogniser is
installed. It is **not speech recognition**: it emits one of a fixed list of
phrases so the audio pipeline can be exercised with no model installed.
`aaron doctor` labels it as such, and `aaron listen` **refuses to dispatch
what it hears** — a fabricated transcript is an instruction, and AARON will not
act on one. `--allow-fake-asr` runs the demonstration anyway.

### The voice path, end to end

The whole chain is real and has been run on real components:

```
speech → Phonon-2 (CPU) → wake phrase → deterministic router → policy →
tool → reply → piper → a local player
```

Two things about it are worth knowing before you start.

**The microphone needs the right backend.** On a PipeWire desktop, ALSA's
`default` device is the *playback* endpoint: a capture stream opens
successfully and then delivers nothing, so the assistant hears silence and
says nobody spoke. Build with `--features aaron-audio/capture-pipewire` (it
needs `libpipewire-0.3` and `libspa-0.2` headers). `aaron doctor` names the
capture host in use and lists this as a problem when it is wrong.

**The wake word is given to the recogniser.** AARON passes its wake phrase
to the engine as a hotword, because a recogniser that mishears it does not
wake the assistant: "aaron" heard as "erin" is a different word, and no
matcher will fire on it. The wake phrase is one word — `aaron` — so a
recogniser that merges words still leaves it recognisable.

To exercise the path without a microphone, `aaron listen --source
file:<path>` plays a WAV through the same pipeline.

---

## Memory and resources

AARON treats memory as a budget. Every queue and retained buffer has a
configured, validated bound; a model is never released while a caller holds it;
and low-memory mode degrades AARON's own behaviour and nothing else.

```console
$ aaron resources status
mode              idle
                  normal operation (idle)
system memory
  total           22.7 GiB
  available       15.0 GiB
  free            66.3%  (low-memory trigger at 12%)
  usable budget   14.0 GiB   (1.0 GiB reserved)
gpu memory
  total           4.0 GiB
  used            39.0 MiB  (99.0% free)
  usable budget   3.2 GiB   (768.0 MiB reserved)
this process
  rss             12.1 MiB
```

AARON never assumes the 4 GB card is available to it, never kills another
application to free memory, and never discards user-approved memories to reduce
RAM use. Full detail: [docs/resources.md](docs/resources.md).

---

## Security model in one paragraph

AARON is deny-by-default. A tool with no entry in `permissions.toml` cannot run.
There is no shell tool, and no tool accepts a free-form command string — that is
asserted structurally by a test that walks every declared argument. Paths are
canonicalised before the containment check, so `..` traversal and symlink
escapes are both rejected; search never follows a symlink at all. Child
processes run with a cleared environment, their own process group, CPU and file
size limits, and are killed by group on timeout. Everything read from a file, a
process or memory is treated as untrusted data: instruction-shaped lines are
stripped before they can reach a response. The audit log is append-only,
redacted and truncated, and records what was requested, what was authorised and
what actually happened.

Details: [docs/security.md](docs/security.md). Adversarial coverage:
[crates/aaron-app/tests/security.rs](crates/aaron-app/tests/security.rs).

---

## Learning from corrections

```console
$ aaron ask "launch pterm"          # misheard
I did not understand that (neural 41% confidence). ...
$ aaron learning pending            # (after you record a correction)
#3  `launch pterm`  -> start_application   (predicted unknown)
$ aaron learning approve 3          # sanitised, appended to the dataset
$ aaron brain train                 # trains a *candidate*
$ aaron learning evaluate           # held-out metrics + regression suite
$ aaron learning promote            # only if every threshold passed
```

Corrections never reach the deployed weights on their own. Promotion is an
explicit command, it checks measured thresholds against a held-out split and a
regression suite, and the previous checkpoint is kept for rollback.

**Measured on the locked held-out benchmark: accuracy 0.731, macro-F1 0.741**,
with 42 of 42 regression cases passing. On the **held-out adversarial safety
benchmark** — written after the model was selected, and scored once — the
model refuses **48.0%** of adversarial input, against 37.3% for the 0.2
release, and the side-effecting false-accept rate fell from 0.284 to **0.186**.

That is still below AARON's own configured promotion bar (`min_macro_f1 = 0.92`,
`min_accuracy = 0.93`), so `aaron learning promote` **correctly refuses** and
AARON keeps using the deterministic router. It refuses on five counts:

```
  - macro-F1 0.7411 is below the required 0.9200
  - accuracy 0.7308 is below the required 0.9300
  - false-accept rate 0.2800 is above the maximum 0.0500
  - safety refusal rate 0.4804 is below the required 0.6000
  - side-effecting false-accept rate 0.1863 is above the maximum 0.0200
```

**The agent-level numbers are the ones that matter, and they are zero**: for
the adversarial benchmark, the agent dispatched a tool for **no** example, and
did nothing that changed state, for either release. The authorisation boundary
is what makes the system safe, and it is independent of the model — which is
why a bad model and a good model are both safe here, and why a low model
false-accept rate would never have been enough on its own.

The most effective next step is harder data, not more of the easy kind: the
remaining errors are out-of-schema sentences the model confidently claims to
understand, and a confident false positive is what runs the wrong tool.

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
  aaron-wake        wake-phrase detection
  aaron-asr         speech recognition backends
  aaron-tts         speech synthesis backends
  aaron-agent       state machine, guards, untrusted-content handling
  aaron-learning    correction queue, promotion, rollback
  aaron-bench       latency and memory benchmarks
  aaron-app         the CLI
```

---

## Documentation

* [architecture.md](docs/architecture.md) — how the pieces fit
* [security.md](docs/security.md) — the threat model and its coverage
* [permissions.md](docs/permissions.md) — what each policy decision means
* [model-training.md](docs/model-training.md) — architecture, dataset, training, evaluation
* [release-0.3-report.md](docs/release-0.3-report.md) — the safety work: metrics, benchmarks, experiments
* [release-0.4-report.md](docs/release-0.4-report.md) — the hold-out fix, the paired statistics, and a measured negative result
* [release-0.5-report.md](docs/release-0.5-report.md) — the readiness contract, three defect fixes, a machine-written corpus, and a measured negative result
* [release-0.5-plan.md](docs/release-0.5-plan.md) — the reviewed plan, and what the review changed
* [dataset-schema.md](docs/dataset-schema.md) — the data format, the quality rules, and the readiness corpus
* [release-0.2-plan.md](docs/release-0.2-plan.md) — this release's findings, experiments and results
* [resources.md](docs/resources.md) — memory budgets and low-memory behaviour
* [troubleshooting.md](docs/troubleshooting.md) — what to do when something fails
* [decisions/](docs/decisions/) — design decisions and why they were made
  (0008 is the readiness contract: a typed verdict and the one-way valve that
  consumes it)

---

## Development

```bash
cargo test --workspace        # 656 tests, no hardware or models required
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
./scripts/benchmark.sh        # measure on this machine
```

Regenerate the shipped dataset:

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