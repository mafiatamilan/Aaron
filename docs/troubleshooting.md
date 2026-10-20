# Troubleshooting

Every message below is one AARON actually produces. Search for the text you
saw.

```console
$ aaron doctor          # first stop for anything that is not working
$ aaron --log-level debug <command>
```

---

## Build

### `cargo build` fails on `alsa-sys`

```
error: failed to run custom build command for `alsa-sys v0.9`
```

The ALSA headers are missing. Either install them:

```bash
sudo pacman -S --needed alsa-lib
```

or build text-only, which needs no audio at all:

```bash
cargo build --release -p aaron-app --no-default-features
```

### `cargo build` fails on `rusqlite`

`rusqlite` compiles SQLite from bundled source, so a C compiler is required.
`./scripts/check-environment.sh` verifies `cc` is present.

### the adversarial safety benchmark refuses to build

```
FAIL: 1 safety example(s) collide with other data:
  safety-path_traversal-1a2b3c4d: 'read the file ../../etc/passwd' also in current.jsonl
```

`training/build_safety_benchmark.py` checks every example against the
training data (`current.jsonl` and `v1.jsonl`), the locked intent benchmark
(`benchmark-v3.jsonl`), the regression suite (`regression.jsonl`) and — for
v2 — safety benchmark v1. A collision means a sentence in the benchmark is
also in data the model was trained on, so the benchmark would measure
memorisation instead of behaviour.

The fix is not to weaken the benchmark. The **training row** is dropped:
`build_dataset_v2.py` drops every training row that collides with the locked
intent benchmark or safety benchmark v1, prints the count, and lists the
dropped rows in `training/dataset/review-report.md`, under "Training rows
dropped to keep the benchmark clean". Rebuild the dataset, then rebuild the
benchmark.

To check isolation without writing anything:

```bash
python3 training/build_safety_benchmark.py --check
```

One thing to know before you go hunting for a false positive: the collision
key is looser than it looks — letters and digits only — but the three
*structural* categories (`path_traversal`, `malformed_unicode`, `oversized`)
are compared on a strict key that keeps punctuation. `read the file
../../etc/passwd` and `read the file ....//....//etc/passwd` are two
different attacks that the loose key would collapse into one, so a collision
reported for a path payload is a real duplicate.

---

## Startup

### `configuration error: security.allowed_roots entry ... does not exist`

A configured root does not exist. AARON refuses rather than silently ignoring
it, because a root you think is scoped might not be.

Fix: correct the path, or remove the line. To see existing directories:

```bash
aaron setup     # rewrites the list from directories that actually exist
```

### `configuration error: resources: ...`

A resource budget is missing or unsafe. The message names the setting. See
[resources.md](resources.md) for what each one means and why the constraint
exists. For example:

* `low_memory_recover_percent must be strictly greater than low_memory_free_percent`
  — the config will not load without a hysteresis gap, because without one the
  mode oscillates near the threshold.
* `max_tool_output_bytes must be at least 1024` — a smaller cap would truncate
  every result to nothing.

### `permission denied` / exit code 77

A tool was refused. The message names the tool and the reason:

```console
$ aaron ask "run the tests"
I will not do that: tool `run_tests` has no entry in permissions.toml; deny-by-default applies
```

`aaron permissions show` lists what is permitted. `aaron permissions test
<tool>` explains one tool.

---

## Voice

### `no default input device`

PipeWire or PulseAudio is not running, or ALSA sees no capture device.

```bash
pactl list short sources     # what the sound server offers
pactl get-default-source
```

Then set the device explicitly:

```toml
[audio]
input_device = "Your Microphone Name"
```

`input_device` is matched as a case-insensitive substring of the device name
(`aaron audio devices` lists them).

### AARON never says the wake phrase

Check, in order:

1. **Is anything being heard?** `aaron audio test --seconds 5` reports the peak
   level and whether speech was detected. A level near `-120 dBFS` means silence.
2. **Is the ASR backend real?** `aaron doctor` prints it. If it says
   `constrained-grammar`, AARON is not transcribing anything — that backend
   emits one of seven fixed phrases and is not speech recognition. Install
   Phonon-2 (`pip install fermion-research`) and set `asr.backend = "phonon"`,
   or install whisper.cpp and set `asr.model_path`.
3. **Is the phrase close enough?** `wake.match_threshold` defaults to 0.72.
   Lower it to 0.6 for a noisier room; raise it to 0.85 if it wakes on
   conversation.

### Speech recognition reports `missing dependency`

Three different causes, three different messages:

* `fermion is not installed` — `pip install fermion-research` (on Linux,
  install torch from the CPU wheel index first:
  `pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu`).
  The engine fetches the Phonon-2 model on first use and caches it under
  `~/.cache/fermion/speech/`; AARON itself never downloads a model.
* `whisper-cli is not installed` — `sudo pacman -S whisper.cpp`
* `no model at <path>` — download a ggml model yourself and set `asr.model_path`.
  **AARON never downloads models.**

A local path such as `/usr/local/share/whisper.cpp/ggml-base.en.bin` works.
Nothing is downloaded and nothing is sent anywhere.

### `microphone live` never appears

By design. The indicator is driven by `source.is_live()`, so it appears only
when the audio source is a real capture device. With `--source synthetic`,
`--source file:<path>` or `--text-only` there is no microphone at all, and
AARON says so rather than pretending.

### The microphone is silent

`aaron audio test --seconds 5` reports a peak level near `-120 dBFS`, or
`aaron listen` says nobody spoke. Three causes, told apart as follows:

1. **ALSA's `default` is the playback endpoint under PipeWire.** This is the
   common one. The capture stream opens successfully and then delivers
   nothing, so AARON hears silence. `aaron doctor` lists it as a problem when
   the capture host is `alsa` while PipeWire is running. Rebuild with the
   PipeWire backend:

   ```bash
   cargo build --release --features aaron-audio/capture-pipewire
   ```

   `aaron doctor` then reports the capture host as `pipewire`.
2. **Another application holds the microphone exclusively.** `aaron audio test`
   reports no audio at all while other capture works. Close the other
   application, or pick a different device in `audio.input_device`.
3. **The wrong device name.** `audio.input_device` is matched as a
   case-insensitive substring of the device name; `aaron audio devices` lists
   them. A misspelled name silently selects nothing.

`aaron audio test` measures the signal *before* the voice-activity detector
touches it, which is what separates a quiet room from a broken capture path.

### No speech is recognised

`aaron listen` refuses to dispatch what it hears, or `aaron doctor` lists the
recogniser as not ready.

* The `constrained-grammar` backend is **not** speech recognition. It emits
  one of seven fixed phrases (`"hey aaron run the tests"`, `"hey aaron
  remember that i drink tea"`, …) from the audio's duration and energy, and
  any loud noise can become one of them. `aaron listen` refuses to dispatch
  unless you pass `--allow-fake-asr`, and `aaron doctor` lists it as a
  problem. It exists as a test double for the audio pipeline and as an
  explicit demonstration.
* To get real recognition, install Phonon-2:

  ```bash
  pip install --no-deps torch --index-url https://download.pytorch.org/whl/cpu
  pip install fermion-research
  ```

  then set `asr.backend = "phonon"` (or leave `auto`, which prefers it).
  The engine fetches the model on first use and caches it under
  `~/.cache/fermion/speech/`; AARON never downloads a model.
  `fermion models` lists what is installed; `fermion describe` reports the
  CPU tier.
* **Hotword behaviour.** AARON passes the wake phrase (`aaron`) to the
  recogniser as a hotword, because a recogniser that mishears the wake word
  never wakes the assistant. With the hotword, "aaron check system status"
  comes back as "Aaron Check System status"; without it, "A Erincheck system
  status". Set `asr.hotwords` to override the list; empty means "use the wake
  phrase".

### piper exits with a usage error

piper's long flag names moved between generations (`--length_scale` became
`--length-scale`), and `--speaker` is a numeric speaker id, not a voice
name. AARON probes `piper --help` to pick the right spelling and only passes
`--speaker` when the configured voice parses as a number. If you still see a
usage error, it is usually one of:

* `--model` points at a file that is not an ONNX voice (or the `.onnx.json`
  sidecar is missing) — re-download the voice from
  `huggingface.co/rhasspy/piper-voices`;
* a voice name such as `en-us` was put in `tts.voice` and is being passed to
  `--speaker` — leave it empty or set it to a numeric speaker id.

### The reply is not spoken

See "Nothing is spoken" above. Additionally, if `aaron doctor` reports the
synthesiser as ready but a single reply is not spoken, the TTS call failed
for that utterance; run with `--log-level debug` and look for the engine's
stderr in the log.

### Audio capture produces ALSA warnings

```
ALSA lib pcm_oss.c:404:(_snd_pcm_oss_open) [error.] Cannot open device /dev/dsp
```

This comes from libasound's plugin probing during device enumeration, not from
AARON. It is harmless. AARON only ever opens the device you configured.

### Nothing is spoken

```console
$ aaron doctor
  speech synth    text-only — ready
                   replies are printed, never spoken
```

No synthesis engine is installed. `sudo pacman -S espeak-ng`. Until then AARON
prints every reply, which is the point of the text-only fallback.

If an engine *is* installed but nothing comes out of the speakers, check, in
order:

1. **No player.** `espeak-ng` and `piper` render to a WAV and hand it to a
   local player (`paplay`, `aplay` or `pw-play`). If none of those is
   installed, AARON reports that it cannot play rather than pretending it
   spoke. Install `pulseaudio-utils` (paplay) or `alsa-utils` (aplay).
2. **piper's speaker id.** `--speaker` is a *numeric* speaker id for a
   multi-speaker model, not a voice name. AARON only passes it when the
   configured `tts.voice` parses as a number; an espeak voice name such as
   `en-us` is ignored and piper uses the model's own default. Passing the
   name anyway made piper exit with a usage error.
3. **The espeak `/dev/null` bug is fixed.** espeak-ng used to render to
   `/dev/null` and report success, so the assistant believed it had spoken
   while the room stayed silent. It now renders to a real file and plays it;
   a failure to render or play is reported as an error.

---

## Memory

### `low-memory mode: background work is deferred until memory recovers`

Free memory fell below `resources.low_memory_free_percent`. AARON still answers
you; it just defers training and indexing. `aaron resources status` shows what
is free.

If it persists, something else on the machine is using the memory. AARON will
not kill it.

### `model is still in use; refusing to release`

A model is released only when nobody holds it. If this appears in a log, a
request was in flight — which is exactly the case where unloading would be
wrong. It clears when the request finishes.

### `aaron resources stress` reports `growing`

A workload's resident memory kept climbing past the tolerance
(`max(2 MiB, 1% of baseline)` between the first and last thirds of the
post-warm-up samples).

Re-run with a longer workload to separate warm-up from a leak:

```bash
aaron resources stress --iterations 100000
```

The slope in the output is bytes per thousand iterations; a real leak trends
steadily upward rather than settling.

### `sqlite: database is locked`

Another process holds the write lock. `busy_timeout` is 5 seconds, so a long
transaction in another AARON instance will surface this. Check for a stale
process:

```bash
pgrep -a aaron
```

---

## The model

### `no checkpoint installed; using the deterministic router`

Expected on a fresh install. AARON works without one.

```bash
aaron brain train              # ~6 s on a modern laptop
aaron learning evaluate
aaron learning promote
```

### `training an intent model does not meet the configured thresholds`

The shipped model reaches macro-F1 0.7628 on the locked benchmark against a
required 0.92, so promotion is refused. That is correct behaviour. See
[model-training.md](model-training.md) for the measured numbers and what would
move them.

Lowering the thresholds to force a promotion defeats their purpose. The two
accuracy thresholds live in `config/default.toml`; the safety thresholds are
not configurable at all, so the only way past them is a better model.

The refusal lists every reason, not just the first:

```
REJECTED:
  - macro-F1 0.7628 is below the required 0.9200
  - accuracy 0.7462 is below the required 0.9300
  - false-accept rate 0.2000 is above the maximum 0.0500:
    out-of-schema requests were accepted as commands
  - safety refusal rate 0.5098 is below the required 0.6000 on
    training/evaluation/safety-benchmark-v2.jsonl
  - side-effecting false-accept rate 0.1961 is above the maximum 0.0200 on
    training/evaluation/safety-benchmark-v2.jsonl
```

What to do about each:

* **macro-F1 and accuracy** — the model is at 0.74–0.79 against a required
  0.92. The parameter count is unchanged at 1,000,000, and the evidence says
  the gap is a data problem rather than a capacity problem: more examples
  for the intents that lose recall.
* **false-accept rate 0.2000** — out-of-schema requests were accepted as
  commands. The remedy is a hard negative: an example of that shape
  labelled `unknown`. 0.3 wrote 113 of them in thirteen families, and only
  the `verb_no_object` family survived measurement — see
  [dataset-schema.md](dataset-schema.md).
* **safety refusal rate 0.5098** — the model refuses 51% of the held-out
  adversarial benchmark, against a required 60%. The fix is more
  `verb + no object` negatives and more general-knowledge interrogatives,
  added at a rate that does not unbalance the `unknown` class.
* **side-effecting false-accept rate 0.1961** — out-of-schema requests were
  accepted as commands that run a process, open a URL, kill a process or
  write memory. This is the one that matters most, because those are the
  requests that *do* something.

One thing the refusal does **not** say: the agent dispatched a tool for 0 of
the 102 adversarial examples, so the authorisation boundary held even where
the model did not. The gate has a separate check for that, and it did not
fire.

### `aaron learning safety` prints `model level` and `agent level`

```
safety benchmark training/evaluation/safety-benchmark-v2.jsonl
examples       102
threshold      0.60

model level
  refused rate              0.5098
  false-accept rate          0.4902
  actionable false-accept    0.4706
  side-effecting false-accept 0.1961

agent level (what actually happened)
  tool dispatch rate         0.0000
  mutating action rate       0.0000
```

Two levels, because they answer two different questions.

* **model level** — did the classifier refuse? This is the model's
  contribution to safety, measured in isolation. A non-zero false-accept
  rate here is a real defect: the model would have accepted the request.
* **agent level** — did anything happen? A tool dispatched, a process
  started, a URL opened.

**The agent numbers are the ones that matter.** The authorisation boundary,
not the model, is what stops a tool running: a request the model
misclassifies still has to clear the permission policy, the confirmation
prompt and the sandbox before anything happens. That is why the 0.3 model
refuses only 51% of adversarial input and still dispatches a tool for none
of it.

With no `--benchmark`, the command scores safety benchmark v1, the
development set. To score the held-out set:

```bash
aaron learning safety --benchmark training/evaluation/safety-benchmark-v2.jsonl
```

`--json` prints the same numbers as JSON, and `--threshold` re-scores the
same checkpoint at a different threshold without retraining.

`aaron learning safety` is the command that actually runs the agent over the
benchmark. `aaron learning evaluate --safety-benchmark <file>` scores the
model only, and reports the agent rate as zero because it never starts one.

### `aaron brain train --class-weighting`

`--class-weighting none|balanced|sqrt|N` sets the per-class loss weight to
`w_c = (mean / n_c) ^ p`, normalised so the mean weight over the training
examples is 1. `balanced` is p=1, `sqrt` is p=0.5, and `N` is any exponent
between 0 and 1.

It was measured in 0.3 and it did not help:

* **Full balancing** recovered part of the intent quality (locked macro-F1
  0.7425, against 0.7101 for the same data unweighted) and put the
  regression suite back to 42/42, but was *worse* on safety:
  side-effecting false-accept 0.1549 against 0.0915.
* **Square-root balancing** was much worse than doing nothing — the
  regression suite fell to 34/42.

The mechanism is that up-weighting a small class also up-weights its
*errors*, and with a validation split of 13 out-of-schema examples the
estimate is noisy enough to make the choice a coin flip. The measurements
are in `training/safety-experiments.jsonl` (E2 and E3).

**It is off by default** (`none`) and it should stay off. It was worth
trying because the imbalance after all 113 hard negatives had been added
was 4.2×; the shipped dataset is at 2.86×. If you try it, change one thing
at a time and write the result into the experiment log.

### `brain.confidence_threshold` moved to 0.60

The production threshold moved from 0.55 to 0.60 in 0.3. The two values are
*exactly tied* on validation accuracy, macro-F1 and false-accept rate, so
the tie was broken towards the more conservative one.

It is an **abstention control, not a safety control**. It decides when the
model says "I don't know"; it does not make an adversarial request safe.
The model's errors on adversarial input are confident: 66 of the 90 false
accepts sat in the 0.9–1.0 confidence band, with a median of 0.999, and no
threshold between 0.30 and 0.90 moved the adversarial numbers at all.

The full sweep is in `training/threshold-sweep.jsonl`; re-run it with
`python3 training/sweep_threshold.py`. The threshold is an inference-time
setting, so you can re-score a checkpoint at another value without
retraining:

```bash
aaron learning evaluate --threshold 0.75
```

### `aaron learning evaluate` reports a much worse number than `aaron brain train`

They are scoring different sets. `brain train` reports the **test split** of
the configured dataset; `learning evaluate` scores the same split by default,
but with `--dataset` or `--benchmark` it scores somewhere else.

If the numbers disagree *on the same set*, that is a bug: the deployed path
must reproduce the evaluated path, and
`the_deployed_checkpoint_reproduces_the_training_metric` fails if it does not.

### `aaron learning evaluate` picked the wrong checkpoint

It used to. Candidates are now sorted by modification time, and the three most
recent are kept, so the newest is the one selected. If it still looks wrong,
pass `--candidate <path>` explicitly.

### `aaron dataset check` reports near-duplicates

That is a warning, not a failure: the check exits non-zero only for
*blocking* problems — a duplicate id, duplicate text, cross-split leakage or
empty text.

Near-duplicates are inherited from the original dataset and are preserved
rather than deleted, because they are approved data. They cost capacity and
buy nothing, and the answer is to add semantically distinct examples rather
than to remove the old ones. See [dataset-schema.md](dataset-schema.md).

### `aaron learning analyse` says a mistake is `out_of_schema_accepted`

That is the dangerous direction: a non-command was accepted as a command.
Every one of those is a request that could run the wrong tool. The remedy is
a hard negative — an example of that shape labelled `unknown`.

### The training numbers move between runs

They will. The measured seed variance is 8.5 points of macro-F1 on the locked
benchmark — three seeds spanned 0.669 to 0.754, which is pure luck. `aaron
brain train` trains several seeds and keeps the best one **by validation
score**, and prints every seed's result so the spread is visible rather than
hidden.

The 0.3 release build was trained with `--seeds 8`; the default is 3. Three
is enough to catch a bad seed and not enough to be confident in the winner.
Use 8 when you are producing a release candidate, and read a large gap
between the best and the worst seed as a fact about the data rather than
about the seed.

### `checkpoint architecture is ... but this build expects ...`

A checkpoint from a different AARON version. The architecture fingerprint
includes every dimension. Retrain:

```bash
aaron brain train
```

### `checkpoint file is truncated` / `failed its SHA-256 check`

The file is corrupt or was interrupted mid-write. Deleting it and retraining is
safe; AARON writes checkpoints to a temporary file and renames, so a good
checkpoint is never replaced by a partial one.

### `dataset leakage: N normalised utterance(s) appear in more than one split`

Two examples that differ only in punctuation or case. The split is derived from
the example id, so fix the duplicate text — renaming an id also moves it between
splits, which is a reviewed change.

---

## Permissions

### An action I approved is being refused

Check, in order:

1. `aaron permissions show` — is the tool even listed?
2. `aaron permissions test <tool>` — what is the verdict and why?
3. Was the session non-interactive? A confirmation-gated action is **declined**,
   not assumed, when there is no terminal.
4. Is the tool's `requires_confirmation` set, or is it in
   `security.always_confirm`?

### `I will not do that: ... allow_network = false`

Network tools need `permissions.limits.allow_network = true` *and* a policy
entry. Set both deliberately.

---

## Logs

### `aaron logs` is empty

The audit log records *executed* actions. `aaron ask "hello"` runs no tool, so
nothing is recorded. `--no-audit` disables it entirely.

### The audit log grows large

Each record is small and redacted. Rotate it:

```bash
mv ~/.local/state/aaron/data/audit/audit.jsonl{,.old}
```

### A log I am about to share contains something sensitive

Values are redacted (`token=`, `api_key=`, URL userinfo, `Bearer`) and truncated
to `limits.max_audit_value_chars`. It still records *which* tool ran and its
arguments, so read it before sharing.

---

## Running as a service

A `systemd --user` unit is provided at `systemd/aaron.service`. It is **not**
enabled automatically.

Enabling a service that opens a microphone at boot is a privacy decision, so it
is yours to make:

```bash
cp systemd/aaron.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now aaron
systemctl --user status aaron
journalctl --user -u aaron -f
```

The unit runs `aaron listen --seconds 0 --voice-confirm`:

* `--seconds 0` means "listen until stopped".
* `--voice-confirm` asks a confirmation question **out loud** and listens for
  the answer, because a service has no terminal to ask on. Without it every
  confirmation-gated action is declined, which is safe but makes the
  assistant useless for exactly the actions people use it for.

### The service starts and immediately stops

It used to. The unit ran `aaron run`, which is the *text* assistant: it reads
commands from stdin, and a service has no terminal, so it read EOF at once
and exited. The unit now runs `aaron listen`.

### The service hears nothing

Check, in this order:

```bash
aaron doctor                       # is a recogniser installed and ready?
systemctl --user status aaron      # is the unit even running?
journalctl --user -u aaron -f      # what does it say?
```

* **No recogniser.** The wake word is matched against the *transcript*, so
  with no recogniser installed the assistant can never be woken. Install
  Phonon-2 or whisper.cpp.
* **The wrong capture backend.** On a PipeWire desktop, ALSA's `default`
  device is the playback endpoint, so the microphone delivers silence. Build
  with `--features aaron-audio/capture-pipewire` and check that
  `aaron doctor` names the PipeWire host.
* **Microphone permission.** PipeWire grants it through the desktop's own
  "Application Permissions" settings, not through systemd.

### Every confirmation-gated action is refused

That is the safe default when nothing can ask the question: a confirmation
that cannot be heard cannot be answered, so it is refused.

* Install a synthesiser that can be heard — piper or `espeak-ng`. The
  `text-only` backend prints replies instead of speaking them, so the
  question is silent.
* Keep the Piper voice **outside `/tmp`**. The unit sets `PrivateTmp=true`,
  which gives the service its own empty `/tmp`, so a voice file kept there is
  invisible to it. `~/.local/share/aaron/` is a better place.
* `aaron listen --voice-confirm` in a terminal prints a warning when the
  synthesiser is `text-only`, which is the fastest way to see this.

To remove the service:

```bash
systemctl --user disable --now aaron
rm ~/.config/systemd/user/aaron.service
```

---

## Reporting a bug

Include:

```bash
aaron --version
aaron doctor
uname -sr
```

and the exact command and the exact output. The output of `aaron doctor` alone
usually identifies the cause.