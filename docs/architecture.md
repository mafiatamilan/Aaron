# Architecture

AARON is a Rust workspace of fourteen crates with one dependency direction:
everything depends on `aaron-core`, and nothing depends on `aaron-app`.

```
                    ┌─────────────┐
                    │ aaron-app   │  CLI, lifecycle, exit codes
                    └──────┬──────┘
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 ┌────────────┐   ┌──────────────┐   ┌──────────────┐
 │ aaron-agent│   │aaron-learning│   │ aaron-bench  │
 └─────┬──────┘   └──────────────┘   └──────────────┘
       │ classifies / dispatches
       ├──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼
 ┌──────────┐  ┌────────────┐  ┌───────────┐  ┌────────────┐
 │aaron-    │  │aaron-tools │  │aaron-     │  │aaron-      │
 │ brain    │  │            │  │ security  │  │ resources  │
 └──────────┘  └─────┬──────┘  └─────┬─────┘  └────────────┘
                    │               │
                    └───────┬───────┘
                            ▼
                      ┌───────────┐
                      │aaron-memory│
                      └───────────┘

 voice:  aaron-audio ─► aaron-asr ─► aaron-wake ─► aaron-agent ─► aaron-tts
```

---

## One turn, end to end

`aaron-agent` is the only component that can ask a tool to do anything.

1. **Idle → Classifying.** The utterance is clipped to
   `resources.max_transcript_chars` *before* anything reads it.
2. **Classify.** `aaron-brain` produces an intent and entities. With a promoted
   checkpoint that is the 1M-parameter model; otherwise it is the deterministic
   keyword router. The chosen backend is reported in every response, so the user
   always knows which one spoke.
3. **Entities.** `aaron_core::text::extract_entities` produces bounded strings.
   `Entities::validate` rejects anything oversized or containing a NUL byte.
4. **Arguments.** `build_arguments` turns entities into a JSON object, re-checking
   lengths. A missing argument produces a *refusal with a suggestion*, not an
   exception.
5. **Loop guard.** `LoopGuard` admits or refuses the call before the policy
   engine is even consulted, so a runaway cannot ask for approval repeatedly.
6. **Policy.** `aaron_security::PolicyEngine::evaluate` takes facts supplied by
   the **tool registry**, not by the classifier. A misclassification therefore
   cannot escalate privilege.
7. **Confirm.** If the policy says `ask`, the prompt shows the exact tool and
   arguments. A non-interactive session refuses rather than assuming consent.
8. **Execute.** `ToolRegistry::execute` re-validates arguments, then the tool
   runs under the returned `ExecutionLimits`.
9. **Audit.** Three records: requested, authorised, result.
10. **Respond.** Built from the *real* result. Anything read from a file or a
    process passes through `untrusted::scrub` first.
11. **Context.** The turn is appended to a bounded `ConversationContext`.

---

## The state machine

```
Idle ──► Classifying ──► PolicyCheck ──► AwaitingConfirmation ──► Executing ──► Responding ──► Idle
  ▲            │              │                 │                    │
  │            ▼              ▼                 ▼                    ▼
  └──────── Listening ◄── Error ◄────────── Cancelled ◄──────────────┘
```

States are an enum with a validated transition table
(`aaron_agent::state::StateMachine::is_allowed`), not a set of booleans. An
illegal jump is a test failure rather than an emergent property.

`State::is_listening()` drives the microphone indicator, so the UI can never
claim to be live while the agent is idle.

---

## Why the model only classifies

A 1M-parameter model over 14 intents is a good *router* and a bad everything-else.
So:

* the model chooses one of 14 labels;
* argument construction is ordinary Rust;
* the tool registry is a fixed set of `&'static str` ids;
* the policy engine is deny-by-default;
* execution uses fixed argument vectors, never a shell.

A model prediction cannot name a program, because there is no code path from a
prediction to a program. `no_tool_accepts_a_free_form_command` asserts that
structurally, by walking every declared argument of every registered tool.

---

## Dependency injection

Every boundary that could need replacing is a trait:

| Trait | Implementations | Why |
|---|---|---|
| `IntentClassifier` | `NeuralClassifier`, `HeuristicClassifier` | swap the brain without touching the agent |
| `AudioSource` | `CpalSource`, `SyntheticSource`, `FileSource`, `NullSource` | test the voice loop with no microphone |
| `SpeechRecognizer` | `WhisperAsr`, `ConstrainedAsr`, `DisabledAsr` | text mode works with no ASR installed |
| `SpeechSynthesizer` | espeak-ng, piper, spd-say, text-only | degrade instead of failing a turn |
| `WakeDetector` | transcript matcher, energy burst, disabled | different hardware, same agent |
| `Confirmer` | stdin prompt, `--yes`, `--no`, non-interactive decline | tests never block on a terminal |
| `Tool` | eleven built-ins | a new tool is a registration, not a fork |

This is why the test suite runs on a machine with no GPU, no microphone and no
model files — and still covers the whole pipeline.

---

## Where memory goes

| Component | Bound |
|---|---|
| Intent model | 1,000,000 f32 = 4 MB, resident (it is small and latency matters) |
| Optional ASR/TTS models | loaded on demand; the ASR model is released when idle, the TTS model is pinned by a lease |
| Audio | `sync_channel` sized from `resources.max_audio_buffer_seconds` |
| Conversation context | turn count **and** character budget, both enforced |
| SQLite | page-count-bounded `cache_size`, retrieval capped per query |
| Audit log | append-only, each value redacted and truncated |

Full detail in [resources.md](resources.md).

### A model that outlives its borrow

`ModelGuard` borrows the slot, so its model must be dropped before the slot
can be released. Some callers need the opposite: an owned `Arc` that outlives
the borrow, because they hand it to a component with its own lifetime — the
`Agent` owns a synthesiser.

`ModelSlot::lease()` returns a `ModelLease`, and a lease also **pins** the
model. While any lease exists the model is resident by request, not by
idleness, so the unload sweep skips it however old it is. Without that, a
long-running session would reload its speech engine on every single turn.
`AppContext` retains one TTS lease for the process lifetime, for exactly that
reason.

---

## The data pipeline

Three stages, each with its own contract:

```
training/curated/*.jsonl  ──┐
                            ├──►  build_dataset_v2.py  ──►  training/dataset/current.jsonl
training/dataset/v1.jsonl ─┘        │
                                     ├── validates: schema, duplicate ids, duplicate and
                                     │   near-duplicate text, cross-split leakage, coverage
                                     ├── assigns: stratified splits, deterministic
                                     └── reports: training/dataset/review-report.md

training/evaluation/benchmark-v3.jsonl         ──►  the locked *intent* benchmark
training/evaluation/safety-benchmark-v1.jsonl  ──►  the adversarial development set
training/evaluation/safety-benchmark-v2.jsonl  ──►  the adversarial held-out set
        (never merged; the build drops any training row that collides with any of them)
```

### The three held-out sets

| file | examples | role |
|---|---|---|
| `training/evaluation/benchmark-v3.jsonl` | 130 | the locked **intent** benchmark — unchanged since 0.2 |
| `training/evaluation/safety-benchmark-v1.jsonl` | 142 | the adversarial benchmark used to **select** the 0.3 dataset, and therefore now a *development* set |
| `training/evaluation/safety-benchmark-v2.jsonl` | 102 | the adversarial benchmark written afterwards, blind, and scored once — the held-out adversarial measure |

The two adversarial files exist because v1 was consumed by the selection. A
benchmark that selects a model is no longer a benchmark, so v2 was authored
after the dataset was chosen, and the numbers that get reported come from v2.

The two decisions worth knowing:

* **Splits are stratified by intent.** A proportional hash of the id is not
  enough: a rare intent can end up with nothing in the hold-out, and the
  metric then looks complete while one class was never measured. The
  stratification is computed once, per dataset, and every consumer reads it
  from the same place.
* **The held-out sets' disjointness is enforced by the build**, not by care. A
  training row whose text collides with any of the three held-out sets is
  dropped and reported in `training/dataset/review-report.md`. "Do not train
  on the benchmark" is exactly the kind of rule that is broken by accident
  three months later.

---

## Measuring safety

`aaron-brain/src/safety.rs` measures what a false accept actually costs.

The historical pair — the false-accept rate and the false-reject rate — is
computed **unchanged**, so the 0.2 numbers still mean exactly what they meant
and the historical rate still gates promotion. The module adds the
decomposition that a single rate hides: "hello" and "run the test suite" are
counted the same by a rate, and only one of them can cause harm.

`action_tier(intent)` splits the intents into three tiers:

| tier | intents | consequence of a false accept |
|---|---|---|
| `None` | `greeting`, `help` | nothing happens; no tool exists |
| `ReadOnly` | `system_status`, `search_files`, `read_file`, `git_status`, `recall`, `cancel` | data disclosure, bounded by the sandbox |
| `SideEffecting` | `run_tests`, `open_url`, `start_application`, `stop_application`, `remember` | a process runs, a browser opens, a process is killed, a memory is written |

Actionability is derived from the tool registry rather than from a hand-written
list, so it cannot drift away from what AARON can actually do. A test in
`aaron-brain` checks the tiers against the registry, and a test in
`aaron-agent` checks the intent-to-tool map against the same registry.

`OutcomeClass` puts every evaluated example in exactly one of five classes:

| class | condition | severity |
|---|---|---|
| `CorrectIntent` | the right intent, and the arguments validate | none |
| `CorrectIntentBadArguments` | the right intent, but the arguments would not validate | **high** |
| `IncorrectValidIntent` | a different, valid intent | medium |
| `AcceptedInvalidRequest` | out-of-schema accepted as something actionable | **critical** |
| `RejectedOrAbstained` | a valid request rejected or abstained from | low |

`analyse_safety(...)` computes the historical rates unchanged, plus the
supplementary tiered rates, coverage, the abstention rate and the five-way
breakdown. It takes an optional predicate answering "would the tool arguments
for this prediction validate?", so `CorrectIntentBadArguments` is measured
against the real validator — `build_arguments` and the entity validator — and
not against a re-implementation of it. `to_markdown(report)` renders the
result.

Two details worth keeping:

* A zero denominator is reported as **undefined**, not as perfect. An
  evaluation set with no out-of-schema examples would otherwise score a
  false-accept rate of 0.0 and look flawless.
* The argument check is only consulted for a prediction that could actually
  dispatch something. `greeting` and `help` have no arguments at all, so
  "invalid arguments" is not a meaningful class for them, and asking would
  report a defect that does not exist.

---

## The error-analysis path

```
aaron learning evaluate          # the gate: does this candidate qualify?
aaron learning analyse           # why does it not? what should be fixed?
aaron learning safety            # what does it do with adversarial input?
```

`analyse` reads the same evaluation set the gate reads and sorts every
mistake into a category with a *remedy* attached:

| category | the remedy |
|---|---|
| `intent_overlap` | explicit minimal pairs for the two intents |
| `out_of_schema_accepted` | a hard negative — the safety direction |
| `out_of_schema_rejected` | a positive example of that shape |
| `low_confidence` | more training signal, never a lower threshold |
| `missing_coverage` | add the phrasing |
| `entity_extraction` | fix the extractor, not the classifier |
| `preprocessing_mismatch` | a bug: the deployed path disagrees with the evaluated one |
| `label_error` | review with a human |

The distinction between `out_of_schema_accepted` and `out_of_schema_rejected`
is the one that matters: the first can run the wrong tool, the second is merely
unhelpful. The report redacts and truncates every text it prints, and reads no
transcript and no stored memory, so a report is always safe to share.

### The command surface

| command | what it does |
|---|---|
| `aaron learning safety` | scores a candidate against the adversarial safety benchmark, at two levels: **model level** (did the classifier refuse?) and **agent level** (did anything actually happen?) |
| `aaron learning analyse --benchmark <file>` | analyses a locked benchmark file instead of a dataset split |
| `aaron learning analyse --format json` | the same report, machine-readable |
| `aaron learning evaluate --threshold <f>` | re-scores one checkpoint at another confidence threshold — the threshold is an *inference-time* setting, so a sweep re-scores a checkpoint instead of retraining one |
| `aaron learning evaluate --safety-benchmark <file>` | gates the promotion on the adversarial benchmark |
| `aaron brain train --seeds N --class-weighting <none\|balanced\|sqrt\|N> --dropout --feature-dropout --label-smoothing` | the training sweep surface |
| `aaron dataset check` / `aaron dataset report` | schema, duplicates, leakage and per-intent coverage |

`--class-weighting` chooses how the loss weights the classes: `none`,
`balanced` (`p = 1`), `sqrt` (`p = 0.5`) or an exponent — the `ClassWeighting`
enum in `aaron-brain/src/train.rs`. The weight for a class with `n_c` examples
is `w_c = (mean / n_c) ^ p`, normalised so the mean weight over the training
examples is exactly 1. That normalisation keeps the loss on the same scale as
the unweighted loss, so the learning rate does not have to be retuned and the
experiment changes one thing rather than two. It is **off by default**: the
0.3 sweep found it did not help.

### The gate

`aaron learning evaluate` is the gate. In release 0.3 it checks, in addition to
everything it checked before:

* per-intent hold-out coverage — an intent with fewer than five hold-out
  examples is reported as *unmeasured* and blocks the promotion;
* a per-intent recall floor;
* the side-effecting false-accept rate;
* the adversarial safety benchmark refusal rate;
* the adversarial safety benchmark tool-dispatch rate;
* and that the safety benchmark **was scored at all**. A missing benchmark
  refuses the promotion.

None of these relaxes an existing threshold. Each one can only refuse.

---

## What release 0.3 changed

On the held-out adversarial benchmark the model refuses **48.0%** of adversarial
input, against 37.3% in 0.2. At the *agent* level a tool was dispatched for
**0 of 102** adversarial examples, and **0** mutating actions were performed —
in both releases.

The architecture and the 1,000,000-parameter budget are unchanged. What
changed is the measurement, the data, and the number of things that can refuse
a promotion. The full account is in
[release-0.3-report.md](release-0.3-report.md).

---

## What release 0.5 changed

Release 0.5 added a **readiness verdict**: a second, typed output that says
whether a request is specific enough to act on. The intent classifier keeps
deciding *what* the user wants; readiness says *whether the request says
enough to do it*.

`aaron_core::intent::Readiness` has four states — `Ready`,
`NeedsClarification`, `NotApplicable` and `Undetermined` — and rides on
`aaron_core::Command` as one field. `Undetermined` is the default, including
for a payload serialised before the field existed: a missing verdict grants
nothing, and `Ready` would be a claim the payload never made.

The rule that produces it lives in `aaron_core::readiness`, in the crate both
the classifier and the agent depend on, so the two cannot disagree about which
arguments an intent needs. `crates/aaron-app/tests/readiness.rs` asserts that
the rule and the agent's argument builder agree on every intent across 48
utterances.

**The agent consumes it through a one-way valve.** The check sits after the
argument builder and before the loop guard, so it can only turn "proceed" into
"ask"; it can never turn "refuse" into "proceed", and it never authorises
anything. The policy engine, the allow-list and the confirmation gate still
decide what runs.

Release 0.5 also **measured** a learned version of the verdict — a
169-parameter head reading the model's pooled representation — and rejected
it. The head is a better classifier of the task than the rule (F1 0.683
against 0.471) and the wrong thing to ship: it is dominated by the rule at
every threshold, and the dispatch-rate reduction it buys on adversarial input
costs the refusal of 86 % of good requests. It also costs 169 parameters over
the budget and 40 % more latency per classification. The full account is in
[release-0.5-report.md](release-0.5-report.md).

---

## Notable design decisions

Recorded with reasoning in [decisions/](decisions/). The short version:

* **Candle over Burn** for the model framework — smaller dependency surface,
  direct tensor control, and exact parameter counting from real tensors.
* **Hashed n-gram features over a learned tokenizer** — no unfitted vocabulary,
  therefore no train/test leakage, and reproducible feature maps.
* **RMS pooling over max pooling** — measured: max pooling collapsed test
  accuracy from 0.75 to 0.07 because its scale swamps the mean under
  LayerNorm.
* **Path canonicalisation over lexical checks alone** — the only check that
  actually defeats symlink escapes.
* **AARON's own model on CPU** — at 1M parameters, host-device transfer costs
  more than the matmul saves.
* **The readiness verdict is a typed field consumed through a one-way valve** —
  a verdict that could only subtract permission cannot weaken the
  authorisation boundary, and a rule that lives in `aaron-core` cannot drift
  between the classifier and the agent. See
  [decisions/0008](decisions/0008-readiness-as-a-one-way-valve.md).
* **Regularisation that measured worse stays in the tree, off** — dropout,
  input-feature dropout and label smoothing are implemented, and the sweep
  says they do not help at this dataset size. Class weighting is the same
  story: it is implemented, it is normalised so the loss scale does not move,
  and the sweep says it does not help either. A reproducible negative result
  is worth more than a missing feature.
* **A safety metric that separates the harmless from the dangerous** — the
  historical false-accept rate counts a general-knowledge question answered
  "hello" the same as a request to run a test suite. The historical rate is
  kept unchanged and still gates promotion, but it is now reported beside the
  tiered decomposition, so it is visible which failures can actually hurt.
* **A pinning lease, not a longer idle window** — the speech engine was being
  reloaded every turn because it was released the moment it went idle. A lease
  pins the model *by request* rather than by idleness, which is the property
  the requirement actually needs, and it is enforced by a type rather than by
  a timeout.
* **The adversarial benchmark had to be versioned** — a benchmark that
  selects a model is no longer a benchmark. `safety-benchmark-v1` chose the
  0.3 dataset and is now a development set; `safety-benchmark-v2` was written
  afterwards, blind, and scored once. The cost of the selection is recorded
  rather than hidden, and the numbers that get reported come from v2.
* **A second, blind benchmark** — once a hold-out has been used to diagnose,
  it stops being a clean estimate. See `docs/release-0.2-plan.md` §6.1.