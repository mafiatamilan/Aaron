# AARON 0.2 — Findings, risks, and plan

This document records what was measured before any change was made, what the
measurements mean, and what will be done about it.

Nothing here is a plan to reach a number. The numbers are the baseline, and the
baseline is reproduced first so that every later claim can be checked against it.

---

## 1. Baseline reproduction

Reproduced on 2026-10-19, Arch Linux, AMD Ryzen 5 7535HS (12 cores), 22.7 GiB
RAM, RTX 2050 (4096 MiB), driver 615.78.08, CUDA UMD 13.4, rustc 1.90.0.

The repository is **not** a git repository (`git status` → `fatal: not a git
repository`). Nothing is committed, so "recent changes" cannot be reconstructed
from history. State before this phase was copied to
`/tmp/opencode/aaron-0.2-backup/` before anything was touched.

### 1.1 Gates

| gate | result |
|---|---|
| `cargo fmt --all --check` | pass |
| `cargo clippy --workspace --all-targets --all-features -- -D warnings` | pass |
| `cargo test --workspace` | **507 passed, 0 failed** (29 test binaries) |
| `cargo build -p aaron-app --no-default-features` | pass |

### 1.2 Model metrics (release build, seed 2690307351, 100 epochs)

Reproduced from a clean `AARON_HOME`, so nothing was inherited:

```
test accuracy 0.7846  macro-F1 0.7899  unknown rate 0.200
regression suite: 41/42 passed
  FAIL `how do i bake sourdough bread` expected unknown got help
REJECTED:
  - macro-F1 0.7899 is below the required 0.9200
  - accuracy 0.7846 is below the required 0.9300
```

Matches the reported history.

### 1.3 Benchmarks

| metric | reported | reproduced |
|---|---|---|
| intent inference median | 35 µs | 35 µs |
| intent inference p95 | 56 µs | **38 µs** |
| wake phrase match median | 67 µs | 65 µs |
| end-to-end text turn median | 22 µs | 22 µs |
| idle RSS | 6–9 MiB | 8.6 MiB baseline, 9.2 MiB peak |
| GPU memory delta | 0 | 39 MiB → 39 MiB, delta 0 |

**Discrepancy: intent inference p95 is 38 µs here, against 56 µs reported.**
The median agrees exactly, so this is not a machine or build difference — p95 is
dominated by scheduler noise and the earlier figure was taken from a run with
more contention. Both are recorded; neither is treated as "the" number, and the
comparison table in §10 reports both runs.

### 1.4 Stress

Five workloads × 20,000 iterations: all `stable`. Peak RSS 9.4 MiB.

---

## 2. Where the 0.790 macro-F1 comes from

Not from a weak architecture. From **contradictory supervision in the dataset**.

### 2.1 The `open the X` family — the noun, not the verb, carries the intent

**Correction to an earlier reading of this data.** A first pass suggested the
dataset contained mislabelled examples — `read_file` rows naming applications
such as "open the file manager" and "open the calculator". Checking the rows
directly showed that is **not** true: every `open the <application>` row is
correctly labelled `start_application`, and every `open the <file>` row is
correctly labelled `read_file`. There is no label error here.

The real defect is different, and the check is worth stating precisely because
the mistake is easy to make: in this family the **verb is identical and only
the noun distinguishes the intent**. `open` appears 12 times under
`read_file` and 14 times under `start_application`.

That makes noun generalisation the whole problem, and the noun space is too
small for it to work:

| intent | distinct nouns | examples |
|---|---|---|
| `read_file` | 41 (mostly with extensions: `config.toml`, `main.rs`, `readme.md`) | 27 |
| `start_application` | 37 (mostly bare: `terminal`, `gimp`, `spotify`) | 20 |

The two noun spaces overlap in *form* but not in *vocabulary*, and the model
cannot learn a general rule from a handful of examples per noun. It memorises
nouns. The held-out example `open the gitignore` names a noun that appears
**nowhere** in the `open the …` training position (the dataset has `.gitignore`
and `find files called .gitignore`, but never `open the gitignore`), so the
model has no evidence about its class and falls to the wrong one.

**The fix is coverage, not correction**: many more bare, extensionless
filenames (`gitignore`, `procfile`, `gemfile`, `editorconfig`) and many more
bare application names, so the model learns the *shape* of each noun class
instead of the individual nouns. The same reasoning explains
`locate the tests directory contents` → `run_tests` at 0.72: `locate` appears
only 7 times in the whole dataset and only under `search_files`, while `tests`
appears 38 times under `run_tests`.

### 2.1b Verb coverage is the other half

Counted across the dataset:

| intent | distinct verbs | dominant |
|---|---|---|
| `cancel` | 5 | `cancel` ×19, `stop` ×7 |
| `run_tests` | 3 | `run` ×38 |
| `open_url` | 2 | `open` ×20 |
| `stop_application` | 2 | `stop` ×10, `kill` ×10 |
| `system_status` | 6 | `check` ×14 |
| `search_files` | 3 | `find` ×23, `search` ×13, `locate` ×7 |
| `remember` | 1 | `remember` ×47 |

`remember` has exactly one verb. `open_url` and `stop_application` have two.
A classifier with one verb per intent cannot be robust to phrasing, and the
held-out set contains phrasings the training set never shows.

### 2.1c The filler-word problem

77 near-duplicate pairs, concentrated exactly where the coverage is thinnest:

| intent | near-duplicate pairs |
|---|---|
| `cancel` | 24 |
| `recall` | 19 |
| `greeting` | 16 |
| `run_tests` | 14 |
| `remember` | 3 |
| `start_application` | 1 |

The pairs are filler-word variants — `cancel` / `cancel it` / `cancel that` /
`cancel please` / `cancel everything`, all at containment similarity 1.0. They
inflate the apparent size of those classes while adding no linguistic
diversity, so the model learns a small set of tokens and nothing about the
phrasings it will actually meet.

The same shape appears elsewhere:

* `show me the file main.rs` (read_file) vs `show me the resources`
  (system_status) vs `show git status` (git_status) — three intents on `show me …`.
* `cancel` recall is **0.200** because half its training phrasings are
  `forget …` / `nevermind`, which collide with `remember` (`forget my last
  request`, `forget it all`) and with `unknown`.

### 2.2 Entity coverage is lopsided

Only 4 of 14 intents contain any example with a literal entity:

| intent | entity-bearing examples |
|---|---|
| open_url | 39 |
| search_files | 29 |
| read_file | 27 |
| remember | 3 |
| **the other 10 intents** | **0** |

So `start_application` has never seen a sentence with a real app entity in a
context it must be distinguished from a filename, and `cancel` has never seen a
hard case at all.

### 2.2b Five intents are not actually being measured

The stratified split guarantees every intent has *some* hold-out examples, but
not enough to be meaningful. Test support below 5 is reported by the new
quality pipeline as *unmeasured*:

| intent | test support | |
|---|---|---|
| `git_status` | 3 | unmeasured |
| `run_tests` | 4 | unmeasured |
| `open_url` | 4 | unmeasured |
| `start_application` | 4 | unmeasured |
| `stop_application` | 4 | unmeasured |
| the other nine | 5–6 | measured |

So a quarter of the classes contribute metrics that are one example away from
being 0% or 100%. `git_status` at F1 0.571 rests on three examples. The
macro-average is being computed over a mixture of solid and anecdotal numbers,
and it is not obvious from the report that this is so.

### 2.3 Confidence is not calibrated to the decision

Of the 14 test-set errors, **6 have the correct intent as top-1** but fall
below the 0.55 threshold and are reported as `unknown`:

| text | top-1 | top-1 prob | reported |
|---|---|---|---|
| `cancel immediately` | cancel | 0.43 | unknown |
| `that's enough` | help | 0.42 | unknown |
| `good day` | greeting | 0.47 | unknown |
| `terminate the drawing app` | stop_application | 0.26 | unknown |
| `show the repo state` | system_status | 0.55 | unknown |
| `recall the wifi name` | system_status | 0.50 | unknown |

The threshold is doing real work here; it is not a free parameter to move.
Lowering it to recover six examples would immediately increase false accepts on
out-of-schema speech, which is the direction that matters for safety. The right
fix is better data, and a *measured* calibration report so the threshold's
effect is visible rather than guessed.

### 2.4 The safety-relevant failures

Two errors are in the direction that can run the wrong tool:

| text | predicted | confidence |
|---|---|---|
| `open the gitignore` | start_application | **0.96** |
| `locate the tests directory contents` | run_tests | **0.72** |
| `how do i change the wallpaper` | system_status | 0.80 |

`open the gitignore` is a *filename* predicted as *an application launch*, at
0.96 confidence — caused directly by the `open the X` ambiguity in §2.1. A
confident misclassification of a file as an app launch is exactly the failure
this project's security posture is supposed to prevent, and it is a data defect,
not a policy defect.

### 2.5 Source of the 0.92 gate

`crates/aaron-core/src/config.rs` → `LearningConfig::default()`:
`min_macro_f1: 0.92`, `min_accuracy: 0.93`, `auto_promote: false`.

Enforced in `crates/aaron-learning/src/lib.rs` →
`LearningEngine::evaluate_candidate`, which appends a failure for each of:
macro-F1 below 0.92, accuracy below 0.93, failure to beat the live model by
`min_improvement`, and any failing regression case. `EvaluationReport::approved()`
is `failures.is_empty()`.

**Not** currently checked (the 0.2 brief requires these, so they will be added
as *additional* gates — never as relaxations of the existing ones):
per-intent minimum recall, per-intent evaluation coverage, false-accept rate on
out-of-schema text, and deployed-path agreement with the evaluated path.

### 2.6 The failing regression case

`how do i bake sourdough bread` → predicted `help` (0.42), expected `unknown`.

Diagnosis: the same sentence appears in the **validation** split of the dataset
labelled `unknown`, so the model has seen the pattern. It still prefers `help`
because `help` is trained on broad "how do I…" phrasings
(`how do you work`, `how do i work`). This is intent overlap between `help`
(which legitimately answers capability questions) and `unknown` (which must
reject general knowledge questions).

Correct fix: more out-of-schema examples beginning with `how do i`, so the model
learns that the *subject* decides, not the interrogative form.

**Not fixed by** editing the regression case, relabelling it, or removing it.

---

## 3. Preprocessing and inference consistency — checked, and consistent

Verified rather than assumed:

* Both training (`classify_one` → `FeatureBatch::from_texts`) and deployment
  (`NeuralClassifier::logits` → `FeatureBatch::from_texts`) enter through the
  single function `aaron_brain::model::encode`. There is one implementation.
* Both call `logits_to_intent(&raw_logits, temperature, threshold)`. The
  double-softmax defect found in 0.1 was fixed and is pinned by
  `the_deployed_checkpoint_reproduces_the_training_metric`.
* `train()` sets `cfg.confidence_threshold` and `cfg.temperature` from the
  live config, so the held-out metric is scored at the deployed threshold.
* The deployed accuracy was re-measured at 0.7846 against a trained metric of
  0.7846 — identical.

**One real defect found:** `brain.max_features` does not control model input.
The sequence length is the compile-time constant `SEQ_LEN = 96`, and
`max_features` is used only to bound *entity extraction*
(`max_features * 8` characters). A configuration key whose name implies it
controls model input, but which does not, is a trap. To be documented and
corrected without changing model behaviour.

---

## 4. Risks

| risk | why it matters | mitigation |
|---|---|---|
| Tuning against the held-out test set | would make the headline number meaningless | freeze the v1 test split as a **locked benchmark**; iterate only on the v2 dev/validation splits |
| "Improving" macro-F1 by weakening the threshold | moves errors from false-reject to false-accept | threshold is not a tuning knob; calibration is *measured and reported* instead |
| Label noise in generated data | synthetic examples that inherit the `open the X` ambiguity would make things worse | generated examples are marked, deduplicated, and reviewed; ambiguous families are made unambiguous rather than duplicated |
| A larger model | changes the 1,000,000 budget contract | budget is fixed and asserted; accuracy work is data-side |
| Regressions in safety | 0.1 added 26 adversarial tests | all must keep passing; new hard-negative tests added alongside |
| User data | memories, audit log, checkpoints | backed up; work uses `AARON_HOME` isolation; no destructive writes |

---

## 5. Plan

### Phase A — data quality pipeline (§3)

`aaron-brain::dataquality`, exposed as `aaron dataset check` / `aaron dataset report`:

schema and label validation · exact and near-duplicate detection (word
3-gram Jaccard) · cross-split leakage · class distribution · per-intent counts
per split · intents with insufficient hold-out coverage · dataset content hash
and version. Every check has a test, including a test that a deliberately
corrupt dataset is rejected.

### Phase B — error-analysis workflow (§4)

`aaron-brain::analysis`, exposed as `aaron learning analyse`, emitting **JSON**
and **Markdown**: accuracy, macro-F1, per-intent precision/recall/F1, confusion
matrix, per-split class counts, expected calibration error and reliability bins,
unknown precision/recall, false-accept and false-reject rates, entity-extraction
error rate, regression results, and representative misclassifications grouped
into actionable categories (intent overlap, missing coverage, entity extraction,
out-of-schema, low confidence, preprocessing mismatch, label error).

**Redaction**: misclassification samples are scrubbed with the existing
`redact_secrets` and truncated. No transcript or memory content is ever read
into a report.

### Phase C — dataset v2 (§3)

1. Freeze `training/dataset/v1.jsonl` and its split as the locked benchmark.
2. Build `training/dataset/current.jsonl` (v2) from:
   * v1 examples, unchanged, with provenance preserved;
   * **disambiguated families** — every `open …`, `show …`, `locate …` example
     made resolvable by cue, never left contradictory;
   * **entity coverage for all entity-taking intents**;
   * **hard negatives** — out-of-schema text that *looks* like a command;
   * **conversational, filled and misspelled** variants;
   * **explicit minimal pairs** for every confusable intent pair.
3. Every added example is marked `source: "generated"` or `"authored"` and is
   deduplicated and validated before training.
4. Review pass: ambiguous families are listed and each is assigned a single
   ruling, applied consistently.

### Phase D — promotion gate hardening (§6)

Add, never replace: per-intent minimum recall, per-intent hold-out coverage,
false-accept ceiling on out-of-schema text, and deployed-path agreement.

### Phase E — security hard negatives (§5, §9)

New adversarial tests for prompt injection in files and transcripts, tool-name
imitation, confirmation bypass, malformed and conflicting entities, low-confidence
predictions, and corrupted checkpoints.

### Phase F — resource and voice verification (§7, §8)

Measure text-only, ASR, TTS and combined voice separately. Report honestly what
the environment can and cannot verify: capture runs against a **real ALSA
device**, but the device is a **null source** (peak −120 dBFS), and **no speech
model or synthesis engine is installed**, so ASR and TTS stay unverified.

### Phase G — benchmarks, docs, CI (§10, §11)

Re-run the full suite under identical conditions, publish a before/after table,
update README and the docs, and extend CI with the new checks.

---

## 6. What actually happened

### 6.1 Accuracy, measured on a benchmark nothing was tuned on

The **locked benchmark** (`training/evaluation/benchmark-v3.jsonl`, 130
examples) was authored *after* the diagnosis but *without* looking at how any
model scored on it, and it is disjoint from the training data by construction
— the build script drops any training row that collides with it.

| model | locked benchmark accuracy | macro-F1 |
|---|---|---|
| 0.1 release (v1 dataset, lr 3e-3) | 0.6154 | 0.6319 |
| + new dataset (v2), lr 3e-3 | 0.6846 | 0.7132 |
| + rebalanced dataset, lr 1e-2 | **0.7538** | **0.7703** |

**+13.8 points of accuracy and +13.8 points of macro-F1 on a benchmark that
was never trained on and never selected on.** The three rows are
single-variable changes: the dataset, then the learning rate.

For continuity with the 0.1 release, the same model on the **v1 hold-out**:
accuracy 0.7846 → **0.8923**, macro-F1 0.7899 → **0.8943**, and the regression
suite went from 41/42 to **42/42**. That hold-out is *not* a clean estimate,
because it was used to diagnose the 0.2 work; the locked benchmark is.

### 6.2 The promotion gate still refuses, and now says why

```
REJECTED:
  - macro-F1 0.7703 is below the required 0.9200
  - accuracy 0.7538 is below the required 0.9300
  - false-accept rate 0.2800 is above the maximum 0.0500:
    out-of-schema requests were accepted as commands
```

The third line is new. The gate gained a safety check, not a relaxation, and
it is the check that matters most: **28% of out-of-schema requests are
accepted as commands.** That is the number to fix next.

### 6.3 What the error analysis found, and what it cost

The systematic analysis (`aaron learning analyse`) is what drove the work.
Two findings changed the direction of the project:

**1. The model memorises.** Train accuracy was exactly **1.0000** with
validation at 0.83. With 1,000,000 parameters and 968 training examples, and
860,160 of those parameters in an embedding table, that is unsurprising.

**2. Regularisation does not help.** Dropout, input-feature dropout and label
smoothing were all implemented — none adds a parameter, so the budget is
untouched — and all were swept:

| regularisation | validation macro-F1 | locked benchmark macro-F1 |
|---|---|---|
| none | **0.8362** | **0.7703** |
| hidden dropout 0.1 | 0.7709 | 0.7227 |
| hidden dropout 0.3 | 0.7707 | 0.6329 |
| feature dropout 0.2 | 0.7981 | 0.7063 |
| feature dropout 0.3 | 0.7991 | 0.6853 |
| label smoothing 0.05 | 0.7996 | 0.7706 |
| label smoothing 0.2 | 0.7972 | 0.7657 |
| all three | 0.7959 | 0.6694 |

Every variant is worse or equal. The reason is visible in the numbers: the
train→validation gap is 16 points, but the validation→benchmark gap is a
further 8 points and is **not** a memorisation effect. It is a distribution
difference between the dataset's phrasing and real phrasing. Regularisation
attacks the first gap and leaves the second one untouched, so it costs
accuracy without buying generalisation.

**The regularisation code stays in the tree, off by default.** A negative
result that is reproducible is worth keeping: the next person to try it will
find the sweep in `training/experiments.jsonl` instead of spending a week
rediscovering it.

### 6.4 The parameter budget is unchanged

1,000,000, verified from the architecture. Dropout adds no tensors, label
smoothing changes no tensors, and the count test still passes.

### 6.5 Seed variance is large, and is now handled

Three seeds spanned 0.669 to 0.754 on the locked benchmark — 8.5 points of
pure luck. `aaron brain train` now trains several seeds and keeps the one
with the best **validation** score. Selection never touches the test split or
the benchmark, and the per-seed results are printed.

### 6.6 Bugs found and fixed in this phase

| bug | consequence | how it was found |
|---|---|---|
| `learning evaluate` picked an arbitrary candidate | you could evaluate a stale checkpoint and believe you evaluated the new one | a stale candidate scored 0.0667 while `brain train` reported 0.7667 |
| checkpoints sorted by filename | `e60` outranked `e200` | the same symptom |
| label smoothing softmaxed over the **batch** axis | the model trained to 0.13 accuracy | the sweep printed a catastrophic number |
| the scrubber removed only the role prefix, not the line | `assistant: acknowledged, running the test suite` survived | a hard-negative test asserted on it |
| `system:` as a substring marker | AARON's own `System status:` heading was deleted from every reply | the same hard-negative suite |
| the promotion gate ignored coverage and false accepts | a model could be promoted with a class that was never measured | reading the gate against the brief |

### 6.7 The remaining blocker, stated honestly

0.92 macro-F1 is **not reached**, and the evidence says why:

* the model memorises the training set completely, and
* the gap to the benchmark is a *phrasing* gap, not a capacity gap, and
* the dominant remaining error is **false accepts**: 28% of out-of-schema
  requests are accepted as commands.

The next data-collection priorities, in order:

1. **Out-of-schema examples that look like commands, at scale.** The false-accept
   rate is the number that matters and it is 28%. Every false accept is a
   request that could run the wrong tool.
2. **Phrasing diversity in the training data** — the validation→benchmark gap
   is 8 points, and it is entirely about how the same request is worded.
3. **More examples for the classes with the worst recall** on the benchmark:
   `greeting` (0.375), `git_status` (0.375), `system_status` (0.556).
4. **Fewer near-duplicates.** The dataset still has 137 near-duplicate pairs,
   all inherited from v1 and all preserved rather than deleted. They cost
   capacity and buy nothing.

None of these is an architecture change, and the parameter budget does not need
to move.

---

## 7. Expected outcome, stated honestly

The gap to 0.92 macro-F1 is **not** primarily a capacity gap. The evidence for
that is the confusion structure in §2: the errors are label contradictions and
coverage holes, not underfitting of the model.

Whether 0.92 is *reachable* is an empirical question that this phase will
answer by measurement, not prediction. If it is not reached, the outcome will be
a documented negative result with the specific data that would be needed —
which is a more useful deliverable than a lowered threshold.