# The custom intent model

AARON's own model is a **1,000,000-trainable-parameter text classifier** over a
closed set of 14 intents. This document explains the architecture, why those
dimensions, how the dataset is built, how training is made reproducible, and
what the model actually achieves.

---

## 1. The architecture

```
utterance
  │
  ├─ hashed features: word unigrams, word bigrams, padded char trigrams
  │
  ├─ Embedding(10240 × 84)                              860,160 params
  │    masked mean-pool  ⊕  masked RMS-pool      →  168
  ├─ LayerNorm(168)                                            336 params
  ├─ Linear(168 → 754) + GELU                              127,426 params
  ├─ LayerNorm(754)                                          1,508 params
  └─ Linear(754 → 14)                                       10,570 params
                                                       ─────────────
                                                        1,000,000
```

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

Padded positions are masked, so `SEQ_LEN = 96` is not a parameter.

### The readiness head, and why it is not in the budget

Release 0.5 built a second output head: a `Linear(2D → 1)` on the **pooled**
vector — the 168-dimensional mean-and-RMS vector, computed *before* `ln1` —
producing one logit that says whether a request is specific enough to act on.
That is 168 weights and one bias, **169 parameters**, and it is *additional* to
the intent classifier:

| architecture | parameters |
|---|---|
| `aaron-intent-1m-v1:10240x84h754c14L96` (intent only) | **1,000,000** |
| `aaron-intent-1m-v1r:10240x84h754c14L96` (with the head) | **1,000,169** |

Landing on exactly 1,000,000 with the head included means giving the 169 back
from somewhere else, and the only solution that keeps the embedding width is
`VOCAB = 10249, HIDDEN = 749` — a different hash space, which changes the
feature map and invalidates every checkpoint. There is no solution that keeps
`VOCAB = 10240`: in closed form `H = (999985 − 10246·D)/(2D+17)`, whose
numerator is negative for `D ≥ 98` and which has no integer solution for
`1 ≤ D ≤ 97`.

**The reallocation was not done**, because the head was measured and
**rejected**: it is a better classifier of the readiness task than the
deterministic rule (F1 0.683 against 0.471) and the wrong thing to ship. It
is dominated by the rule at every threshold, and the dispatch-rate reduction it
buys on adversarial input costs the refusal of 86 % of good requests. It also
costs 40 % more latency per classification, for 168 multiply-adds, because a
separate tensor operation costs more than the arithmetic does.

The head is nevertheless **implemented, measured and kept**, because the
measurement is the point: `crates/aaron-brain/src/readiness.rs` trains it, the
checkpoint format carries it, and the promotion gate refuses a candidate over
the budget — so a head-bearing checkpoint cannot be promoted by accident. See
[release-0.5-report.md](release-0.5-report.md).

The budget is **verified from the architecture**, not asserted in a comment:

```rust
#[test]
fn parameter_count_is_exactly_one_million() {
    assert_eq!(expected_parameter_count(), 1_000_000);
}
```

`expected_parameter_count()` sums `parameter_specs()`, the same list that drives
checkpoint validation. `aaron model` prints the same figure at runtime, and CI
runs this test on its own job.

---

## 2. Why these dimensions

`C = 14` is fixed by the closed `Intent` enum, which is a contract shared by the
brain, the agent, the dataset and the evaluator. Adding an intent changes the
head and therefore the budget, which is why the parameter count is a test.

That leaves

```
V·D + 2D + (2D·H + H) + 2H + (H·C + C) = 1,000,000
```

Searched over plausible `(V, D, H)`:

| constraint | choice |
|---|---|
| `V` a power-of-two hash space, so bucketing is cheap and uniform | `10240` |
| `D` wide enough that hashed features do not collide constantly | `84` |
| residual lands on an exact integer | `H = 754` |

Embedding storage is 86% of the budget. That is the correct trade for a
1M-parameter *text* classifier: it needs a wide input vocabulary far more than
it needs depth. A deep model on a 10240-entry bag of hashed n-grams would spend
its capacity memorising the dataset.

`V = 10240` also gives a collisions-per-feature ratio of about 8% across the
roughly 2000 distinct features in the dataset, which is acceptable and keeps
near-synonyms sharing a bucket.

---

## 3. Features

`aaron_core::text::features` produces, for each utterance:

* every word as a unigram (`w:check`),
* every adjacent pair as a bigram (`b:check the`),
* every padded character trigram of every word (`c:^ch`, `c:che`, …).

Each feature is hashed with **FNV-1a 64** into `[0, V)`. FNV-1a rather than
`DefaultHasher` because the standard library's `DefaultHasher` explicitly does
not guarantee stability across releases, and a feature map that shifts between
Rust versions would silently invalidate every checkpoint.

Why hashed n-grams rather than a learned tokenizer:

* no unfitted vocabulary, therefore **no train/test leakage**;
* deterministic and reproducible on any machine;
* character trigrams give tolerance to typos and ASR errors with no extra
  parameters.

Cost: ~8% bucket collisions, and no cross-word morphological generalisation.

---

## 4. Pooling: why RMS, not max

This was measured, not assumed.

The architecture concatenates two pooling statistics to reach `2D`. The obvious
second statistic is **max**, and it is wrong here:

* max is a single winning feature per dimension, so one unusual word decides
  the class;
* its scale is roughly an order of magnitude larger than the mean;
* LayerNorm then normalises over the whole 168-vector, shrinking the mean half
  until it carries almost no signal.

Measured test accuracy during the pooling investigation, on the shipped
dataset under the then-current (proportional, pre-stratification) split:

| second statistic | test accuracy |
|---|---|
| max, scale 1.0 | 0.067 |
| max, scale 0.5 | 0.067 |
| max, scale 0.25 | 0.067 |
| mean (max discarded) | 0.750 |
| **RMS** | **0.767** |

Those numbers are not directly comparable with the headline result below
(0.7846), which was measured later under the corrected stratified split and
with the tuned defaults. The comparison that matters here is *within* the
table: max pooling collapses to near-chance at every scale while mean-only and
RMS do not, which is a difference of two orders of magnitude and not a
split artefact.

**RMS pooling** — the per-dimension root-mean-square of the embeddings — is the
same order of magnitude as the mean, so both halves survive LayerNorm. It also
removes the masking sentinel entirely: padded rows are exactly zero after
masking, so they contribute nothing to either statistic.

Two bugs were found by this experiment and are worth recording:

1. **The sentinel had the wrong sign.** Adding `+1e9` to padded positions made
   `max` a constant, and LayerNorm's scale invariance then washed the whole
   second half out — the model trained to 70% on an effectively mean-pooled
   network and nobody knew.
2. **The sentinel poisoned the gradient.** Multiplying the mask by `-1e9`
   inside the autograd graph multiplied the incoming gradient by a billion and
   training diverged on the first step. The sentinel is now built from a
   **detached** mask, so the gradient of `max` flows only into the winning
   embedding.

---

## 5. Initialisation

Deterministic, from AARON's own SplitMix64 rather than `rand`, whose generators
are explicitly not reproducible across versions.

| tensor | init |
|---|---|
| `emb.weight` | `N(0, 0.05)` |
| `fc1.weight`, `fc2.weight` | He: `N(0, sqrt(2/fan_in))` |
| LayerNorm weight / bias | `1.0` / `0.0` |

The embedding standard deviation is much tighter than He would suggest. Its rows
are pooled by RMS as well as by mean, and a large initial scale makes the
pooled vector dominated by outliers. `0.05` keeps the two halves comparable.
`splitsmix64_matches_known_vectors` pins the PRNG against reference values.

---

## 6. The dataset

`training/dataset/current.jsonl`, **1,243 examples**, built by
`training/build_dataset_v2.py` from three sources:

| source | rows | what it is |
|---|---|---|
| `inherited` | 641 | the frozen v1 dataset, unchanged, ids unchanged |
| `curated` | 413 | hand-written, for releases 0.2 and 0.3 |
| `template` | 189 | a shared frame with a curated noun substituted |

Every row is reviewed in `training/dataset/review-report.md`, which the build
regenerates. The format and the rules are in
[dataset-schema.md](dataset-schema.md).

### What release 0.2 added, and why

The v1 dataset was generated from templates, and the quality pipeline found
three concrete defects in it.

**1. The noun space was too small to generalise over.** `read_file` and
`start_application` share the verb `open`, so only the noun separates them.
v1 had 41 file nouns and 37 application nouns, almost none extensionless, and
the held-out set contains nouns that appear nowhere in that position. The
model memorised nouns instead of learning the noun *class*. Fixed by adding
24 extensionless filenames and 46 application names, in two contexts each.

**2. Verb coverage was one to three verbs per intent.** `remember` had
exactly one verb; `open_url` and `stop_application` had two. A class with one
verb is one phrasing away from being entirely out of distribution. Fixed by
adding ten phrasings for each thin class.

**3. 77 near-duplicate pairs**, concentrated in the thinnest classes
(`cancel` 24, `recall` 19, `greeting` 16, `run_tests` 14). They were filler
variants — `cancel`, `cancel it`, `cancel that`, `cancel please` — which
inflate a class without adding linguistic diversity.

**And one finding that changed the project.** The error analysis showed that
9 of 10 mistakes shared a shape: a *frame* that several intents use, where
only the object decides. "show me …" is used by `read_file`, `git_status`,
`system_status`, `search_files` and `help`. "stop …" is used by
`stop_application` and `cancel`. "check …" is used by three intents. The
model had no way to learn that the frame carries no information, because no
two intents ever appeared with the same frame.

The fix is **contrastive frames**: the same frame, side by side, for every
intent that legitimately uses it. Sixty of them.

### Class balance

The first 0.2 build was imbalanced at **5.14×** — `start_application` had 252
examples and `git_status` had 49, and `git_status` had a recall of 0.200. The
imbalance was self-inflicted: five sentence frames over 46 application names
produced 230 rows. Two frames instead of five, plus targeted additions to the
thin classes, brought the ratio to **2.56×**.

### Splits

| split | rule | count |
|---|---|---|
| train | 80% per intent | 968 |
| validation | 10% per intent | 121 |
| test | 10% per intent | 121 |

Stratified by intent, ordered by `fnv1a64(id)` so the hold-out does not
reshuffle when examples are added, and rounded to nearest with a floor of one
— because truncating `(n*10)/100` gives *zero* hold-out examples for any
intent with fewer than ten rows, which silently removes that intent from the
measured set. That bug shipped once; a test now sweeps every group size from
5 to 40.

The v1 dataset is frozen as `training/dataset/v1.jsonl` and is the locked
historical benchmark.

### A second locked benchmark

`training/evaluation/benchmark-v3.jsonl`, 130 hand-written examples, 8–10 per
intent plus 25 out-of-schema.

It exists because the v1 hold-out was used to *diagnose* the 0.2 work, and a
set that has been used to diagnose is no longer a clean estimate of
generalisation. The v3 benchmark was written after the diagnosis but without
looking at how any model scored on it, and it is **disjoint from the training
data by construction** — the build script drops any training row that
collides with it, and `python3 training/build_benchmark.py --check` verifies
it.

### Leakage checks

`Dataset::validate` rejects duplicate ids, two utterances that normalise to
the same text, any normalised utterance appearing in more than one split, and
an intent with examples but none in the training split. The build script
refuses to write a dataset that fails any of these.

## 7. Training

Entirely in Rust. No Python at inference, at build time, or during training.

```bash
aaron brain train --epochs 100 --lr 1e-2 --weight-decay 1e-2 --seeds 3
```

| hyperparameter | value | why |
|---|---|---|
| seeds | 3 | the measured seed variance is 8.5 points; the winner is chosen on **validation** |
| seed | `2690307351` | the base seed; the others are `seed+1`, `seed+2` |
| epochs | 100 | early stopping reaches its best epoch well before this |
| batch size | 16 | |
| learning rate | **1e-2** | selected by sweep on validation; was 3e-3 when the dataset had 641 examples |
| weight decay | 1e-2 | selected by sweep; 1e-1 and 3e-1 were clearly worse |
| patience | 20 | early stopping on validation macro-F1 |
| optimiser | AdamW | from `candle_nn`, no custom optimiser |
| dropout | 0 | swept, and it does not help — see §7.2 |
| feature dropout | 0 | swept, and it does not help |
| label smoothing | 0 | swept; 0.05 is within noise of none |

### 7.1 The sweep

`training/run_experiments.py` runs one variable at a time and appends every
result to `training/experiments.jsonl`. Selection is on **validation**; the
locked benchmark is scored for the record and never used to choose anything.

```
$ python3 training/run_experiments.py --suite lr
experiment              val acc   val F1  test acc  test F1  bench acc  bench F1    sec
lr-5e-2                  0.7686   0.7652    0.7107   0.7143     0.6462    0.6582   20.3
lr-3e-2                  0.6364   0.6049    0.5868   0.5612     0.5692    0.5685    8.7
lr-2e-2                  0.7851   0.7935    0.7438   0.7396     0.6846    0.7113    9.6
lr-1e-2                  0.8347   0.8362    0.7603   0.7685     0.7538    0.7703   21.0
lr-5e-3                  0.8017   0.8028    0.7521   0.7473     0.7462    0.7636    9.8
lr-3e-3                  0.7769   0.7993    0.7273   0.7333     0.7231    0.7370   13.2
lr-1e-3                  0.8099   0.8161    0.7603   0.7625     0.7462    0.7667   13.9
lr-3e-4                  0.7934   0.8122    0.7273   0.7287     0.7231    0.7541   24.6
```

The 0.1 release's learning rate of 3e-3 was chosen when the dataset had 641
examples. With 1,210 it moved to 1e-2, worth **+3 points** of macro-F1 on the
locked benchmark. Learning rates of 2e-2 and above destabilise training.

The base was moved from 3e-3 to 1e-2 after this suite, and every later suite
runs at that base. The move is recorded in `run_experiments.py` rather than
left implicit, because a suite run at a different base than the one before it
is not comparable.

### 7.2 Regularisation: a negative result

Dropout on the hidden layer, dropout on the *input features*, and label
smoothing are all implemented. **None of them adds a parameter**, so the
1,000,000 budget is untouched, and all three are off by default.

They were added because the model was measured at **train accuracy 1.0000**
with validation at 0.83 — 860,160 embedding parameters memorising 968
examples. The sweep says they do not help:

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

The reason is visible in the numbers. The train→validation gap is 16 points,
but the validation→benchmark gap is a *further* 8 points and is not a
memorisation effect: it is a difference between the dataset's phrasing and
real phrasing. Regularisation attacks the first gap and leaves the second
untouched, so it costs accuracy without buying generalisation.

The code stays in the tree because a reproducible negative result is worth
more than a missing feature: the next person to try it finds the sweep in
`training/experiments.jsonl` instead of spending a week rediscovering it.

### 7.3 Seed variance

Three seeds on the same data spanned **0.669 to 0.754** on the locked
benchmark. A single training run is a lottery, and a number reported from one
run is partly seed luck.

`aaron brain train --seeds N` trains N models and keeps the one with the best
**validation** macro-F1, then validation accuracy. The test split and the
benchmark are never consulted for the choice. Every seed's scores are printed:

```
seed results (selected on validation macro-F1):
  2690307351   val F1 0.8362  val acc 0.8347  best epoch  64   <- selected
  2690307352   val F1 0.7902  val acc 0.7934  best epoch  20
  2690307353   val F1 0.6706  val acc 0.6777  best epoch  17
```

### 7.4 Determinism

Initialisation from SplitMix64, batch order reshuffled every epoch from the
same PRNG stream, fixed-size CPU reductions, and the regularisation masks
drawn from the same stream. `training_is_deterministic` asserts that two runs
produce bit-identical weights.

Training memory is separate from inference memory and is released afterwards:
the optimiser state and the `VarMap` live only inside `train()`. Inference
rebuilds the model from a checkpoint into a plain `HashMap<String, Tensor>`.

---

## 8. Checkpoints

One file, with a manifest:

```
[u32 LE]  manifest length
[JSON    ] format, arch id, parameter count, seed, epochs, metrics,
           per-tensor {shape, offset, length, sha256}, and a sha256 of the data
[f32 LE  ] concatenated tensor data, little-endian
```

Loading answers three questions before a single weight is used:

1. **Is this the architecture AARON expects?** `arch_id()` includes every
   dimension: `aaron-intent-1m-v1:10240x84h754c14L96`.
2. **Does it contain exactly the declared tensors?** A missing or extra tensor
   is an error.
3. **Is every byte the one that was written?** Per-tensor SHA-256 plus a whole
   block SHA-256.

Corrupting one byte of the last weight produces
`tensor 'fc2.bias' failed its SHA-256 check; the file is corrupt`. Saving is
write-then-rename, so an interrupted save never replaces a good checkpoint with
a truncated one.

---

## 8b. The 0.3 changes: safety-first data and a measured threshold

### The threshold moved from 0.55 to 0.60

The full sweep is in `training/threshold-sweep.jsonl`; the method is in
`training/sweep_threshold.py`. **0.55 and 0.60 are exactly tied on
validation** — accuracy 0.8347, macro-F1 0.8362, false-accept 0.1540, all
identical — so the tie-break is the more conservative threshold, which cannot
reduce measured validation performance and abstains more often on out-of-schema
input.

**The threshold is an abstention control, not a safety control.** The sweep
measured the safety benchmark's rates at thirteen thresholds and *not one
moved*: the model's errors on adversarial input are confident (66 of 90 false
accepts sat in the 0.9–1.0 confidence band, median 0.999), so no threshold
reaches them. See [release-0.3-report.md](release-0.3-report.md) §7.

### The hard negatives that were kept, and the ones that were not

113 hard negatives were written for release 0.3, in thirteen families. Only
one family survived measurement: **a command verb with no object** — "open it",
"run that", "kill the process", "remember that". The model had learned *verb
present ⇒ command*, and the fix is 31 examples showing that the object is
what makes it a command.

The other twelve families are kept in
`training/curated/experimental/rejected-families.jsonl` with the measurement
that rejected them. All thirteen together made the model *worse* — locked
macro-F1 0.71 against 0.77, and the regression suite fell to 41/42 — because
the `unknown` class grew past twice the size of the others and the model
learned *short ⇒ unknown*, which breaks short legitimate commands.

### Class weighting was tried and did not help

`ClassWeighting` implements `w_c = (mean / n_c) ^ p`, normalised so the mean
weight is 1 and the loss scale is unchanged. The measured imbalance justified
it (4.2× after the hard negatives were added). Full balancing recovered part
of the intent quality but was worse on safety; square-root balancing dropped
the regression suite to 34/42.

It stays in the code, **off by default**, because a reproducible negative
result is worth more than a missing feature.



`aaron learning evaluate` reports, from the held-out split only:

* accuracy and macro-F1 (over all 14 classes, and over the classes present);
* per-intent precision, recall and F1, with support counts;
* the full 14×14 confusion matrix;
* the out-of-distribution rejection rate;
* the regression suite (42 hand-picked hard cases).

## 9. Evaluation

### Measured result, release 0.3

Release build, 8 seeds selected on validation, 100 epochs, threshold 0.60.

| set | accuracy | macro-F1 | what it is |
|---|---|---|---|
| **locked benchmark v3** (130) | **0.7462** | **0.7628** | the intent measure: never trained on, never selected on |
| **safety benchmark v2** (102) | — | — | the adversarial measure: written after the selection, scored once |

Against the previous release, on the locked benchmark:

| model | accuracy | macro-F1 | false-accept |
|---|---|---|---|
| 0.1 release | 0.6154 | 0.6319 | — |
| 0.2 release | 0.7538 | 0.7703 | 0.2800 |
| **0.3 release** | 0.7462 | 0.7628 | **0.2000** |

0.3 trades 0.8 points of intent accuracy for 8 points of false-accept rate
and 11.8 points of adversarial refusal. The full before/after, including the
held-out safety benchmark, is in
[release-0.3-report.md](release-0.3-report.md) §10.

Per-intent, on the locked benchmark:

| intent | support | recall | precision | f1 |
|---|---|---|---|---|
| open_url | 8 | 1.000 | 1.000 | 1.000 |
| stop_application | 8 | 1.000 | 1.000 | 1.000 |
| remember | 8 | 0.875 | 0.875 | 0.875 |
| recall | 8 | 0.875 | 1.000 | 0.933 |
| run_tests | 8 | 0.875 | 1.000 | 0.933 |
| cancel | 8 | 0.875 | 0.875 | 0.875 |
| help | 8 | 0.750 | 0.857 | 0.800 |
| search_files | 8 | 0.750 | 0.750 | 0.750 |
| read_file | 8 | 0.750 | 0.750 | 0.750 |
| start_application | 8 | 0.875 | 0.636 | 0.737 |
| system_status | 9 | 0.556 | 0.500 | 0.526 |
| git_status | 8 | 0.375 | 0.600 | 0.462 |
| greeting | 8 | 0.375 | 0.750 | 0.500 |
| unknown | 25 | 0.720 | 0.581 | 0.643 |

Safety-relevant rates on the locked benchmark:

| metric | value |
|---|---|
| false-accept rate (out-of-schema accepted as a command) | **0.280** |
| false-reject rate (real commands rejected as out-of-schema) | 0.124 |
| unknown precision | 0.581 |
| unknown recall | 0.720 |
| regression suite | **42 / 42** |

**The false-accept rate is the number that matters.** 28% of out-of-schema
requests are accepted as a command, and every one of those is a request that
could run the wrong tool. It is the top item in the next round of data work.

The regression suite went from 41/42 to **42/42** without editing the suite,
without relabelling anything, and without touching the held-out data: the
`contrastive frames` and the interrogative-form positives fixed both failing
cases (`any problems with the machine` and `do i have uncommitted changes`)
as a side effect of covering the *shapes* they belong to.

### The error-analysis workflow

`aaron learning analyse` produces a machine-readable (JSON) and a readable
(Markdown) report:

```
aaron learning analyse                                   # the dev test split
aaron learning analyse --benchmark training/evaluation/benchmark-v3.jsonl
aaron learning analyse --format json --out report.json
```

It contains accuracy and macro-F1, per-intent precision/recall/F1, the
confusion matrix, per-split class counts, expected calibration error and
reliability bins, unknown precision and recall, false-accept and false-reject
rates, entity-extraction error rate, the regression result, and every
misclassification grouped into an actionable category:

| category | meaning | the remedy |
|---|---|---|
| `intent_overlap` | two command intents confused | explicit minimal pairs |
| `out_of_schema_accepted` | non-command accepted as a command | a hard negative — **safety** |
| `out_of_schema_rejected` | real command rejected | a positive example of that shape |
| `low_confidence` | right at the top, below the threshold | more training signal, *not* a lower threshold |
| `missing_coverage` | the phrasing is simply absent | add it |
| `entity_extraction` | right intent, wrong entity | fix the extractor, not the classifier |
| `preprocessing_mismatch` | the deployed path disagrees with the evaluated one | a bug, not a data problem |
| `label_error` | the label is probably wrong | review it with a human |

Every text in a report is redacted with `redact_secrets` and truncated. No
transcript and no stored memory is ever read, so a report is always safe to
share.

### Three bugs this evaluation caught

**1. The deployed classifier was softmaxed twice.** `aaron learning evaluate`
reported 0.0667 while `aaron brain train` reported 0.7667 — same weights, same
file. `NeuralClassifier::classify` passed *probabilities* into
`logits_to_intent`, which applies its own softmax. Softmax of a softmax
compresses everything toward 1/14, so every utterance scored ≈ 0.17, below the
0.55 threshold, and the deployed model answered `unknown` to everything.
Training was measured through `score`, which uses raw logits and was correct
all along.

That is the dangerous shape of bug: training looked healthy, the reported
number was real, and only the *deployed* path was broken. Two tests now pin
it: `classification_is_not_double_softmaxed` and
`the_deployed_checkpoint_reproduces_the_training_metric`.

**2. The hold-out did not measure every class.** The split was
`fnv1a64(id) % 100`, which is proportional but not stratified: `run_tests`
happened to get zero test examples, so its per-intent row was reported with
support 0 and an F1 of 0 — indistinguishable in the output from a class the
model genuinely fails. Stratifying by intent raised macro-F1 from 0.705 to
0.790 and made the table honest.

The first version of the stratification had its own bug: it computed the
hold-out size as `(n * 10) / 100`, which truncates to **zero** for any intent
with fewer than ten examples, so those intents got no test split at all — the
same failure by a different route. The size is now rounded to nearest and
floored at one, and `even_a_small_intent_gets_a_hold_out` sweeps every group
size from 5 to 40.

**3. `learning evaluate` could evaluate a stale checkpoint.** Candidates were
sorted by *filename*, and filenames embed the epoch count, so `e60` sorted
above `e200`. You could evaluate a stale candidate and believe you had
evaluated the new one — and did, until the numbers came back 0.0667 against a
training report of 0.7667. Candidates are now sorted by modification time,
the three most recent are kept, and a mismatch is reported.

### The model is not promoted

AARON's configured bar is `min_macro_f1 = 0.92`, `min_accuracy = 0.93`.
Measured macro-F1 on the locked benchmark is 0.770, so `aaron learning promote`
**correctly refuses**, and the deployed router remains the deterministic one.
This is reported rather than hidden, and lowering the thresholds to make the
model deploy would defeat their purpose.

The gate now reports *why* it refused, on three counts:

```
REJECTED:
  - macro-F1 0.7703 is below the required 0.9200
  - accuracy 0.7538 is below the required 0.9300
  - false-accept rate 0.2800 is above the maximum 0.0500:
    out-of-schema requests were accepted as commands
```

The third is new in 0.2. The gate gained three checks — per-intent hold-out
coverage, a per-intent recall floor, and a ceiling on the false-accept rate —
and **none of them relaxes an existing threshold**. Every one can only refuse
a promotion.

### What would move the number

In order of expected value, and each one is a **data** priority rather than an
architecture one:

1. **Out-of-schema examples that look like commands, at scale.** The
   false-accept rate is 0.28 and every false accept is a request that could
   run the wrong tool. This is the safety number, and it is the worst number
   in the report.
2. **Phrasing diversity.** The validation→benchmark gap is 8 points and is
   entirely about *how the same request is worded*.
3. **More examples for the classes with the worst recall** on the benchmark:
   `greeting` (0.375), `git_status` (0.375), `system_status` (0.556).
4. **Fewer near-duplicates.** The dataset still carries 137 near-duplicate
   pairs, all inherited from v1 and all preserved rather than deleted. They
   cost capacity and buy nothing.
5. **A larger vocabulary is not the answer** — it would make overfitting worse.
6. **Fewer embedding parameters** (smaller `V`) with more hidden width, which
   would trade memorisation for capacity. That changes the budget arithmetic
   and needs a fresh search over `(V, D, H)`.

The 0.2 release already established that regularisation does not close the
gap (§7.2), so the answer is more and better data, not a different model.

---

## 10. Optional speech models are separate

The 1,000,000 budget covers **only** the model above.

| Component | Parameters | Licence | Notes |
|---|---|---|---|
| Whisper tiny (ggml) | 39 M | MIT (code), weights per upstream | ~75 MB |
| Whisper base (ggml) | 74 M | as above | ~142 MB |
| Whisper small | 244 M | as above | ~466 MB |
| Whisper medium | 769 M | as above | ~1.5 GB |
| Whisper large-v3 | 1.55 B | as above | ~3.1 GB |
| Piper voice (VITS) | ~30 M | MIT (code), per-voice | varies |
| espeak-ng | none | GPL-3.0 | formant synthesis, no model file |

AARON downloads none of these and reports the absence rather than pretending.
See [models/manifest.toml](../models/manifest.toml).

On an RTX 2050 with 4 GB, a Whisper `base` model runs comfortably on the GPU
with roughly 1 GB of VRAM, but AARON does not build whisper.cpp for you and
does not manage its VRAM. If you do, `asr.use_gpu = true` passes the flag.

---

## 11. Reproducing

```bash
python3 training/generate_dataset.py > training/dataset/current.jsonl
python3 training/generate_regression.py > training/evaluation/regression.jsonl
cargo test -p aaron-brain
cargo build --release
./target/release/aaron brain train --epochs 100
./target/release/aaron learning evaluate
```

Everything is seeded. Two runs with the same dataset, seed and configuration
produce identical weights, which `training_is_deterministic` verifies.