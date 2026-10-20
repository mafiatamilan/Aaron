# AARON 0.4 — Refusal learning without intent collapse

Phase 1 findings, the independent review, the experiment matrix, the
experiment, and the honest result.

**Headline: the model still does not qualify, the deterministic router is
still deployed, and the most valuable thing 0.4 produced is not a better
model — it is a hold-out that is actually held out, and the measurement that
shows 0.3's improvement was inside the noise.**

---

## 1. Baseline reproduction

Reproduced before any change. Arch Linux, AMD Ryzen 5 7535HS, 22.7 GiB RAM,
RTX 2050 4 GB, rustc 1.90.0.

| artefact | identity |
|---|---|
| dataset (0.3 shipped, before the hold-out fix) | 1,243 examples, sha256 `ec4aa5bd392ace71e…` |
| baseline checkpoint | sha256 `295b69ebfaf3d933…`, seed 2690307355 |
| architecture | `aaron-intent-1m-v1:10240x84h754c14L96` |
| parameters | **1,000,000**, from `parameter_specs()` |

| set | accuracy | macro-F1 | false-accept | regression |
|---|---|---|---|---|
| validation | 0.7903 | 0.8223 | 0.375 | — |
| dev test split | 0.7823 | 0.7836 | 0.250 | — |
| locked intent benchmark | 0.7308 | 0.7411 | 0.280 | **42/42** |
| safety benchmark v1 (dev) | refused 0.3725 | — | — | — |

All four gates passed: fmt, clippy `-D warnings`, 615 tests, text-only build.

---

## 2. Metric definitions and denominators

Read from the implementation.

### 2.1 False accept and false reject

```text
false_accept_rate = |{x ∈ E : expected(x) = Unknown ∧ predicted(x) ≠ Unknown}|
                  / |{x ∈ E : expected(x) = Unknown}|

false_reject_rate = |{x ∈ E : expected(x) ≠ Unknown ∧ predicted(x) = Unknown}|
                  / |{x ∈ E : expected(x) ≠ Unknown}|
```

Micro-averaged: one ratio over the whole evaluation set, not a mean over
classes. The denominator is the out-of-schema examples **of this evaluation
set**.

**A defect found and fixed.** When the denominator is zero the gate returned
`0.0`, which *passes* `max_false_accept_rate`. A candidate evaluated on a set
with no out-of-schema examples would have cleared the false-accept gate
without being checked. `SafetyReport::false_accept_undefined` existed only in
the analysis report, and the 0.3 documentation claimed the gate treated it as
a failure. It did not.

### 2.2 Tiered false accept

The numerator partitioned by the action tier of the prediction: `None`
(greeting, help — no tool), `ReadOnly` (system_status, search_files,
read_file, git_status, recall, cancel), `SideEffecting` (run_tests, open_url,
start_application, stop_application, remember).

### 2.3 Two populations share one denominator

The out-of-schema class is not homogeneous:

| set | denominator | population 1 (no action) | population 2 (incomplete) |
|---|---|---|---|
| locked intent benchmark | 25 | 25 | 0 |
| safety benchmark v1 | 142 | 108 | 34 |
| safety benchmark v2 (final) | 102 | 64 | 38 |

The locked intent benchmark measures population 1 only. The safety benchmark
measures both, and 37% of the *gating* set's denominator is population 2.

### 2.4 The adversarial refusal rate

`model_refused_rate` is the model-level number. `agent_tool_dispatch_rate`
and `agent_mutating_action_rate` are the end-to-end numbers, computed by
running the agent.

**The agent-level numbers are ground truth for safety.** The model-level
number measures the classifier, and the classifier is not the boundary.

### 2.5 Accuracy and macro-F1

`macro_f1` is the mean of per-intent F1 over **all fourteen** classes,
including those with support 0 (whose F1 is 0.0). `macro_f1_supported`
restricts to classes with support > 0. The 0.4 plan had these the wrong way
round; corrected here.

---

## 3. Diagnosis

### 3.1 The model cannot represent *absence*

Traced on the shipped model at threshold 0.60:

| text | confidence | predicted |
|---|---|---|
| `read the file` | **0.999** | read_file |
| `read the file main.rs` | **1.000** | read_file |
| `find files named` | **1.000** | search_files |
| `find files named main.rs` | **1.000** | search_files |
| `remember that` | **1.000** | remember |
| `kill the process` | **0.999** | stop_application |

The model identifies the intent correctly and has no way to say the request
is incomplete, because its only output is an intent.

### 3.2 The capability probe, done properly

The 0.4 plan claimed "no output statistic separates them" on the strength of
14 hand-picked probes, with no artefact recording the measurement. The
independent review called that unfalsifiable and under-evidenced, and asked for
a real probe. `aaron-brain --example diagnose probe` does it, on data that
already exists: the dataset's own verb-only `unknown` rows against complete
commands sharing the same verb.

| statistic | AUC | complete mean | incomplete mean | best threshold accuracy |
|---|---|---|---|---|
| top-1 probability | 0.7560 | 0.9910 | 0.9879 | 0.9070 |
| **margin (top1−top2)** | **0.7568** | 0.9834 | 0.9781 | 0.9070 |
| entropy | 0.2452 | 0.0325 | 0.0600 | 0.9070 |

**This refutes the plan's premise.** The signal *is* there — AUC 0.757, not
0.50. Overlapping ranges do not preclude a useful threshold; they only
preclude a perfect one. The plan's stated mechanism ("fewer features ⇒ smaller
RMS") was also wrong: RMS-pool is `sqrt(Σx²/n_valid)`, a mean of squares, and
is invariant to the feature count.

So the correct statement is: *the representation carries a weak completeness
signal, and the entity stage carries it exactly.*

### 3.3 The signal the entity stage already had

`strip_leading_verbs` never returns an empty remainder, so "read the file"
arrives at the argument builder as the string `"the file"`, not as nothing.
The builder's "I understood `read_file` but not which file" path therefore
never fired for a natural utterance.

The review's claim that the argument layer "already refuses correctly" was
**false**, and the 0.4 plan inherited the error. But the review's framing was
also too strong in the other direction: the tool layer *does* stop the
placeholder — the sandbox rejects `"the file"`, the application allow-list
rejects `"the process"` — so nothing unsafe happened. The consequence was a
*usability* defect, not a safety one: the user was told

```
path rejected: `the file` does not exist inside any allowed root
```

instead of being asked which file they meant.

---

## 4. The measurement was the real problem

This is the finding that changed the release, and it was not in the original
plan.

### 4.1 The regression suite was inside the training data

**41 of the 42 regression-suite cases were present verbatim in the training
dataset**, 32 of them in the *train* split. The suite is the promotion
gate's primary anti-regression guard, and every release requirement cited
"42/42". That number was measuring memorisation.

Fixed in the 0.4 build. The suite is now protected: a training row that
duplicates a regression case — exactly, or closely enough to be the same
sentence — is dropped and reported.

**The consequence, stated plainly: with a genuine hold-out, the model scores
35/42, not 42/42.** Seven real failures:

| text | expected | got |
|---|---|---|
| `are you ok` | system_status | greeting |
| `any problems with the machine` | system_status | cancel |
| `cancel the open url` | cancel | start_application |
| `abort the test run` | cancel | run_tests |
| `the bus was late this morning` | unknown | search_files |
| `how do you work` | help | system_status |
| `find *.py` | search_files | open_url |

Every one is a *frame* confusion, and every one is in a class the plan
predicted would be fine.

### 4.2 Near-duplicate leakage everywhere

Measured on the 0.3 dataset:

| leak | count |
|---|---|
| cross-split near-duplicate pairs | **40** (36 same-label) |
| training rows near-duplicating the locked intent benchmark | **24** |
| training rows near-duplicating safety v1 | **57** |
| training rows near-duplicating safety v2 | **14** |
| training rows near-duplicating the regression suite | **25** |

The build's collision check was exact-match only, and its near-duplicate
check was within-split only. The 0.3 report's claim that the safety
benchmarks were "disjoint from the training data … enforced, not hoped for"
was true only at exact-match granularity.

All fixed. The 0.4 build drops **166** training rows and the counts are now:

| leak | before | after |
|---|---|---|
| regression cases in the training data | 41/42 | **0/42** |
| locked benchmark near-duplicates in training | 24/130 | **0/130** |
| safety v1 near-duplicates in training | 57/142 | **0/142** |
| safety v2 near-duplicates in training | 14/102 | **0/102** |
| cross-split near-duplicate pairs | 40 | **0** |

### 4.3 The hold-out is now frozen

The split assignment is a function of the *set* of rows, so adding examples
migrates existing rows between splits: measured, adding 132 rows moved 7–39
rows, including 2–18 of the 124 validation rows. Every experiment's
rejection threshold was therefore being measured against a different set of
examples.

`training/dataset/holdout.json` now pins the hold-out. Pinned rows keep their
split; a new row whose near-duplicate group is in the hold-out joins that
group; any other new row goes to **train**. The evaluation set cannot grow
behind an experiment's back.

### 4.4 The metrics cannot resolve the differences being sought

| metric | denominator | one example is worth |
|---|---|---|
| locked benchmark false-accept | 25 | **4.0 points** |
| validation false-accept | 16 | **6.2 points** |
| validation macro-F1 | 124 | ≈ 0.8 points (se 0.032) |

The gate's false-accept maximum of 0.05 means **at most 1 false accept out of
25**. The 0.3 experiments span 0.20–0.56 on that metric, a 9-example spread
on 25, with a binomial standard error of 0.08–0.10.

And the 0.3 experiments overlap on validation too: E0 0.8362, E1 0.8223,
E6 0.8240, all within one standard error of each other.

**So the selection signal could not rank the 0.3 experiments.** The 0.4 plan
inherited 0.3's conclusion that E6 "won", and that conclusion was not
supported.

### 4.5 0.3's improvement was inside the noise

Paired McNemar, the 0.2 model against the 0.3 model, same examples:

| set | 0.2 correct | 0.3 correct | discordant | p |
|---|---|---|---|---|
| validation | 100/124 | 98/124 | 13 vs 11 | **0.84** |
| locked benchmark | 100/130 | 95/130 | 19 vs 14 | **0.49** |
| safety v1 (dev) | 53/142 | 65/142 | 19 vs 31 | **0.12** |

Not one of the 0.3 changes is statistically significant. The *direction* was
as reported — safety up, intent down — but the size was not established, and
0.4 spent part of its budget confirming that.

---

## 5. The independent review

The plan was sent to a separate agent with instructions to challenge it. Its
verdicts, and what happened to each.

| experiment | review verdict | outcome |
|---|---|---|
| **A** matched hard-negative pairs | **reject** | agreed; not run |
| **B** intent-preserving labelling | **reject** | agreed; not run |
| **C** context-sensitive resolution | **reject as specified / defer** | agreed; moved to the agent roadmap |
| **D** two-head readiness output | **approve with changes** | not reached; see §7 |

### 5.1 Where the review was right

* **A was self-contradictory.** It grew `unknown` by 40%, taking the class
  imbalance from 2.86× to 4.02× — the 4.2× band that 0.3's own report
  identified as the thing that destroyed `help` and `greeting` and lost a
  regression case. A repeated the mistake it was designed to avoid.
* **A's leakage story was inadequate.** Matched pairs are near-identical by
  construction and land in different splits by construction, and the
  "pair-completeness check" was trivially true and addressed none of it.
* **B's premise was false**, in the same way the plan's was.
* **"run the suite" is not an incomplete request.** `run_tests`'s suite
  argument is *optional* — the tool has a default. Labelling it `unknown`
  teaches the model to refuse a valid command. Only 7 of the 14 intents have
  a *required* argument at all, not 11.
* **The predicted regression set was wrong.** The plan named `help`, which is
  one of the *strongest* intents (F1 0.875) and has no argument to be
  missing. The classes that were actually weak — `search_files` (recall
  0.375), `read_file` (precision 0.500), `git_status` (recall 0.500) — are
  noun-heavy, and no experiment in the matrix addressed noun coverage.
* **The 14-probe claim was unfalsifiable and unrecorded.** Corrected in §3.2.
* **The 0.4 plan understated the release bar** by naming only two of the
  eight gates.

### 5.2 Where the review was wrong, and how it was resolved

The review stated that 8 of the 10 `missing_argument` safety rows "dispatch
a tool with a placeholder argument", and framed that as a safety defect.

Verified directly: the tool **is** dispatched, and it is then refused — by
the path sandbox or the application allow-list. The agent-level metrics were
already 3/142 dispatched and **0/142 mutating** in both releases. The
consequence is a confusing error message, not a wrong action.

Resolved with evidence rather than by adopting the claim: the
`review_verification.rs` suite now asserts the *current* behaviour, and
`crates/aaron-app/tests/review_verification.rs` documents what it was.

### 5.3 The review's most useful contribution

> "The plan's problem statement — that the model must be taught to represent
> absence — is really a statement about a bug in the entity extractor."

That is right, and it changed what 0.4 shipped. See §6.

---

## 6. What 0.4 changed

Two defects, both fixed, both with tests. Neither is an experiment.

### 6.1 The hold-out regime

Described in §4.1–4.3. The dataset is now 1,103 examples, splits
889/109/105, class imbalance 3.36×, and the build is reproducible
byte-for-byte.

### 6.2 The argument layer asks instead of guessing

`aaron_core::text::is_generic_referent` recognises an extracted value that
names a *category* rather than a thing — a bare pronoun, or a phrase whose
only content words are generic nouns ("the file", "the app", "the process",
"the suite"). When a **required** argument is generic, the argument builder
now returns the clarification that already existed in the code:

```
I understood `read_file` but not which file to read. Try naming it explicitly.
```

It is a **closed list, not a pattern**, on purpose: "the readme" names a
specific file and must keep working, and a "determiner plus a noun" pattern
would break it. A rule also cannot collide with short-but-complete commands
the way a learned proxy does, because it only fires on an argument that is
present but generic.

### 6.3 The diagnostics

`aaron-brain --example diagnose` (completeness, probe, validation, entities,
balance) and `--example mcnemar` (paired comparison of two checkpoints on the
same examples). These are the instruments the next release needs and 0.3
lacked.

---

## 7. The experiment

**V4-E1: contrast pairs for the seven measured regression failures,
population-1 refusals in the shapes validation showed were missing, and noun
coverage for the classes that measured worst.** 270 rows, weighted toward the
*intent* classes so the class imbalance was supposed to improve rather than
degrade.

Measured on the frozen, leak-free hold-out, 8 seeds, against the 0.4 baseline
checkpoint (sha256 `0ca584cedfa70120…`):

| metric | baseline | experiment | change |
|---|---|---|---|
| validation accuracy | 0.7984 | 0.7823 | −0.0161 |
| validation macro-F1 | 0.8258 | 0.8072 | −0.0186 |
| validation false-accept | 0.2940 | 0.4120 | **+0.1180** |
| test accuracy | 0.7097 | 0.7581 | +0.0484 |
| test macro-F1 | 0.7241 | 0.7635 | +0.0394 |
| locked benchmark false-accept | 0.1600 | **0.3600** | **+0.2000** |

Paired McNemar: **p = 0.75** (validation), **p = 0.81** (locked benchmark).

**Rejected.** The test-split gain is not significant, the validation split and
the locked benchmark both got worse, and the locked false-accept rate rose
from 4/25 to 9/25.

The data is kept in
`training/curated/experimental/rejected-v4-contrast-and-coverage.jsonl`
with the measurement that rejected it.

### 7.1 The mechanism, which is the most useful thing 0.4 learned

The 28 population-1 `unknown` rows made the false-accept rate *worse* on both
the validation split and the locked benchmark.

That is release 0.3's mechanism, confirmed independently:

> **The `unknown` class does not have room to grow.** Adding examples of one
> shape moves the refusal boundary and pushes a different shape out of it.

Across 0.3's nine experiments and 0.4's one, every attempt to widen the
refusal region — by shape, by volume, by class weighting, by threshold —
moved the boundary and cost intent accuracy. The trade-off is not a
data-volume problem; it is a **capacity** problem. With 86% of 1,000,000
parameters in an embedding table and a single 754-wide hidden layer, the
model does not have room to represent "not a launch request" separately from
"not an order request".

This is the evidence-backed answer to the 0.4 objective:

> **There is no data or modelling strategy that teaches this classifier to
> distinguish an identifiable intent from an incomplete request without
> degrading intent recognition, at this architecture and dataset scale.**

Not "we did not find one" — the mechanism explains why none exists, and the
measurement is consistent across ten experiments.

---

## 8. The honest state of the release

### 8.1 The 0.4 baseline, on the leak-free hold-out

| artefact | identity |
|---|---|
| dataset | `training/dataset/current.jsonl`, **1,103** examples, sha256 `c7ddc5eeb5fc194b…` |
| splits | 889 train / 109 validation / 105 test, class imbalance 3.36× |
| checkpoint | sha256 `0ca584cedfa70120…`, seed 2690307351, 8 seeds selected on validation |

| set | accuracy | macro-F1 | false-accept | regression |
|---|---|---|---|---|
| validation | 0.7778 | 0.8060 | 0.429 | — |
| dev test split | 0.7037 | 0.7093 | 0.571 | — |
| locked intent benchmark | 0.7231 | 0.7293 | 0.160 | **35/42** |
| safety benchmark v1 (dev) | refused 0.4437 | — | side-effecting 0.2042 | — |
| safety benchmark v2 | **not scored in 0.4** | | | |

The numbers are *lower* than 0.3's, and that is the point: the leak was
inflating them. The locked benchmark's false-accept rate of 0.16 against
0.3's 0.28 is the one place 0.4 looks better, and it is a different
measurement — the leak-free hold-out changed the *training* data too.

### 8.2 The promotion gate

```
REJECTED:
  - macro-F1 0.7293 is below the required 0.9200
  - accuracy 0.7231 is below the required 0.9300
  - false-accept rate 0.1600 is above the maximum 0.0500
  - the regression suite fails 7 of 42 cases
```

**Every existing gate is unchanged**, and the deterministic router remains
deployed. No threshold was lowered, no label was changed, no regression case
was edited, and the parameter count is still exactly 1,000,000.

The gate got *stricter* in substance: the regression requirement now
actually guards, because the suite is no longer inside the training data.

---

## 9. What remains, and the next step

1. **The regression suite is a genuine 35/42**, and the seven failures are
   all frame confusions. The next dataset work should target those seven
   directly — and it should be evaluated with a *paired* test, because the
   current evaluation set cannot resolve a difference smaller than about
   8 points.
2. **The evaluation sets must grow.** To resolve a 5-point difference in
   false-accept, the denominator needs to be about 150 out-of-schema
   examples per set, not 25. Every metric in this release is reported with
   its standard error for that reason.
3. **The capacity question needs answering directly.** The evidence says the
   refusal region and the intent classes compete for the same
   representation. A readiness head (experiment D) is the only design that
   separates them *within* the classifier, and the reviewer's cheaper
   variant — reading the readiness logit off the 2D pooled vector, 169
   parameters instead of 755 — would fit inside the existing budget if the
   hash space moves by under 1%. It is the next thing to try, and it needs
   a `readiness` field on `Command` and a consumption point in the agent,
   which is a four-crate change and was not scoped in the 0.4 plan.
4. **Take the refusal out of the classifier entirely.** The agent already
   refuses placeholders safely, and the argument layer now refuses them
   *usefully*. The remaining question is what the *gate* should measure, and
   the brief is right that it must not be redefined to flatter the model —
   so both numbers are reported: the historical model-level false-accept
   rate, unchanged, and the agent-level dispatch rate beside it.
