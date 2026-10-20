# AARON 0.5 — Explicit Readiness Modeling

**Status: complete. The readiness head is REJECTED; the deterministic router
remains deployed and no model is promoted.** The findings, the measurements
and the decision are in `docs/release-0.5-report.md`. This document is the
plan that was reviewed, and the record of what changed when it was.

Round 1 returned **REJECT** on the safety argument, the experiment design and
internal consistency; §2.10 records what was wrong and what changed. Two
things happened after that revision which the plan did not anticipate and
which changed the release's shape:

* **The corpus is now partly machine-written.** The shipped dataset contains
  exactly one usable readiness example, so a head trained on it learns a
  near-constant label. Six free models — DeepSeek among them — wrote the
  contrastive pairs, and a different model family audited them blind. The
  audit rejected 19 rows, eleven of them in the *hand-written* corpus, where
  it found a systematic labelling drift.
* **A silent logit-versus-probability bug** was live for one measurement
  round and was caught by a cross-check written for the purpose. The numbers
  it produced looked plausible and described a different model. The threshold
  sweep moved into the library so that the sweep and the metrics cannot
  disagree.

**Objective.** Evaluate a separate 169-parameter readiness head that reads the
existing pooled representation, against the deterministic completeness handling
that already ships. The intent classifier keeps deciding *what* the user wants;
the readiness head estimates *whether the request says enough to act on it*.

**The design was a hypothesis, not a conclusion**, and it was measured rather
than assumed. The head turned out to learn the task well (F1 0.683 against
the rule's 0.471) and to be the wrong thing to ship anyway: it is dominated
by the rule at every threshold, and the one thing it buys — a 92 % cut in the
agent's dispatch rate on adversarial input — costs the refusal of 86 % of
perfectly good requests.

---

## 1. Baseline, hashes and verification status

Recorded before any change, so that every number in this release has a
provenance.

| artefact | sha256 |
|---|---|
| `training/dataset/current.jsonl` (1,103 rows) | `c7ddc5eeb5fc194b89c9442cf0d7e47ba908dafefd59f0cf9941c47c5e77ae6f` |
| `training/dataset/holdout.json` (214 pinned rows) | `6f56cbe7b576713c92327c7f06554729cddd2eadfaa61f28a1153e12c0ac5e8e` |
| `training/evaluation/benchmark-v3.jsonl` (locked intent) | `de81f12f0892453ae3f6b5f1ab770e85bf27bb026a78ee78e3ef6501c805e4ee` |
| `training/evaluation/safety-benchmark-v1.jsonl` (dev) | `9dd63460d363959775c98615214fef687989d411e9fe994fe6bdb2620354ae3a` |
| `training/evaluation/safety-benchmark-v2.jsonl` (**held out**) | `b54377f886ca51ffc2080c67385ccefc3db085eb586e94ca33a61b339d3b4f38` |
| `training/evaluation/regression.jsonl` (42 cases) | `d2c3eced6910763c7c9c7d766193ce2be0a442708732bf94e2e7578bdbd64ff4` |
| 0.4 baseline checkpoint | `0ca584cedfa701201d687442a5178ac92284224a44b38e0c3f70788b827c407a` |

0.4 baseline metrics (from `docs/release-0.4-report.md`):

| set | accuracy | macro-F1 | false-accept | regression |
|---|---|---|---|---|
| validation | 0.7778 | 0.8060 | 0.429 | — |
| locked intent benchmark | 0.7231 | 0.7293 | 0.160 | **35/42** |
| safety benchmark v1 (dev) | refused 0.4437 | — | side-effecting 0.2042 | — |

**Averaging convention.** Macro-F1 is the unweighted mean of the per-intent F1
scores over all 14 intents, including intents with no support in a given set
(they score 0.0). This is the convention `crates/aaron-brain/src/eval.rs`
already implements; it is stated here because the brief requires it and because
0.4 reported the same number.

**Test suite:** 627 passing, 0 failing at plan time (the 0.4 report records 615;
the difference is this release's new tests).

---

## 2. Phase 1 findings

All findings below were verified by reading the source or by computing from
the data, not by inference from the 0.4 report.

### 2.1 The seven genuine regression failures are intent confusions

Measured with the real extractor (`extract_entities`) and the real argument
builder, not an ad-hoc re-implementation:

| utterance | expected | predicted | extracted entities | argument builder |
|---|---|---|---|---|
| `are you ok` | system_status | greeting | *(none)* | — (no tool) |
| `any problems with the machine` | system_status | cancel | *(none)* | — (no tool) |
| `cancel the open url` | cancel | start_application | `application = "cancel the open url"` | accepts |
| `abort the test run` | cancel | run_tests | `suite = "abort the test run"` | accepts |
| `the bus was late this morning` | unknown | search_files | `text = "the bus was late this morning"` | accepts |
| `how do you work` | help | system_status | *(none)* | — (no tool) |
| `find *.py` | search_files | open_url | *(none)* | refuses (no URL) |

In every case where an argument is extracted, it is usable. `find *.py` is the
one row whose *behaviour* is a clarification rather than a dispatch, because
the predicted intent `open_url` has a required argument that is absent — but
the expected intent was `search_files`, so it remains an intent error.

**These are frame confusions inside the 14-way softmax, not completeness
failures.** A readiness head cannot fix them; it does not address the failure
mode they represent. Targeting them is separate work (contrastive intent
data) and this release does not attempt it.

### 2.2 Defect found and fixed: the generic-referent rule over-refuses

The 0.4 rule rejected *any* argument that resolved to a closed list of generic
nouns. Measured consequences on the shipped dataset:

- `recall my name` → refused with *"I understood `recall` but not what to look
  for"*, although the user named the thing.
- `search your memory for my project` → refused.
- `search for large files in the project` → refused.
- `launch code` → refused.

The rule conflated two different questions. *"Is this a reference to a
particular object?"* is the right question for a path, an application or a
search pattern. *"Did the user actually say anything?"* is the right question
for a free-text argument, because a topic is content, not a reference.

**Fix, landed.** `crates/aaron-agent/src/agent.rs` now has three argument
gates:

| gate | applies to | rejects |
|---|---|---|
| `need` | every argument | absent or empty |
| `need_referent` | path, application, search pattern | generic referent |
| `need_content` | remember text, recall query | contentless (filler-only) |

`crates/aaron-core/src/text.rs` gains `is_contentless` (a filler/stop-word
list) and its `GENERIC_REFERENT_NOUNS` list is trimmed of words that are
plausible identifiers (`name`, `code`, `text`, `line`, `number`, `one`).

A second, smaller defect fell out of the same analysis: the entity extractor
strips fillers before verbs, so `remember that` left the bare word `That`
behind and the agent stored it as a memory. `need_content` catches it.

### 2.3 Defect found and fixed: the trainer ignores the frozen hold-out

**This was the most consequential finding in the inspection.** Release 0.4
froze the hold-out in `training/dataset/holdout.json` and wrote a `split`
field into every dataset row. The trainer read neither.

`Dataset::split_of` (`crates/aaron-brain/src/dataset.rs`) fell back to
`assign_stratified`, which re-derives splits from the id hash *per intent*.
The builder (`training/build_dataset_v2.py::assign_splits`) hashes the *lowest
id of a near-duplicate group*, so the two algorithms disagree whenever a
near-duplicate group exists.

Measured on `current.jsonl` by reimplementing both algorithms:

```
rows=1103   trainer-split != file-split: 32
pinned hold-out rows: 214   of which the TRAINER puts in train: 12
```

Twelve rows that the 0.4 report treated as held out were **trained on**, and
the trainer's own validation/test split contained 14 rows that are not pinned.
The reported validation macro-F1 of 0.8060 was measured on a set differing
from the frozen hold-out in 26 of its 214 rows.

**Consequences.**

1. The 0.4 frozen-hold-out guarantee held for the builder but not end to end
   into training.
2. The 0.4 checkpoint cannot be honestly re-evaluated on the frozen hold-out,
   because it was trained on 12 of those rows. Every comparison in this
   release must use a **freshly trained** baseline.
3. Model selection and early stopping use validation macro-F1, so the wrong
   rows were steering the training run itself.

**Fix, landed.** `Example` gains `#[serde(default)] pub split: Option<Split>`;
the dataset file's `split` is authoritative and `assign_stratified` remains
the fallback for programmatically-built datasets. A partially-declared split
is an error. `crates/aaron-app/tests/holdout.rs` pins the behaviour: the
trainer's split is the declared split, the trainer's hold-out is exactly the
214 pinned rows, and no pinned row is trained on.

### 2.4 Finding: there is almost no readiness supervision

The readiness label of a row is a property of `(intent, entities)`, decided by
exactly the code the agent dispatches through:

```
if intent requests no action:                NotApplicable
else if build_arguments(intent, entities):   Ready
else:                                        NeedsClarification
```

Run over `training/dataset/current.jsonl` with the real extractor and the real
argument builder (`crates/aaron-agent/examples/readiness_labels.rs`):

| label | before §2.2 | after §2.2 |
|---|---|---|
| Ready | 855 | 862 |
| NeedsClarification | 8 | **1** |
| NotApplicable | 240 | 240 |

**Before the §2.2 fix, all eight NeedsClarification rows were false positives
of the over-broad rule.** After the fix exactly one remains: `search for the
file` (intent `search_files`, split `train`).

Split distribution after the fix: train 1, validation 0, **test 0**.

**A head trained on this would be trained on a near-constant label with zero
held-out positives.** It would learn to say "Ready" and assert a readiness it
has no evidence for — which is worse than having no head, because a head that
always says "Ready" can never ask a question.

This is why the release needs an authored corpus (§5) and why the corpus is a
first-class deliverable rather than a detail.

### 2.5 Finding: the readiness head's value proposition is narrower than first stated

**Round-1 correction.** The first draft of this plan claimed that a readiness
head would let the assistant answer *"which file should I read?"* instead of
*"I did not understand that"* for a fragment such as `read the document`.
**That claim was wrong, and the review caught it.** Two independent reasons:

1. `crates/aaron-agent/src/agent.rs` short-circuits `Intent::Unknown`
   *before* the argument builder, so a readiness verdict on an `unknown` intent
   is never read at the proposed placement.
2. If the intent head says `read_file` instead, the deterministic rule
   **already** refuses: `extract_entities` yields `path = "the document"`,
   `document` is in `GENERIC_REFERENT_NOUNS`, and `need_referent` refuses with
   *"I understood `read_file` but not which file to read."*

So the "unknown → targeted clarification" proposition is unreachable and the
"read_file → targeted clarification" case is already served by arm B. **The
head is never consulted in either case.**

**The proposition that survives.** Measured with the real code:

| utterance | extracted argument | rule verdict | really complete? |
|---|---|---|---|
| `read the document` | `path = "the document"` | refuses | no |
| `read the first 20 lines` | `path = "the first 20 lines"`, `count = 20` | **accepts** | **no** |
| `read` | `path = "read"` | **accepts** | **no** |
| `open the file called` | `path = "open the file called"` | **accepts** | **no** |
| `search the web` | `text = "the web"` | **accepts** | **no** |
| `read the pattern` | `path = "the pattern"` | **accepts** | **no** |
| `read the window` | `path = "the window"` | **accepts** | **no** |

Three structural families the closed list misses:

1. **the argument is the verb itself** (`read` → `path = "read"`) — the
   extractor strips the verb only when something follows it;
2. **the argument is a determiner plus a quantity** (`the first 20 lines`);
3. **the argument is a generic noun outside the closed list** (`the pattern`,
   `the web`, `the window`).

**The head's proposition is therefore:** *a learned head can detect an
unresolved reference that a closed noun list misses.* The control that must be
beaten is not "no readiness handling" but **an extended deterministic list**,
which catches family 3 for zero parameters. The head's case is families 1 and
2 — patterns a list cannot express.

**The head cannot move the model-level false-accept metric** for a fixed
architecture and a fixed trunk, because that metric is `expected == Unknown ∧
predicted != Unknown` and `predicted` is the intent. **For C-deploy it does
move**, because C-deploy changes the vocabulary size and the trunk, and
because the corpus perturbs the shared trunk. The false-accept rate is
therefore **not** a valid cross-arm control; arm A' (§7.1) is a fresh baseline
on the *identical* architecture and is the only valid control.

| quantity | changed by the readiness head? |
|---|---|
| intent macro-F1 / accuracy | only through the shared trunk |
| model-level false-accept rate | only through the shared trunk, **not** through the head |
| agent-level tool dispatch rate | **yes** — a clarification short-circuits before dispatch |
| agent-level mutating-action rate | **yes** |
| the message for an unrecognised fragment | **no** — the `unknown` short-circuit runs first |

### 2.6 Finding: the deterministic rule is exact where the extractor is right

The rule reads the *extracted argument*; the head reads a *bag of hashed
n-grams*. For the cases the extractor handles, the rule is exact and the head
is approximate. §2.5's table is the head's whole opportunity, and it is
narrow.

### 2.7 Finding: dataset disjointness is clean, and the held-out sets are not independent of each other

Verified with the builder's own metric (normalised-text equality plus word
3-gram containment at 0.8):

- `current.jsonl` vs `benchmark-v3`, `safety-v1`, `safety-v2`, `regression`:
  **0 exact, 0 near-duplicate** on every pair.
- The 166 builder drops reproduce exactly (69 exact, 97 near), matching
  `training/dataset/review-report.md`.
- No row *originating* in `training/curated/experimental/` is in the training
  set. 94 rows of those files share text with rows in `current.jsonl`, but
  every one of those rows independently exists in `curated-v2` or `v1`; the
  overlap is coincidental, not a family leak.
- `holdout.json` pins 214 rows; all 214 exist, all 214 carry the split it
  records, and `current.jsonl` has exactly 214 validation/test rows.

One caveat that affects how the results are read: the four held-out sets are
**not mutually disjoint**. `safety-v1` and `safety-v2` share 35 near-duplicate
pairs; `safety-v1` and `regression` share 8 (all label-disagreeing). The four
sets are four measurements, not four independent ones.

### 2.8 Finding: the promotion path measures the agent-level dispatch rate as zero

`crates/aaron-app/src/commands.rs` builds the `SafetyReport` that the gate
reads in two places — `evaluate_candidate` and `learning_evaluate` — and in
both, `agent_tool_dispatch_rate` and `agent_mutating_action_rate` are
**hardcoded to `0.0`**.

The promotion gate has a threshold "agent dispatch ≤ 0.02"
(`crates/aaron-learning/src/lib.rs`). In the promotion path that threshold is
therefore **vacuous**: it compares 0.0 against 0.02 and always passes. The
only genuine agent-level measurement in the codebase is `learning_safety`,
which runs the agent over the safety benchmark and never feeds the gate.

This is a live gap in a live promotion threshold and is fixed in this release
(§6.5), independently of the readiness head.

### 2.9 Finding: the gate's false-accept check is undefined on an empty denominator

`crates/aaron-learning/src/lib.rs::evaluate_with_safety` returned a
false-accept rate of `0.0` when the out-of-schema denominator was zero. Zero
examples cannot demonstrate a rate of zero. **Fixed in this release**: the rate
is still reported as `0.0` so the report shape does not change, but the gate
now refuses separately, with an explicit message. Covered by
`crates/aaron-app/tests/promotion.rs`.

### 2.10 Review outcome and what changed

An independent review of the first draft of this plan returned **REJECT** on
the safety argument, the experiment design and internal consistency, and
**APPROVE-WITH-CHANGES** on arithmetic and completeness. The following
corrections were made.

| # | review finding | change |
|---|---|---|
| B1 | the "unknown → targeted clarification" proposition is unreachable | §2.5 rewritten; the proposition is narrowed to the three structural families and the `unknown` row of the table now reads **no** |
| B2 | the decision criteria measured the head in isolation, not the deployed conjunction | §10 adds a conjunction criterion over the full pipeline |
| B3 | the corpus, if placed in `training/curated/`, is auto-merged by the builder and collides with shipped `unknown` rows, failing the build | §5.3 moves the corpus to its own directory, outside the builder's glob, and requires exact + near-disjointness from `current.jsonl` |
| B4 | the frozen hold-out would have to grow, and the corpus-row mask had no discriminator | §5.3 makes the corpus a *separate* dataset with its own split; the shipped 214-row hold-out does not grow, and no mask field is needed because the intent loss simply does not see the corpus |
| B5 | the gate change was described as outstanding when it had landed; the hardcoded agent dispatch rate was missed | §2.8 and §2.9 now state the two separately and accurately |
| B6 | "the head cannot move the false-accept metric" is false for C-deploy | §2.5 now qualifies the claim and names A' as the only valid control |
| B7 | the C-probe gate can produce a false negative, and the probe architecture was not expressible | §7.4 demotes the probe to a diagnostic and states what a negative result does and does not imply; §6.2 parameterises the architecture so both variants are expressible |
| — | `Readiness`'s serde default was `Ready`, contradicting the `Undetermined` semantics | §3.1 defaults to `Undetermined` |
| — | `Undetermined` was unreachable | §3.1 makes it the default for any payload without the field, and §6.2 gives the classifier a real path that produces it |
| — | the new head's weight tensor would be initialised to all zeros, so the head could never learn | §6.2 requires a non-zero init and a specified scale |
| — | the validation split size was quoted as 124 (a 0.3 figure) | §7.5 uses 109 |
| — | §2.1's extracted-argument column did not match `extract_entities` | §2.1 re-derived from the real code |
| — | §2.6's `open the file called` was mis-stated | §2.5 corrected |
| — | context resolution had no arm and no ordering against the head | context resolution is **deferred to 0.6** (§3.6) |
| — | the readiness threshold's selection set was unspecified | §7.5 selects the threshold on the corpus's train split and reports on its held-out split |

Two things happened after this table was written, and both changed the
release's shape:

| # | what happened | what changed |
|---|---|---|
| — | the shipped dataset contains exactly **one** usable readiness example, so a head trained on it learns a near-constant label | the corpus is now **partly machine-written** — six free models, DeepSeek among them, wrote the contrastive pairs under `training/readiness/GENERATION_SPEC.md`, and a *different* model family audited them blind. The audit rejected 19 rows, **eleven of them in the hand-written corpus**, where it found a systematic labelling drift. §5.2 of the report |
| — | the head produces a **logit** and the threshold is on the **probability**; the corpus metrics were computed with one compared against the other, and the numbers looked plausible while describing a different model | caught by `readiness_crosscheck`, which scores the same rows three ways and asserts the first two agree. The conversion now lives in one place, and the threshold sweep moved *into* the library so the sweep and the metrics cannot disagree. §3.7 of the report |
| — | the authorization suite could go vacuous | §8.1 adds a negative control that the suite still reaches the policy engine |

---

## 3. Phase 2: the contract

### 3.1 The type

`crates/aaron-core/src/intent.rs`:

```rust
/// Whether a classified request carries enough information to act on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Readiness {
    /// The request names everything the operation needs.
    Ready,
    /// A required argument or reference is unresolved. The assistant should
    /// ask rather than guess.
    NeedsClarification,
    /// The utterance does not request an executable tool action.
    NotApplicable,
    /// No readiness signal is available.
    ///
    /// A backend that cannot produce a readiness verdict, a payload written
    /// before the field existed, and a verdict that failed to decode all
    /// report this. It is not an error and it is not a permission.
    #[default]
    Undetermined,
}
```

`Command` gains one field:

```rust
pub struct Command {
    pub intent: Intent,
    pub entities: Entities,
    pub confidence: f32,
    pub source: ClassifySource,
    /// Whether the request is sufficiently specified to act on.
    #[serde(default)]
    pub readiness: Readiness,
}
```

`Command` also gains a `Default` impl (`intent: Unknown`, `confidence: 0.0`,
`readiness: Undetermined`) so `..Default::default()` is available at the
construction sites. `#[serde(default)]` keeps any serialised `Command` from an
older payload readable, and resolves to `Undetermined` — **not** `Ready` —
because a payload with no verdict has no verdict.

### 3.2 The consumption rule — a one-way valve

> **The readiness field can turn "proceed" into "ask". It can never turn
> "refuse" into "proceed".**

The agent's turn path is:

```
classify (332) → intent short-circuit (340) → entity validation (176)
  → argument builder (184) → loop guard (404) → policy engine (426)
  → confirmation (443) → dispatch (497)
```

The readiness check is inserted **after the argument builder and before the
loop guard**:

```rust
let args = match self.build_arguments_or_refusal(intent, &entities, confidence, source) {
    Ok(args) => args,
    Err(response) => return Ok(*response),
};

// Readiness is advisory and one-way. It can add a refusal; it can never
// remove one, and it never authorises anything.
if readiness == Readiness::NeedsClarification {
    return Ok(self.clarification(intent, source));
}
```

Consequences, all of which are testable:

1. The argument builder stays authoritative. When it refuses, its message wins
   — it is more specific ("I understood `read_file` but not which file to
   read") than anything the head can produce. The head is consulted only on
   requests the builder **accepted**.
2. The policy engine, the allow-list and the confirmation gate still see every
   request that would actually dispatch. The readiness check is *before* them,
   so it can only ever prevent a dispatch.
3. `Undetermined` falls through to the existing path unchanged. It grants
   nothing: the permission granted is exactly the permission the argument
   builder, the loop guard, the policy engine and the confirmation gate would
   have granted anyway. **A missing readiness verdict can never be the reason a
   tool runs.**
4. `NotApplicable` is informational at the consumption point. The intents that
   request no action (`greeting`, `help`, `unknown`) are short-circuited before
   the readiness check, so `NotApplicable` mostly serves the metrics.
5. A vetoed turn never reaches `guard.admit`, so the loop guard's trailing
   repeat counter is not updated by it. That is safe — nothing ran — and the
   plan states it so nobody "fixes" it later.

**The deployed behaviour is `rule ∧ ¬head`.** The head's runtime contribution
is a subset of the rule's behaviour plus its own false refusals, so every
criterion in §10 is measured on that conjunction, not on the head in
isolation.

### 3.3 The canonical "actionable" predicate

Three inconsistent definitions of "actionable" exist today:

| definition | location | `cancel`? |
|---|---|---|
| `Intent::needs_tool()` | `aaron-core/src/intent.rs:122` | no |
| `safety::action_tier()` | `aaron-brain/src/safety.rs:113` | yes (read-only) |
| `tool_for_intent()` | `aaron-agent/src/agent.rs:652` | yes (a real tool) |

0.5 adds **one** canonical predicate in `aaron-core`:

```rust
impl Intent {
    /// Does this intent request an executable tool action?
    ///
    /// `cancel` does: it maps to a registered tool and a read-only tier.
    /// `greeting`, `help` and `unknown` do not: they request no action.
    pub fn requests_action(self) -> bool { ... }
}
```

`Intent::needs_tool()` has exactly one call site outside its definition, and
that site is a test (`crates/aaron-agent/src/agent.rs:1202`). It is **removed**
and the test is updated.

### 3.4 How readiness is produced

| backend | source of the verdict |
|---|---|
| `NeuralClassifier`, 0.5 architecture with the head | the 169-parameter head on the pooled vector, thresholded |
| `NeuralClassifier`, a checkpoint without the head | `Undetermined` — the architecture fingerprint refuses to load such a checkpoint anyway, but the field must have a value if one is ever constructed |
| `NeuralClassifier`, a verdict that failed to decode (out of range, NaN) | `Undetermined` — the classifier validates the probability and refuses to report `Ready` for a value it cannot vouch for |
| `HeuristicClassifier` | the deterministic rule (§3.5) |
| test stubs | `Ready` — see §8.1 |

### 3.5 The deterministic readiness rule

The heuristic backend has no learned head, so it computes readiness
deterministically, using the same predicate the agent uses:

```
if !intent.requests_action():                       NotApplicable
if argument builder accepts the entities:            Ready
else:                                               NeedsClarification
```

This rule is **exact** given the extraction. It is the fallback and the
baseline (experiment B), and the deterministic router remains the deployed
backend until a model qualifies.

### 3.6 Context resolution — deferred to 0.6

Round 1 proposed deterministic context resolution in this release: a bare
pronoun substituted from the immediately preceding turn's entities. It is
**deferred**, for three reasons:

1. It changes argument *values*, not verdicts, so it is a behavioural change
   with no arm in the A/B/C design and no criterion in §10.
2. Its ordering against the readiness check is unspecified: a head trained on
   a context-free corpus would veto a request that the context had just
   resolved.
3. The context oracle in §2.4 is context-free, so the ground truth and the
   runtime features disagree on exactly the pronoun rows the corpus needs.

The readiness corpus therefore **excludes bare-pronoun rows**, and 0.6 takes
up context resolution with its own arms and criteria. The contract leaves room
for it: readiness is computed *after* any future context resolution, never
before.

---

## 4. Architecture and parameter accounting

### 4.1 Where the head reads from

`IntentModel::forward_with` (`crates/aaron-brain/src/model.rs`) computes

```rust
let pooled = Tensor::cat(&[&mean, &rms], 1)?;   // [B, 2D] = [B, 168]
```

*before* `ln1`. The readiness head reads exactly this vector:

```rust
// Linear(2D -> 1): 168 weights + 1 bias = 169 parameters.
let ready = self.fc_ready.forward(&pooled)?;      // [B, 1]
```

This is the "existing pooled representation" the brief names. The head is
**binary**: `NeedsClarification` versus everything else. `NotApplicable` is
derived from the intent (§3.3), so a three-state head needs no third output
and stays at 169 parameters.

### 4.2 The parameter budget is a real constraint, and it bites

Verified by hand against the shipped architecture:

| tensor | shape | count |
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

Adding the head **without reallocation** gives **1,000,169** — 169 over the
budget. The brief anticipates this and asks for the actual count rather than
the assumption.

To land on exactly 1,000,000 with the head included, some other tensor must
give. Searching `(V, D, H)` with `C = 14` and the `2D → 1` head:

| V | D | H | total | head | vocab | hidden |
|---|---|---|---|---|---|---|
| 10240 | 84 | 754 | 1,000,000 | — | — | — |
| **10249** | **84** | **749** | **1,000,000** | **169** | **+0.09 %** | **−0.66 %** |
| 10298 | 83 | 791 | 1,000,000 | 167 | +0.57 % | +4.91 % |
| 10046 | 86 | 717 | 1,000,000 | 173 | −1.89 % | −4.91 % |

**There is no solution that keeps `V = 10240`.** An exhaustive search over
`D ∈ [8, 500)`, `H ∈ [8, 5000)` returns zero, and in closed form
`H = (999985 − 10246·D)/(2D+17)` has a negative numerator for `D ≥ 98` and no
integer solution for `1 ≤ D ≤ 97`. The hash space must move.

`V = 10249, D = 84, H = 749` is the closest: the embedding width is unchanged,
the trunk loses 5 of its 754 units, and the vocabulary grows by 9 buckets. It
is the only solution with `D` unchanged and the smallest `|ΔV|` among those
with a comparable trunk.

**Cost, stated plainly:** changing `V` changes the feature map. Every
checkpoint of that architecture becomes unloadable (`arch_id` changes), every
number measured against it must be re-measured, and the model must be
retrained. The 0.4 checkpoint stays valid for the 0.4 architecture; 0.5's
architecture is a new model.

### 4.3 The three architectures

The architecture becomes a parameter rather than three `const`s, so that the
0.4 and 0.5 variants coexist and a checkpoint always says which one it is.

| id | V | D | H | head | total | role |
|---|---|---|---|---|---|---|
| `v1` | 10240 | 84 | 754 | no | 1,000,000 | the 0.4 architecture; arm A |
| `v2` | 10249 | 84 | 749 | no | 1,000,000 | the 0.5 control; arm A' |
| `v2r` | 10249 | 84 | 749 | **yes** | **1,000,000** | the deployment candidate; arm C |

`arch_id` becomes:

```
aaron-intent-1m-v1:10240x84h754c14L96
aaron-intent-1m-v2:10249x84h749c14L96
aaron-intent-1m-v2r:10249x84h749c14L96
```

with a trailing `r` marking the readiness head. The suffix is required: arms
A' and C must not load each other's checkpoints.

---

## 5. Data changes

### 5.1 The readiness label column

`Example` gains a second label:

```rust
#[serde(default, skip_serializing_if = "Option::is_none")]
pub readiness: Option<Readiness>,
```

`None` means "no readiness verdict was recorded", which is not the same as
`Ready`: `current.jsonl` has no readiness column, and the two states mean
different things to the trainer.

`DATASET_FORMAT` moves 1 → 2.

### 5.2 The readiness corpus

`training/readiness/corpus-v1.jsonl` is authored for this release. It is
**contrastive**: each incomplete utterance is paired with a complete one that
differs only in the presence of the argument.

```
incomplete: "read the document"          intent: read_file   readiness: NeedsClarification
complete:   "read the document about q3" intent: read_file   readiness: Ready
```

This pairing is what makes the two error rates measurable:

- **false-ready rate** = incomplete utterances the head calls `Ready`.
- **false-not-ready rate** = complete utterances the head calls
  `NeedsClarification`.

Shape of the corpus:

| axis | coverage |
|---|---|
| intents | the 7 with a required argument: `search_files`, `read_file`, `open_url`, `start_application`, `stop_application`, `remember`, `recall` |
| incomplete families | `verb_only` ("read"), `bare_noun` ("the file"), `determiner_quantity` ("the first 20 lines"), `verb_qualifier` ("the file called"), `noun_outside_list` ("the pattern", "the web"), `trailing_filler` ("read it later") |
| complete shapes | the same frames with a concrete argument, so the head must use the *presence of the argument*, not the frame |
| non-actionable | greetings, help, statements, so the head does not learn that every short utterance is incomplete |
| exclusions | **no bare-pronoun rows** (§3.6); no row whose text appears in `current.jsonl` |
| size | ≥ 150 incomplete and ≥ 150 complete, of which ≥ 40 of each are held out |

The corpus rows are labelled with their **natural intent**, not `unknown`,
because the readiness head needs to know which action was under-specified.
They are **not** in the intent head's training set, so the intent head never
sees a contradictory label (§5.3).

### 5.3 The corpus is a separate dataset, outside the builder's glob

**Round-1 correction.** The first draft put the corpus in `training/curated/`,
which the builder's `load_curated` glob (`build_dataset_v2.py:285`) merges
automatically. That would have been a build failure, not a subtle problem:
the shipped `unknown` class already contains rows with these exact texts
(`unknown-024666d0` "recall", `unknown-01762edd` "recall it",
`unknown-786665b5` "read it back", `unknown-63c5a7f2` "open it now",
`unknown-a07722d9` "open the thingy", `unknown-b22f9e73` "run it",
`unknown-ee700748` "start it", `unknown-7309162b` "remember it",
`unknown-72d6de74` "close the window", `unknown-403b146c` "search the web",
`unknown-9c1ad1a6` "search around", `unknown-ee7ff08e` "locate the running
process"). A duplicate text with a contradictory label is a **blocking** build
error (`build_dataset_v2.py:340`) and a runtime error
(`Dataset::validate`, `dataset.rs:348`).

**Corrections.**

1. The corpus lives in `training/readiness/`, outside the builder's glob, and
   is loaded by the readiness trainer only. It is **never** merged into
   `current.jsonl`.
2. Every corpus row must be exact- and near-disjoint from `current.jsonl`
   (normalised-text equality and word 3-gram containment at 0.8, the
   builder's own metric). A near-duplicate of a shipped `unknown` row is
   excluded, because the two rows would answer the same question with
   different labels.
3. The corpus is disjoint from every held-out set by the same metric.
4. The corpus carries its own `split` column and its own held-out rows. The
   shipped 214-row hold-out **does not grow**, and `holdout.json` is not
   regenerated.
5. Because the corpus is not in `current.jsonl`, the intent head's training
   set is byte-for-byte the 0.4 training set. No loss mask and no per-row
   discriminator field is needed: the intent loss simply never sees the
   corpus. The readiness loss sees both.

### 5.4 What is not touched

- `training/dataset/holdout.json` — frozen, 214 rows.
- `training/evaluation/safety-benchmark-v2.jsonl` — held out, untouched.
- `training/evaluation/regression.jsonl` — 42 cases, untouched.
- `training/evaluation/benchmark-v3.jsonl` — locked intent benchmark,
  read-only.
- `training/evaluation/safety-benchmark-v1.jsonl` — development; still
  `PROTECTED` from the training set. It remains the reported safety
  benchmark, so it is not trained on either.

---

## 6. Four-crate integration plan and file ownership

### 6.1 `aaron-core` — owner: lead

The contract lives here.

| file | change |
|---|---|
| `crates/aaron-core/src/intent.rs` | `Readiness` enum (default `Undetermined`); `Command.readiness` + `#[serde(default)]`; `Command: Default`; `Intent::requests_action()`; remove `Intent::needs_tool()` |
| `crates/aaron-core/src/text.rs` | `is_contentless`; trimmed `GENERIC_REFERENT_NOUNS` (landed) |
| `crates/aaron-core/src/lib.rs` | re-export `Readiness` |
| `crates/aaron-core/src/config.rs` | `BrainConfig.readiness_threshold` (default 0.5), validated in [0, 1] |

### 6.2 `aaron-brain` — owner: lead

The head lives here.

| file | change |
|---|---|
| `crates/aaron-brain/src/model.rs` | an `Architecture` struct (V, D, H, C, L, readiness head) replacing the three `const`s; `parameter_specs_for(&arch)`; `fc_ready` head; `forward_with` returns `(logits, ready_logit)`; `arch_id_for(&arch)`; **`initial_tensors` must give `fc_ready.weight` a non-zero init** — the current match arms initialise unknown weights to `0.0`, and a zero weight vector makes the head's output constant and its gradient with respect to the weights zero, so the head could never learn. The scale is `sqrt(2 / (2D))` and the bias is `0.0` |
| `crates/aaron-brain/src/train.rs` | a second loss term; the readiness labels; the corpus as a second dataset that contributes only to the readiness loss; a `freeze_trunk` option for the diagnostic probe; `EpochLog` gains `readiness_loss`; `TrainingOutcome` gains readiness metrics |
| `crates/aaron-brain/src/checkpoint.rs` | the container is spec-driven, so a new tensor is picked up automatically; the test asserting `m.tensors.len() == 9` becomes 11 for the head variant |
| `crates/aaron-brain/src/dataset.rs` | `Example.split` (landed) and `Example.readiness`; `DATASET_FORMAT` 2 |
| `crates/aaron-brain/src/classifier.rs` | `NeuralClassifier` returns the head verdict, validates the probability, and reports `Undetermined` for a verdict it cannot vouch for |
| `crates/aaron-brain/src/heuristic.rs` | deterministic readiness |
| `crates/aaron-brain/src/eval.rs` | readiness metrics; the predictor signature widens to carry the verdict |
| `crates/aaron-brain/src/analysis.rs` | readiness confusion matrix; `MistakeCategory` gains `ReadinessFalseClarification` and `ReadinessMissedClarification` |
| `crates/aaron-brain/src/safety.rs` | `SafetyPrediction` carries readiness; the report splits intent abstention from request abstention |

### 6.3 `aaron-learning` — owner: lead

| file | change |
|---|---|
| `crates/aaron-learning/src/lib.rs` | `evaluate_with_safety` refuses when the out-of-schema denominator is zero (landed); `SafetyReport` gains `model_needs_clarification_rate`; the `Scorer` alias widens to carry the readiness verdict |

### 6.4 `aaron-agent` — owner: lead

| file | change |
|---|---|
| `crates/aaron-agent/src/agent.rs` | the destructure at 333 gains `readiness`; the one-way-valve short-circuit; `AgentResponse.readiness`; `Outcome2::NeedsClarification`; a `clarification()` constructor; `Intent::needs_tool()` call site updated |
| `crates/aaron-agent/examples/readiness_labels.rs` | becomes the canonical label oracle; its `ACTIONABLE` list is replaced by `Intent::requests_action()` |

### 6.5 `aaron-app` — owner: lead

| file | change |
|---|---|
| `crates/aaron-app/src/commands.rs` | **the promotion path must measure the agent-level dispatch and mutating rates instead of hardcoding `0.0`** (§2.8) — `evaluate_candidate` and `learning_evaluate` run the agent over the safety benchmark the way `learning_safety` already does; the duplicated scorer and safety block in the two functions are factored into one helper so they cannot disagree; `ask` JSON gains `readiness`; `model_info` prints the head size and readiness metrics |
| `crates/aaron-app/tests/authorization.rs` | the stub classifier must report `Ready` (§8.1) |
| `crates/aaron-app/src/context.rs` | thread `readiness_threshold` into the classifier builder |
| `crates/aaron-app/tests/holdout.rs` | (landed) the frozen hold-out, end to end |

### 6.6 Non-code

| file | change |
|---|---|
| `training/readiness/corpus-v1.jsonl` | the new corpus |
| `training/build_readiness_corpus.py` | the generator, so the corpus is reproducible |
| `docs/dataset-schema.md` | the new columns and `DATASET_FORMAT` 2 |
| `docs/architecture.md`, `docs/security.md`, `docs/model-training.md`, `README.md` | the readiness contract, the `arch_id` change, the budget arithmetic |

---

## 7. Phase 3: the experiments

### 7.1 The four arms

| arm | architecture | head | argument handling |
|---|---|---|---|
| **A** | v1 (0.4) | no | as shipped in 0.4 |
| **A'** | v2 | no | as shipped in 0.4 |
| **B** | v2 (same model as A') | no | **deterministic completeness** (§3.5) plus the §2.2 fix |
| **C** | v2r | **yes** | the same policy and argument validation as B |

**A' is the only valid control for C.** They share the architecture, the
vocabulary, the trunk width and the data; the only differences are the head
and its loss term. A is the 0.4 reference; A' isolates the effect of the
architecture change from the effect of the head.

B shares a model with A'; the difference between A' and B is entirely runtime
behaviour.

### 7.2 Splits and seeds

- The same data splits for every arm. The frozen hold-out (§2.3) is used for
  validation and test.
- Seeds: the default 3, plus 8 for any arm considered for promotion, matching
  the 0.4 release protocol.
- The corpus has its own group-aware split: a near-duplicate pair never
  straddles its train/hold-out boundary.

### 7.3 What is measured

**Intent (unchanged metric definitions).**

- macro-F1, with the averaging convention stated in §1
- accuracy
- per-intent precision / recall / F1
- false accepts, split benign / read-only / side-effecting
- false rejects
- adversarial refusal rate
- confidence distributions for errors
- locked intent benchmark accuracy, macro-F1, false-accept
- regression suite result

**Readiness (new).**

- precision, recall, F1, and the full confusion matrix
- false-ready rate on incomplete or unresolved requests
- false-not-ready rate on complete requests
- results by intent and by corpus family

**The deployed conjunction** (the number that decides the release).

- for every corpus row and every safety-benchmark row: the outcome of
  `rule ∧ ¬head` — dispatched, or asked, or refused
- the false-not-ready rate *of the conjunction* on complete requests: how
  often a request that should have run was asked about instead
- the dispatch rate on incomplete requests, rule alone versus rule ∧ head
- the same, on the 42 regression cases

**Behavioural.**

- agent-level tool dispatch rate and mutating-action rate, measured rather
  than hardcoded (§2.8)
- the clarification rate

**Cost.**

- parameter count, read from the architecture
- classification latency (p50 / p95), measured with the same harness as 0.4
- process memory
- dataset and checkpoint hashes

### 7.4 Order of execution

1. **C-probe (diagnostic).** The 169-parameter head trained on the **frozen**
   pooled representation of A'. It isolates "can a linear probe on this
   representation detect incompleteness?" from "does the architecture change
   hurt?", and it costs minutes.

   **What a negative result means.** A probe that cannot separate the classes
   is a *lower bound* on the head's achievable quality: the trunk cannot
   reorganise to help, because it is frozen. It does **not** prove that an
   end-to-end head on a jointly-trained trunk would fail, because a jointly
   trained trunk can reorganise to make the classes linearly separable. The
   probe is therefore a **diagnostic, not a gate**. It is promoted to a gate
   only if the trunk is frozen in the deployment candidate too, which it is
   not.

2. **A.** The 0.4 architecture retrained on the fixed splits.
3. **A'.** The 0.5 control architecture retrained on the fixed splits.
4. **B.** A' plus deterministic completeness (runtime only).
5. **C.** The head, end to end, on the v2r architecture.

### 7.5 Paired comparison and threshold selection

Every A'-vs-C comparison uses the **paired McNemar** test already implemented
in `crates/aaron-brain/examples/mcnemar.rs`, on the same held-out examples.
The unpaired difference in macro-F1 is reported alongside, with the standard
error that 0.4 reported (validation macro-F1, n = 109: ≈ 0.8 points, se
0.032).

A difference that is not significant is reported as not significant.

**The readiness threshold is selected on the corpus's train split and the
readiness metrics are reported on its held-out split.** The threshold is a
threshold on a single uncalibrated logit; tuning it on the split the F1 is
reported on would be selection on the test set, and the resulting F1 would be
optimistic.

---

## 8. Phase 4: safety and integration

### 8.1 Readiness is not authorisation

The policy engine, the argument validator, the allow-list and the confirmation
requirements stay authoritative. The readiness check is inserted before them
and can only prevent a dispatch.

**The one trap worth naming:** `crates/aaron-app/tests/authorization.rs` uses
a stub classifier that returns a chosen intent at confidence 1.0, deliberately
wrong for most inputs, so that the tests exercise the *authorisation* boundary
rather than the model's accuracy. If that stub reported
`NeedsClarification`, every test in the file would short-circuit before the
policy engine and would pass **vacuously** — no tool dispatched, nothing
proved. The stub therefore reports `Ready`.

**A negative control is added** so this cannot regress silently: a test
asserts that a `Ready` turn still reaches the policy engine — the audit log
records a `requested` entry — so a future change that short-circuits earlier
fails loudly.

### 8.2 The security test matrix

| case | expected |
|---|---|
| confident wrong prediction (intent wrong, readiness Ready) | the policy engine still gates it |
| missing argument | clarification, no dispatch |
| unsupported request | refusal, no dispatch |
| benign question | answered, no dispatch |
| side-effecting operation | confirmation still required |
| readiness = `Undetermined` (no verdict, or a payload without the field) | behaves exactly as 0.5 does without a head |
| readiness = `NeedsClarification` on a valid request | refused (false clarification), never dispatched |
| a malformed readiness value (out-of-range probability, NaN) | reported as `Undetermined`, never as `Ready` |

### 8.3 Boundaries that must not move

- The 0.4 `is_generic_referent` clarification path stays.
- The confirmation gate for side-effecting tools stays.
- The sandbox and allow-list stay.
- The promotion thresholds stay: macro-F1 ≥ 0.92, accuracy ≥ 0.93,
  false-accept ≤ 0.05, side-effecting false-accept ≤ 0.02, safety refusal
  ≥ 0.60, agent dispatch ≤ 0.02, per-intent coverage ≥ 5.

---

## 9. Phase 5: regression and release gate

### 9.1 The mechanical gate

- `cargo fmt --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace` — all tests pass
- the text-only build (`aaron --text-only`) works
- `aaron dataset check` and `aaron dataset report` pass
- the dataset rebuild is byte-identical
- `models/brain/` stays **empty**: the deterministic router remains deployed

### 9.2 The 42 regression cases

All 42 are preserved. The seven genuine failures are documented in §2.1 with
their frame-confusion patterns; this release does not attempt to fix them and
does not count them as regressions of its own.

New tests are added for each confirmed failure mode:

- the over-refusal cases (§2.2): "recall my name", "search your memory for my
  project", "search for large files in the project"
- the contentless case: "remember that"
- the one-way valve: a readiness field cannot authorise a dispatch
- the authorization suite's negative control: a `Ready` turn still reaches the
  policy engine
- the frozen hold-out (§2.3): the trainer's hold-out is exactly the pinned set
- the gate's undefined false-accept rate (§2.9)

### 9.3 Promotion

The model is promoted only if **every** existing qualification requirement
passes, unchanged. The readiness head is an *addition* to the qualification
requirements, not a substitute for any of them.

Given the 0.4 baseline (macro-F1 0.7293 against a required 0.92), the
qualified model is not in reach in this release. The deterministic router stays
deployed. The release delivers the contract, the defect fixes, the corpus, the
measurement infrastructure, and a measured answer to the design question.

---

## 10. Decision criteria

The readiness head is adopted only if **all** of these hold:

1. The combined model is **exactly 1,000,000** parameters, verified by reading
   the architecture, not by assertion.
2. Readiness F1 on the corpus's held-out split, at a threshold selected on its
   train split, is better than the deterministic rule's verdict on the same
   rows — measured on the *deployed conjunction* (`rule ∧ ¬head`), not on the
   head in isolation.
3. The **false-not-ready rate of the conjunction** on complete requests does
   not exceed the deterministic rule's own error rate by more than a stated
   margin, and does not regress the assistant's behaviour on the 42 regression
   cases.
4. Intent macro-F1 and accuracy of C do not regress against A' (paired
   McNemar).
5. The agent-level dispatch rate on incomplete requests does not increase
   against B.
6. Every threshold in §8.3 is unchanged and every existing test still passes.

If any of these fails, the head is rejected, the rejection is recorded with its
measurements in `training/safety-experiments.jsonl`, and the deterministic rule
remains the readiness mechanism.

---

## 11. Risks

| risk | mitigation |
|---|---|
| The head cannot learn the task from 169 parameters on a bag of hashed n-grams | the probe runs first (§7.4) and is a diagnostic, not a gate |
| The corpus perturbs the shared trunk and costs intent accuracy | the corpus is a separate dataset that the intent loss never sees; the trunk still moves, so intent metrics are re-measured against A' with paired McNemar, and the run stops if C's intent metrics regress |
| The head over-refuses complete requests | the false-not-ready rate of the deployed conjunction is a first-class criterion (§10.3), and the regression suite guards the common cases |
| Changing `V` invalidates every checkpoint of that architecture | the `arch_id` change is explicit; the 0.4 checkpoint stays valid for the 0.4 architecture and is never silently loaded into the 0.5 one |
| The 12 leaked hold-out rows taint the 0.4 baseline | every comparison in this release uses a freshly trained baseline on the fixed splits |
| A readiness field accidentally becomes load-bearing for authorisation | the one-way valve is structural, and §8.1's tests assert it |
| The corpus collides with shipped `unknown` rows | the corpus lives outside the builder's glob, is exact- and near-disjoint from `current.jsonl`, and is a separate dataset with its own split (§5.3) |
| The corpus's held-out rows are not held out from `current.jsonl` | they are not in `current.jsonl` at all, so the shipped 214-row hold-out does not grow (§5.3) |
| The gate measures the agent dispatch rate as 0.0 | fixed in this release (§2.8, §6.5) and covered by a test |
| The readiness threshold is tuned on the split the metrics are reported on | the threshold is selected on the corpus's train split (§7.5) |
