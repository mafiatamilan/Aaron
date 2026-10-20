# AARON 0.5 — Explicit Readiness Modeling

**Decision: the readiness head is REJECTED. The deterministic router remains
deployed. No model is promoted.**

The hypothesis was that a 169-parameter head reading the model's pooled
representation would tell, better than a closed list of generic nouns,
whether a request names enough to act on. It was measured, not assumed, and
the corpus it was measured on was **written by DeepSeek and three other model
families** and then **audited blind**.

The head learns the task. Its F1 on the held-out corpus is **0.683 against
the deterministic rule's 0.471**. And it is still the wrong thing to ship:
there is no threshold at which it is better than the rule on both axes, and the
one thing it buys — an 83.3 % cut in the agent's dispatch rate on adversarial
input — costs the refusal of 0.86 of perfectly good requests.

The release still ships five things that matter:

1. **The readiness contract** — a typed, four-state verdict on
   `aaron_core::Command`, consumed by the agent through a one-way valve that
   can only turn "proceed" into "ask", never "refuse" into "proceed".
2. **Three defect fixes** the readiness work exposed: the generic-referent
   rule over-refused free-text arguments ("recall my name"); the trainer
   ignored the frozen hold-out, training on 12 of its 214 rows; and the
   promotion gate's "agent dispatch ≤ 0.02" threshold was compared against a
   hardcoded `0.0` and could never fail.
3. **A 647-row contrastive corpus**, 53.5 % (346 of 647) of it machine-written
   and audited blind by a different model family than the one that wrote it.
4. **The measurement apparatus** — a frozen-trunk head trainer, a threshold
   sweep computed in one place, and a blind cross-check that caught a silent
   logit-versus-probability bug in the middle of the release.
5. **A measured negative result** with the numbers that produce it.

---

## 1. The baseline

### 1.1 Artefacts

| artefact | sha256 |
|---|---|
| `training/dataset/current.jsonl` (1,103 rows) | `c7ddc5eeb5fc194b89c9442cf0d7e47ba908dafefd59f0cf9941c47c5e77ae6f` |
| `training/dataset/holdout.json` (214 pinned rows) | `6f56cbe7b576713c92327c7f06554729cddd2eadfaa61f28a1153e12c0ac5e8e` |
| `training/readiness/corpus-v1.jsonl` (319, hand-written) | `9575e34e19ddddeac19c277d8f47aa368da75bcf46f4151346459aa59a98f827` |
| `training/readiness/corpus-v2.jsonl` (681, merged) | `2b9c54563d5a5c49b4f1f4b51f3350c65d7cfd9ae83bba7b8166897e610228c5` |
| **`training/readiness/corpus-v3.jsonl` (647, audited)** | `a8e0456a0c80fd45e1e747e34970353a8c29c605b1944ea6109ecf438cd919d5` |
| `training/readiness/rejected-vraft-audit.jsonl` (19 rejected) | `4a570ca03c57e45ee61ce25106a43850e832c4f2079f8b8fcbf6205fa6391066` |
| `training/evaluation/benchmark-v3.jsonl` (locked intent) | `de81f12f0892453ae3f6b5f1ab770e85bf27bb026a78ee78e3ef6501c805e4ee` |
| `training/evaluation/safety-benchmark-v1.jsonl` (dev) | `9dd63460d363959775c98615214fef687989d411e9fe994fe6bdb2620354ae3a` |
| `training/evaluation/safety-benchmark-v2.jsonl` (**held out**) | `b54377f886ca51ffc2080c67385ccefc3db085eb586e94ca33a61b339d3b4f38` |
| `training/evaluation/regression.jsonl` (42 cases) | `d2c3eced6910763c7c9c7d766193ce2be0a442708732bf94e2e7578bdbd64ff4` |
| 0.4 baseline checkpoint | `0ca584cedfa701201d687442a5178ac92284224a44b38e0c3f70788b827c407a` |
| **0.5 baseline checkpoint** (arm A', retrained on the fixed hold-out) | `26450f5997c3b55fc2a6ba095a1bffa97879500fce5eb126b8264342c36db321` |
| readiness probe checkpoint (rejected) | `845a8a749131a7575f96c29fe61a5baa1190888751144716a38da159163433ac` |

`training/evaluation/safety-benchmark-v2.jsonl` was not used for training or for
selecting anything in this release; it is, however, the promotion gate's
default benchmark (`crates/aaron-app/src/commands.rs`, around line 1426) and
the corpus's disjointness check opens it, so the gate's own measurements
involve it.

### 1.2 The 0.4 baseline, as reported

| set | accuracy | macro-F1 | false-accept | regression |
|---|---|---|---|---|
| validation | 0.7778 | 0.8060 | 0.429 | — |
| locked intent benchmark | 0.7231 | 0.7293 | 0.160 | **35/42** |
| safety benchmark v1 (dev) | refused 0.4437 | — | side-effecting 0.2042 | — |

**Macro-F1 convention:** the unweighted mean of the per-intent F1 scores over
all 14 intents, including intents with no support in a given set (they score
0.0). This is `crates/aaron-brain/src/eval.rs`'s convention.

### 1.3 A caveat that applies to every 0.4 number

**The 0.4 validation numbers were measured on a hold-out that was not the
frozen one.** The trainer re-derived the splits from the id hash, per intent,
while the builder hashes the lowest id of a near-duplicate group. The two
algorithms disagree on 32 of 1,103 rows, and 12 rows that `holdout.json` pins
as validation or test were in the *training* split.

```
rows=1103   trainer-split != file-split: 32
pinned hold-out rows: 214   of which the TRAINER puts in train: 12
```

Two consequences: the 0.4 checkpoint **cannot** be honestly re-scored on the
frozen hold-out, because it was trained on 12 of those rows; and the
validation macro-F1 of 0.8060 was measured on a set differing from the frozen
hold-out in 26 of its 214 rows, with early stopping steered by the wrong rows.

Fixed in this release (`crates/aaron-app/tests/holdout.rs`).

### 1.4 The 0.5 baseline (arm A')

The 0.4 architecture, retrained on the fixed hold-out: 100 epochs, three
seeds, lr 1e-2, weight decay 1e-2, selection on validation macro-F1.

| set | n | accuracy | macro-F1 | false-accept | regression |
|---|---|---|---|---|---|
| validation | 109 | 0.7064 | 0.7447 | 0.429 | 32/42 |
| test | 105 | 0.7333 | 0.7351 | 0.429 | 32/42 |
| locked intent benchmark | 130 | 0.7308 | 0.7552 | 0.3200 | 32/42 |
| safety benchmark v1 | 142 | refused 0.4577 | — | side-effecting 0.1972 | — |

**Paired comparison against the 0.4 checkpoint, on the locked benchmark** (the
only set neither model was trained on):

```
A (0.4 baseline)   correct 94/130   false accepts 4/25
B (0.5 baseline)   correct 95/130   false accepts 8/25
paired: both right 78, both wrong 19, A only 16, B only 17
McNemar: chi2 = 0.000 (continuity-corrected), p ~= 1.0000
=> the difference is not significant at this sample size
```

**The two baselines are statistically indistinguishable.** The hold-out fix did
not change the model's quality; it changed the honesty of the measurement. The
locked benchmark's false-accept rate moved from 0.160 to 0.320 — four rows out
of twenty-five out-of-schema examples — and the paired test says that is noise.
One false accept is 4.0 points on that denominator.

The regression suite moved from 35/42 to 32/42, with a *different* set of
failures: three of 0.4's failures are fixed (`abort the test run`,
`the bus was late this morning`, `how do you work`), four still fail, and six
are new. 35 + 3 − 6 = 32. That is seed variance, which the 0.4 report already
measured at 0.669–0.754 on the locked benchmark across three seeds.

---

## 2. What shipped

### 2.1 The readiness contract

`aaron_core::intent::Readiness` — a typed verdict, four states:

| state | meaning |
|---|---|
| `Ready` | the request names everything the operation needs |
| `NeedsClarification` | a required argument or reference is unresolved |
| `NotApplicable` | the utterance requests no executable tool action |
| `Undetermined` | no verdict is available |

`Undetermined` is the **default**, including for a serialised `Command`
written before the field existed. A missing verdict is not `Ready`: a payload
with no verdict has no verdict, and `Ready` would be a claim the payload never
made.

`Command` gains `readiness: Readiness` with `#[serde(default)]`, and
`Command` gains a `Default` impl so construction sites can use
`..Default::default()`.

### 2.2 The consumption rule — a one-way valve

> **The readiness field can turn "proceed" into "ask". It can never turn
> "refuse" into "proceed", and it never authorises anything.**

The agent's turn path:

```
classify (384) → intent short-circuit (392) → entity validation (206)
  → argument builder (469) → [READINESS CHECK] (478) → loop guard (489)
  → policy engine (512) → confirmation (540) → dispatch (590)
```

`Intent::Unknown` is handled earlier: at line 423 the check runs *before* any
argument builder (there is none for an unknown intent) and can turn "I did not
understand" into a clarification. That branch cannot dispatch, so the one-way
valve survives.

The check sits **after** the argument builder and **before** the loop guard,
so:

* the argument builder's message wins when both fire — *"I understood
  `read_file` but not which file to read"* is more useful than anything a
  readiness verdict can say;
* for recognised intents the readiness check only ever sees requests the
  builder *accepted*, and can only refuse them (for `Intent::Unknown` it runs
  earlier, at line 423, and still only refuses);
* the policy engine, the allow-list and the confirmation gate still see every
  request that would actually dispatch. The check is *before* them, so it
  prevents a dispatch rather than permitting one;
* a vetoed turn never reaches `guard.admit`, so the loop guard's trailing
  repeat counter is not updated by it. That is safe — nothing ran — and it is
  stated here so nobody "fixes" it later.

`Undetermined` falls through to the existing path unchanged.

**One known interaction.** Because the check runs *before* the policy engine,
a readiness veto can preempt a policy denial: the user gets "could you be
more specific?" instead of "I will not do that: deny-by-default". Measured on
the regression suite, the head turns four policy denials into clarifying
questions. Nothing is dispatched either way, so the authorisation boundary is
untouched — but the *message* is less informative, and a user who supplies the
extra detail gets refused a moment later. This is why the head is not simply
dropped in: the placement wants revisiting if a readiness head is ever
adopted.

### 2.3 The canonical readiness rule

The rule lives in `aaron_core::readiness`, in the crate both the classifier
and the agent depend on:

```
if the intent requests no action:                 NotApplicable
if a required argument is absent:                  NeedsClarification
if a required argument is a generic referent:      NeedsClarification
if a required argument carries no content:         NeedsClarification
otherwise:                                         Ready
```

Three argument kinds, because one rule cannot answer both questions:

| kind | applies to | rejects |
|---|---|---|
| `Content` | remember text, recall query | an argument that carries no content at all |
| `Referent` | path, application, search pattern | a generic referent ("the file") |
| `Value` | url, and the optional fields | nothing; the shape is the tool's business |

The argument slot table is the *intended* single source of truth for both the
classifier's verdict and the agent's argument builder, but the two are not
mechanically coupled: `crates/aaron-agent/src/agent.rs`'s `build_arguments`
re-encodes the same rules, and the only coupling is the test in
`crates/aaron-app/tests/readiness.rs`, which asserts that the two **agree on
every intent across 45 utterances**. It passes.

One known drift: `readiness_of` rejects a present-but-empty *optional* `Value`,
while the builder filters empty optionals and accepts them. The post-builder
readiness check catches the case, so the effect is an over-refusal, not a
dispatch hole.

### 2.4 The four-crate integration

| crate | what changed | files |
|---|---|---|
| **`aaron-core`** | the contract | `intent.rs` (`Readiness`, `Command.readiness`, `Command: Default`, `Intent::requests_action()`, `Intent::needs_tool()` removed); `readiness.rs` (new: the slot table and the rule); `text.rs` (`is_contentless`, trimmed `GENERIC_REFERENT_NOUNS`); `lib.rs` |
| **`aaron-brain`** | the head | `model.rs` (`Heads`, `fc_ready`, `forward_with_readiness`, `arch_id`, `initial_tensors`); `readiness.rs` (new: the corpus, the trainer, the metrics, the sweep); `train.rs`; `checkpoint.rs` (`readiness_head` in the manifest); `classifier.rs` (the verdict, and one forward pass); `heuristic.rs` (the deterministic verdict) |
| **`aaron-agent`** | the consumer | `agent.rs` (`Outcome2::NeedsClarification`, `AgentResponse.readiness`, the one-way valve, `clarification()`); `state.rs` |
| **`aaron-app`** | the surface | `commands.rs` (`readiness` in the `ask` JSON, the measured agent-level safety report, the shared `score_safety_benchmark` helper, `--readiness-threshold`); `context.rs` (`agent_with`); `tests/authorization.rs` (the stub reports `Ready`); `tests/holdout.rs`, `tests/readiness.rs`, `tests/readiness_corpus.rs`, `tests/safety_measurement.rs`, `tests/promotion.rs` |

`aaron-learning` was touched too: the gate's `SafetyReport` semantics and the
parameter-budget check.

---

## 3. Defect fixes

### 3.1 The generic-referent rule over-refused (fixed)

The 0.4 rule rejected *any* argument that resolved to a closed list of generic
nouns. Measured consequences on the shipped dataset:

| utterance | 0.4 behaviour | 0.5 behaviour |
|---|---|---|
| `recall my name` | refused: *"not what to look for"* | accepted |
| `search your memory for my project` | refused | accepted |
| `search for large files in the project` | refused | accepted |
| `launch code` | refused | accepted |
| `remember that` | stored the word *"That"* as a memory | refused with a clarification |

The rule conflated two questions. *"Is this a reference to a particular
object?"* is right for a path, an application or a search pattern. *"Did the
user actually say anything?"* is right for a free-text argument, because a
topic is content, not a reference. The fix is the three-gate split in §2.3,
plus `is_contentless` for the second case, plus trimming
`GENERIC_REFERENT_NOUNS` of words that are plausible identifiers (`name`,
`code`, `text`, `line`, `number`, `one`).

**All eight of 0.4's NeedsClarification rows were false positives of the
over-broad rule.** After the fix exactly one remains: `search for the file`.

### 3.2 The trainer ignored the frozen hold-out (fixed)

See §1.3. Fixed by making the dataset file's `split` authoritative, with a
partial declaration refused as an error and `assign_stratified` kept as the
fallback for datasets built in memory.

### 3.3 The promotion gate measured the agent dispatch rate as zero (fixed)

`crates/aaron-app/src/commands.rs` built the `SafetyReport` that the gate
reads with `agent_tool_dispatch_rate: 0.0` and
`agent_mutating_action_rate: 0.0`, **hardcoded**, in both `evaluate_candidate`
and `learning_evaluate`.

The gate has a threshold "agent dispatch ≤ 0.02". It was comparing zero
against a maximum and always passing: **a candidate that steered the
assistant into dispatching tools for adversarial input would have been
promoted.** This was the most consequential of the release's findings, and it
was found by the review, not by the readiness work.

The measurement is now real, taken against the **candidate** being promoted,
with the real policy engine, allow-list, sandbox and confirmation requirements,
and a non-interactive confirmer so that scoring a hundred adversarial inputs
neither blocks on stdin nor runs a mutating tool.

The definition is stated precisely: *"the policy engine permitted a tool"* —
the request got past classification, argument validation and the policy
engine, and only the user's confirmation stood between it and running. It is
deliberately not "a tool ran", because measuring that would require granting
the confirmation, which would mean running `run_tests`, `open_url`,
`stop_application` and `remember` on adversarial input.

`learning safety` now reports three numbers: the rate the gate reads, the rate
at which a tool *actually ran*, and the mutating rate.

**The same bug was in `learning safety`**, where the agent level was measured
with the *deployed* classifier while the model level was measured with the
candidate. The two levels were describing two different systems. Fixed.

### 3.4 The gate's false-accept rate was undefined on an empty denominator (fixed)

`evaluate_with_safety` returned a false-accept rate of `0.0` when the
out-of-schema denominator was zero. Zero examples cannot demonstrate a rate of
zero. The rate is still reported as `0.0` so the report shape does not
change, but the gate now refuses separately, with an explicit message.

### 3.5 The promotion path did not enforce the parameter budget (fixed)

The readiness head is 169 parameters *additional* to the intent classifier, so
a checkpoint carrying it has 1,000,169 parameters — and it would have passed
every other gate. The promotion path now refuses any candidate above
`PARAMETER_BUDGET`.

### 3.6 A double forward pass (fixed)

Measuring the latency of the head exposed it: `classify` computed the intent
logits in one forward pass and the readiness logit in a second. Measured at
**1.49 ms against 0.54 ms** — a 168-multiply-add head more than doubling the
cost of a classification. Both now come from one pass.

### 3.7 A silent logit-versus-probability bug (found and fixed mid-release)

The head produces a **logit**; the threshold is on the **probability**. A
logit of 0.30 is a probability of 0.57, so comparing the two directly makes
every threshold below 0.5 fire on almost everything.

This was live for one measurement round. The corpus metrics were computed with
the logit compared against a probability threshold, and the numbers looked
plausible: they described a model that fires on everything below a 0.5
threshold, and they were attributed to a head that does not.

It was caught by `crates/aaron-brain/examples/readiness_crosscheck.rs`, which
scores the same rows three ways — through the classifier, through the
probability, and through the logit — and asserts the first two agree. On the
shipped corpus they disagreed by a factor of more than two: on validation 46
rows were asked through the probability and 23 through the logit; on test, 45
and 16.

Three things came out of it:

* the conversion is now written out in one place, `head_probabilities`, with a
  comment saying why;
* the threshold sweep moved *into* the library, so the sweep and the metrics
  are computed from the same model and the same probabilities and cannot
  disagree — two pieces of code reporting two different numbers for one
  measurement is a bug with no way to tell which side is wrong;
* `crates/aaron-brain/src/readiness.rs` has unit tests for the conversion, the
  confusion rates and the head's parameter count.

---

## 4. The corpus

### 4.1 Why it had to be machine-written

The readiness label of a row is a property of `(intent, entities)`. Run over
the shipped dataset with the real extractor and the real argument builder,
the labels are:

| label | count |
|---|---|
| Ready | 862 |
| **NeedsClarification** | **1** |
| NotApplicable | 240 |

**One.** The shipped dataset contains one usable example of the case the head
exists to detect. A head trained on that learns a near-constant label, and a
head that always says "Ready" is worse than no head, because it can never ask
a question.

The corpus also has to be *contrastive*: each incomplete utterance paired with
a complete one that differs only in the presence of the argument. The
complete rows are the hard negatives — a head that learned "short utterance →
incomplete" would ace them and still be useless.

Writing 600 contrastive pairs by hand is possible but slow, and it is exactly
the kind of mechanical, well-specified, volume work a language model is good
at. So the corpus is **partly model-written**.

### 4.2 How it was made

**Generation.** `training/readiness/GENERATION_SPEC.md` defines the label,
the output format, the six families and the rules. Six slices were generated
by four distinct free models (DeepSeek three times, plus mimo, longcat and
nemotron):

| slice | model | rows |
|---|---|---|
| `read_file` | `mimo-v2.6-flash-free` | 80 |
| `search_files` | `longcat-2.5-preview-free` | 80 |
| `open_url` | `deepseek-v4.1-flash-free` | 80 |
| `start_application` | `deepseek-v4.1-flash-free` | 80 |
| `stop_application` | `deepseek-v4.1-flash-free` | 80 |
| `remember` / `recall` | `nemotron-3.5-super-vl-free` | 160 |

**Mechanical validation** (`training/build_readiness_corpus_v2.py`) — the model
does not get to decide any of this:

* the format, the intent (derived from the slice, or from the leading verb for
  the one slice that holds two intents) and the family;
* the pair structure: an incomplete row must be followed by a complete row in
  the same family and intent;
* uniqueness, per intent;
* exact- and near-disjointness from the shipped dataset, the locked benchmark,
  both safety benchmarks, the regression suite and the accepted curated files,
  using the builder's own metric;
* no bare pronouns standing in for the argument.

116 candidate pairs were dropped: 91 for overlapping a protected set, and the
rest for a verb that did not fit its slice.

**Blind audit** (`training/apply_readiness_audit.py`). Every row was then
audited by a *different* model family, blind: the auditor sees the utterance
and the intent, and never the label it is being asked to confirm.

| auditor | rows | ready | needs clarification |
|---|---|---|---|
| `beam-free` | 250 | 131 | 119 |
| `nemotron-3.5-super-vl-free` | 281 | 122 | 159 |
| `minimax-m3.1-flash-free` | 76 | 44 | 32 |
| `beam-free` (second pass) | 74 | 30 | 44 |
| `beam-free` (third pass) | 2 | 2 | 0 |

**662 of 681 audited rows agreed (97.2 %).** Nineteen rows were rejected, and
fifteen more went with them as their pair partners.

**The audit earned its keep.** Of the 19 rejections, **11 were in the
hand-written rows** — mine, not the models' — and they were a *systematic*
drift rather than random noise. I had written "stop the server", "stop the
daemon", "terminate the worker" and "end the editor" as **complete** requests,
on the reading that "the server" is the one running here. The auditors
applied the corpus's own stated principle consistently — a noun that names a
*kind* of thing is incomplete — and marked them incomplete. They were right,
and a corpus that contradicts itself teaches a classifier the annotator's
inconsistency rather than the concept. The eight generated rows the audit rejected were `close the foreground
window` plus seven `search for the …` rows (board minutes, code review,
meeting transcript, privacy policy, release notes, style guide, user manual).

**A discrepancy worth flagging.** Thirty-four rows were dropped from the
corpus (19 rejected plus their 15 pair partners), but the artefact
`training/readiness/rejected-vraft-audit.jsonl` holds only the 19 directly
rejected ones. That is not a data loss: the partners are removed from
`corpus-v3.jsonl` by `rejected_ids`, but `training/apply_readiness_audit.py`
never writes them to the artefact, because the partner-adding loop tests
`if row["id"] not in {…}` against the set built from the *rejected* list,
which every partner already fails. I have not fixed that script — it belongs
to someone else — so the artefact under-counts the drop by 15 rows. A reader
reconciling "34 dropped" against "19 in the file" should read this, not assume
the numbers are wrong.

### 4.3 The final corpus

| | rows |
|---|---|
| hand-written | 301 |
| model-generated | 346 |
| **total** | **647** |
| by split | 443 train / 109 validation / 95 test |
| NeedsClarification / Ready / NotApplicable | 322 / 315 / 10 |

`training/readiness/corpus-v3.jsonl` is the shipped corpus. `corpus-v1` and
`corpus-v2` are kept so the hand-written and merged corpora can be measured
against the audited one.

`crates/aaron-app/tests/readiness_corpus.rs` pins seven properties of it: the
hold-out is big enough to measure a rate, the corpus is disjoint from every
protected set, no row uses a bare pronoun as its argument, no text is repeated
for one intent, no text carries contradictory labels, and every intent/family
combination holds both sides of the contrast.

---

## 5. The experiment

### 5.1 The design

| arm | model | argument handling | role |
|---|---|---|---|
| **A'** | v1 architecture, retrained on the fixed hold-out | as shipped | the baseline |
| **B** | the same model as A' | deterministic completeness (the rule) | the runtime baseline |
| **C** | A' plus the 169-parameter head, frozen trunk | the same | the hypothesis |

The head is a `Linear(2D → 1)` on the **pooled** representation — the
168-dimensional mean-and-RMS vector, computed before `ln1` — producing one
logit. It is trained on a **frozen trunk**: the optimiser holds only
`fc_ready`, so the intent classifier's outputs are bit-identical before and
after, and the comparison against A' is clean by construction.

The threshold is selected on the corpus's **training** split and the metrics
are reported on its held-out splits. Tuning it anywhere else would be
selection on the test set.

### 5.2 The result — the head learns the task

**Test split (95 rows), at the threshold selected on the training split
(0.39).** False-ready rates are over the 51 incomplete rows; false-not-ready
rates are over the 44 complete rows (43 ready + 1 not_applicable).

| | false-ready | false-not-ready | precision | recall | F1 |
|---|---|---|---|---|---|
| readiness head | 0.1961 | 0.6364 | 0.5942 | 0.8039 | **0.6833** |
| deterministic rule | 0.6863 | 0.0227 | 0.9412 | 0.3137 | 0.4706 |

**Validation split (109 rows).** False-not-ready over 55 complete rows (54
ready + 1 not_applicable).

| | false-ready | false-not-ready | precision | recall | F1 |
|---|---|---|---|---|---|
| readiness head | 0.1667 | 0.5273 | 0.6081 | 0.8333 | **0.7031** |
| deterministic rule | 0.7037 | 0.0000 | 1.0000 | 0.2963 | 0.4571 |

**The head is a much better classifier of the readiness task than the rule.**
It cuts the false-ready rate from 0.686 to 0.196 and raises F1 from 0.471 to
0.683.

Two caveats, both of which favour the head and still do not save it. First,
the rule is scored with a privileged input: `rule_verdict` calls
`readiness_of(example.intent, …)`, so it sees the gold intent while the head
sees only the text. The comparison is therefore favourable to the head, which
strengthens rather than weakens the conclusion. Second, the head was not
trained to convergence: the checkpoint's manifest records `best_epoch: 199`
against a 200-epoch cap, so the run ended on the cap, not on the patience
criterion.

And that is exactly the problem.

### 5.3 The result — and why it is still the wrong thing to ship

A classifier is not a rate; it is a **curve**, and what ships is a point on it.
The point is chosen by the cost of each error, and the two costs are not
equal:

* a **false-ready** (dispatching an under-specified request) is caught by the
  tool's own validator — "path rejected: `the file` does not exist" — so the
  cost is a technical error message;
* a **false-not-ready** (refusing a complete request) is a needless question
  asked of a user who was clear. It happens on *every* ordinary request.

So the operating point has to be compared at a matched false-not-ready rate,
not at a matched F1. **The threshold sweep** (test split; the rule sits at
false-ready 0.6863, false-not-ready 0.0227):

| threshold | false-ready | false-not-ready | F1 | |
|---|---|---|---|---|
| 0.45 | 0.2941 | 0.4773 | 0.6667 | 21× the rule's false-not-ready rate |
| 0.50 | 0.3725 | 0.2955 | 0.6667 | 13× |
| 0.55 | 0.6078 | 0.1591 | 0.5128 | 7× |
| **0.60** | **0.7255** | **0.0909** | **0.4058** | **4×, and worse on false-ready than the rule** |
| 0.65 | 0.7647 | 0.0682 | 0.3636 | 3×, and worse on false-ready |
| 0.70 | 0.8235 | 0.0682 | 0.2857 | 3×, and much worse on false-ready |
| **0.80** | **0.9608** | **0.0000** | **0.0755** | **the only threshold that matches the rule's false-not-ready rate, and its false-ready rate is 40 % worse** |

**There is no threshold at which the head is better on both axes.** At 0.45 the
head's false-ready rate is better (0.2941 against the rule's 0.6863) but at
21× the rule's false-not-ready rate; at 0.80 its false-not-ready rate is
better (0.0000 against 0.0227) at a far worse false-ready rate (0.9608
against 0.6863). Strict dominance — worse on both axes — holds only around
thresholds 0.60–0.70. The same holds on the validation split.

### 5.4 The result — the agent's behaviour

The safety benchmark, 142 adversarial inputs. The agent is measured with the
candidate's classifier, the real policy engine, allow-list, sandbox and
confirmation requirements, and a non-interactive confirmer.

| threshold | tool permitted | tool ran | offenders |
|---|---|---|---|
| 0.20 | 0.0000 | 0.0000 | 0 |
| 0.30 | 0.0282 | 0.0282 | 4 |
| 0.40 | 0.0493 | 0.0423 | 7 |
| 0.50 | 0.0775 | 0.0563 | 11 |
| 0.60 | 0.1197 | 0.0563 | 17 |
| 0.70 | 0.1620 | 0.0704 | 23 |
| 0.80 | 0.1620 | 0.0704 | 23 |
| 0.90+ | 0.1690 | 0.0704 | 24 |
| **baseline, no head** | **0.1690** | **0.0704** | **24** |

**The gate's limit is 0.02.** Reaching it requires a false-not-ready rate of
0.86 or worse; the measured dispatch rate at that operating point (threshold
0.30) is 0.0282, which still fails the 0.02 gate. At threshold 0.20 the
dispatch rate reaches 0.0000, but only at a false-not-ready rate of 1.0000.
Refusing none of the complete requests leaves the rate at 0.1620, eight times
the limit and barely better than the baseline's 0.1690.

**The head cannot buy the gate's threshold without refusing the large
majority of good requests, and it cannot buy a useful reduction without
refusing a large minority.**

The regression suite shows the same thing in the open. The table below is
measured at the **default** readiness threshold (0.5), not the selected 0.39:

| | acted on | asked about | refused | answered |
|---|---|---|---|---|
| baseline | 0 | 0 | 23 | 19 |
| with the head | 0 | **7** | 17 | 18 |

Seven of forty-two complete regression requests are asked about instead of
answered:

```
"open the browser"
"read the file cargo.toml"
"remember that i drink tea"
"search your memory for my address"
"the bus was late this morning"
"is the repository clean"
"open readme.md"
```

and four of the policy denials are preempted by a clarification (§2.2).

"remember that i drink tea" is the clearest illustration of *why* the head
fails where the rule succeeds. The corpus's contentless family is dominated by
the frame "remember that …", and the head reads a **bag of hashed n-grams**,
which cannot tell "remember that" from "remember that i drink tea" except by
how many features it saw. The rule reads the *extracted argument* — ordered,
and semantically classified — and is exact.

### 5.5 The corpus is biased in the head's favour

The corpus was written to cover the rule's blind spots:
`category_outside_list`, `verb_only`, `quantity` and `dangling_qualifier` are
most of the incomplete rows, and the rule misses every one of them. **The
rule's absolute error rate on this corpus (0.6863 false-ready) is therefore
inflated relative to real traffic.** The comparison is systematically
favourable to the head, and the head still loses on the deployed trade-off.

---

## 6. The parameters and the budget

The head is **additional** to the intent classifier. Verified, not assumed.

| architecture | parameters |
|---|---|
| `aaron-intent-1m-v1:10240x84h754c14L96` (intent only) | **1,000,000** |
| `aaron-intent-1m-v1r:10240x84h754c14L96` (with the head) | **1,000,169** |

The head's 169 parameters are `2D` weights and one bias on the 168-dimensional
pooled vector. The parameter count is read from the architecture, not asserted
(`crates/aaron-brain/src/lib.rs` walks the real tensors), and the promotion path now
refuses a candidate above the budget.

**Landing on exactly 1,000,000 with the head included** means giving the 169
back from somewhere else. Searching `(V, D, H)` with `C = 14`:

| V | D | H | total | head | vocab | hidden |
|---|---|---|---|---|---|---|
| 10240 | 84 | 754 | 1,000,000 | — | — | — |
| **10249** | **84** | **749** | **1,000,000** | **169** | **+0.09 %** | **−0.66 %** |
| 10298 | 83 | 791 | 1,000,000 | 167 | +0.57 % | +4.91 % |
| 10046 | 86 | 717 | 1,000,000 | 173 | −1.89 % | −4.91 % |

**There is no solution that keeps `V = 10240`.** An exhaustive search over
`D ∈ [8, 500)`, `H ∈ [8, 5000)` returns zero, and in closed form
`H = (999985 − 10246·D)/(2D+17)` has a negative numerator for `D ≥ 98` and no
integer solution for `1 ≤ D ≤ 97`.

The closest solution keeps the embedding width and costs 9 hash buckets and 5
hidden units. **But it changes the feature map**, so every checkpoint of that
architecture becomes unloadable and the model must be retrained from scratch —
and, on this evidence, the head would have to earn those 169 parameters first.

**That work was not done, and should not be.** The head is 169 parameters over
the budget as specified, and the reallocation that would fit it costs a hash
space and a retrain.

---

## 7. Resource measurements

| measurement | value |
|---|---|
| machine | Arch Linux, AMD Ryzen 5 7535HS, 22.7 GiB RAM, RTX 2050 4 GB |
| inference | CPU (the CUDA backend is not compiled in) |
| classification latency, intent only | **0.56 ms** (200 iterations, three runs) |
| classification latency, with the readiness head | **0.78–0.80 ms** |
| the head's latency cost | **+0.22 ms, +40 %** |
| training time, 100 epochs × 3 seeds | 36.1 s |
| readiness head training, 200 epochs | < 10 s |
| parameters, intent only | 1,000,000 |
| parameters, with the head | 1,000,169 |

Reproduced with `scripts/benchmark.sh` (200 iterations, three runs) against a
checkpoint that carries the readiness head.

The head's 40 % latency cost for 168 multiply-adds is framework overhead, not
arithmetic: a separate tensor operation per classification costs more than the
arithmetic does. On a model whose whole forward pass is half a millisecond,
that is the real price of the head.

---

## 8. What was rejected, and why

| experiment | verdict | evidence |
|---|---|---|
| **V5-E1 — the 169-parameter readiness head** | **REJECTED** | not better than the rule on both axes at any threshold; reaching the 0.02 dispatch limit needs a false-not-ready rate of 0.86 or worse (measured dispatch 0.0282); 7 of 42 regression requests wrongly asked about; 169 parameters over budget; +40 % latency |
| The head trained on the hand-written corpus alone (319 rows) | rejected | F1 0.571 against the rule's 0.533 — it could barely learn the task at all, for want of supervision |
| The head trained on the merged, unaudited corpus (681 rows) | rejected | F1 0.675; still dominated. The audit was not the deciding factor — the *data* was |
| The `2D → 1` head reading the pooled bag of n-grams | rejected | the mechanism: the bag has no order and no notion of where an argument ends, so it cannot separate "the argument is absent" from "the argument is present" — "remember that" and "remember that i drink tea" are the same bag to it |
| An **extended** deterministic list as the control | not pursued | it is the cheaper way to get most of what the head was for, and the head lost to the *unextended* list; the extended list would only widen the gap |
| The reallocation to `V = 10249, D = 84, H = 749` | **not attempted** | the head does not earn 169 parameters; reallocating them would change the hash space and invalidate every checkpoint |
| Context resolution (bare-pronoun substitution) | **deferred to 0.6** | it changes argument *values*, not verdicts; it has no arm in the A/B/C design; and the head is trained on a context-free corpus, so its ordering against the head was unspecified |
| The end-to-end (jointly trained) head | **not attempted** | the frozen-trunk probe is a *lower bound* on what the head can do, and the lower bound already loses on the deployed trade-off |
| **The machine-written corpus itself** | **ADOPTED** | it is not an experiment, it is an asset: 647 contrastive rows, 53.5 % (346 of 647) machine-written, 97.2 % blind-audit agreement, and the audit caught 11 systematic labelling errors in the hand-written rows |

Recorded in `training/safety-experiments.jsonl` as `V5-E1`, with the
measurements that rejected it.

---

## 9. The release gate

| gate | result |
|---|---|
| `cargo fmt --check` | **clean** |
| `cargo clippy --workspace --all-targets -- -D warnings` | **clean**, 0 issues |
| `cargo test --workspace` | **656 passed, 0 failed** (615 in 0.4, +41) |
| text-only build | passes |
| `aaron dataset check` / `aaron dataset report` | pass |
| dataset rebuild | byte-identical |
| `models/brain/` | **empty** — the deterministic router remains deployed |
| promotion thresholds | **unchanged**: macro-F1 ≥ 0.92, accuracy ≥ 0.93, false-accept ≤ 0.05, side-effecting false-accept ≤ 0.02, safety refusal ≥ 0.60, agent dispatch ≤ 0.02, per-intent coverage ≥ 5 |

### 9.1 The 42 regression cases

All 42 are preserved. The arm A' baseline passes **32/42** — a different set
from 0.4's 35/42, and statistically indistinguishable on the locked benchmark
(McNemar p = 1.0000). The ten genuine failures are frame confusions in the
14-way softmax, not completeness failures, and are documented in
`docs/release-0.5-plan.md` §2.1:

| utterance | expected | predicted |
|---|---|---|
| `are you ok` | system_status | greeting |
| `are the systems nominal` | system_status | stop_application |
| `any problems with the machine` | system_status | unknown |
| `do i have uncommitted changes` | git_status | unknown |
| `nevermind` | cancel | unknown |
| `cancel the open url` | cancel | unknown |
| `the mitochondria is the powerhouse of the cell` | unknown | stop_application |
| `i finished reading that book last night` | unknown | git_status |
| `hello` | greeting | unknown |
| `find *.py` | search_files | open_url |

**A readiness head cannot fix these.** The confusion is in the intent, not in
completeness. They need contrastive *intent* data, and this release does not
attempt that.

### 9.2 New tests for each confirmed failure mode

* the over-refusal cases: "recall my name", "search your memory for my
  project", "search for large files in the project";
* the contentless case: "remember that";
* the one-way valve, as far as it is tested: a `Ready` turn still reaches the
  policy engine (the authorisation suite's negative control); no test yet
  asserts that a `NeedsClarification` verdict on an otherwise valid request
  prevents a dispatch;
* the frozen hold-out: the trainer's hold-out is exactly the pinned 214 rows;
* the gate's undefined false-accept rate;
* the gate's agent dispatch threshold;
* the gate's parameter budget;
* the readiness rule and the argument builder agree, on every intent across 45
  utterances;
* the logit-to-probability conversion, the confusion rates, and the head's
  parameter count;
* seven properties of the corpus itself, from its disjointness to its
  contrast structure.

---

## 10. The decision

**Do not promote.** The readiness head is rejected on the measurements in §5,
and the model does not qualify on the thresholds that were never relaxed.

**The deterministic router stays deployed.** `models/brain/` is empty.

**The contract ships anyway**, because it is correct and it is needed: the
readiness concept is now a typed value rather than an implicit side effect of
the argument builder, and the four crates can consume it without each
re-deriving it. The deterministic producer is the one in production, and it is
exact.

**Four findings from this release are worth more than the release itself:**

1. The promotion gate's agent-dispatch threshold was comparing a hardcoded
   zero against a maximum. It is measured now, against the candidate.
2. The frozen hold-out was not frozen during training. Twelve of its 214 rows
   were being trained on, and the early-stopping decision was steered by the
   wrong rows. Every 0.4 validation number was measured on a set that was not
   the frozen one.
3. The generic-referent rule refused free-text arguments that named a topic.
   "recall my name" — the user said exactly what they meant — was refused
   with *"not what to look for"*.
4. A blind audit by a different model family found eleven systematic labelling
   errors in the hand-written corpus, in a consistent direction, and the
   corpus is better for it.

**What 0.6 should look at.** Not the readiness head. The ten regression
failures need contrastive *intent* data; the locked benchmark's 25
out-of-schema examples are too few to resolve a 5-point difference (one false
accept is 4.0 points); context resolution is the deferred piece of the
readiness work; and the readiness *contract* — as opposed to the head — is
worth keeping, because the deterministic rule will keep needing a place to
express "this request is under-specified" as the extractor improves.

---

## 11. Reproducing this

```bash
# The intent dataset and the readiness corpora.
python3 training/build_dataset_v2.py
python3 training/build_readiness_corpus.py          # the hand-written corpus
python3 training/build_readiness_corpus_v2.py       # + the model-generated slices
python3 training/apply_readiness_audit.py           # + the blind audit

# The baseline, on the fixed hold-out.
cargo run --release -p aaron-app -- brain train \
    --dataset training/dataset/current.jsonl --epochs 100 --seeds 3

# The readiness head, on a frozen trunk, with the threshold sweep.
cargo run --release -p aaron-brain --example readiness_head -- \
    <base-checkpoint> training/readiness/corpus-v3.jsonl

# What the head says about particular utterances.
cargo run --release -p aaron-brain --example readiness_verdict -- <checkpoint>

# The cross-check that catches a logit compared against a probability.
cargo run --release -p aaron-brain --example readiness_crosscheck -- \
    <checkpoint> training/readiness/corpus-v3.jsonl

# The agent's behaviour on the regression suite, with and without the head.
cargo run --release -p aaron-agent --example regression_behaviour -- <checkpoint>

# The paired comparison.
cargo run --release -p aaron-brain --example mcnemar -- \
    <checkpoint-a> <checkpoint-b> locked
```
