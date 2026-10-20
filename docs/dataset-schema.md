# Dataset schema and data-quality rules

The dataset is the thing the model is made of, and it is the easiest part of
AARON to get quietly wrong. This document is the contract: the file format,
what each field means, what the quality pipeline checks, and the rules for
adding data.

## 1. The dataset as it stands

Release 0.3 ships `training/dataset/current.jsonl` with **1,243 examples**
(1,210 in 0.2). The SHA-256 of the file begins `ec4aa5bd392ace71e`, and that
prefix is quoted in the release report so every number can be tied back to a
file.

The class imbalance is **2.86×** — 163 rows of `unknown` against 57 of
`greeting` — where 0.2 was 2.56×. It rose because the 0.3 addition is 31
`unknown` rows against 10 rows of other intents: the largest class grew and
the smallest ones did not. §1.2 explains why only a small number of the new
negatives was kept.

### 1.1 The hard negatives that were kept

`training/curated/curated-v3-hard-negatives.jsonl` — 41 rows, written by
`training/build_hard_negatives.py`.

They address one measured defect, and only one. The model had learned
**"a command verb with no object is read as a command"**: it predicted
`start_application` at 0.999 confidence for "open it", and `remember` at
1.000 for "remember that". The training data always paired a verb with a
complete object — "open the terminal", "run the tests", "remember that my
wifi is called aurora" — so the model learned *verb present ⇒ command*,
regardless of whether an object was there.

The file holds two families:

| family | rows | what it is |
|---|---|---|
| `verb_no_object` | 31 | a command verb with no object, labelled `unknown` |
| `control` | 10 | complete commands that must keep working |

The `control` rows are the mirror image, and they are not decoration. A model
that learns "refuse more" and nothing else is just as broken; the controls
are what stop the class from collapsing onto short utterances.

Every row in the file carries a **`family`** field. It is not part of the
training format (§2) — it is a grouping, used by the build's
`--exclude-family` option and by the experiments that chose the file.

### 1.2 The hard negatives that were rejected

`training/curated/experimental/rejected-balance-additions.jsonl` — 238 rows,
also measured and rejected. The idea was sound (the hard negatives grew
`unknown` from 163 to 194 and the imbalance with it), but the additions were
mostly the same *shape* as existing examples — "show me the file X" for a dozen
X — so they added volume without coverage and every measure got worse: locked
accuracy 0.6385, regression 37/42, safety refusal 0.4510. More rows of the same
kind is not more data.

`training/curated/experimental/rejected-families.jsonl` — 72 rows, kept
deliberately.

These families were written, added, trained and measured, and they made
things worse. Adding all thirteen families grew the `unknown` class past
twice the size of the others, and the model learned **"short ⇒ unknown"**,
which broke short legitimate commands. The locked benchmark macro-F1 fell
0.77 → 0.71 and the regression suite fell to 41/42.

They are kept in the tree so the experiment can be repeated, and so the
reasoning is reviewable rather than folklore. The file's own header says
how to re-enable one.

The build does not read them. `build_dataset_v2.py` globs
`training/curated/*.jsonl`, and this file is in `training/curated/experimental/`,
a subdirectory the glob does not reach: it is excluded by construction, not
by a flag someone could forget to pass.

## 2. File format

JSON Lines, one example per line. Blank lines and lines beginning with `#`
are ignored, so a header comment is allowed.

```json
{"id":"system_status-3f1a9c2e","text":"check the system status","intent":"system_status","source":"curated","notes":"", "split":"train"}
```

| field | type | meaning |
|---|---|---|
| `id` | string | Unique. `{intent}-{8 hex}` of the FNV-1a-64 hash of the text, low 32 bits. |
| `text` | string | The utterance, as a user would say it. |
| `intent` | string | One of the fourteen values in the `Intent` enum. |
| `source` | string | `inherited`, `curated` or `template`. See §4. |
| `notes` | string | Free text. Why this example exists, what it was added for. |
| `split` | string | `train`, `validation` or `test`. Assigned, never chosen by hand. |
| `family` | string | Optional. The group a hard-negative row belongs to (`verb_no_object`, `control`, …). Only hard-negative rows have it. Used by `--exclude-family`; never read by the trainer. |

The `split` field is **written by the build script** and is **authoritative**.
The trainer reads it, and a dataset whose rows disagree with it is refused.

That was not true until release 0.5. The trainer used to recompute the
assignment from the id hash, per intent, while the build script hashes the
*lowest id of a near-duplicate group*. The two algorithms disagreed on 32 of
the shipped dataset's 1,103 rows, and 12 rows that `holdout.json` pins as
validation or test were in the **training** split. The frozen hold-out was
frozen for the builder and not for the trainer, which meant the validation
macro-F1 and the early-stopping decision were steered by rows the hold-out
was supposed to exclude.

`crates/aaron-app/tests/holdout.rs` pins the behaviour: the trainer's split is
the declared split, the trainer's hold-out is exactly the 214 pinned rows, and
no pinned row is trained on.

## 3. The intent set

Fixed, and closed. Fourteen values:

```
greeting  help  system_status  remember  recall  search_files  read_file
git_status  run_tests  open_url  start_application  stop_application
cancel  unknown
```

`unknown` is a first-class class. It is the answer for anything that is not a
command AARON can carry out, and a request that is confidently misrouted is
worse than one that is refused.

## 4. Provenance: `source`

| value | meaning | who reviews it |
|---|---|---|
| `inherited` | a row from the frozen v1 dataset, unchanged | already reviewed |
| `curated` | a whole sentence written by hand for this release | a human reads it |
| `template` | a shared sentence frame with a curated noun substituted | a human reads **every** row |

`template` exists because some coverage can only be expressed as a family:
every application name, every filename. Those families are the ones that hide
errors, because one mistake in a frame is repeated across every row of it.
They are listed separately in the review report for exactly that reason.

The review report is `training/dataset/review-report.md`, regenerated by
`python3 training/build_dataset_v2.py`. It lists every curated row, every
duplicate dropped in the merge, and every training row dropped to keep the
held-out sets clean.

## 5. Splits

Stratified by intent. Within an intent, examples are ordered by
`fnv1a64(id)` and the first 80% train, next 10% validation, last 10% test.

* **Ordering by hash, not by file position**, so adding examples does not
  reshuffle the existing hold-out.
* **Rounded to nearest, floored at one.** Truncating `(n * 10) / 100` gives
  zero hold-out examples for any intent with fewer than ten rows, which
  silently removes that intent from the measured set. That bug shipped once
  and is now tested against every group size from 5 to 40.
* **An intent with fewer than five examples stays entirely in training.** A
  one-example hold-out measures nothing and destabilises the macro average.

Roles:

| split | used for |
|---|---|
| `train` | fitting the weights |
| `validation` | early stopping, hyperparameter selection, seed selection |
| `test` | the development hold-out: error analysis drives data iteration here |
| the locked benchmark | the final report, and nothing else |
| the safety benchmarks | adversarial refusal, at model level and agent level |

The locked benchmark and the two safety benchmarks are *separate files* that
are never merged into the training data. The build script drops any training
row that collides with one, so they cannot drift apart by accident. §6
describes them and the rule.

## 6. The held-out sets

Three files are held out of the training data. None of them is ever merged
into the dataset, and a training row that collides with one is dropped from
the training data rather than being allowed to stay.

| file | examples | what it measures | status |
|---|---|---|---|
| `training/evaluation/benchmark-v3.jsonl` | 130 | does AARON understand the requests people make? | locked; nothing is tuned against it |
| `training/evaluation/safety-benchmark-v1.jsonl` | 142, in 13 categories | can anything make AARON do something it should not? | **development set** — it selected the 0.3 dataset |
| `training/evaluation/safety-benchmark-v2.jsonl` | 102, in 13 categories | the same question, asked in new sentences | **held out** — written after the selection, scored once |

The intent benchmark and the adversarial benchmark ask different questions,
and they are kept apart on purpose. A model tuned for the first can be
catastrophically bad at the second, and a model that scores well on the
second has not necessarily learned anything about the first. Averaging them
would hide both.

Every example in a safety benchmark is **out of schema**: the expected
answer is `unknown`, because AARON has no tool that does what is being
asked, the request is malformed, or the request is trying to talk its way
past the boundary. The thirteen categories are attack *shapes* — path
traversal, a URL carrying credentials, an instruction embedded in a file, a
claim of authority, role play, oversized input, and so on. A shape with one
example in it measures the author's imagination rather than the model's
behaviour, so most shapes have several.

Both safety benchmarks are also disjoint from the locked intent benchmark
and from the regression suite.

### 6.1 The drop rule

A training row that collides with a held-out set is **dropped from the
training data and reported**. Never the other way round: the benchmark is not
weakened to accommodate the training data.

`build_dataset_v2.py` protects the locked intent benchmark and safety
benchmark v1. It prints how many rows it dropped and lists them in the review
report, under "Training rows dropped to keep the benchmark clean".
`build_safety_benchmark.py` refuses to write at all if any of its examples
collide with the training data (`current.jsonl`, `v1.jsonl`), the locked
intent benchmark, the regression suite, or — for v2 — v1.

### 6.2 Two keys, not one

The collision check has two keys, because one key cannot do both jobs.

* **Loose key** — letters and digits only. This is what cross-file collision
  checks use. A false collision is the safe direction: refusing to write
  because two sentences normalise alike is much better than training on the
  benchmark.
* **Strict key** — punctuation kept, only whitespace collapsed.

The loose key is too lossy for payloads that *are* their punctuation. It
collapses `read the file ../../etc/passwd` and
`read the file ....//....//etc/passwd` into one string, and those are two
different attacks.

So the three **structural** categories — `path_traversal`, `malformed_unicode`
and `oversized` — are compared on the strict key. Every other category uses
the loose key.

The same split applies inside a benchmark file: rows are de-duplicated on the
strict key, and an input that normalises to nothing (a whitespace-only
utterance, a run of zero-width characters) is keyed on its raw form, because
those are distinct attacks too.

## 7. The quality pipeline

`aaron-brain::dataquality`, exposed as:

```
aaron dataset check          # validate; exits non-zero on a blocking problem
aaron dataset check --json   # the same report, machine-readable
aaron dataset report         # the coverage table
```

### Checks

| finding | blocking | what it means |
|---|---|---|
| `duplicate_id` | yes | the same id appears twice |
| `duplicate_text` | yes | two rows normalise to the same text |
| `leakage` | yes | one utterance appears in more than one split |
| `empty_text` | yes | nothing is left after normalisation |
| `near_duplicate` | no | two rows in one split are very similar |
| `insufficient_coverage` | no | an intent has too few examples to train or measure |
| `skewed_split` | no | the splits are badly unbalanced |

### Near-duplicate detection

Word 3-gram shingles, compared with the **stronger of Jaccard and
containment**.

Jaccard alone is the wrong measure for short commands. "read the file
main.rs" has three 3-grams; adding the word "please" produces four, of which
three are shared — Jaccard says 0.75, containment says 1.0. Every paraphrase
in a command set is of that shape: *the same sentence plus a filler word*.
Containment catches exactly that and leaves genuinely different commands
alone, because two unrelated commands share no 3-grams at all.

The threshold is 0.8, chosen by inspection against the shipped dataset.

### The content hash

`aaron dataset check` prints a SHA-256 over the canonical example content:
ids, normalised texts and labels, sorted.

* Two datasets with the same hash train the same model.
* Two reports with different hashes are not comparable, and this is what
  makes that checkable.
* Reformatting or reordering the file does not change the hash; changing a
  label does.

## 8. Rules for adding data

1. **Never add an example to the hold-out you intend to measure on.** The
   build script refuses a dataset where a normalised utterance appears in two
   splits, so this cannot happen silently.
2. **Never train on the locked benchmark.** The build script drops any
   colliding training row and reports it.
3. **Never weaken a metric to accommodate a model.** The hold-out is
   stratified, and the intent set is fixed.
4. **Mark generated rows.** Anything not written by hand is `template`, and
   every `template` row appears in the review report.
5. **Add shapes, not copies.** A near-duplicate of an existing row adds
   nothing: it inflates the class without adding linguistic diversity. The
   0.2 release added *contrastive* pairs — the same frame used by several
   intents — because that is what the error analysis said was missing.
6. **Record why.** The `notes` field is the only record of why an example
   exists. A dataset whose examples all say "" is a dataset nobody can
   improve on next time.
7. **A benchmark that selects a model stops being a benchmark.** Safety
   benchmark v1 was used to choose the 0.3 dataset, so from 0.3 onwards it is
   a development set: it may be looked at, argued with and tuned against,
   and it is not evidence of anything. The held-out measure is a *new* file,
   authored after the selection, from new sentences, and scored once. That
   is v2.
8. **Never add to a held-out set — write a new one.** The old file stays
   frozen so the numbers already reported against it stay reproducible.
9. **Structural payloads are compared on the strict key** (§6.2). A path that
   differs only in punctuation is a different attack, and the check has to be
   able to say so.
10. **Keep the rejected data.** When a family of examples measures as a
    regression, the file goes to `training/curated/experimental/` with the
    measurement that rejected it, rather than into the bin. The 0.3
    rejected-families file (§1.2) is the worked example.

## 9. Building

```bash
python3 training/build_curated.py          # regenerate the curated additions
python3 training/build_hard_negatives.py   # regenerate the 0.3 hard negatives
python3 training/build_dataset_v2.py       # merge, validate, write, and report
python3 training/build_benchmark.py        # regenerate the locked benchmark
python3 training/build_safety_benchmark.py # regenerate the adversarial benchmark
```

`build_safety_benchmark.py` takes `--version 1|2` — which of the two files to
write — and `--check`, which verifies isolation and writes nothing.

`build_dataset_v2.py` takes two options that exist so the 0.3 experiment can
be repeated:

| option | effect |
|---|---|
| `--exclude <stem>` | leave one curated file out of the merge, e.g. `curated-v3-hard-negatives` |
| `--exclude-family <family>` | leave one family of curated rows out, e.g. `verb_no_object` |

Together they rebuild the previous dataset version exactly, so a
before-and-after comparison is a command rather than an archaeology project.
The rejected families in `training/curated/experimental/` need no option:
they live in a subdirectory the build does not glob.

The build is deterministic: running it twice produces a byte-identical file,
and CI checks that.

`build_dataset_v2.py` refuses to write a dataset that has a blocking problem.
The 0.1 release's filler-word problem — 77 near-duplicate pairs, mostly
filler variants of "cancel" — is preserved rather than deleted, because those
rows are approved data. It is reported as a warning on every run, and the
0.2 release's answer to it was to *add* semantically distinct examples rather
than to remove the old ones.

---

## 10. The readiness corpus

A **second**, separate dataset: `training/readiness/corpus-v3.jsonl`. It is
not part of the intent dataset and is never merged into it.

It exists because a second classifier output — *is this request specific
enough to act on?* — needs supervision that the intent dataset does not
provide. Run over `current.jsonl` with the real extractor and the real
argument builder, the intent dataset contains **one** usable example of an
under-specified request. A head trained on that learns a near-constant label,
and a head that always says "Ready" can never ask a question.

### Why it is a separate file

Merging it into `current.jsonl` would be a **blocking build error**, not a
stylistic choice. The shipped `unknown` class already contains rows with the
same texts ("recall", "recall it", "open it now", "search the web"), and a
duplicate text with a contradictory label is rejected by both the build
script and the runtime loader. The corpus also needs its own held-out split,
and the shipped 214-row hold-out must not grow.

### Format

| field | string | meaning |
|---|---|---|
| `id` | string | Unique. `{intent}-{8 hex}` of the text's hash. |
| `text` | string | The utterance. |
| `intent` | string | The intent the utterance carries. |
| `readiness` | string | `needs_clarification`, `ready` or `not_applicable`. |
| `family` | string | The shape of the utterance. See below. |
| `kind` | string | `incomplete`, `complete` or `non_actionable`. |
| `pair` | string | The pair this row belongs to. Pairs are atomic. |
| `split` | string | `train`, `validation` or `test`. |
| `source` | string | `authored` or `generated`. |

### The families

| family | shape | does the deterministic rule catch it? |
|---|---|---|
| `verb_only` | the verb, nothing after it | no |
| `bare_noun` | determiner plus a category noun | yes |
| `quantity` | determiner plus a quantity | no |
| `dangling_qualifier` | "the file called" | no |
| `category_outside_list` | a category noun the closed list does not contain | no |
| `contentless` | a free-text argument with no content | yes |
| `non_actionable` | greetings, help, statements | not applicable |

The families the rule misses are the head's whole opportunity, and they are
also the families a closed list cannot express. The corpus is deliberately
biased towards them, so the rule's error rate on it is *inflated* relative to
real traffic.

### It is contrastive, and it is partly machine-written

Each incomplete utterance is paired with a complete one that differs only in
the presence of the argument. The complete rows are the hard negatives: a head
that learned "short utterance → incomplete" would ace them and still be
useless.

The pairs were written by language models — DeepSeek among them — and then
**audited blind** by a *different* model family, which saw the utterance and
the intent and never the label. Where the two disagreed the row was dropped,
and its pair partner went with it: a complete row whose incomplete
counterpart is gone is a negative with no positive to contrast against.

That audit rejected 19 rows, **eleven of them in the hand-written corpus**,
where it found a systematic labelling drift rather than noise.

`crates/aaron-app/tests/readiness_corpus.rs` pins the corpus's properties:
its disjointness from every protected set, its hold-out size, the absence of
bare pronouns, the uniqueness of its texts, the absence of contradictory
labels, and the contrast structure of every family.

### Building it

```bash
python3 training/build_readiness_corpus.py        # the hand-written corpus
python3 training/build_readiness_corpus_v2.py     # + the generated slices
python3 training/apply_readiness_audit.py         # + the blind audit
```
