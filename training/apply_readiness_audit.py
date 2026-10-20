#!/usr/bin/env python3
"""Apply the blind audit to the readiness corpus, and keep what survives.

The corpus is partly model-generated, and a generated label is only as good
as the model that wrote it. So every row is audited by a *different* model
family, blind: the auditor sees the utterance and the intent, and never the
label it is being asked to confirm.

A row where the two disagree is **dropped**. It is not a tie to be broken:
the two judgements are about the same question, from independent sources, and
a disagreement means the row is ambiguous. Ambiguous rows are exactly the
rows a classifier cannot be taught from, and a classifier taught from them
learns the annotator's uncertainty rather than the concept.

The split is **inherited**, not recomputed. The split is a function of the
near-duplicate group's lowest id, so removing rows could move a whole group
to a different split, and the corpus's held-out set would no longer be
comparable with the one that was measured. Dropping rows must not move the
evaluation.

Outputs:
  training/readiness/corpus-v3.jsonl        the audited corpus
  training/readiness/rejected-vraft-audit.jsonl   the rows that were dropped
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READINESS = os.path.join(REPO, "training", "readiness")
AUDIT_DIR = os.path.join(READINESS, "generated", "audit")

CORPUS_IN = os.path.join(READINESS, "corpus-v2.jsonl")
CORPUS_OUT = os.path.join(READINESS, "corpus-v3.jsonl")
REJECTED_OUT = os.path.join(READINESS, "rejected-vraft-audit.jsonl")

VERDICTS = {
    "a": os.path.join(AUDIT_DIR, "a-verdict.jsonl"),
    "b": os.path.join(AUDIT_DIR, "b-verdict.jsonl"),
    "c": os.path.join(AUDIT_DIR, "c-verdict.jsonl"),
    "d": os.path.join(AUDIT_DIR, "d-verdict.jsonl"),
    "e": os.path.join(AUDIT_DIR, "e-verdict.jsonl"),
}


def main() -> int:
    with open(CORPUS_IN, encoding="utf-8") as handle:
        corpus = [json.loads(line) for line in handle if line.strip() and not line.startswith("#")]

    verdicts: dict[str, str] = {}
    for name, path in VERDICTS.items():
        if not os.path.isfile(path):
            print(f"error: {path} is missing; the audit has not run")
            return 1
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if row["id"] in verdicts and verdicts[row["id"]] != row["verdict"]:
                    print(f"error: two auditors disagree about `{row['id']}`")
                    return 1
                verdicts[row["id"]] = row["verdict"]

    # The pair is atomic: a rejected row takes its contrast with it. A
    # complete row whose incomplete counterpart is gone is a negative with no
    # positive to contrast against, and the contrast is the whole point of
    # the corpus.
    rejected_ids: set[str] = set()
    partner_of: dict[str, str] = {}
    for row in corpus:
        pair = row.get("pair")
        if pair:
            partner_of.setdefault(pair, row["id"])
    by_pair: dict[str, list[str]] = {}
    for row in corpus:
        pair = row.get("pair")
        if pair:
            by_pair.setdefault(pair, []).append(row["id"])

    kept: list[dict] = []
    rejected: list[dict] = []
    unaudited = 0

    for row in corpus:
        verdict = verdicts.get(row["id"])
        if verdict is None:
            unaudited += 1
            kept.append(row)
            continue
        # The auditor answers a two-state question — "is this specific enough
        # to act on?" — and the corpus carries three states. `NotApplicable`
        # means the utterance requests no action at all, which is also "the
        # head has nothing to say", so it corresponds to the auditor's
        # `ready`. Only the `NeedsClarification` boundary is a real
        # disagreement.
        agrees = verdict == row["readiness"] or (
            row["readiness"] == "not_applicable" and verdict == "ready"
        )
        if agrees:
            kept.append(row)
        else:
            row = dict(row)
            row["audit_verdict"] = verdict
            row["audit_rejection"] = "the blind audit disagreed with the label"
            rejected.append(row)
            rejected_ids.add(row["id"])
            pair = row.get("pair")
            if pair:
                for partner in by_pair.get(pair, []):
                    if partner != row["id"]:
                        rejected_ids.add(partner)

    if rejected_ids:
        kept = [r for r in kept if r["id"] not in rejected_ids]
        for row in rejected:
            if row["id"] not in {r["id"] for r in rejected}:
                rejected.append(dict(row, audit_rejection="the pair partner was rejected"))

    with open(CORPUS_OUT, "w", encoding="utf-8") as handle:
        handle.write("# AARON readiness corpus, release 0.5 (v3: audited)\n")
        handle.write("# Not part of the intent training set. See the generator.\n")
        for row in kept:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    with open(REJECTED_OUT, "w", encoding="utf-8") as handle:
        handle.write("# Rows rejected by the blind readiness audit, release 0.5\n")
        handle.write("# Kept for the record; not used for training or evaluation.\n")
        for row in rejected:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    print(f"corpus: {CORPUS_OUT}")
    print(f"rows: {len(kept)}  (of {len(corpus)}; {len(rejected)} rejected by the audit)")
    if unaudited:
        print(f"  {unaudited} rows had no verdict and were kept unaudited")
    print()
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_split: dict[str, Counter] = defaultdict(Counter)
    for row in kept:
        by_source[row["source"]][row["readiness"]] += 1
        by_split[row["split"]][row["readiness"]] += 1
    print("by source")
    for source in sorted(by_source):
        c = by_source[source]
        print(
            f"  {source:<12} needs_clarification {c['needs_clarification']:>4}  "
            f"ready {c['ready']:>4}  not_applicable {c['not_applicable']:>4}"
        )
    print()
    print("by split")
    for split in sorted(by_split):
        c = by_split[split]
        total = sum(c.values())
        print(
            f"  {split:<12} needs_clarification {c['needs_clarification']:>4}  "
            f"ready {c['ready']:>4}  not_applicable {c['not_applicable']:>4}   (of {total})"
        )
    print()
    print(f"rejected: {REJECTED_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
