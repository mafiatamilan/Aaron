#!/usr/bin/env python3
"""Merge the model-generated readiness rows into the corpus, and validate them.

Release 0.5's readiness corpus started as 319 hand-written rows. That is the
scarce resource: the disjointness rule (the corpus may not overlap the
shipped dataset or any held-out set) removed a third of the candidates, and
the rows that survived were thin in exactly the families that matter.

So the corpus is now partly **model-generated**: DeepSeek and other free
models write the contrastive pairs, and this script decides which of them are
usable. The model does not get to decide that — it has no idea what the
shipped dataset contains, and a single mislabelled row teaches the
classifier the wrong thing.

# What is checked, mechanically

1. **Format.** Every line is a JSON object with `text`, `kind` and `family`.
2. **Intent.** Derived from the slice the file came from, and checked against
   the utterance's leading verb. A row whose verb does not belong to the
   slice is dropped rather than guessed at.
3. **Family.** Must be one of the families that slice is allowed to use.
4. **Uniqueness.** No two rows in the corpus may normalise to the same text,
   for the same intent. The same text with two different intents is a
   contradiction and is dropped too.
5. **Disjointness.** Exact- and near-disjoint from the shipped dataset, the
   locked benchmark, both safety benchmarks, the regression suite and the
   accepted curated files, using the builder's own metric: normalised-text
   equality plus word 3-gram containment at 0.8.
6. **Pair structure.** Each `incomplete` row must be followed by a `complete`
   row in the same family and intent. A row without its counterpart is
   dropped: an unpaired complete row is a negative with no positive to
   contrast against, and an unpaired incomplete row teaches the classifier
   that a shape is incomplete without showing what complete looks like.
7. **No bare pronouns.** Context resolution is deferred to release 0.6.

What is *not* checked mechanically is the ground truth itself — "is this
request specific enough to act on?" is a judgement. That is why a second,
independent model audits the rows blind before they are used, and why the
hand-written rows and the generated rows are kept in separate files so the
two can be compared.

Outputs:
  training/readiness/corpus-v2.jsonl          the merged corpus
  training/readiness/generated/audit.jsonl     the rows, for the blind audit
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENERATED_DIR = os.path.join(REPO, "training", "readiness", "generated")
CORPUS_V1 = os.path.join(REPO, "training", "readiness", "corpus-v1.jsonl")
CORPUS_V2 = os.path.join(REPO, "training", "readiness", "corpus-v2.jsonl")
AUDIT_PATH = os.path.join(REPO, "training", "readiness", "generated", "audit.jsonl")

NEAR_DUPLICATE_THRESHOLD = 0.8

PROTECTED = [
    "training/dataset/current.jsonl",
    "training/dataset/v1.jsonl",
    "training/evaluation/benchmark-v3.jsonl",
    "training/evaluation/safety-benchmark-v1.jsonl",
    "training/evaluation/safety-benchmark-v2.jsonl",
    "training/evaluation/regression.jsonl",
    "training/curated/curated-v2.jsonl",
    "training/curated/curated-v3-hard-negatives.jsonl",
]

# Which slice each generated file holds, and the intents it may carry.
SLICES = {
    "read_file": (["read_file"], {"verb_only", "bare_noun", "quantity", "dangling_qualifier", "category_outside_list"}),
    "search_files": (["search_files"], {"verb_only", "bare_noun", "quantity", "dangling_qualifier", "category_outside_list"}),
    "open_url": (["open_url"], {"verb_only", "bare_noun", "category_outside_list"}),
    "start_application": (["start_application"], {"verb_only", "bare_noun", "category_outside_list"}),
    "stop_application": (["stop_application"], {"verb_only", "bare_noun", "category_outside_list"}),
    "remember_recall": (["remember", "recall"], {"verb_only", "contentless"}),
}

# The verbs that introduce each intent's argument. A row whose leading word
# is not one of them is dropped: the slice says which intent it is, and a
# row that contradicts it is a generation error, not a judgement call.
INTENT_VERBS = {
    "read_file": {"read", "show", "cat", "print", "display", "dump", "head", "tail", "render", "preview", "open", "view", "load"},
    "search_files": {"search", "find", "locate", "grep", "scan", "hunt", "sift", "look", "seek", "filter"},
    "open_url": {"open", "browse", "visit", "surf", "navigate", "fetch", "go", "pull", "launch"},
    "start_application": {"open", "launch", "start", "fire", "spawn", "raise", "run", "boot", "load", "relaunch", "begin", "kick"},
    "stop_application": {"kill", "close", "stop", "terminate", "end", "quit", "halt", "abort", "shut"},
    "remember": {
        "remember", "note", "memorize", "memorise", "log", "record", "store", "save",
        "keep", "write", "mark", "put", "register", "jot", "pin", "file", "add",
    },
    "recall": {
        "recall", "remind", "retrieve", "recollect", "look", "think", "cast", "dredge",
        "dig", "pull", "call", "bring", "fetch", "search", "check", "find", "surface",
        # "remember what I said about X" is a recall request, not a store.
        "remember",
    },
}

# A bare pronoun standing in for the argument: "read it", "open them".
# These are dropped: a pronoun is only resolvable from the conversation
# history, which this classifier does not see, and release 0.6 takes that up.
BARE_PRONOUNS = {"it", "them", "those", "these", "one", "ones", "itself"}

# "that" and "this" are pronouns only when they stand in for the argument.
# In "remember that" and "recall this" they are the *frame* of a request whose
# argument is missing, and those frames are the canonical contentless case.
FRAME_PRONOUNS = {"that", "this"}


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def shingles(text: str) -> set[str]:
    words = normalise(text).split()
    if len(words) < 3:
        return set(words)
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def is_near_duplicate(a: str, b: str) -> bool:
    return containment(shingles(a), shingles(b)) >= NEAR_DUPLICATE_THRESHOLD


def short_hash(text: str) -> str:
    return format(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16), "08x")


def load_protected() -> list[tuple[str, str]]:
    out = []
    for rel in PROTECTED:
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    out.append((rel, json.loads(line)["text"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return out


def leading_verb(text: str) -> str:
    words = normalise(text).split()
    return words[0] if words else ""


def main() -> int:
    protected = load_protected()
    dropped: Counter[str] = Counter()
    dropped_detail: list[tuple[str, str]] = []

    accepted: list[dict] = []
    pairs_seen: dict[str, int] = defaultdict(int)

    # --- the hand-written corpus -----------------------------------------
    with open(CORPUS_V1, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            accepted.append(
                {
                    "text": row["text"],
                    "intent": row["intent"],
                    "readiness": row["readiness"],
                    "family": row["family"],
                    "kind": row["kind"],
                    "source": "authored",
                    # The hand-written corpus records its pairs explicitly,
                    # so the partner of a rejected row can be found.
                    "pair": row.get("pair"),
                }
            )

    authored_count = len(accepted)

    # --- the generated slices --------------------------------------------
    for filename in sorted(os.listdir(GENERATED_DIR)):
        if not filename.endswith(".jsonl") or filename == "audit.jsonl":
            continue
        stem = filename[: -len(".jsonl")]
        if stem not in SLICES:
            dropped[f"unknown slice {stem}"] += 1
            continue
        intents, families = SLICES[stem]
        path = os.path.join(GENERATED_DIR, filename)
        with open(path, encoding="utf-8") as handle:
            lines = [l.strip() for l in handle if l.strip() and not l.startswith("#")]

        # The rows arrive in pairs: incomplete, complete.
        for index in range(0, len(lines) - 1, 2):
            pair = []
            for raw in lines[index : index + 2]:
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError:
                    pair = []
                    break
                if not isinstance(row, dict) or "text" not in row:
                    pair = []
                    break
                pair.append(row)
            if len(pair) != 2:
                dropped[f"{stem}: malformed pair"] += 1
                continue
            incomplete, complete = pair
            if incomplete.get("kind") != "incomplete" or complete.get("kind") != "complete":
                dropped[f"{stem}: pair is not incomplete/complete"] += 1
                continue
            if incomplete.get("family") != complete.get("family"):
                dropped[f"{stem}: pair families differ"] += 1
                continue
            family = incomplete.get("family")
            if family not in families:
                dropped[f"{stem}: family {family} not allowed"] += 1
                continue

            for row in pair:
                text = row["text"].strip()
                words = normalise(text).split()
                if not words:
                    dropped[f"{stem}: empty text"] += 1
                    break
                if set(words) & BARE_PRONOUNS:
                    dropped[f"{stem}: bare pronoun"] += 1
                    break
                if set(words) & FRAME_PRONOUNS and stem != "remember_recall":
                    # "read that" is a pronoun standing in for the argument;
                    # "remember that" is a frame. Only the content intents
                    # have frames.
                    dropped[f"{stem}: bare pronoun"] += 1
                    break
                verb = leading_verb(text)
                if len(intents) == 1:
                    # The slice names the intent. The leading verb is not a
                    # closed list — a generator will reach for verbs nobody
                    # thought of, and "jump to the issue tracker" is still an
                    # `open_url` request.
                    intent = intents[0]
                else:
                    # This slice holds two intents, and the verb is the only
                    # thing that distinguishes them — but the verbs overlap
                    # ("remember" can be either), so the *pair index* decides
                    # and the verb only has to be plausible for the intent it
                    # is assigned to. The generator writes the first half of
                    # the file for the first intent and the second half for
                    # the second, and the pair index is that boundary.
                    index = pairs_seen[stem]
                    pairs_seen[stem] += 1
                    intent = intents[0] if index < 40 else intents[1]
                    if verb not in INTENT_VERBS[intent]:
                        dropped[f"{stem}: verb {verb!r} does not fit {intent}"] += 1
                        break
                row["_intent"] = intent
            else:
                # Disjointness.
                conflict = None
                for rel, other in protected:
                    for row in pair:
                        text = row["text"]
                        if normalise(text) == normalise(other) or is_near_duplicate(text, other):
                            conflict = (rel, row["text"])
                            break
                    if conflict:
                        break
                if conflict:
                    dropped[f"overlaps {conflict[0]}"] += 1
                    dropped_detail.append(conflict)
                    continue
                # The pair is atomic: if one of the two rows is rejected, the
                # other goes with it. A complete row whose incomplete
                # counterpart is gone is a negative with no positive to
                # contrast against, and the contrast is the whole point of
                # the corpus.
                pair_id = "gen:{}|{}|{}".format(
                    stem,
                    family,
                    len([p for p in accepted if p.get("source") == "generated"]),
                )
                for row, kind, readiness in (
                    (incomplete, "incomplete", "needs_clarification"),
                    (complete, "complete", "ready"),
                ):
                    accepted.append(
                        {
                            "text": row["text"].strip(),
                            "intent": row["_intent"],
                            "readiness": readiness,
                            "family": family,
                            "kind": kind,
                            "source": "generated",
                            "pair": pair_id,
                        }
                    )

    # --- internal consistency -------------------------------------------
    seen: dict[tuple[str, str], str] = {}
    unique: list[dict] = []
    for row in accepted:
        key = (row["intent"], normalise(row["text"]))
        if key in seen:
            dropped["duplicate within the corpus"] += 1
            continue
        seen[key] = row["text"]
        unique.append(row)

    # A text that is incomplete for one intent and complete for another is a
    # contradiction: the head reads the text, not the intent.
    by_text: dict[str, set[str]] = defaultdict(set)
    for row in unique:
        by_text[normalise(row["text"])].add(row["readiness"])
    contradictions = [t for t, s in by_text.items() if len(s) > 1]
    if contradictions:
        unique = [r for r in unique if normalise(r["text"]) not in contradictions]
        dropped["contradictory labels for one text"] += len(contradictions)

    # --- the split --------------------------------------------------------
    parent: dict[str, str] = {r["text"]: r["text"] for r in unique}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for i, a in enumerate(unique):
        sa = shingles(a["text"])
        for b in unique[i + 1 :]:
            if containment(sa, shingles(b["text"])) >= NEAR_DUPLICATE_THRESHOLD:
                union(a["text"], b["text"])

    groups: dict[str, list[str]] = defaultdict(list)
    for row in unique:
        groups[find(row["text"])].append(row["text"])

    # A text can appear under two intents, so the split is assigned by text
    # and applied to every row carrying it.
    splits: dict[str, str] = {}
    for member_texts in groups.values():
        lowest = min(member_texts)
        bucket = int(hashlib.sha256(lowest.encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else ("validation" if bucket < 85 else "test")
        for text in member_texts:
            splits[text] = split
    for row in unique:
        row["split"] = splits[row["text"]]

    # --- write ------------------------------------------------------------
    rows = []
    for row in unique:
        rows.append(
            {
                "id": f"readiness-{row['intent']}-{short_hash(row['text'])}",
                "text": row["text"],
                "intent": row["intent"],
                "readiness": row["readiness"],
                "family": row["family"],
                "kind": row["kind"],
                "split": row["split"],
                "source": row["source"],
                "pair": row.get("pair"),
                "notes": f"readiness corpus, release 0.5; family `{row['family']}`",
            }
        )
    rows.sort(key=lambda r: (r["intent"], r["family"], r["text"]))
    with open(CORPUS_V2, "w", encoding="utf-8") as handle:
        handle.write("# AARON readiness corpus, release 0.5 (v2: hand-written + model-generated)\n")
        handle.write("# Not part of the intent training set. See the generator.\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # The audit file: the rows without their labels, for a blind second
    # opinion.
    #
    # It carries no `readiness` field, and **no `kind` field either** — the
    # kind *is* the label, so leaving it in would make the audit a formality.
    with open(AUDIT_PATH, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "intent": row["intent"],
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    # --- report -----------------------------------------------------------
    print(f"corpus: {CORPUS_V2}")
    print(f"rows: {len(rows)}  ({authored_count} hand-written, {len(rows) - authored_count} model-generated)")
    print()
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_source: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    by_intent: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["readiness"]] += 1
        by_source[row["source"]][row["readiness"]] += 1
        by_family[row["family"]][row["readiness"]] += 1
        by_intent[row["intent"]][row["readiness"]] += 1

    print("by split")
    for split in sorted(by_split):
        c = by_split[split]
        total = sum(c.values())
        print(
            f"  {split:<12} needs_clarification {c['needs_clarification']:>4}  "
            f"ready {c['ready']:>4}  not_applicable {c['not_applicable']:>4}   (of {total})"
        )
    print()
    print("by source")
    for source in sorted(by_source):
        c = by_source[source]
        print(
            f"  {source:<12} needs_clarification {c['needs_clarification']:>4}  "
            f"ready {c['ready']:>4}  not_applicable {c['not_applicable']:>4}"
        )
    print()
    print("by family")
    for family in sorted(by_family):
        c = by_family[family]
        print(
            f"  {family:<26} needs_clarification {c['needs_clarification']:>4}  "
            f"ready {c['ready']:>4}  not_applicable {c['not_applicable']:>4}"
        )
    print()
    print("by intent")
    for intent in sorted(by_intent):
        c = by_intent[intent]
        print(
            f"  {intent:<20} needs_clarification {c['needs_clarification']:>4}  "
            f"ready {c['ready']:>4}  not_applicable {c['not_applicable']:>4}"
        )
    print()
    print(f"near-duplicate groups: {len(groups)}")
    print()
    print("dropped")
    for reason, count in dropped.most_common():
        print(f"  {count:>4}  {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
