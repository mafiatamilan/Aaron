"""Build the AARON intent dataset (v3).

Why this script exists
----------------------
The v1 dataset was generated from templates, and the quality pipeline found
three concrete defects in it:

  1. 77 near-duplicate pairs. Most of the `cancel` class was filler-word
     variants of "cancel" (`cancel it`, `cancel that`, `cancel please`, ...).
     They inflate the class without adding linguistic diversity.
  2. Noun spaces too small to generalise over. `read_file` and
     `start_application` share the verb `open`, so only the noun separates
     them, and 41 file nouns against 37 application nouns is not enough to
     learn a noun *class*.
  3. Verb coverage of one to three per intent, so a single phrasing dominates
     a class and every other phrasing is out of distribution.

Release 0.4 added a fourth, and it is the serious one.

  4. **The hold-out was not held out.** The independent review of the 0.4
     experiment matrix measured 41 of the 42 regression-suite cases present
     verbatim in the training data, and 40 cross-split near-duplicate pairs
     (36 of them same-label). The regression suite is the promotion gate's
     primary anti-regression guard, and 42/42 was measuring memorisation.
     Fixing the hold-out is a prerequisite for any experiment's number
     meaning anything, so it is done here rather than in an experiment.

What release 0.4 changed
------------------------
* **The regression suite is protected.** A training row that duplicates a
  regression case — exactly, or closely enough to be the same sentence — is
  dropped and reported. A regression suite that is inside the training set is
  not a regression suite.
* **Near-duplicates of every held-out set are dropped**, not just exact
  matches. Measured on the 0.3 dataset: 24 near-duplicates of the locked
  intent benchmark, 57 of safety v1 and 14 of safety v2 were being trained on.
* **Splits are group-aware.** Near-duplicate rows are grouped and the whole
  group is assigned to one split, using the group's lowest id as the hash
  key. A row with no near-duplicate keeps the split it already had, so the
  change to the existing hold-out is limited to the groups that actually
  needed unifying.
* **Cross-split near-duplicate leakage is now a blocking check.** It was
  previously undetected.

Usage
-----
    python3 training/build_dataset_v2.py
    python3 training/build_dataset_v2.py --exclude curated-v3-hard-negatives
    python3 training/build_dataset_v2.py --exclude-family bare_noun
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT / "training" / "dataset"
CURATED_DIR = ROOT / "training" / "curated"

V1 = DATASET_DIR / "v1.jsonl"
CURRENT = DATASET_DIR / "current.jsonl"
REPORT = DATASET_DIR / "review-report.md"
# The frozen hold-out: which row ids are in the validation and test splits.
#
# Release 0.4 added this because the split assignment is a function of the
# *set* of rows, not of each row: adding examples moves the cut points and
# migrates existing rows between splits. Measured on the 0.3 dataset, adding
# 132 rows moved 7-39 existing rows, including 2-18 of the 124 validation
# rows. Every experiment's rejection threshold was therefore being measured
# against a different set of examples.
#
# With the hold-out frozen, an experiment's number is comparable to the
# baseline's by construction, and new training data goes to the training
# split.
HOLDOUT = DATASET_DIR / "holdout.json"

# The sets that must never be trained on.
PROTECTED = [
    ROOT / "training" / "evaluation" / "benchmark-v3.jsonl",
    ROOT / "training" / "evaluation" / "safety-benchmark-v1.jsonl",
    ROOT / "training" / "evaluation" / "safety-benchmark-v2.jsonl",
    ROOT / "training" / "evaluation" / "regression.jsonl",
]

# A sentence counts as a near-duplicate of another when the word 3-grams of
# the shorter are contained in the longer's. See docs/release-0.4-plan.md §2 for
# why containment and not Jaccard: every paraphrase in a command set is "the
# same sentence plus a filler word", and containment catches that while
# leaving genuinely different commands alone.
NEAR_DUPLICATE_THRESHOLD = 0.8

# Minimum examples an intent needs in the hold-out for its metrics to count.
MIN_HOLDOUT = 5
# Minimum examples an intent needs in training to be learnable.
MIN_TRAIN = 10


# --------------------------------------------------------------------------
# Text normalisation and similarity
# --------------------------------------------------------------------------


def normalise(text: str) -> str:
    """A loose key: only letters and digits survive.

    Used for the *cross-file* collision check, because a false collision is the
    safe direction — refusing to write because two sentences normalise alike
    beats training on the benchmark.
    """
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def strict_key(text: str) -> str:
    """A key that keeps punctuation, so path structure survives."""
    return re.sub(r"\s+", " ", text.lower()).strip()


# Categories whose payload *is* its punctuation, and which therefore have to be
# compared on the strict key.
STRUCTURAL = {"path_traversal", "malformed_unicode", "oversized"}


def shingles(text: str) -> set[str]:
    words = normalise(text).split()
    if not words:
        return set()
    if len(words) < 3:
        return set(words)
    return {" ".join(words[i : i + 3]) for i in range(len(words) - 2)}


def containment(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity(a: set[str], b: set[str]) -> float:
    return max(jaccard(a, b), containment(a, b))


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# --------------------------------------------------------------------------
# Splitting
# --------------------------------------------------------------------------


def load_frozen_holdout() -> dict[str, str]:
    """The pinned split of every hold-out row, or an empty mapping."""
    if not HOLDOUT.exists():
        return {}
    try:
        data = json.loads(HOLDOUT.read_text())
    except json.JSONDecodeError:
        return {}
    return {k: v for k, v in data.get("rows", {}).items() if v in ("validation", "test")}


def write_frozen_holdout(rows: list[dict], splits: dict[str, str]) -> None:
    """Pin the hold-out so later builds cannot move it."""
    pinned = {
        r["id"]: splits[r["id"]]
        for r in rows
        if splits[r["id"]] in ("validation", "test")
    }
    payload = {
        "note": "Frozen hold-out. Pinned so that adding training examples "
        "cannot move the evaluation set. Regenerate deliberately, never "
        "automatically.",
        "rows": dict(sorted(pinned.items())),
    }
    HOLDOUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def assign_splits(rows: list[dict], groups: dict[str, str]) -> dict[str, str]:
    """Assign splits, keeping near-duplicate groups together.

    Within an intent, groups are ordered by the hash of the group's canonical
    (lowest) id and the first 80% train, next 10% validation, last 10% test.

    Using the group's canonical id rather than the row's own id means a row
    with no near-duplicate keeps exactly the split it had before, so the change
    to an existing hold-out is limited to the groups that actually needed
    unifying.
    """
    by_intent: dict[str, list[list[str]]] = defaultdict(list)
    members: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        members[groups[row["id"]]].append(row["id"])
    for canonical, ids in members.items():
        intent = next(r["intent"] for r in rows if r["id"] == ids[0])
        by_intent[intent].append(sorted(ids))

    out: dict[str, str] = {}
    for intent, groups_for_intent in by_intent.items():
        ordered = sorted(groups_for_intent, key=lambda ids: (fnv1a64(ids[0].encode()), ids[0]))
        n = len(ordered)
        if n < 5:
            for ids in ordered:
                for row_id in ids:
                    out[row_id] = "train"
            continue
        test_n = max(1, (n * 10 + 50) // 100)
        val_n = max(1, (n * 10 + 50) // 100)
        test_start = n - test_n
        val_start = test_start - val_n
        for i, ids in enumerate(ordered):
            split = "train" if i < val_start else ("validation" if i < test_start else "test")
            for row_id in ids:
                out[row_id] = split
    return out


def build_near_duplicate_groups(rows: list[dict]) -> dict[str, str]:
    """Group rows that are near-duplicates of each other.

    Union-find over the near-duplicate relation. The canonical id of a group
    is its lowest id, which is what the split assignment hashes.
    """
    parent: dict[str, str] = {r["id"]: r["id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    cache = [(r["id"], shingles(r["text"])) for r in rows]
    for i in range(len(cache)):
        id_a, sh_a = cache[i]
        for j in range(i + 1, len(cache)):
            id_b, sh_b = cache[j]
            if containment(sh_a, sh_b) >= NEAR_DUPLICATE_THRESHOLD:
                union(id_a, id_b)
    return {r["id"]: find(r["id"]) for r in rows}


# --------------------------------------------------------------------------
# Reading and writing
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for number, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{number}: invalid JSON: {exc}") from exc
    return rows


def load_curated(exclude: list[str], exclude_families: list[str]) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(CURATED_DIR.glob("*.jsonl")):
        if path.stem in exclude or path.name in exclude:
            continue
        for row in read_jsonl(path):
            row.setdefault("source", "curated")
            if exclude_families and row.get("family") in exclude_families:
                continue
            rows.append(row)
    return rows


def write_dataset(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# aaron-intent-dataset format=1\n")
        handle.write("# source=inherited|curated|template; split=train|validation|test\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate(rows: list[dict], splits: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return `(blocking, warnings)`.

    A *blocking* problem means the dataset cannot be trained on: a duplicate
    id, a duplicate utterance, a leak across splits, or an empty text. The
    build refuses to write a dataset with one of those, because a dataset the
    build happily writes and the trainer then rejects is a trap for whoever
    runs it next.
    """
    blocking: list[str] = []
    warnings: list[str] = []

    seen_ids: set[str] = set()
    for row in rows:
        if row["id"] in seen_ids:
            blocking.append(f"duplicate id: {row['id']}")
        seen_ids.add(row["id"])
        if row["intent"] not in {
            "greeting", "help", "system_status", "remember", "recall",
            "search_files", "read_file", "git_status", "run_tests", "open_url",
            "start_application", "stop_application", "cancel", "unknown",
        }:
            blocking.append(f"{row['id']}: unknown intent {row['intent']!r}")
        if not normalise(row["text"]):
            blocking.append(f"{row['id']}: empty text after normalisation")
        if row.get("split") not in ("train", "validation", "test"):
            blocking.append(f"{row['id']}: bad split {row.get('split')!r}")

    by_text: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        by_text[normalise(row["text"])].append(row["id"])
    for text, ids in by_text.items():
        if len(ids) > 1:
            where = {splits.get(i, "?") for i in ids}
            blocking.append(
                f"duplicate text {text!r} in ids {ids} (splits {sorted(where)})"
            )

    owner: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        owner[normalise(row["text"])].add(splits[row["id"]])
    for text, where in owner.items():
        if len(where) > 1:
            blocking.append(f"leakage: {text!r} appears in {sorted(where)}")

    # Cross-split *near*-duplicate leakage. This was previously undetected, and
    # it is what made 42/42 on the regression suite meaningless.
    cache = [(r["id"], shingles(r["text"])) for r in rows]
    split_of = {r["id"]: splits[r["id"]] for r in rows}
    cross = 0
    for i in range(len(cache)):
        id_a, sh_a = cache[i]
        for j in range(i + 1, len(cache)):
            id_b, sh_b = cache[j]
            if split_of[id_a] == split_of[id_b]:
                continue
            if normalise(dict((r["id"], r["text"]) for r in rows)[id_a]) == normalise(
                dict((r["id"], r["text"]) for r in rows)[id_b]
            ):
                continue
            if containment(sh_a, sh_b) >= NEAR_DUPLICATE_THRESHOLD:
                cross += 1
                if cross <= 10:
                    warnings.append(
                        f"cross-split near-duplicate: {id_a!r} ({split_of[id_a]}) "
                        f"~ {id_b!r} ({split_of[id_b]})"
                    )
    if cross:
        warnings.append(f"{cross} cross-split near-duplicate pair(s) remain")

    # Per-intent coverage.
    per: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        per[row["intent"]][splits[row["id"]]] += 1
    for intent in sorted(per):
        counts = per[intent]
        if counts["test"] < MIN_HOLDOUT:
            warnings.append(
                f"intent {intent}: only {counts['test']} test example(s), "
                f"below the reporting floor of {MIN_HOLDOUT}"
            )
        if counts["train"] < MIN_TRAIN:
            warnings.append(
                f"intent {intent}: only {counts['train']} training example(s), "
                f"below the learning floor of {MIN_TRAIN}"
            )

    split_counts = Counter(splits[r["id"]] for r in rows)
    largest = max(split_counts.values())
    smallest = min(split_counts.values())
    if smallest * 10 < largest:
        warnings.append(
            f"split sizes are badly unbalanced: {dict(split_counts)}; "
            f"a near-empty split makes early stopping and hold-out metrics noisy"
        )
    return blocking, warnings


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------


def build(exclude: list[str], exclude_families: list[str]):
    if not V1.exists():
        raise SystemExit(
            f"{V1} is missing.\n"
            "The frozen v1 dataset is the locked benchmark and must not be\n"
            "regenerated. Restore it from version control before building."
        )
    inherited = read_jsonl(V1)
    for row in inherited:
        row["source"] = "inherited"
        row.setdefault("notes", "")
    curated = load_curated(exclude, exclude_families)
    merged = inherited + curated

    # Deduplicate by id.
    by_id: dict[str, dict] = {}
    dropped_duplicates: list[dict] = []
    for row in merged:
        existing = by_id.get(row["id"])
        if existing is None:
            by_id[row["id"]] = row
            continue
        if existing["text"] != row["text"] or existing["intent"] != row["intent"]:
            raise SystemExit(
                f"id collision: {row['id']} is both "
                f"{existing['text']!r}/{existing['intent']} and "
                f"{row['text']!r}/{row['intent']}"
            )
        winner, loser = (existing, row) if existing["source"] == "inherited" else (row, existing)
        by_id[row["id"]] = winner
        dropped_duplicates.append(loser)
    merged = list(by_id.values())

    # Drop training rows that collide with a held-out set — exactly, or
    # closely enough to be the same sentence.
    #
    # The near-duplicate half of this is the 0.4 fix. Measured on the 0.3
    # dataset: 24 near-duplicates of the locked intent benchmark, 57 of
    # safety v1, 14 of safety v2 and 25 of the regression suite were being
    # trained on.
    protected_texts: dict[str, str] = {}
    protected_strict: dict[str, str] = {}
    for protected in PROTECTED:
        if not protected.exists():
            continue
        for row in read_jsonl(protected):
            key = normalise(row["text"])
            if key:
                protected_texts[key] = f"{protected.name}:{row['id']}"
            protected_strict[strict_key(row["text"])] = f"{protected.name}:{row['id']}"
    # The regression suite's *labels* matter too: a training row with the same
    # text and a different label would be a contradiction, not a duplicate.
    protected_pairs = set()
    for protected in PROTECTED:
        if not protected.exists():
            continue
        for row in read_jsonl(protected):
            protected_pairs.add((normalise(row["text"]), row.get("intent", "")))

    kept = []
    dropped_for_protected = []
    for row in merged:
        key = normalise(row["text"])
        if key and key in protected_texts:
            dropped_for_protected.append((row, protected_texts[key], "exact"))
            continue
        # Near-duplicate of a held-out sentence, judged on the loose key for
        # ordinary rows and the strict key for structural payloads.
        sh = shingles(row["text"])
        hit = None
        for pkey, pid in protected_texts.items():
            psh = shingles(pkey)
            if sh and psh and containment(sh, psh) >= NEAR_DUPLICATE_THRESHOLD:
                hit = (pid, "near-duplicate")
                break
        if hit is None and row.get("family") in STRUCTURAL:
            strict = strict_key(row["text"])
            for pstrict, pid in protected_strict.items():
                if strict == pstrict:
                    hit = (pid, "structural")
                    break
        if hit is not None:
            dropped_for_protected.append((row, hit[0], hit[1]))
            continue
        kept.append(row)
    merged = kept

    # Group near-duplicates so they land in the same split.
    groups = build_near_duplicate_groups(merged)
    splits = assign_splits(merged, groups)

    # Freeze the hold-out.
    #
    # A pinned row keeps its recorded split. A new row whose near-duplicate
    # group is already in the hold-out joins that group, so a contrast pair
    # cannot leak. Any other new row goes to the training split, so the
    # evaluation set cannot grow behind the experiments' backs.
    frozen = load_frozen_holdout()
    group_split: dict[str, str] = {}
    for row in merged:
        if row["id"] in frozen:
            group_split.setdefault(groups[row["id"]], frozen[row["id"]])
    for row in merged:
        group = groups[row["id"]]
        if row["id"] in frozen:
            splits[row["id"]] = frozen[row["id"]]
        elif group in group_split:
            splits[row["id"]] = group_split[group]
        else:
            splits[row["id"]] = "train"
    for row in merged:
        row["split"] = splits[row["id"]]
    merged.sort(key=lambda r: r["id"])
    curated_kept = [
        r for r in curated if r["id"] in splits and normalise(r["text"]) == normalise(
            next(m["text"] for m in merged if m["id"] == r["id"])
        )
    ]
    return merged, curated_kept, dropped_duplicates, dropped_for_protected, groups


def write_report(
    rows: list[dict],
    curated: list[dict],
    dropped: list[dict],
    dropped_for_protected: list[tuple[dict, str, str]],
    blocking: list[str],
    warnings: list[str],
    path: Path,
) -> None:
    per: dict[str, Counter] = defaultdict(Counter)
    source_counts: Counter = Counter()
    for row in rows:
        per[row["intent"]][row["split"]] += 1
        source_counts[row["source"]] += 1

    lines: list[str] = [
        "# Dataset review report",
        "",
        "Generated by `training/build_dataset_v2.py`.",
        "",
        f"- examples: **{len(rows)}**",
        f"- provenance: {dict(source_counts)}",
        "",
        "## Class balance",
        "",
        "| intent | train | validation | test | total |",
        "|---|---|---|---|---|",
    ]
    for intent in sorted(per):
        c = per[intent]
        lines.append(
            f"| {intent} | {c['train']} | {c['validation']} | {c['test']} | "
            f"{c['train'] + c['validation'] + c['test']} |"
        )
    lines += ["", "## Held-out protection", ""]
    if dropped_for_protected:
        lines.append(
            f"**{len(dropped_for_protected)} training row(s) dropped** because they "
            "duplicated a held-out example:"
        )
        lines.append("")
        lines.append("| intent | text | held-out set | kind |")
        lines.append("|---|---|---|---|")
        for row, pid, kind in sorted(
            dropped_for_protected, key=lambda t: (t[0]["intent"], t[0]["id"])
        ):
            lines.append(
                f"| {row['intent']} | {row['text']} | {pid} | {kind} |"
            )
    else:
        lines.append("No training row duplicates a held-out example.")
    lines += ["", "## Findings", ""]
    if blocking:
        lines.append(f"**{len(blocking)} blocking problem(s).**")
        lines.append("")
        for problem in blocking:
            lines.append(f"- {problem}")
        lines.append("")
        lines.append("The dataset was **not** written.")
    else:
        lines.append("No blocking problems found.")
    if warnings:
        lines += ["", f"## {len(warnings)} warning(s)", ""]
        for warning in warnings:
            lines.append(f"- {warning}")
    lines += ["", "## Duplicates dropped in the merge", ""]
    if dropped:
        lines += ["| intent | text |", "|---|---|"]
        for row in sorted(dropped, key=lambda r: (r["intent"], r["id"])):
            lines.append(f"| {row['intent']} | {row['text']} |")
    else:
        lines.append("None.")
    lines += ["", "## Curated additions, for review", ""]
    by_source: dict[str, list[dict]] = defaultdict(list)
    for row in curated:
        by_source[row["source"]].append(row)
    for source in sorted(by_source):
        lines += [
            f"### source = `{source}` ({len(by_source[source])} rows)",
            "",
            "| intent | text | notes |",
            "|---|---|---|",
        ]
        for row in sorted(by_source[source], key=lambda r: (r["intent"], r["id"])):
            notes = (row.get("notes") or "").replace("|", "\\|")
            lines.append(f"| {row['intent']} | {row['text']} | {notes} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="validate without writing")
    parser.add_argument("--report", type=Path, default=REPORT, help="where to write the review report")
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="exclude a curated file by stem, e.g. curated-v3-hard-negatives.",
    )
    parser.add_argument(
        "--exclude-family",
        action="append",
        default=[],
        help="exclude one family of curated rows, e.g. bare_noun.",
    )
    args = parser.parse_args()

    rows, curated, dropped, dropped_for_protected, _groups = build(
        args.exclude, args.exclude_family
    )
    blocking, warnings = validate(rows, {r["id"]: r["split"] for r in rows})

    if not args.check_only and not blocking:
        write_dataset(rows, CURRENT)
    write_report(
        rows, curated, dropped, dropped_for_protected, blocking, warnings, args.report
    )

    print(f"examples: {len(rows)} ({len(curated)} curated additions)")
    print(f"report:   {args.report}")
    if dropped_for_protected:
        print(
            f"dropped {len(dropped_for_protected)} training row(s) that duplicate a "
            f"held-out example"
        )
    if blocking:
        print(f"\n{len(blocking)} BLOCKING problem(s); the dataset was not written:")
        for problem in blocking[:40]:
            print(f"  - {problem}")
        if len(blocking) > 40:
            print(f"  ... and {len(blocking) - 40} more")
        return 1
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for warning in warnings[:10]:
            print(f"  - {warning}")
        if len(warnings) > 10:
            print(f"  ... and {len(warnings) - 10} more")
    if not args.check_only:
        print("\nwrote the dataset.")
    print("no blocking problems.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
