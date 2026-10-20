#!/usr/bin/env python3
"""Build the readiness corpus for release 0.5.

The corpus teaches a *second* output: whether a request is specific enough to
act on. It is deliberately **not** merged into `training/dataset/current.jsonl`
— it lives in its own directory, outside the builder's `curated/*.jsonl` glob,
because:

* its rows carry a different label (readiness), not just an intent;
* merging them would put rows with a *natural* intent next to the shipped
  `unknown` rows with the same text, which is a duplicate-text build error
  with a contradictory label;
* the corpus needs its own held-out split, and the shipped hold-out must not
  grow.

Every row comes in a **pair**: an incomplete utterance and a complete one
that differs only in the presence of the argument.

    incomplete: "read the file"        intent: read_file  readiness: needs_clarification
    complete:   "read the readme"      intent: read_file  readiness: ready

The pairing is what makes the two error rates measurable:

* the **false-ready rate** — incomplete rows the head calls `Ready`;
* the **false-not-ready rate** — complete rows the head calls
  `NeedsClarification`.

The complete rows are the *hard* negatives: the same frame with an argument
present. A head that learned "short utterance → incomplete" would ace them
and still be useless, so they are the rows that make the benchmark mean
something.

# The ground truth

An utterance is `needs_clarification` when it requests an action and supplies
no argument that identifies a particular object. It is `ready` when it does.

The line is deliberately conservative: "the readme" names one specific file
in a project and must keep working, so it is *ready*. "the file" names a
category and is not.

# The families

| family | shape | does the shipped rule catch it? |
|---|---|---|
| `verb_only` | the verb and nothing else | no — the extractor hands the verb back as the argument |
| `bare_noun` | a determiner plus a category noun | yes — the noun is on the closed list |
| `quantity` | a determiner plus a quantity | no |
| `dangling_qualifier` | "the file called" | no |
| `category_outside_list` | a determiner plus a category noun the closed list does not contain | no — this is the rule's blind spot |
| `contentless` | a free-text argument that carries no content | yes |

The families the rule misses are the head's whole opportunity, and they are
also the families a closed list cannot express. That is the comparison the
release measures: a learned head against an extended deterministic list.

# Disjointness

The corpus must be exact- and near-disjoint from every shipped dataset and
every held-out evaluation set, using the builder's own metric: normalised-text
equality plus word 3-gram containment at 0.8. A near-duplicate of a shipped
`unknown` row is excluded, because the two would answer the same question
with different labels.

Bare-pronoun rows are excluded on purpose: context resolution is deferred to
release 0.6, and a pronoun is only resolvable with the conversation history,
which the readiness head does not see.

# The split

The corpus is split 70/15/15, not the dataset's 80/10/10. It is small, and
the held-out measurement needs enough rows in each class to mean anything: at
80/10/10 a 600-row corpus would hold out 30 rows per class, and one row is
4.0 points of false-accept rate on the locked benchmark. 70/15/15 holds out
about 45 per class instead.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_PATH = os.path.join(REPO, "training", "readiness", "corpus-v1.jsonl")

NEAR_DUPLICATE_THRESHOLD = 0.8

# The sets the corpus must not overlap.
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


# ---------------------------------------------------------------------------
# The builder's similarity metric, copied from `training/build_dataset_v2.py`
# so the corpus is filtered with exactly the rule the dataset uses.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# The pairs: (incomplete, complete), grouped by intent and family.
# ---------------------------------------------------------------------------

PAIRS: dict[tuple[str, str], list[tuple[str, str]]] = {}


def add(intent: str, family: str, pairs: list[tuple[str, str]]) -> None:
    PAIRS.setdefault((intent, family), []).extend(pairs)


# --- read_file: the argument is a path ---------------------------------------
add(
    "read_file",
    "verb_only",
    [
        ("read", "read main.rs"),
        ("show", "show the changelog"),
        ("cat", "cat the manifest"),
        ("print", "print the licence"),
        ("display", "display the config"),
        ("dump", "dump the journal"),
        ("head", "head the notice"),
        ("tail", "tail the logfile"),
        ("render", "render the template"),
        ("preview", "preview the sketch"),
    ],
)
add(
    "read_file",
    "bare_noun",
    [
        ("read the file", "read the readme"),
        ("read the document", "read the changelog"),
        ("read the script", "read the bootstrap.sh"),
        ("read the folder", "read the fixtures folder"),
        ("read the directory", "read the examples directory"),
        ("read the project", "read the aaron project"),
        ("read the path", "read the /etc path"),
        ("read the command", "read the makefile command"),
        ("read the item", "read the invoice item"),
        ("read the thing", "read the parcel thing"),
        ("read the app", "read the settings app"),
        ("read the site", "read the marketing site"),
        ("read the link", "read the reference link"),
        ("read the website", "read the vendor website"),
        ("read the suite", "read the integration suite"),
    ],
)
add(
    "read_file",
    "quantity",
    [
        ("read the first 20 lines", "read the first 20 lines of main.rs"),
        ("read the last 10 lines", "read the last 10 lines of the changelog"),
        ("read the top 5 lines", "read the top 5 lines of the config"),
        ("read the next 3 paragraphs", "read the next 3 paragraphs of the paper"),
        ("read the first hundred bytes", "read the first hundred bytes of main.rs"),
        ("read the last two kilobytes", "read the last two kilobytes of the log"),
        ("read the first 40 lines", "read the first 40 lines of the changelog"),
        ("read the middle 10 lines", "read the middle 10 lines of the config"),
        ("read the final paragraph", "read the final paragraph of the report"),
        ("read the first two pages", "read the first two pages of the paper"),
        ("search for the top 5 matches", "search for the top 5 matches in src"),
        ("show the first 3 results", "show the first 3 results in the log"),
    ],
)
add(
    "read_file",
    "dangling_qualifier",
    [
        ("read the file called", "read the file called main.rs"),
        ("read the document titled", "read the document titled notes"),
        ("read the script named", "read the script named bootstrap.sh"),
        ("read the folder called", "read the folder called fixtures"),
        ("read the report titled", "read the report titled summary"),
        ("read the file named", "read the file named main.rs"),
        ("read the log containing", "read the log containing the panic"),
        ("read the entry mentioning", "read the entry mentioning the timeout"),
        ("read the list including", "read the list including the backups"),
        ("read the record showing", "read the record showing the error"),
    ],
)
add(
    "read_file",
    "category_outside_list",
    [
        ("read the spreadsheet", "read the spreadsheet budget.csv"),
        ("read the invoice", "read the invoice invoice-0042.pdf"),
        ("read the receipt", "read the receipt receipt-march.txt"),
        ("read the transcript", "read the transcript meeting.md"),
        ("read the recording", "read the recording session.log"),
        ("read the manuscript", "read the manuscript chapter-one.md"),
        ("read the screenshot", "read the screenshot desktop.png"),
        ("read the notebook", "read the notebook analysis.ipynb"),
        ("read the gist", "read the gist snippet.rs"),
        ("read the blueprint", "read the blueprint design.svg"),
        ("read the roadmap", "read the roadmap roadmap.md"),
        ("read the checklist", "read the checklist release.md"),
        ("read the schedule", "read the schedule march.md"),
        ("read the budget", "read the budget budget.csv"),
        ("read the diary", "read the diary journal.md"),
        ("read the contract", "read the contract lease.pdf"),
        ("read the agenda", "read the agenda agenda.md"),
        ("read the manifest", "read the manifest Cargo.toml"),
        ("read the schema", "read the schema schema.sql"),
        ("read the dashboard", "read the dashboard panel.html"),
        ("read the plugin", "read the plugin plugin.json"),
        ("read the theme", "read the theme theme.toml"),
        ("read the snippet", "read the snippet example.rs"),
        ("read the template", "read the template base.html"),
    ],
)

# --- search_files: the argument is a pattern ---------------------------------
add(
    "search_files",
    "verb_only",
    [
        ("search", "search main.rs"),
        ("find", "find the todo marker"),
        ("locate", "locate the config file"),
        ("grep", "grep the unwrap calls"),
        ("scan", "scan the rust sources"),
        ("hunt", "hunt the compiler warning"),
        ("sift", "sift the changelog entries"),
    ],
)
add(
    "search_files",
    "bare_noun",
    [
        ("search for the file", "search for the readme"),
        ("search for the document", "search for the changelog"),
        ("search for the folder", "search for the migrations folder"),
        ("search for the project", "search for the aaron project"),
        ("search for the script", "search for the build script"),
        ("search for the item", "search for the todo item"),
        ("search for the app", "search for the settings app"),
        ("search for the site", "search for the marketing site"),
        ("search for the link", "search for the reference link"),
        ("search for the suite", "search for the integration suite"),
    ],
)
add(
    "search_files",
    "category_outside_list",
    [
        ("search for the pattern", "search for the pattern *.toml"),
        ("search for the glob", "search for the glob **/*.rs"),
        ("search for the extension", "search for the extension .md"),
        ("search for the filename", "search for the filename Cargo.toml"),
        ("search for the regex", "search for the regex ^pub fn"),
        ("search for the wildcard", "search for the wildcard *.log"),
        ("search for the template", "search for the template base.html"),
        ("search for the suffix", "search for the suffix .rs"),
        ("search for the prefix", "search for the prefix src/"),
        ("search for the annotation", "search for the annotation @deprecated"),
        ("search for the macro", "search for the macro println"),
        ("search for the import", "search for the import serde"),
    ],
)
add(
    "search_files",
    "dangling_qualifier",
    [
        ("search for the file called", "search for the file called main.rs"),
        ("search for the pattern named", "search for the pattern named todo"),
        ("search for the glob matching", "search for the glob matching *.rs"),
        ("search for the file named", "search for the file named main.rs"),
        ("search for the folder containing", "search for the folder containing fixtures"),
    ],
)

# --- open_url: the argument is a URL -----------------------------------------
add(
    "open_url",
    "verb_only",
    [
        ("browse", "browse example.com"),
        ("visit", "visit wikipedia.org"),
        ("surf", "surf news.ycombinator.com"),
        ("navigate", "navigate crates.io"),
        ("fetch", "fetch example.org"),
    ],
)
add(
    "open_url",
    "bare_noun",
    [
        ("open the url", "open https://example.com"),
        ("open the site", "open https://wikipedia.org"),
        ("open the link", "open https://crates.io"),
        ("open the website", "open https://rust-lang.org"),
        ("open the web page", "open https://example.com/docs"),
        ("open the address", "open https://example.com/index.html"),
        ("open the homepage", "open https://example.org"),
    ],
)
add(
    "open_url",
    "category_outside_list",
    [
        ("open the blog post", "open https://example.com/post"),
        ("open the wiki page", "open https://en.wikipedia.org/wiki/Rust"),
        ("open the api docs", "open https://docs.rs/candle"),
        ("open the issue", "open https://github.com/aaron-assistant/aaron/issues/1"),
        ("open the release notes", "open https://example.com/releases"),
        ("open the pull request", "open https://github.com/aaron-assistant/aaron/pull/7"),
        ("open the changelog", "open https://example.com/changelog"),
        ("open the documentation", "open https://example.com/docs"),
        ("open the reference", "open https://example.com/reference"),
        ("open the tutorial", "open https://example.com/tutorial"),
        ("open the discussion", "open https://example.com/discussion"),
    ],
)

# --- start_application: the argument is an application -----------------------
add(
    "start_application",
    "verb_only",
    [
        ("launch", "launch the terminal"),
        ("start", "start the browser"),
        ("fire", "fire the terminal"),
        ("spawn", "spawn the editor"),
        ("raise", "raise the browser"),
    ],
)
add(
    "start_application",
    "bare_noun",
    [
        ("open the app", "open the terminal"),
        ("open the application", "open the browser"),
        ("launch the program", "launch the terminal"),
        ("start the process", "start the browser"),
        ("open the script", "open the terminal"),
        ("open the frontend", "open the browser"),
        ("launch the item", "launch the terminal"),
        ("start the thing", "start the browser"),
        ("open the command", "open the terminal"),
        ("open the project", "open the browser"),
    ],
)
add(
    "start_application",
    "category_outside_list",
    [
        ("open the editor", "open vscode"),
        ("open the mail client", "open thunderbird"),
        ("open the music player", "open spotify"),
        ("open the file manager", "open thunar"),
        ("open the pdf reader", "open okular"),
        ("open the chat client", "open signal"),
        ("open the note app", "open obsidian"),
        ("open the image viewer", "open eog"),
        ("open the debugger", "open gdb"),
        ("open the package manager", "open pacman"),
        ("open the music daemon", "open spotifyd"),
    ],
)

# --- stop_application: the argument is an application ------------------------
add(
    "stop_application",
    "verb_only",
    [
        ("kill", "kill the terminal"),
        ("close", "close the browser"),
        ("stop", "stop the server"),
        ("terminate", "terminate the service"),
        ("end", "end the editor"),
        ("quit", "quit the browser"),
    ],
)
add(
    "stop_application",
    "bare_noun",
    [
        ("kill the process", "kill the terminal"),
        ("close the application", "close the browser"),
        ("stop the program", "stop the daemon"),
        ("terminate the script", "terminate the worker"),
        ("kill the item", "kill the stray editor"),
        ("stop the project", "stop the build server"),
        ("kill the app", "kill the terminal"),
        ("close the program", "close the browser"),
        ("stop the application", "stop the builder"),
        ("terminate the thing", "terminate the service"),
    ],
)
add(
    "stop_application",
    "category_outside_list",
    [
        ("kill the editor", "kill vscode"),
        ("kill the mail client", "kill thunderbird"),
        ("kill the music player", "kill spotify"),
        ("kill the file manager", "kill thunar"),
        ("kill the pdf reader", "kill okular"),
        ("kill the chat client", "kill signal"),
        ("kill the image viewer", "kill eog"),
        ("kill the debugger", "kill gdb"),
        ("kill the package manager", "kill pacman"),
    ],
)

# --- remember: the argument is free text -------------------------------------
add(
    "remember",
    "verb_only",
    [
        ("remember", "remember my birthday is in June"),
        ("note", "note the boiler is broken"),
        ("memorize", "memorize the wifi password"),
        ("log", "log the office closes at six"),
    ],
)
add(
    "remember",
    "contentless",
    [
        ("remember that", "remember that the boiler is broken"),
        ("remember this", "remember this is important"),
        ("remember it", "remember it is in the drawer"),
        ("remember the thing", "remember the thing about the boiler"),
        ("remember the stuff", "remember the stuff about the boiler"),
        ("remember that one", "remember that one about the boiler"),
    ],
)

# --- recall: the argument is free text ---------------------------------------
add(
    "recall",
    "verb_only",
    [
        ("recall", "recall my birthday"),
        ("remind", "remind me about the dentist"),
        ("retrieve", "retrieve my passport number"),
    ],
)
add(
    "recall",
    "contentless",
    [
        ("recall that", "recall that the boiler is broken"),
        ("recall this", "recall what this was about"),
        ("recall it", "recall what it was about"),
        ("recall the thing", "recall the thing about the boiler"),
        ("recall the stuff", "recall the stuff about the boiler"),
        ("recall that one", "recall that one about the boiler"),
    ],
)

# Non-actionable utterances: the head must never call these incomplete.
NON_ACTIONABLE = [
    ("greeting", "hello there"),
    ("greeting", "good morning"),
    ("greeting", "hey aaron"),
    ("greeting", "hi"),
    ("greeting", "greetings"),
    ("greeting", "morning"),
    ("greeting", "good evening"),
    ("help", "what can you do"),
    ("help", "help"),
    ("help", "what are my capabilities"),
    ("help", "list your commands"),
    ("help", "show me the help"),
    ("help", "how do you work"),
    ("unknown", "the bus was late this morning"),
    ("unknown", "it is raining outside"),
    ("unknown", "she left the office at noon"),
    ("unknown", "the meeting starts at nine"),
    ("unknown", "he bought a new bicycle"),
    ("unknown", "that was a long journey"),
    ("unknown", "they finished the migration last week"),
    ("unknown", "the package arrived yesterday"),
    ("unknown", "we should leave before six"),
    ("unknown", "the report was long"),
]


# ---------------------------------------------------------------------------
# Build.
# ---------------------------------------------------------------------------

def short_hash(text: str) -> str:
    return format(int(hashlib.sha256(text.encode()).hexdigest()[:8], 16), "08x")


def load_protected() -> dict[str, list[str]]:
    protected: dict[str, list[str]] = {}
    for rel in PROTECTED:
        path = os.path.join(REPO, rel)
        if not os.path.isfile(path):
            continue
        texts = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    texts.append(json.loads(line)["text"])
                except (json.JSONDecodeError, KeyError):
                    continue
        protected[rel] = texts
    return protected


def main() -> int:
    protected = load_protected()

    rows: list[dict] = []
    dropped: Counter[str] = Counter()
    dropped_texts: list[tuple[str, str, str]] = []
    kept_incomplete = 0
    kept_complete = 0

    def conflicts(text: str) -> str | None:
        for rel, texts in protected.items():
            for other in texts:
                if normalise(text) == normalise(other) or is_near_duplicate(text, other):
                    return rel
        return None

    # The same complete utterance may be the counterpart of more than one
    # incomplete one, so the complete side is deduplicated. The incomplete
    # side is not: two identical incomplete utterances would be two rows
    # teaching the same thing.
    seen_positive: set[tuple[str, str]] = set()
    seen_negative: set[tuple[str, str]] = set()
    positive_texts: dict[str, str] = {}
    negative_texts: dict[str, str] = {}

    for (intent, family), pairs in PAIRS.items():
        for index, (incomplete, complete) in enumerate(pairs):
            # The pair is atomic: if one of the two rows is rejected, the
            # other goes with it. A complete row whose incomplete
            # counterpart is gone is a negative with no positive to contrast
            # against, and the contrast is the whole point of the corpus.
            pair_id = f"{intent}|{family}|{index}"
            for text, readiness, kind in (
                (incomplete, "needs_clarification", "incomplete"),
                (complete, "ready", "complete"),
            ):
                key = (intent, normalise(text))
                if kind == "incomplete":
                    if key in seen_positive:
                        continue
                    seen_positive.add(key)
                    positive_texts[normalise(text)] = intent
                else:
                    if key in seen_negative:
                        continue
                    seen_negative.add(key)
                    negative_texts[normalise(text)] = intent

                conflict = conflicts(text)
                if conflict:
                    dropped[conflict] += 1
                    dropped_texts.append((text, conflict, kind))
                    continue
                rows.append(
                    {
                        "id": f"readiness-{intent}-{short_hash(text)}",
                        "text": text,
                        "intent": intent,
                        "readiness": readiness,
                        "family": family,
                        "kind": kind,
                        "source": "authored",
                        "pair": pair_id,
                        "notes": f"readiness corpus, release 0.5; family `{family}`",
                    }
                )
                if kind == "incomplete":
                    kept_incomplete += 1
                else:
                    kept_complete += 1

    # A text cannot be incomplete for one intent and complete for another:
    # that would be a contradiction, and the head would be taught both.
    contradictions = [
        (text, positive_texts[text], negative_texts[text])
        for text in positive_texts
        if text in negative_texts and positive_texts[text] != negative_texts[text]
    ]
    if contradictions:
        for text, pos, neg in contradictions:
            print(
                f"error: {text!r} is incomplete for `{pos}` but complete for `{neg}`",
                file=sys.stderr,
            )
        return 1

    for intent, text in NON_ACTIONABLE:
        conflict = conflicts(text)
        if conflict:
            dropped[conflict] += 1
            dropped_texts.append((text, conflict, "non_actionable"))
            continue
        rows.append(
            {
                "id": f"readiness-{intent}-{short_hash(text)}",
                "text": text,
                "intent": intent,
                "readiness": "not_applicable",
                "family": "non_actionable",
                "kind": "non_actionable",
                "source": "authored",
                "notes": "readiness corpus, release 0.5; family `non_actionable`",
            }
        )

    # --- internal consistency --------------------------------------------
    seen: dict[tuple[str, str], str] = {}
    for row in rows:
        key = (row["intent"], normalise(row["text"]))
        if key in seen:
            print(
                f"error: {row['text']!r} appears twice for `{row['intent']}`",
                file=sys.stderr,
            )
            return 1
        seen[key] = row["text"]

    # --- the split --------------------------------------------------------
    # Group near-duplicates together, then assign whole groups, so a pair
    # never straddles the boundary.
    parent: dict[str, str] = {r["id"]: r["id"] for r in rows}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_id = {r["id"]: r for r in rows}
    for i, a in enumerate(rows):
        sa = shingles(a["text"])
        for b in rows[i + 1 :]:
            if containment(sa, shingles(b["text"])) >= NEAR_DUPLICATE_THRESHOLD:
                union(a["id"], b["id"])

    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        groups[find(row["id"])].append(row["id"])

    # The split is a function of the group's lowest id, so adding a row does
    # not move the others. The corpus is split 70/15/15: it is small, and a
    # 10% hold-out would be too thin to measure a rate.
    for member_ids in groups.values():
        lowest = min(member_ids)
        bucket = int(hashlib.sha256(lowest.encode()).hexdigest()[:8], 16) % 100
        split = "train" if bucket < 70 else ("validation" if bucket < 85 else "test")
        for member in member_ids:
            by_id[member]["split"] = split

    # --- write ------------------------------------------------------------
    os.makedirs(os.path.dirname(CORPUS_PATH), exist_ok=True)
    rows.sort(key=lambda r: (r["intent"], r["family"], r["text"]))
    with open(CORPUS_PATH, "w", encoding="utf-8") as handle:
        handle.write("# AARON readiness corpus, release 0.5\n")
        handle.write("# Not part of the intent training set. See the generator.\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    # --- report -----------------------------------------------------------
    print(f"corpus: {CORPUS_PATH}")
    print(f"rows: {len(rows)}")
    print()
    by_split: dict[str, Counter] = defaultdict(Counter)
    by_family: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_split[row["split"]][row["readiness"]] += 1
        by_family[row["family"]][row["readiness"]] += 1

    print("by split")
    for split in sorted(by_split):
        counts = by_split[split]
        total = sum(counts.values())
        print(
            f"  {split:<12} needs_clarification {counts['needs_clarification']:>4}  "
            f"ready {counts['ready']:>4}  not_applicable {counts['not_applicable']:>4}   (of {total})"
        )
    print()
    print("by family")
    for family in sorted(by_family):
        counts = by_family[family]
        print(
            f"  {family:<26} needs_clarification {counts['needs_clarification']:>4}  "
            f"ready {counts['ready']:>4}  not_applicable {counts['not_applicable']:>4}"
        )
    print()
    print("by intent")
    by_intent: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        by_intent[row["intent"]][row["readiness"]] += 1
    for intent in sorted(by_intent):
        counts = by_intent[intent]
        print(
            f"  {intent:<20} needs_clarification {counts['needs_clarification']:>4}  "
            f"ready {counts['ready']:>4}  not_applicable {counts['not_applicable']:>4}"
        )
    print()
    print(f"near-duplicate groups: {len(groups)}")
    print(f"kept: {kept_incomplete} incomplete, {kept_complete} complete")
    if dropped:
        print("dropped for overlap with a protected set:")
        for rel, count in dropped.most_common():
            print(f"  {count:>4}  {rel}")
    if os.environ.get("AARON_SHOW_DROPPED"):
        print()
        print("dropped rows:")
        for text, rel, kind in dropped_texts:
            print(f"  {kind:<15} {text!r}  ({rel})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
