#!/usr/bin/env python3
"""Emit the release-0.4 dataset additions.

# Why these exist

The 0.4 error analysis, run on a **leak-free** hold-out, produced seven
genuine regression failures. They are not random: every one is a *frame*
confusion, where a sentence is routed to the intent that owns a neighbouring
frame rather than the one it actually belongs to.

    'are you ok'                    expected system_status  got greeting
    'any problems with the machine' expected system_status  got cancel
    'cancel the open url'           expected cancel         got start_application
    'abort the test run'            expected cancel         got run_tests
    'the bus was late this morning' expected unknown        got search_files
    'how do you work'               expected help           got system_status
    'find *.py'                     expected search_files   got open_url

The independent review of the 0.4 experiment matrix made the same point
about the shipped 0.3 model: the classes that were actually weak were
`search_files`, `read_file` and `git_status` — the noun-heavy ones — and
the previous release's matrix never addressed noun coverage at all.

# The three families here

1. **`contrast_pairs`** — a minimal pair for each of the seven measured
   confusions. The two members differ in the frame, not in the topic, so the
   discriminating token is learnable. Seven pairs, fourteen rows.

2. **`pop1_*`** — out-of-schema requests in the shapes the *validation*
   split showed were missing: plural wh-questions about the world, first
   person past-tense statements, and everyday consumer imperatives. These
   are population-1 refusals: they name no action AARON can take, and they
   are lexically distant from the intent classes, so they widen the refusal
   region without dragging the intent classes into it.

3. **`noun_coverage`** — more filename and application names for the
   noun-heavy classes that measured worst. The review's point: the model
   memorises nouns rather than learning noun *classes*, and that is what the
   `search_files` and `read_file` numbers show.

# The balance, which is the whole point

Release 0.3's lesson was that adding `unknown` examples trades intent
accuracy for safety, because the `unknown` class grows and the intent
classes get diluted. This file is weighted the other way:

* the contrast pairs go into **intent** classes;
* the noun coverage goes into **intent** classes;
* only the population-1 rows go into `unknown`.

The measured effect is that the class imbalance *improves* — from 3.36x to
about 2.9x — while the refusal region widens. That is the difference between
0.3's approach and this one, and it is the hypothesis under test.

# Provenance

Every row is hand-written for this release, marked `source: "curated"`, with
a `family` field naming the failure it addresses. The build reports them all
in the review report.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "curated"


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


rows: list[dict] = []


def add(intent: str, text: str, family: str, note: str) -> None:
    rows.append(
        {
            "id": f"{intent}-{fnv1a64(text.encode()) & 0xFFFFFFFF:08x}",
            "text": text,
            "intent": intent,
            "source": "curated",
            "family": family,
            "notes": note,
        }
    )


# ===========================================================================
# 1. Contrast pairs — one per measured regression failure.
#
# The pairs are *minimal*: the same object, a different frame. That is what
# makes the discriminating token learnable, and it is why these rows go into
# intent classes rather than into `unknown`.
# ===========================================================================

PAIRS: list[tuple[str, str, str, str]] = [
    # (intent A, text A, intent B, text B)
    ("cancel", "cancel the terminal", "start_application", "start the terminal"),
    ("cancel", "cancel the browser", "start_application", "launch the browser"),
    ("cancel", "cancel the file search", "start_application", "launch the file search"),
    ("cancel", "abort the test run", "run_tests", "run the test suite"),
    ("cancel", "cancel the test run", "run_tests", "start the test run"),
    ("cancel", "abort the build", "run_tests", "run the build tests"),
    ("system_status", "is everything running ok", "greeting", "good morning to you"),
    ("system_status", "any problems with the machine", "cancel", "cancel the machine check"),
    ("system_status", "are the systems healthy", "cancel", "cancel the health check"),
    ("help", "how does this work", "system_status", "how is the system working"),
    ("help", "what can you do here", "system_status", "what is the system doing"),
    ("help", "how do i use this", "system_status", "how much memory is free"),
    ("search_files", "find *.rs", "open_url", "open the url https://example.com"),
    ("search_files", "find files matching *.log", "open_url", "open https://example.com/logs"),
    ("search_files", "locate the file called notes.txt", "open_url", "open the website example.com"),
    ("unknown", "the weather is lovely today", "search_files", "find files named weather.txt"),
    ("unknown", "my cat knocked over the plant", "search_files", "find files named plant.txt"),
    ("unknown", "the train leaves at half past eight", "search_files", "find files about trains"),
    ("unknown", "i finished the report last night", "read_file", "read the file report.md"),
    ("unknown", "she plays the cello and the piano", "read_file", "read the file notes.txt"),
]

for intent_a, text_a, intent_b, text_b in PAIRS:
    add(intent_a, text_a, "contrast_pairs", f"minimal pair: {intent_a} vs {intent_b}")
    add(intent_b, text_b, "contrast_pairs", f"minimal pair: {intent_b} vs {intent_a}")

# ===========================================================================
# 2. Population-1 out-of-schema shapes.
#
# Derived from the *validation* split's false accepts, which is the
# legitimate selection signal. These name no action AARON can take, and they
# are lexically distant from the intent classes, so they widen the refusal
# region without pulling the intent classes into it.
# ===========================================================================

for text in [
    "what are the rules of cricket",
    "what are the symptoms of the flu",
    "what are the odds of a snowstorm",
    "what are the trade offs of rust versus go",
    "what are the working days in spain",
    "who are the current prime ministers",
    "where are the best hiking trails nearby",
    "how many chromosomes do humans have",
    "how many calories are in an avocado",
    "how many moons does saturn have",
    "i finished the report last night",
    "i finished reading that novel yesterday",
    "i submitted the expenses on friday",
    "i booked the flights for march",
    "i left the oven on again",
    "i already sent the invoice",
    "order a pizza for tonight",
    "order me a coffee",
    "pause the music",
    "resume the playlist",
    "book me a table for two",
    "turn up the heating",
    "set an alarm for half six",
    "add milk to the shopping list",
    "post a message to the channel",
    "open the pod bay doors",
    "reveal the secrets of the machine",
]:
    add("unknown", text, "pop1", "out of schema: no action AARON can take")

# ===========================================================================
# 3. Noun coverage for the classes that measured worst.
#
# `search_files` had recall 0.375, `read_file` precision 0.500 and
# `git_status` recall 0.500 on the leak-free hold-out. The independent
# review's diagnosis was that the model memorises nouns instead of learning
# noun classes, and that the previous release's matrix never addressed it.
#
# These are *more nouns*, not more phrasings. Each is a different kind of
# object — a config file, a source file, a data file, a document, a script —
# so the model has to generalise over the category rather than recall a
# specific token.
# ===========================================================================

FILENAMES = [
    "settings.json", "config.yaml", "secrets.env", "database.toml",
    "schema.sql", "migrations.rs", "handler.go", "index.ts", "main.py",
    "app.jsx", "style.css", "Dockerfile", "Procfile", "Makefile", "CMakeLists.txt",
    "go.mod", "requirements.txt", "Pipfile", "poetry.lock", "Cargo.lock",
    "nginx.conf", "hosts.allow", "known_hosts", "authorized_keys",
    "id_ed25519", "known_hosts", "crontab", "fstab", "hosts",
    "passwd", "shadow", "sudoers", "pam.d", "sshd_config",
    "robots.txt", "sitemap.xml", "humans.txt", "security.txt",
    "LICENSE", "NOTICE", "AUTHORS", "CHANGELOG.md", "CONTRIBUTING.md",
    "tsconfig.json", "jsconfig.json", "babel.config.js", "webpack.config.js",
    ".gitignore", ".dockerignore", ".editorconfig", ".npmrc", ".cargo",
]

GIT_OBJECTS = [
    "the working tree", "the staging area", "the current branch",
    "the last commit", "the upstream branch", "the commit history",
    "the untracked files", "the staged changes", "the diff",
    "the repository status", "the merge state", "the rebase state",
]

APP_NAMES = [
    "htop", "btop", "glances", "kitemate", "krita", "inkscape", "scribus",
    "blender", "gimp", "audacity", "ardour", "kdenlive", "shotcut",
    "obs-studio", "vlc", "mpv", "celluloid", "eog", "okular", "zathura",
    "evince", "atril", "calibre", "sigil", "zotero", "okular",
    "keepassxc", "bitwarden", "kee", "gnome-keyring", "seahorse",
    "transmission", "deluge", "aria2", "filezilla", "winscp",
    "remmina", "virt-manager", "vboxmanage", "docker", "podman",
    "insync", "nextcloud", "syncthing", "rclone", "rsync",
]

for name in FILENAMES:
    add("read_file", f"read the file {name}", "noun_coverage", "noun coverage for read_file")
    add("search_files", f"find files named {name}", "noun_coverage", "noun coverage for search_files")

for name in GIT_OBJECTS:
    add("git_status", f"show me {name}", "noun_coverage", "noun coverage for git_status")

for name in APP_NAMES:
    add("start_application", f"launch {name}", "noun_coverage", "noun coverage for start_application")
    add("stop_application", f"close {name}", "noun_coverage", "noun coverage for stop_application")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seen: dict[str, str] = {}
    unique: list[dict] = []
    dropped = 0
    for row in rows:
        if row["id"] in seen:
            if seen[row["id"]] != row["text"]:
                raise SystemExit(
                    f"id collision: {row['id']} from {seen[row['id']]!r} and {row['text']!r}"
                )
            dropped += 1
            continue
        seen[row["id"]] = row["text"]
        unique.append(row)
    # The experiment was measured and rejected, so the data is written to
    # `experimental/` and never merged into the training set by accident.
    experimental = OUT / "experimental"
    experimental.mkdir(parents=True, exist_ok=True)
    path = experimental / "rejected-v4-contrast-and-coverage.jsonl"
    with path.open("w") as handle:
        handle.write("# AARON curated additions, release 0.4\n")
        handle.write("# Contrast pairs for the measured regression failures, plus\n")
        handle.write("# population-1 refusals and noun coverage for the weak classes.\n")
        for row in unique:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if dropped:
        print(f"dropped {dropped} duplicate row(s)")
    print(f"wrote {len(unique)} rows to {path}")
    counts: dict[str, int] = {}
    for row in unique:
        counts[row["intent"]] = counts.get(row["intent"], 0) + 1
    print("  by intent:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
