#!/usr/bin/env python3
"""Author the release-0.2 held-out benchmark.

This file is a **permanently locked benchmark**. It is not part of the
training dataset, it is never merged into `training/dataset/current.jsonl`,
and the build script refuses to run if any of its rows appears in the
training data.

Why a second benchmark
----------------------
The v1 hold-out (65 examples) was used to diagnose the 0.2 work: the error
analysis in `docs/release-0.2-plan.md` was produced by scoring against it.
Once a set has been used to *diagnose*, it is no longer a clean estimate of
generalisation for any model that was changed in response. Scoring the final
model on it would flatter it.

So this benchmark was authored **after** the diagnosis but **without** looking
at how any model scored on it, and it is frozen from here on. It is written
from the same category analysis — intent overlap, out-of-schema handling,
short commands, interrogatives — so it probes the same weaknesses, but with
different content.

Every intent gets at least eight examples, so no class is measured on a
handful of rows. The out-of-schema class gets examples that deliberately look
like commands, because a false accept is the failure that can run the wrong
tool.

Usage:
    python3 training/build_benchmark.py            # write the benchmark
    python3 training/build_benchmark.py --check    # verify it is disjoint
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training" / "evaluation" / "benchmark-v3.jsonl"

INTENTS = [
    "greeting", "help", "system_status", "remember", "recall", "search_files",
    "read_file", "git_status", "run_tests", "open_url", "start_application",
    "stop_application", "cancel", "unknown",
]


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for byte in data:
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


# --------------------------------------------------------------------------
# The benchmark.
#
# Written to exercise the *categories* the error analysis found, not the
# specific sentences that failed:
#
#   intent_overlap             commands whose discriminating token is shared
#   out_of_schema_rejected     real commands shaped like questions
#   out_of_schema_accepted     out-of-schema text shaped like commands
#   missing_coverage           phrasings the v1/v2 data does not contain
#   short commands             one and two word commands
# --------------------------------------------------------------------------

CASES: list[tuple[str, str]] = [
    # --- greeting -------------------------------------------------------
    ("greeting", "howdy partner"),
    ("greeting", "good night to you"),
    ("greeting", "hello aaron there"),
    ("greeting", "anybody there"),
    ("greeting", "afternoon"),
    ("greeting", "morning to you"),
    ("greeting", "hey hey"),
    ("greeting", "greetings and salutations"),
    # --- help -----------------------------------------------------------
    ("help", "what are you for"),
    ("help", "which things can you handle"),
    ("help", "how do i speak to you"),
    ("help", "enumerate your commands"),
    ("help", "which requests do you understand"),
    ("help", "what should i ask you"),
    ("help", "print your cheat sheet"),
    ("help", "what can this assistant do"),
    # --- system_status --------------------------------------------------
    ("system_status", "how is the desktop doing"),
    ("system_status", "show me the vital signs"),
    ("system_status", "is everything running smoothly"),
    ("system_status", "how much ram is available"),
    ("system_status", "are things nominal"),
    ("system_status", "give me a health report on the hardware"),
    ("system_status", "is the cpu under load"),
    ("system_status", "how much disk is left"),
    ("system_status", "what is the load average"),
    # --- remember -------------------------------------------------------
    ("remember", "remember that the wifi is called aurora"),
    ("remember", "remember that my dentist is on tuesday"),
    ("remember", "make a note that the office shuts at six"),
    ("remember", "remember that i cannot eat shellfish"),
    ("remember", "keep in mind that the train departs at 7am"),
    ("remember", "remember that the repo is named aaron"),
    ("remember", "note it down that i favour dark mode"),
    ("remember", "remember that the heating is broken"),
    # --- recall ---------------------------------------------------------
    ("recall", "what do you remember about my wifi"),
    ("recall", "look up what you know about my dentist"),
    ("recall", "search your memory for my train time"),
    ("recall", "what did i say about the release"),
    ("recall", "remind me about the berlin trip"),
    ("recall", "pull out what you know about my tablets"),
    ("recall", "search your memory for my bank account"),
    ("recall", "dig out what you kept about my passport"),
    # --- search_files ---------------------------------------------------
    ("search_files", "where is the file called app.py"),
    ("search_files", "hunt for files named launch.json"),
    ("search_files", "look for any toml files"),
    ("search_files", "track down the containerfile"),
    ("search_files", "scan the tree for ruby files"),
    ("search_files", "sift through the tree for rust files"),
    ("search_files", "list every file ending in .yaml"),
    ("search_files", "show me what is in the lib directory"),
    # --- read_file ------------------------------------------------------
    ("read_file", "open the pipfile"),
    ("read_file", "read the file called gitmodules"),
    ("read_file", "show me the file named brewfile"),
    ("read_file", "what is in the .env.sample"),
    ("read_file", "print the first 40 lines of lib.rs"),
    ("read_file", "read the top of the history file"),
    ("read_file", "display the contents of taskfile"),
    ("read_file", "open the guardfile"),
    # --- git_status -----------------------------------------------------
    ("git_status", "which branch is active"),
    ("git_status", "is there anything staged"),
    ("git_status", "what would be committed"),
    ("git_status", "give me the change summary"),
    ("git_status", "how many files are not tracked"),
    ("git_status", "are there pending commits"),
    ("git_status", "what is in the working directory"),
    ("git_status", "is the repo tidy"),
    # --- run_tests ------------------------------------------------------
    ("run_tests", "execute the rust test suite"),
    ("run_tests", "run the shell tests"),
    ("run_tests", "kick off the smoke tests"),
    ("run_tests", "run the tests for the tools crate"),
    ("run_tests", "fire up the integration tests"),
    ("run_tests", "go ahead and run the test suite"),
    ("run_tests", "trigger the full test suite"),
    ("run_tests", "start the test harness"),
    # --- open_url -------------------------------------------------------
    ("open_url", "browse to https://example.org"),
    ("open_url", "navigate to https://docs.rs/serde"),
    ("open_url", "bring up https://wiki.archlinux.org"),
    ("open_url", "go to https://lib.rs"),
    ("open_url", "pull up https://example.io/guide"),
    ("open_url", "load https://en.m.wikipedia.org"),
    ("open_url", "visit https://example.net/guide"),
    ("open_url", "take me to https://github.com/aaron-assistant/docs"),
    # --- start_application ----------------------------------------------
    ("start_application", "launch glances"),
    ("start_application", "open keepass"),
    ("start_application", "start the filezilla client"),
    ("start_application", "run scribus"),
    ("start_application", "fire up the calibre ebook library"),
    ("start_application", "open the dconf viewer"),
    ("start_application", "launch the virtualbox manager"),
    ("start_application", "open nerd fonts viewer"),
    # --- stop_application ------------------------------------------------
    ("stop_application", "close the bittorrent client"),
    ("stop_application", "kill gimp"),
    ("stop_application", "terminate the profiler"),
    ("stop_application", "end the archive manager"),
    ("stop_application", "shut down the calendar client"),
    ("stop_application", "quit the audio editor"),
    ("stop_application", "halt the process watcher"),
    ("stop_application", "get rid of the video player"),
    # --- cancel ---------------------------------------------------------
    ("cancel", "never mind that"),
    ("cancel", "actually forget it"),
    ("cancel", "hold on stop there"),
    ("cancel", "scratch that i was wrong"),
    ("cancel", "call off the download"),
    ("cancel", "stand down now"),
    ("cancel", "no wait not this"),
    ("cancel", "put that back please"),
    # --- unknown: out of schema, but command-shaped ----------------------
    ("unknown", "delete the build folder"),
    ("unknown", "post the invoice by email"),
    ("unknown", "install the missing library"),
    ("unknown", "increase the speaker volume"),
    ("unknown", "empty the waste basket"),
    ("unknown", "rename the field in the record"),
    ("unknown", "publish the crate to the index"),
    ("unknown", "restart the queue service"),
    ("unknown", "change my default editor"),
    ("unknown", "schedule a call for friday"),
    # --- unknown: questions about the world -------------------------------
    ("unknown", "how do i replace a washing machine pump"),
    ("unknown", "how do i sharpen a kitchen knife"),
    ("unknown", "what is the speed of light"),
    ("unknown", "why do onions make you cry"),
    ("unknown", "who wrote the book of mormon"),
    ("unknown", "how many continents are there"),
    ("unknown", "what is the boiling point of mercury"),
    ("unknown", "should i buy a house or rent"),
    ("unknown", "how do i unclog a kitchen sink"),
    ("unknown", "what is a good gift for a new colleague"),
    # --- unknown: polite requests AARON has no intent for -----------------
    ("unknown", "can you book me a taxi"),
    ("unknown", "could you feed the cat"),
    ("unknown", "would you mind taking out the bins"),
    ("unknown", "please fetch my coat from the hall"),
    ("unknown", "i would like to order a sandwich"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify disjointness only")
    args = parser.parse_args()

    seen: set[str] = set()
    rows = []
    for intent, text in CASES:
        if intent not in INTENTS:
            raise SystemExit(f"unknown intent {intent!r}")
        key = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
        if key in seen:
            raise SystemExit(f"duplicate benchmark text: {text!r}")
        seen.add(key)
        rows.append(
            {
                "id": f"{intent}-b{fnv1a64(text.encode()) & 0xFFFFFFFF:08x}",
                "text": text,
                "intent": intent,
                "source": "benchmark",
                "notes": "release 0.2 locked held-out benchmark; never trained on",
            }
        )

    # The benchmark must not overlap the training data.
    dataset = ROOT / "training" / "dataset" / "current.jsonl"
    if dataset.exists():
        training = {
            re.sub(r"[^a-z0-9]+", " ", json.loads(line)["text"].lower()).strip()
            for line in dataset.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
        overlap = seen & training
        if overlap:
            raise SystemExit(
                f"benchmark overlaps the training data ({len(overlap)} example(s)): "
                f"{sorted(overlap)[:5]}"
            )
        print(f"disjoint from {len(training)} training example(s)")

    if args.check:
        print("check passed")
        return 0

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as handle:
        handle.write("# AARON locked held-out benchmark, release 0.2\n")
        handle.write("# Never merged into the training dataset. Do not train on this file.\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} benchmark examples to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
