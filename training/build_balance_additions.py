#!/usr/bin/env python3
"""Emit the release-0.3 class-balance additions — measured and REJECTED.

# Why these exist

Adding the hard negatives grew the `unknown` class from 163 to 194 while the
other thirteen classes stayed where they were. The measured class imbalance
went from 2.56x to 4.09x, and the effect on the model was not subtle: the
classes whose complete commands are *short* lost the most.

The mechanism is specific. The hard negatives are fragments — "the file", "the
app", "open it" — and a bag-of-features model has no way to tell a fragment
from a short command except by length. So it learned *short => unknown*, and
`read_file` recall fell from 0.750 to 0.375 while `search_files` fell from
0.800 to 0.500.

Two things fix it, and this file is one of them:

1. **Controls** — the same verbs with a complete object, so the *object* is the
   discriminative feature rather than the length. Those are in
   `curated-v3-hard-negatives.jsonl`.
2. **Balance** — more examples for the classes the hard negatives starved.
   This file. A dataset whose largest class is four times its smallest has
   a majority class, and a majority class is a class the model learns.

# Rules for this file

* Every example is a **complete, realistic command** for its intent.
* No filler-word variants. "check system status please" and "check system
  status now" are the same example twice, and a dataset full of them measures
  the author's patience rather than the model's understanding.
* Each class gets a genuinely different set of situations: different verbs,
  different objects, different levels of formality.
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


def add(intent: str, text: str, note: str) -> None:
    rows.append(
        {
            "id": f"{intent}-{fnv1a64(text.encode()) & 0xFFFFFFFF:08x}",
            "text": text,
            "intent": intent,
            "source": "curated",
            "family": "balance",
            "notes": note,
        }
    )


# --- greeting: 57 examples, the smallest class --------------------------
for text in [
    "hi there",
    "morning all",
    "good afternoon",
    "evening",
    "hello aaron",
    "hey there",
    "hiya aaron",
    "howdy",
    "greetings",
    "hi aaron",
    "morning",
    "hey",
    "is it me you are expecting",
    "good day",
    "hello again",
    "anyone home",
    "aaron are you there",
    "just saying hello",
    "well hello",
    "hi hello hi",
]:
    add("greeting", text, "balance: greeting had 57 examples, the smallest class")

# --- help: 58 examples ---------------------------------------------------
for text in [
    "what can you do",
    "what are you capable of",
    "which commands do you understand",
    "how do I speak with you",
    "what should I ask you",
    "list your abilities",
    "show me your cheat sheet",
    "what do you understand",
    "help me out here",
    "what can I say to you",
    "give me the command list",
    "what are your options",
    "how do you work",
    "what is in the manual",
    "walk me through what you can do",
    "what are your supported commands",
    "print the help",
    "tell me what you are good for",
]:
    add("help", text, "balance: help had 58 examples")

# --- open_url: 64 examples ------------------------------------------------
for text in [
    "open the website https://example.com",
    "open the website https://example.org",
    "open the website https://docs.rs/candle",
    "open the website https://crates.io",
    "open the website https://www.rust-lang.org",
    "open the website https://github.com/aaron-assistant/aaron",
    "open the website https://en.wikipedia.org",
    "open the website https://man.archlinux.org",
    "open the website https://news.ycombinator.com",
    "open the website https://lib.rs",
    "browse to https://example.com",
    "navigate to https://example.org",
    "take me to https://crates.io",
    "load https://docs.rs/serde",
    "visit https://example.net",
    "pull up https://man.archlinux.org",
    "go to https://www.rust-lang.org/learn",
    "open the url https://example.com/docs",
    "show me https://example.io",
    "bring up https://example.com",
]:
    add("open_url", text, "balance: open_url had 64 examples")

# --- stop_application: 64 examples ----------------------------------------
for text in [
    "kill the terminal",
    "close the browser",
    "stop the file manager",
    "quit the text editor",
    "terminate the media player",
    "end the music player",
    "shut down the mail client",
    "halt the code editor",
    "get rid of the image viewer",
    "close the calculator",
    "kill the torrent client",
    "stop the system monitor",
    "close the spreadsheet",
    "quit the video editor",
    "terminate the database tool",
    "end the archive manager",
    "shut down the calendar",
    "halt the drawing app",
    "get rid of the task manager",
    "close the screenshot tool",
]:
    add("stop_application", text, "balance: stop_application had 64 examples")

# --- git_status: 70 examples -----------------------------------------------
for text in [
    "show git status",
    "check the working tree",
    "is anything staged",
    "which branch am I on",
    "what would git commit",
    "how many files are untracked",
    "give me the diff summary",
    "what changed since the last commit",
    "any uncommitted changes",
    "is the repo clean",
    "what is staged right now",
    "show me the modified files",
    "tell me the current branch",
    "which files need committing",
    "what does git think",
    "show the change summary",
    "do I have conflicts",
    "what is in the index",
    "list the untracked files",
    "any staged changes",
]:
    add("git_status", text, "balance: git_status had 70 examples")

# --- run_tests: 73 examples -----------------------------------------------
for text in [
    "run the tests",
    "run the cargo test suite",
    "run the unit tests",
    "execute the test suite",
    "kick off the tests",
    "start the test runner",
    "run the integration tests",
    "fire up the test suite",
    "run the tests for the brain crate",
    "trigger the workspace tests",
    "go ahead and run the tests",
    "run the python tests",
    "run the security tests",
    "start the test suite please",
    "run the regression tests",
    "run every test",
    "run the doc tests",
    "launch the test suite",
    "run the test suite in release mode",
    "please run the tests",
]:
    add("run_tests", text, "balance: run_tests had 73 examples")

# --- recall: 75 examples ---------------------------------------------------
for text in [
    "what do you remember about my wifi",
    "what do you know about my dentist",
    "search your memory for my train time",
    "what did I tell you about the rollout",
    "remind me what I said about lisbon",
    "pull up what you know about my medication",
    "check your memory for my bank details",
    "dig through your memory for my passport",
    "go through your notes about my prescription",
    "look up what you know about my landlord",
    "what have you stored about my commute",
    "retrieve what you know about my allergies",
    "fetch from memory what I said about zsh",
    "what is in your memory about my flights",
    "do you remember my morning routine",
    "what did you note about my standup",
    "find what you saved about my holiday",
    "what did you write down about the boiler",
    "recite what you have on my passport",
    "show me what you kept about my wifi",
]:
    add("recall", text, "balance: recall had 75 examples")

# --- system_status: 77 examples --------------------------------------------
for text in [
    "check system status",
    "how is the machine doing",
    "give me the vitals",
    "is the machine healthy",
    "how much memory is free",
    "what is the cpu usage",
    "report the hardware health",
    "how much disk space is left",
    "are the systems nominal",
    "what is the load average",
    "check the machine health",
    "how much swap is in use",
    "give me the resource overview",
    "what does the gpu look like",
    "is anything eating the cpu",
    "show me the system statistics",
    "any unusual resource usage",
    "how many cores are busy",
    "what is the memory pressure",
    "quick health check on the box",
]:
    add("system_status", text, "balance: system_status had 77 examples")

# --- search_files: 81 examples ---------------------------------------------
for text in [
    "find files named main.rs",
    "find files named config.toml",
    "find files named *.py",
    "find files named readme.md",
    "find files called setup.sh",
    "locate the file named notes.txt",
    "search for files called Cargo.toml",
    "which files are named Dockerfile",
    "show me the files in the src directory",
    "list the files in the tests directory",
    "find anything named .gitignore",
    "look for toml files",
    "hunt down the dockerfile",
    "which files end in .yaml",
    "search the project for rust files",
    "list every file matching *.json",
    "show me the files that changed yesterday",
    "find files named justfile",
    "locate any shell scripts",
    "which files are named pyproject.toml",
]:
    add("search_files", text, "balance: search_files had 81 examples")

# --- read_file: 130 examples, but it lost precision ------------------------
for text in [
    "read the file main.rs",
    "read the file config.toml",
    "read the file notes.txt",
    "read the file readme.md",
    "read the file cargo.toml",
    "read the file src/lib.rs",
    "read the file setup.sh",
    "read the file pyproject.toml",
    "open the file Dockerfile",
    "show me the file justfile",
    "read the first 20 lines of main.rs",
    "read the first 50 lines of config.toml",
    "read the top of readme.md",
    "show me the contents of Cargo.toml",
    "print the file setup.sh",
    "read the file .gitignore",
    "read the file src/main.rs",
    "read the file requirements.txt",
    "read the file tsconfig.json",
    "read the file index.js",
]:
    add("read_file", text, "balance: read_file lost precision to the hard negatives")

# --- remember: 78 examples --------------------------------------------------
for text in [
    "remember that my wifi is called aurora",
    "remember that the dentist is on tuesday",
    "remember that the office closes at six",
    "remember that i am allergic to shellfish",
    "remember that the train leaves at 7am",
    "remember that the repo is called aaron",
    "remember that i prefer dark mode",
    "remember that the heating is broken",
    "remember that my landlord is mrs chen",
    "remember that the deployment is on tuesday",
    "note that the boiler needs servicing",
    "keep in mind that the standup is at nine",
    "write down that i take the early train",
    "store the fact that the spare key is under the mat",
    "remember that the conference is in lisbon",
    "remember that my daughter is called nell",
    "remember that the cafe on the corner does lunch",
    "remember that the printer is on the third floor",
    "remember that i am on the night shift",
    "remember that the code review is on friday",
]:
    add("remember", text, "balance: remember had 78 examples")

# --- cancel: 87 examples ------------------------------------------------------
for text in [
    "cancel that",
    "cancel it",
    "stop that",
    "never mind",
    "actually forget it",
    "hold on leave it",
    "scratch that i changed my mind",
    "call off the search",
    "stand down",
    "no wait not that",
    "put that back",
    "undo that please",
    "cancel the file search",
    "abort the current operation",
    "disregard that request",
    "forget the whole thing",
    "i changed my mind",
    "leave it be",
    "revert that",
    "never mind that",
]:
    add("cancel", text, "balance: cancel had 87 examples")


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
    # This file was MEASURED AND REJECTED: the additions were mostly the same
    # shape as existing examples, so they added volume without coverage and
    # every measure got worse. It is written to `experimental/` so it can never
    # be merged into the training data by accident — which is what happened
    # when it was written to the curated directory and then regenerated.
    experimental = OUT / "experimental"
    experimental.mkdir(parents=True, exist_ok=True)
    path = experimental / "rejected-balance-additions.jsonl"
    with path.open("w") as handle:
        handle.write(
            "# AARON curated class-balance additions — MEASURED AND REJECTED in 0.3.\n"
            "#\n"
            "# The idea was sound: the hard negatives grew the `unknown` class from\n"
            "# 163 to 194 and the imbalance from 2.56x to 4.09x, so the other\n"
            "# thirteen classes were given more data to compensate.\n"
            "#\n"
            "# It made every measure worse:\n"
            "#\n"
            "#   locked benchmark  accuracy 0.6385  macro-F1 0.6466  false-accept 0.4000\n"
            "#   regression suite  37/42\n"
            "#   safety benchmark  refused 0.4510  side-effecting false-accept 0.2353\n"
            "#\n"
            "# against the shipped configuration's 0.7308 / 0.7411 / 0.2800, 42/42 and\n"
            "# 0.4804 / 0.1863. The additions were mostly the same *shape* as existing\n"
            "# examples — \"show me the file X\" for a dozen X — so they added volume\n"
            "# without adding coverage. More rows of the same kind is not more data.\n"
            "#\n"
            "# Kept so the experiment can be repeated.\n"
        )
        for row in unique:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    if dropped:
        print(f"dropped {dropped} duplicate row(s)")
    print(f"wrote {len(unique)} REJECTED balance rows to {path}")


if __name__ == "__main__":
    main()
