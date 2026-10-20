#!/usr/bin/env python3
"""Emit the curated dataset additions.

The sentences here are hand-written. The script exists to give every row a
stable id, a `source` marker and a note, so the additions are reviewable in a
diff and the review report can list them.

`source` semantics, and the distinction that matters:

  curated   the whole sentence was written by hand for this release
  template  the *sentence frame* is shared across a family and only the
            substituted part (a filename, an application name) was curated

Families of the second kind are exactly the ones that need a human eye, so
they are listed separately in the review report.
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


def make_id(intent: str, text: str) -> str:
    """Match the v1 id scheme exactly.

    v1 ids are `{intent}-{low 32 bits of fnv1a64(text)}`. Using the same
    scheme means a curated sentence that happens to repeat a v1 sentence
    produces the *same id*, so the merge deduplicates it instead of creating
    two rows with the same text and different ids.
    """
    return f"{intent}-{fnv1a64(text.encode()) & 0xFFFFFFFF:08x}"


rows: list[dict] = []


def add(intent: str, text: str, source: str = "curated", notes: str = "") -> None:
    rows.append(
        {
            "id": make_id(intent, text),
            "text": text,
            "intent": intent,
            "source": source,
            "notes": notes,
        }
    )


def family(intent: str, frames: list[str], items: list[str], notes: str) -> None:
    """Emit one row per (frame, item) pair, with the item substituted.

    Used only where the varying part is a curated noun — a filename or an
    application name — and the surrounding sentence is shared.
    """
    for frame in frames:
        for item in items:
            add(intent, frame.format(item=item), source="template", notes=notes)


# ===========================================================================
# 1. Noun-class coverage.
#
# `open`/`show`/`launch` are shared by read_file, start_application and
# open_url, so only the noun separates them. The v1 dataset had 41 file nouns
# and 37 application nouns, almost none of them extensionless, and the
# held-out set contains nouns that appear nowhere in that position. These
# families widen both noun spaces and, crucially, give the model examples of
# each noun *class* rather than each noun.
# ===========================================================================

BARE_FILENAMES = [
    "gitignore", "dockerignore", "editorconfig", "procfile", "gemfile",
    "rakefile", "vagrantfile", "caddyfile", "brewfile", "justfile",
    "makefile", "readme", "changelog", "contributing", "authors",
    "hosts", "fstab", "crontab", "profile", "bashrc", "zshrc",
    "keyboard shortcuts", "hosts file", "passwd file", "sudoers file",
    "shell history", "boot loader config", "kernel parameters",
]

FILE_FRAMES = [
    "open the {item}",
    "read the {item}",
    "show me the {item}",
    "what is in the {item}",
]

family(
    "read_file",
    FILE_FRAMES,
    BARE_FILENAMES,
    "noun-class coverage: extensionless filenames, so `open <noun>` can be "
    "told apart from `open <application>`",
)

APP_NAMES = [
    "htop", "btop", "krita", "blender", "inkscape", "vlc", "totem", "obs",
    "kdenlive", "transmission", "filezilla", "qemu", "wireshark", "postman",
    "dbeaver", "figma", "signal", "element", "okular", "zathura", "evince",
    "mpv", "audacity", "ardour", "etcher", "virtualbox", "remmina",
    "keepassxc", "bitwarden", "syncthing", "nextcloud",
    "libreoffice writer", "libreoffice calc", "libreoffice impress",
    "gnome calculator", "kde konsole", "code blocks", "jetbrains toolbox",
    "dbeaver community", "mongodb compass", "docker desktop",
]

# Two frames, not five.
#
# The first version used five frames over 46 applications, which produced 230
# rows and made `start_application` a fifth of the whole dataset. The class
# imbalance that caused was measurable: `git_status` had 49 examples and a
# recall of 0.200, while `start_application` had 252. Two frames still give
# every application name two contexts, which is what the noun-class signal
# needs, without the class swamping the rest.
APP_FRAMES = [
    "open {item}",
    "launch {item}",
]

family(
    "start_application",
    APP_FRAMES,
    APP_NAMES,
    "noun-class coverage: bare application names, so `open <noun>` can be told "
    "apart from `open <filename>`",
)

# ===========================================================================
# 2. Verb coverage.
#
# v1 had one verb for `remember`, two for `open_url` and `stop_application`,
# three for `run_tests` and `search_files`. A class with one verb is one
# phrasing away from being entirely out of distribution.
# ===========================================================================

for text in [
    "store the fact that my wifi is called aurora",
    "note that the dentist is on tuesday",
    "keep in mind that the build server is in berlin",
    "make a note that i prefer dark mode",
    "write down that the wifi is called aurora",
    "log the fact that the office closes at six",
    "put in your memory that the spare key is under the mat",
    "save this for later the train leaves at half eight",
    "i want you to remember that the boiler is broken",
    "add to your memory that i am allergic to penicillin",
]:
    add("remember", text, notes="verb coverage: `remember` had exactly one verb in v1")

for text in [
    "what have you stored about my wifi",
    "look up my wifi name in your memory",
    "dig through your memory for my dentist",
    "pull up what you know about my commute",
    "remind me what i said about berlin",
    "check your memory for my bank details",
    "go through your notes about my medication",
    "fetch from memory what i said about rust",
    "what is in your memory about my flights",
    "search your memory for my passport number",
]:
    add("recall", text, notes="verb coverage: `recall` was dominated by `remember` and `what`")

for text in [
    "browse to https://example.com",
    "bring up https://example.org",
    "navigate to https://docs.rs/candle",
    "go to https://wikipedia.org",
    "pull up https://man.archlinux.org",
    "load https://crates.io",
    "take me to https://example.net",
    "visit https://example.com/docs",
    "head to https://github.com/aaron-assistant/aaron",
    "show me the page at https://example.io/guide",
]:
    add("open_url", text, notes="verb coverage: `open_url` had only `open` and `show`")

for text in [
    "close spotify",
    "kill the terminal",
    "terminate the browser",
    "end the music player",
    "shut down firefox",
    "quit gimp",
    "stop the drawing app",
    "halt the terminal emulator",
    "get rid of the code editor",
    "wind up the media player",
]:
    add("stop_application", text, notes="verb coverage: `stop_application` had only `stop` and `kill`")

for text in [
    "execute the cargo test suite",
    "run the python tests",
    "kick off the unit tests",
    "fire up the test suite",
    "go ahead and run the integration tests",
    "start the test runner",
    "run the tests in release mode",
    "trigger the test suite",
    "run cargo test please",
    "run the test suite for the brain crate",
]:
    add("run_tests", text, notes="verb coverage: `run_tests` had only `run` and `start`")

for text in [
    "where is the file called main.rs",
    "hunt for files named config.toml",
    "look for readme files",
    "track down the dockerfile",
    "dig out any toml files",
    "scan for py files",
    "sift through the project for rust files",
    "list the files matching *.json",
    "show me every file ending in .md",
    "which files are called .gitignore",
]:
    add("search_files", text, notes="verb coverage: `search_files` had only `find`, `search`, `locate`")

for text in [
    "how is the machine doing",
    "give me the vitals",
    "report the hardware health",
    "what does the load average look like",
    "tell me about resource usage",
    "is everything healthy",
    "give me a system overview",
    "how much swap is in use",
    "any unusual network activity",
    "what is the disk usage",
]:
    add("system_status", text, notes="verb coverage: `system_status` was dominated by `check`")

for text in [
    "which branch is checked out",
    "is anything staged",
    "what would git commit",
    "give me the diff summary",
    "how many files are untracked",
    "what does the working tree look like",
    "any conflicts in the repo",
    "show me the last commit",
    "which files changed since the last commit",
    "is the repository in a clean state",
]:
    add("git_status", text, notes="verb coverage: `git_status` was dominated by `check` and `show`")

# ===========================================================================
# 3. Minimal pairs.
#
# Same verb, different object, different intent. These are the pairs a
# bag-of-features model gets wrong, because the discriminative token is one
# word among many.
# ===========================================================================

for text, intent, note in [
    ("stop the music player", "stop_application", "minimal pair: `stop` + app vs `stop` alone"),
    ("stop", "cancel", "minimal pair: `stop` alone is cancel, `stop <app>` is stop_application"),
    ("stop the file search", "cancel", "minimal pair: `stop` + a task is cancel"),
    ("cancel the download", "cancel", "minimal pair: `cancel` + a task is cancel"),
    ("cancel spotify", "stop_application", "minimal pair: `cancel` + app is stop_application"),
    ("run spotify", "start_application", "minimal pair: `run` + app vs `run` + tests"),
    ("run the tests", "run_tests", "minimal pair: `run` + app vs `run` + tests"),
    ("search for the file", "search_files", "minimal pair: `search` + local vs `search` + web"),
    ("search the web", "unknown", "minimal pair: there is no web-search intent"),
    ("check status", "system_status", "minimal pair: `check` + machine vs `check` + git"),
    ("check git status", "git_status", "minimal pair: `check` + machine vs `check` + git"),
    ("check the repository", "git_status", "minimal pair: `check` + machine vs `check` + repo"),
    ("show me the file main.rs", "read_file", "minimal pair: `show` + file vs `show` + git"),
    ("show me the git log", "git_status", "minimal pair: `show` + file vs `show` + git"),
    ("show me the resources", "system_status", "minimal pair: `show` + file vs `show` + machine"),
    ("open the file main.rs", "read_file", "minimal pair: `open` + file vs `open` + app"),
    ("open the terminal", "start_application", "minimal pair: `open` + file vs `open` + app"),
    ("open the website example.com", "open_url", "minimal pair: `open` + url vs `open` + file"),
    ("open config.toml", "read_file", "minimal pair: `open` + url vs `open` + file"),
    ("what do you remember about tea", "recall", "minimal pair: `remember` as noun vs verb"),
    ("remember that i drink tea", "remember", "minimal pair: `remember` as noun vs verb"),
    ("forget my last request", "cancel", "minimal pair: `forget` + request is cancel"),
    ("forget that i drink tea", "unknown", "minimal pair: `forget` + fact has no intent"),
    ("give me a status report", "system_status", "minimal pair: `report` + machine"),
    ("give me the git report", "git_status", "minimal pair: `report` + machine vs + git"),
    ("list the files", "search_files", "minimal pair: `list` + files"),
    ("list your commands", "help", "minimal pair: `list` + files vs `list` + capabilities"),
    ("what can you do", "help", "minimal pair: `what can` + capabilities"),
    ("what can you find", "unknown", "minimal pair: `what can` + capabilities"),
]:
    add(intent, text, notes=note)

# ===========================================================================
# 4. Hard negatives.
#
# Out-of-schema requests that *look* like commands. A false accept is the
# failure that matters: it is what runs the wrong tool. Every one of these
# must be `unknown`.
# ===========================================================================

for text in [
    "open the pod bay doors",
    "delete all my files",
    "send an email to bob",
    "increase the volume",
    "close the window",
    "what time is it",
    "set an alarm for seven in the morning",
    "add a contact to my phone",
    "put the kettle on",
    "turn the lights off",
    "lock the door",
    "print the document",
    "empty the recycle bin",
    "rename the folder",
    "move the file to the desktop",
    "zip up the project",
    "commit the changes for me",
    "push to the remote",
    "create a new branch",
    "install the package",
    "update the dependencies",
    "deploy to production",
    "restart the service",
    "change my password",
    "transfer the money",
    "book a flight",
    "order a pizza",
    "call my mother",
    "water the plants",
    "clean the room",
    "do the dishes",
    "buy milk",
    "walk the dog",
    "post a message to the channel",
    "create a branch called feature",
    "rebase onto main",
    "squash my commits",
    "sign the commit",
    "tag the release",
    "generate the documentation",
    "bump the version number",
    "publish the crate",
    "format the code",
    "lint the project",
    "profile the binary",
    "benchmark the model",
    "export the database",
    "import the bookmarks",
    "take a screenshot",
    "record the screen",
    "play a video",
    "pause the music",
    "skip the track",
]:
    add("unknown", text, notes="hard negative: looks like a command, no such intent")

# Questions about the world that begin like a capability question.
for text in [
    "how do i change the oil in a car",
    "how do i tie a bowline knot",
    "how do i prune a rose bush",
    "how do i file my taxes",
    "how do i make sourdough starter",
    "how do i train for a marathon",
    "how do i replace a bicycle brake",
    "how do i set up a fish tank",
    "how do i unclog a sink",
    "how do i season a cast iron pan",
    "what is the capital of peru",
    "what is the speed of sound",
    "why is the sky blue",
    "why do leaves change colour",
    "when did the titanic sink",
    "who wrote the origin of species",
    "how many bones are in the human hand",
    "what is the difference between a virus and a bacterium",
    "should i learn rust or go",
    "is it safe to eat rice that has been left out",
    "what is a good beginner programming language",
]:
    add("unknown", text, notes="hard negative: `how do i` about the world, not about AARON")

# Capability questions that really are about AARON.
for text in [
    "what can you do",
    "what commands do you understand",
    "how do i use you",
    "what are you able to do",
    "list your capabilities",
    "what do you know how to do",
    "help me understand what you can do",
    "what can you help me with",
    "tell me what you are able to do",
]:
    add("help", text, notes="capability question, positive class")

# ===========================================================================
# 5. Conversational and spoken-style variants.
#
# The voice path produces these, and v1 contains almost no filler, no
# politeness, no ASR-style misspellings and no self-corrections.
# ===========================================================================

for text, intent in [
    ("um can you check the system status please", "system_status"),
    ("pls run the tests", "run_tests"),
    ("chk system status", "system_status"),
    ("hey aaron whats up", "greeting"),
    ("morning", "greeting"),
    ("hiya", "greeting"),
    ("good afternoon to you", "greeting"),
    ("yo status", "system_status"),
    ("aaron run the test suite", "run_tests"),
    ("ok show me git status", "git_status"),
    ("hmm open the website https://example.com", "open_url"),
    ("could you possibly show git status", "git_status"),
    ("would you mind checking the system", "system_status"),
    ("hey what can you do", "help"),
    ("so whats the git status", "git_status"),
    ("can u find files named main.rs", "search_files"),
    ("pls remember that the wifi is called aurora", "remember"),
    ("right then run the tests", "run_tests"),
    ("hang on what was the git status", "git_status"),
    ("sorry i meant the system status", "system_status"),
]:
    add(intent, text, notes="conversational / spoken style")

# Self-corrections and hesitation, which the voice path produces constantly.
for text, intent in [
    ("actually never mind", "cancel"),
    ("hold on forget it", "cancel"),
    ("lets not do that", "cancel"),
    ("scratch that i changed my mind", "cancel"),
    ("no wait leave it", "cancel"),
    ("i mean no dont", "cancel"),
    ("on second thought no", "cancel"),
    ("actually keep going", "unknown"),
]:
    add(intent, text, notes="self-correction / hesitation")

# ===========================================================================
# 6. Cancel coverage.
#
# v1's cancel class was 24 near-duplicate filler variants, and its recall was
# 0.200 on the held-out set. These are semantically distinct ways to withdraw
# a request, plus the pairs that keep cancel apart from stop_application.
# ===========================================================================

for text in [
    "cancel the file search",
    "cancel the test run",
    "cancel the download",
    "abort the request",
    "call off the search",
    "stop the current task",
    "halt what you are doing",
    "stop everything right now",
    "stand down",
    "leave it be",
    "forget the whole thing",
    "undo the last thing",
    "take that back",
    "never mind i am wrong",
    "that is enough",
    "enough of that",
    "forget what i just said",
    "disregard that request",
    "i changed my mind",
    "no more of that",
    "stop the current operation",
    "cancel my request",
    "pull the plug",
    "abort the mission",
    "lets start over",
]:
    add("cancel", text, notes="cancel coverage: semantically distinct withdrawal phrasings")

# ===========================================================================
# 7. Entity coverage for the intents that take entities.
# ===========================================================================

for text in [
    "remember that my wifi network is called aurora",
    "remember that the code review is on friday",
    "remember that my phone number ends in 4821",
    "remember that the spare key is in the drawer",
    "remember that i take the train at 7am",
    "remember that the project is called aaron",
    "remember that the password hint is my first pet",
    "remember that the deployment is on tuesday",
]:
    add("remember", text, notes="entity coverage: `remember` had almost no entity examples")

for text in [
    "what do you remember about my wifi network",
    "what do you remember about the code review",
    "what do you remember about my train time",
    "look up what you know about my spare key",
    "search your memory for the project name",
    "what did i tell you about the deployment",
]:
    add("recall", text, notes="entity coverage: `recall` entities were mostly `my <thing>`")

for text in [
    "run the cargo test suite",
    "run the python test suite",
    "run the frontend tests",
    "run the tests for the security crate",
    "run the integration test suite",
    "run the workspace tests",
]:
    add("run_tests", text, notes="entity coverage: named test suites")

for text in [
    "open https://example.com",
    "open https://example.com/docs",
    "open https://www.rust-lang.org",
    "open https://github.com/aaron-assistant/aaron",
    "open https://docs.rs/candle/latest",
    "open https://man.archlinux.org/man/pacman",
    "open https://news.ycombinator.com",
    "open https://crates.io/crates/candle-core",
    "open https://example.com:8080/health",
    "open https://en.wikipedia.org/wiki/Rust_(programming_language)",
]:
    add("open_url", text, notes="entity coverage: URLs in varied shapes")

# ===========================================================================
# 8. `open the <noun>` disambiguation pairs.
#
# The noun is the only cue, so the pairs have to be explicit and balanced:
# for every filename there is an application, and vice versa.
# ===========================================================================

for text, intent in [
    ("open the editor", "start_application"),
    ("open the editorconfig", "read_file"),
    ("open the notes app", "start_application"),
    ("open the notes file", "read_file"),
    ("open the browser", "start_application"),
    ("open the browser source", "read_file"),
    ("open the terminal", "start_application"),
    ("open the terminal config", "read_file"),
    ("open the spreadsheet", "start_application"),
    ("open the spreadsheet formula", "read_file"),
    ("open the pdf viewer", "start_application"),
    ("open the pdf metadata", "read_file"),
]:
    add(intent, text, notes="`open the <noun>` disambiguation pair")

# ===========================================================================
# 9. Greeting coverage.
#
# v1's greeting class was 16 near-duplicate filler variants, and its recall
# was 0.800. These are distinct greeting shapes.
# ===========================================================================

for text in [
    "hello there",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "greetings",
    "morning",
    "howdy",
    "hiya",
    "hello aaron",
    "hey there aaron",
    "is anyone there",
    "good day to you",
    "top of the morning to you",
    "hello anyone around",
    "anyone home",
    "wake up aaron",
    "aaron you there",
    "knock knock",
]:
    add("greeting", text, notes="greeting coverage: distinct greeting shapes")


# ===========================================================================
# 10. Contrastive frames.
#
# The error analysis found nine `intent_overlap` mistakes, and they all share
# a shape: a *frame* that several intents use, where only the object decides.
# "show me ..." is used by read_file, git_status, system_status, search_files
# and help. "stop ..." is used by stop_application and cancel. "check ..." is
# used by system_status, git_status and run_tests.
#
# The fix is not more examples of each intent — it is examples of the *same
# frame* side by side, so the model learns that the frame carries no
# information and the object carries all of it.
# ===========================================================================

CONTRASTIVE: list[tuple[str, str, str]] = [
    # --- "show me" -----------------------------------------------------
    ("show me the file main.rs", "read_file", "frame `show me` used by five intents"),
    ("show me the git log", "git_status", "frame `show me` used by five intents"),
    ("show me the resources", "system_status", "frame `show me` used by five intents"),
    ("show me every file ending in .md", "search_files", "frame `show me` used by five intents"),
    ("show me the command list", "help", "frame `show me` used by five intents"),
    ("show me the wifi name", "recall", "frame `show me` used by five intents"),
    # --- "what does" ---------------------------------------------------
    ("what does git think", "git_status", "frame `what does` is not file-reading"),
    ("what does the load average look like", "system_status", "frame `what does` is not file-reading"),
    ("what does the cpu usage look like", "system_status", "frame `what does` is not file-reading"),
    ("what does the help command say", "help", "frame `what does` is not file-reading"),
    # --- "stop" --------------------------------------------------------
    ("stop nautilus", "stop_application", "frame `stop <app>` vs `stop <task>`"),
    ("stop the music player", "stop_application", "frame `stop <app>` vs `stop <task>`"),
    ("stop the file search", "cancel", "frame `stop <app>` vs `stop <task>`"),
    ("stop the download", "cancel", "frame `stop <app>` vs `stop <task>`"),
    # --- "locate" ------------------------------------------------------
    ("locate the tests directory contents", "search_files", "frame `locate` is not running tests"),
    ("locate the source directory contents", "search_files", "frame `locate` is not running tests"),
    ("locate the missing file", "search_files", "frame `locate` is not running tests"),
    ("locate the running process", "unknown", "frame `locate` is not running tests"),
    # --- "check" -------------------------------------------------------
    ("check performance", "system_status", "frame `check` used by three intents"),
    ("check the test results", "run_tests", "frame `check` used by three intents"),
    ("check the repository", "git_status", "frame `check` used by three intents"),
    ("check the disk space", "system_status", "frame `check` used by three intents"),
    # --- "how is / how do i" --------------------------------------------
    ("how is the machine doing", "system_status", "frame `how is` is not a world question"),
    ("how is the repository doing", "git_status", "frame `how is` is not a world question"),
    ("how do i use you", "help", "frame `how do i` is not always out of schema"),
    ("how do i bake bread", "unknown", "frame `how do i` is not always out of schema"),
    # --- "remember" ----------------------------------------------------
    ("remember that my name is david", "remember", "frame `remember` as verb vs noun"),
    ("remember what i told you about the wifi", "recall", "frame `remember` as verb vs noun"),
    ("what do you remember about my dentist", "recall", "frame `remember` as verb vs noun"),
    ("remember this for me", "remember", "frame `remember` as verb vs noun"),
    # --- "open" --------------------------------------------------------
    ("open the terminal", "start_application", "frame `open` used by four intents"),
    ("open config.toml", "read_file", "frame `open` used by four intents"),
    ("open https://example.com", "open_url", "frame `open` used by four intents"),
    ("open the pod bay doors", "unknown", "frame `open` used by four intents"),
    # --- "halt" / "call off" -------------------------------------------
    ("halt what you are doing", "cancel", "frame `halt` is not a capability question"),
    ("call off the search", "cancel", "frame `call off` is not a capability question"),
    # --- one-word commands ----------------------------------------------
    ("evening", "greeting", "one-word greeting"),
    ("afternoon", "greeting", "one-word greeting"),
    ("git", "git_status", "one-word command"),
    ("status", "system_status", "one-word command"),
    ("tests", "run_tests", "one-word command"),
    ("browser", "start_application", "one-word command"),
    ("greetings", "greeting", "one-word greeting"),
    # --- "take that back" and friends -----------------------------------
    ("put it back", "cancel", "withdrawal phrasing"),
    ("undo that please", "cancel", "withdrawal phrasing"),
    ("revert that", "cancel", "withdrawal phrasing"),
    # --- "save this for later" ------------------------------------------
    ("save this for later the dentist is at noon", "remember", "remembering phrasing"),
    ("keep this in mind the boiler is broken", "remember", "remembering phrasing"),
    # --- "track down" ---------------------------------------------------
    ("track down the containerfile", "search_files", "search phrasing"),
    ("track down the missing symbol", "search_files", "search phrasing"),
    # --- "go through your notes" ----------------------------------------
    ("go through your notes about my prescription", "recall", "recall phrasing"),
    ("go through your notes about my landlord", "recall", "recall phrasing"),
    # --- "which files changed" ------------------------------------------
    ("which files have changed since the last commit", "git_status", "git phrasing"),
    ("which branch am i on right now", "git_status", "git phrasing"),
    # --- "are we running out of" ----------------------------------------
    ("are we low on memory", "system_status", "interrogative system phrasing"),
    ("is the machine running out of disk space", "system_status", "interrogative system phrasing"),
    ("is the server overheating", "system_status", "interrogative system phrasing"),
    # --- "top of the morning" -------------------------------------------
    ("top of the morning to you", "greeting", "greeting phrasing"),
    ("good day to you", "greeting", "greeting phrasing"),
    # --- "do i have" ----------------------------------------------------
    ("do i have any staged changes", "git_status", "interrogative git phrasing"),
    ("do i need to commit anything", "git_status", "interrogative git phrasing"),
    # --- "fire up" ------------------------------------------------------
    ("fire up the torrent client", "start_application", "start phrasing"),
    ("fire up the screenshot tool", "start_application", "start phrasing"),
]

for text, intent, note in CONTRASTIVE:
    add(intent, text, notes=note)

# ===========================================================================
# 11. Class balance.
#
# The v2 dataset measured a 5.14x imbalance, with `git_status` at 49 examples
# and a recall of 0.200 while `start_application` had 252. The small classes
# are the ones whose metrics were least trustworthy, so they are the ones
# that need more data. These additions are ordinary commands, not exotic
# ones: the point is coverage, not novelty.
# ===========================================================================

for text in [
    "what is staged right now",
    "which files are staged for commit",
    "how many commits since the last tag",
    "give me the recent commit list",
    "is there anything to push",
    "what is the current head commit",
    "tell me the branch and its upstream",
    "do the tests pass on this branch",
    "what changed in the last commit",
    "list the modified files",
    "how many lines are staged",
    "any new files git has not seen",
    "what does the index contain",
    "is the index dirty",
    "show me what git would commit",
]:
    add("git_status", text, notes="class balance: git_status had 49 examples and recall 0.200")

for text in [
    "what can this thing do",
    "which requests do you handle",
    "what should i ask you",
    "print your cheat sheet",
    "enumerate your commands",
    "what is in the quick reference",
    "how do i speak to you",
    "what are you for",
    "give me a list of what you can do",
    "which verbs do you understand",
]:
    add("help", text, notes="class balance: help had 51 examples")

for text in [
    "how much ram is available",
    "how many cores does this machine have",
    "give me the system overview",
    "what is the cpu temperature",
    "how much swap is in use",
    "is everything running smoothly",
    "how is the desktop doing",
    "tell me the memory usage",
    "any unusual processes running",
    "what does the uptime look like",
    "how much space is left on the disk",
    "give me the resource snapshot",
    "is the gpu busy right now",
    "how is the load average",
    "check the health of the machine",
]:
    add("system_status", text, notes="class balance: system_status had 61 examples")

for text in [
    "which files are called composer.json",
    "hunt for files named helmfile.yaml",
    "scan the tree for shell scripts",
    "track down the containerfile",
    "dig out any config files",
    "which files end in .lock",
    "list the files in the assets directory",
    "show me everything in the templates folder",
    "find the files that mention the word timeout",
    "search for large files in the project",
    "which folders contain markdown",
    "where are the yaml files",
    "show me the files that were changed yesterday",
    "look for the file that has the schema",
    "which scripts are executable",
]:
    add("search_files", text, notes="class balance: search_files had 64 examples")

for text in [
    "run the clippy check",
    "run the doc tests",
    "run the benchmark suite",
    "start the test suite for the brain crate",
    "kick off the regression tests",
    "run the linter tests",
    "execute the whole test suite",
    "run the tests in release mode",
    "fire up the test runner please",
    "go ahead and run the unit tests",
    "run the integration test suite",
    "run cargo test for me",
]:
    add("run_tests", text, notes="class balance: run_tests had 61 examples")

for text in [
    "browse to https://example.com",
    "navigate to https://docs.rs/candle",
    "bring up https://man.archlinux.org",
    "go to https://crates.io",
    "pull up https://example.org/guide",
    "load https://en.wikipedia.org",
    "visit https://example.net/docs",
    "take me to https://github.com/aaron-assistant/aaron",
]:
    add("open_url", text, notes="class balance: open_url had 57 examples")

for text in [
    "close the torrent client",
    "kill krita",
    "terminate the debugger",
    "end the file manager",
    "shut down the email client",
    "quit the video editor",
    "halt the system monitor",
    "get rid of the music player",
    "stop nautilus",
    "close the browser",
    "kill the docker daemon",
    "end the background job",
]:
    add("stop_application", text, notes="class balance: stop_application had 53 examples")

for text in [
    "never mind",
    "actually forget it",
    "hold on leave it",
    "scratch that i changed my mind",
    "call off the search",
    "stand down",
    "no wait not that",
    "put that back",
    "undo that please",
    "revert that",
    "cancel the file search",
    "abort the request",
    "stop the current task",
    "halt what you are doing",
    "disregard that request",
]:
    add("cancel", text, notes="class balance: cancel had 80 examples but 24 near-duplicate pairs")

for text in [
    "what do you remember about my wifi",
    "look up what you know about my dentist",
    "search your memory for my train time",
    "what did i tell you about the deployment",
    "remind me what i said about berlin",
    "pull up what you know about my medication",
    "check your memory for my bank details",
    "dig through your memory for my passport",
    "go through your notes about my prescription",
    "go through your notes about my landlord",
]:
    add("recall", text, notes="class balance: recall had 68 examples and 19 near-duplicate pairs")

for text in [
    "remember that the wifi is called aurora",
    "remember that my dentist is on tuesday",
    "note that the office closes at six",
    "remember that i am allergic to shellfish",
    "store the fact that the train leaves at 7am",
    "remember that the repo is named aaron",
    "write down that i favour dark mode",
    "remember that the heating is broken",
    "keep in mind that the train departs at 7am",
    "note it down that i prefer dark mode",
]:
    add("remember", text, notes="class balance: remember had 69 examples and one verb")

for text in [
    "hiya",
    "good evening",
    "hey aaron",
    "evening",
    "afternoon",
    "morning to you",
    "hello hello",
    "howdy partner",
    "good night to you",
    "hello aaron there",
    "greetings and salutations",
    "anyone around",
]:
    add("greeting", text, notes="class balance: greeting had 56 examples and 16 near-duplicate pairs")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    # The same sentence may be reached from two different themes. That is a
    # duplicate, not a collision: it is dropped, and reported. A *different*
    # sentence hashing to the same id would be a real bug.
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
    if dropped:
        print(f"dropped {dropped} duplicate row(s)")
    path = OUT / "curated-v2.jsonl"
    with path.open("w") as handle:
        handle.write("# AARON curated dataset additions, release 0.2\n")
        for row in unique:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(unique)} curated rows to {path}")


if __name__ == "__main__":
    main()
