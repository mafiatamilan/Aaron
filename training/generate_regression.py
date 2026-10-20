#!/usr/bin/env python3
"""Generate the AARON regression suite.

These are the utterances that are *hard*: near-misses between two intents, and
utterances outside the schema entirely. A candidate checkpoint must get them
right before `aaron learning promote` will accept it, so a promotion cannot
quietly regress a case somebody already reported.

Usage:
    python3 training/generate_regression.py > training/evaluation/regression.jsonl
"""

import json
import sys

CASES = [
    # `system_status` vs `help`: "are you ok" is a status question, not a greeting.
    ("are you ok", "system_status"),
    ("how is the system", "system_status"),
    ("system status", "system_status"),
    ("are the systems nominal", "system_status"),
    ("any problems with the machine", "system_status"),
    ("do i have uncommitted changes", "git_status"),
    ("is the repository clean", "git_status"),
    # `remember` vs `recall`: the verb decides.
    ("remember that i drink tea", "remember"),
    ("what do you remember about tea", "recall"),
    ("do you remember my dog", "recall"),
    ("make a note that i use a standing desk", "remember"),
    ("search your memory for my address", "recall"),
    # `open_url` vs `start_application`.
    ("open https://example.com", "open_url"),
    ("open the browser", "start_application"),
    ("launch the terminal", "start_application"),
    # `search_files` vs `read_file`.
    ("find files named main.rs", "search_files"),
    ("read the file cargo.toml", "read_file"),
    ("open readme.md", "read_file"),
    ("where is the dockerfile", "search_files"),
    # `cancel` must win over everything.
    ("cancel that", "cancel"),
    ("stop that", "cancel"),
    ("nevermind", "cancel"),
    ("cancel the open url", "cancel"),
    ("abort the test run", "cancel"),
    # Not commands at all.
    ("the mitochondria is the powerhouse of the cell", "unknown"),
    ("what is the capital of australia", "unknown"),
    ("tell me a joke about penguins", "unknown"),
    ("my cat knocked over the plant again", "unknown"),
    ("the bus was late this morning", "unknown"),
    ("i finished reading that book last night", "unknown"),
    ("it has been a long week", "unknown"),
    ("how do i bake sourdough bread", "unknown"),
    # Greeting vs help.
    ("hello", "greeting"),
    ("good morning", "greeting"),
    ("what can you do", "help"),
    ("how do you work", "help"),
    # Read-file argument extraction.
    ("read the first 20 lines of dockerfile", "read_file"),
    ("show me the file main.rs", "read_file"),
    # Search argument extraction.
    ("search for *.rs in the project", "search_files"),
    ("find *.py", "search_files"),
    # Run tests.
    ("run cargo test", "run_tests"),
    ("run the test suite", "run_tests"),
]


def fnv1a64(text: str) -> int:
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def main() -> int:
    for text, intent in CASES:
        sys.stdout.write(
            json.dumps(
                {
                    "id": f"{intent}-{fnv1a64(text) & 0xFFFFFFFF:08x}",
                    "text": text,
                    "intent": intent,
                    "source": "authored",
                    "notes": "regression suite; must not regress on promotion",
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())