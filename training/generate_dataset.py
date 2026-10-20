#!/usr/bin/env python3
"""Generate the AARON v1 intent dataset.

This script is *authored input*, not a model: every line below is a sentence a
person might plausibly say or type to an assistant. It exists so the dataset is
reproducible and reviewable in a diff rather than being an opaque binary blob.

Usage:
    python3 training/generate_dataset.py > training/dataset/current.jsonl

Determinism: the output depends only on this file. Re-running it produces a
byte-identical dataset, and `aaron brain train` is seeded separately, so the
whole training pipeline is reproducible.

Design rules the generated data follows:
  * every intent has a comparable number of examples;
  * phrasings vary in formality, length, punctuation and filler words;
  * each intent includes near-miss negatives, so the model has to learn the
    distinction rather than a keyword;
  * text is lowercase-normalised only by the model's own feature extractor, so
    the raw text keeps realistic capitalisation and punctuation.
"""

import json
import sys

# Each entry is (intent, [utterances]). Roughly 45 per intent.
DATA = {
    "greeting": [
        "hello", "hi there", "hey", "good morning", "good evening",
        "hey aaron", "hi aaron", "hello aaron", "morning", "good afternoon",
        "greetings", "howdy", "hi", "yo", "hello there aaron",
        "hey, are you there", "good to see you", "hi, anyone home",
        "just saying hello", "hello again", "hi again", "nice to meet you aaron",
        "hello aaron how are you", "hey aaron what's up", "hi aaron you around",
        "morning aaron", "evening", "hello there", "hiya", "sup",
        "hello assistant", "hi assistant", "good day", "greetings aaron",
        "hello there how are things", "hey are you awake", "hi are you listening",
        "knock knock", "hello hello", "hi hi", "good morning aaron",
        "hey aaron good morning", "hi aaron good evening",
        "just checking in", "hello again aaron", "hi again aaron",
    ],
    "help": [
        "help", "what can you do", "what are your capabilities",
        "how do you work", "list your commands", "what commands do you understand",
        "show me the help", "what are you able to do", "give me the command list",
        "i forgot what you can do", "remind me what you support",
        "what do you know how to do", "how do i use you", "usage",
        "what are my options", "which commands exist", "what does aaron understand",
        "help please", "can you help me", "i need help",
        "show help", "print the manual", "what should i say",
        "list everything you can do", "what are the available intents",
        "how do i talk to you", "give me an example command",
        "what's the syntax", "what syntax do you use",
        "show the quick reference", "quick reference please",
        "i don't know what to ask you", "what are you good at",
        "explain what you do", "tell me about yourself",
        "help me understand you", "what does aaron do", "how can i use aaron",
        "list your features", "features list", "show capabilities",
        "help with commands", "command overview", "overview of commands",
        "what do the commands look like", "examples please",
    ],
    "system_status": [
        "check system status", "system status", "how is the system",
        "check the machine", "machine status", "how are things running",
        "are you ok", "system health", "check the computer",
        "how much memory is left", "what is your cpu usage", "check resource usage",
        "report the system status", "give me a status report",
        "how is the pc doing", "check performance", "show me system stats",
        "is the system healthy", "status check", "run a health check",
        "how much disk space is left", "check the gpu", "is the gpu busy",
        "how many cores do you have", "what temperature is the cpu",
        "check on the machine", "give me the hardware summary",
        "what is the memory usage", "how is the load average",
        "check system health", "system check please", "quick status check",
        "are the systems nominal", "any problems with the machine",
        "show me the resources", "how are the resources doing",
        "check the system resources", "give me the vitals",
        "host status", "machine health", "what's the system doing",
        "are we running out of memory", "how is the cpu doing",
        "check cpu and memory", "status of this computer", "give me diagnostics",
    ],
    "remember": [
        "remember that i drink tea", "remember that my name is david",
        "remember this that i prefer dark mode", "remember that the wifi password is hunter2",
        "remember that i work in berlin", "remember that i use neovim",
        "remember that my birthday is in june", "remember that i am allergic to nuts",
        "note that i take the bus to work", "remember that the wifi is called office5g",
        "remember that i hate monday meetings", "make a note that i use a standing desk",
        "remember my api key is abc123", "remember that i cycle to work",
        "remember that my dog is called rex", "remember that i am learning rust",
        "remember that the spare key is under the mat", "remember that i take vitamins at eight",
        "remember that i do not eat mushrooms", "remember that my daughter starts school in september",
        "keep in mind that i am vegetarian", "remember that i live at number 12",
        "remember that i have a meeting every monday at nine",
        "remember that i switched to linux last year",
        "remember that i do not like cilantro", "remember that my commute is 40 minutes",
        "store that i prefer tabs over spaces", "remember that the printer is on the second floor",
        "remember that i take the train on fridays", "remember that my email ends in example.com",
        "remember that i am allergic to penicillin", "remember that i read before bed",
        "remember that my project is called aaron", "remember that i owe rent on the first",
        "remember that i run every morning", "remember that i prefer dark roast coffee",
        "remember that the office closes at six", "remember that i am on the committee",
        "remember that my flight is at seven in the morning", "remember that i have a check up in may",
        "remember that i am learning spanish", "remember that i use a standing desk mat",
        "remember that my phone number ends in 4821", "remember that i do not drive",
        "remember that i like window seats", "remember that the code review is on thursday",
        "remember that i use i3 window manager", "remember that i keep backups on sunday",
        "remember that my subscription renews in august", "remember that i am allergic to latex",
        "remember that i prefer to meet in the morning",
    ],
    "recall": [
        "what do you remember about tea", "recall my wifi password",
        "what do you know about my birthday", "do you remember my dog",
        "search your memory for my address", "what did i say about neovim",
        "look up in memory my email", "from memory what is my name",
        "what do you remember about coffee", "recall the wifi name",
        "what do you know about my job", "do you remember my commute",
        "what have i told you about my allergies", "search memory for the printer",
        "what do you recall about my birthday", "look up what you know about my car",
        "what do you remember about my commute", "do you know anything about my family",
        "search your memory for my password", "recall my name",
        "what do you know about berlin", "do you remember the office hours",
        "what did i tell you my dog is called", "search memory for dark mode",
        "what do you recall about my meetings", "look up my bank details in memory",
        "what do you know about my training schedule", "do you remember my standing desk",
        "search your memory for my flight", "what do you remember about my medication",
        "recall the things i told you about linux", "what do you know about my nickname",
        "do you remember what i do for work", "search memory for subscription",
        "what have i stored about my car", "do you remember my birthday",
        "look up what i said about rust", "what do you recall about my weekend",
        "search your memory for the account number", "what do you know about my hobbies",
        "do you remember my morning routine", "recall the wifi details",
        "what do you remember about my allergy", "do you know my name",
        "search your memory for my project", "what did i tell you about my rent",
        "do you remember anything about my diet", "look up my preferences in memory",
        "what do you recall about my phone", "search memory for team meeting",
        "what do you know about my passport", "do you remember my language",
    ],
    "search_files": [
        "find files named main.rs", "find *.py", "search for files called readme.md",
        "locate the file config.toml", "where is cargo.lock",
        "find files matching *.json", "search for *.rs in the project",
        "find documents named invoice", "look for a file called notes.txt",
        "find all *.log files", "search files for setup.sh",
        "where is the main file", "find the dockerfile", "locate *.yaml files",
        "find files named test.rs", "search for *.toml in my project",
        "find files ending in .md", "locate a file called makefile",
        "search for files called package.json", "find *.c files",
        "where are the css files", "find files named readme",
        "search for *.html files", "find the LICENSE file",
        "locate files named tsconfig.json", "search files for docker-compose.yml",
        "find *.sh scripts", "where is the changelog", "find files called .gitignore",
        "search for *.sql files", "find the settings file",
        "locate *.png images", "find files named requirements.txt",
        "search for *.rs in my code", "find the schema file",
        "where is the install script", "find files called pyproject.toml",
        "search for *.md documentation", "locate the tests directory contents",
        "find files named .env.example", "search files for build.rs",
        "find *.ini files", "where is the main.rs file",
        "search for files named justfile", "find the vendored dependencies",
        "locate *.rs files in the src directory", "find files called rust-toolchain.toml",
        "search for *.tex files", "find the license header",
        "where is my resume", "find files named favicon.ico",
    ],
    "read_file": [
        "read the file cargo.toml", "open readme.md", "read config.toml",
        "show me the file main.rs", "read the first 20 lines of dockerfile",
        "open the file license", "cat changelog.md", "read setup.sh",
        "read the file .gitignore", "open notes.txt", "read the first 50 lines of main.rs",
        "show the contents of pyproject.toml", "read src/lib.rs",
        "open the makefile", "read the file rust-toolchain.toml", "cat the license file",
        "read the first 100 lines of readme.md", "open the gitignore",
        "read config/default.toml", "show me package.json", "read the file justfile",
        "open the test file", "read src/main.rs", "cat .env.example",
        "read the first 10 lines of the changelog", "open docker-compose.yml",
        "show the contents of build.rs", "read the requirements file",
        "open tsconfig.json", "read the file main.rs please", "show me the dockerfile",
        "read the top of readme.md", "open the file notes", "read the settings file",
        "cat the top of main.rs", "read the file license file", "open the changelog",
        "show me the first lines of config.toml", "read the file pyproject.toml",
        "read the source file", "open the file index.js", "read app.py",
        "read the first 25 lines of setup.sh", "open the readme",
        "show me the file src/lib.rs", "read the file dockerfile",
    ],
    "git_status": [
        "git status", "check git status", "show the git status",
        "what is the git status", "any uncommitted changes",
        "do i have uncommitted changes", "check the repository status",
        "git status of this repo", "show me what changed in git",
        "what does git think", "are there staged changes", "check for dirty files",
        "show git diff status", "git state", "what is in the working tree",
        "any modified files in git", "check the branch status",
        "show the git summary", "how many files changed",
        "is the repository clean", "show me untracked files",
        "git status please", "check if anything needs committing",
        "what is staged", "show the repo state", "check version control status",
        "do i need to commit anything", "what has changed since the last commit",
        "show the current branch", "what branch am i on",
        "give me a git summary", "check the tree state",
    ],
    "run_tests": [
        "run the tests", "run cargo test", "run the test suite",
        "run all the tests", "run tests please", "start the test suite",
        "run the unit tests", "run pytest", "run npm test",
        "run the test command", "execute the tests", "run tests now",
        "run the cargo tests", "run the whole suite", "tests please",
        "run the integration tests", "run the tests for this crate",
        "can you run the tests", "run the test suite and tell me",
        "run cargo test quiet", "run the linter and the tests",
        "run the tests in debug mode", "run the quick tests",
        "run tests", "test everything", "run the test",
        "run the tests and show me the failures", "start the test run",
        "run the test suite for aaron", "run tests with output",
        "run the doctests", "run clippy and the tests",
        "run the tests again", "run the tests once more",
        "please run the test suite", "run the benchmark tests",
        "run the regression tests", "run cargo nextest",
        "run the tests and summarise", "run the unit tests only",
        "run the tests and time them", "run the smoke tests",
    ],
    "open_url": [
        "open https://example.com", "open the website https://rust-lang.org",
        "browse to https://news.ycombinator.com", "go to the site https://github.com",
        "open https://docs.rs/candle/latest", "visit https://archlinux.org",
        "open the url https://example.org/docs", "show me https://wikipedia.org",
        "open https://this-week-in-rust.org", "browse to https://crates.io",
        "open https://localhost:3000", "go to https://arxiv.org/abs/1706.03762",
        "open https://www.rust-lang.org/learn", "visit https://doc.rust-lang.org",
        "open https://github.com/huggingface/candle", "open the website example.com",
        "go to https://en.wikipedia.org/wiki/Rust", "open https://jsonplaceholder.typicode.com",
        "browse https://developer.mozilla.org", "open https://lib.rs",
        "open https://github.com/aaron-assistant/aaron", "show me https://man.archlinux.org",
        "go to https://wiki.archlinux.org", "open https://www.reddit.com/r/rust",
        "browse to https://news.ycombinator.com/newest", "visit https://status.github.com",
        "open https://example.com/a?b=c", "open https://localhost:8080/health",
        "go to https://www.kernel.org", "open https://code.visualstudio.com",
        "show me https://docs.python.org/3", "open https://stackoverflow.com/questions/1",
        "browse to https://www.phoronix.com", "open https://sourceforge.net",
        "visit https://www.rust-lang.org", "open https://crates.io/crates/candle-core",
        "go to https://archlinux.org/packages/extra/x86_64/whisper.cpp",
        "open https://example.com/", "open the website https://aaron.local/docs",
        "browse https://www.gnu.org", "show https://example.net/index.html",
    ],
    "start_application": [
        "start firefox", "launch the browser", "open the terminal",
        "launch code", "start the text editor", "open spotify",
        "launch discord", "start nautilus", "open the file manager",
        "launch obsidian", "start krita", "open gimp",
        "launch the terminal", "start the browser", "open chromium",
        "launch the file manager", "start the music player", "open thunderbird",
        "launch slack", "start zoom", "open the calculator",
        "launch the editor", "start the pdf viewer", "open libreoffice",
        "launch steam", "start the screenshot tool", "open the settings app",
        "launch the terminal emulator", "start the system monitor",
        "open the web browser", "launch my notes app", "start the media player",
        "open the task manager", "launch blender", "start inkscape",
        "open the mail client", "launch the code editor", "start the file browser",
        "open the image viewer", "launch the archive manager", "start the clock",
        "open the webcam app", "launch the drawing app", "start the torrent client",
    ],
    "stop_application": [
        "kill firefox", "close the terminal", "stop spotify",
        "quit discord", "terminate code", "kill chromium",
        "close the file manager", "stop the browser", "quit the music player",
        "kill obsidian", "stop nautilus", "close the text editor",
        "terminate the download", "kill slack", "quit gimp",
        "stop the music", "close the pdf viewer", "kill the torrent client",
        "quit the terminal", "stop the code editor", "close krita",
        "kill the camera app", "quit libreoffice", "stop steam",
        "close the image viewer", "kill the calculator", "quit the mail client",
        "stop the game", "close everything in chromium", "kill the media player",
        "quit the notes app", "stop the archive manager", "kill blender",
        "close the system monitor", "stop the task manager", "quit the screenshot tool",
        "close the settings app", "kill the clock", "stop the webcam",
        "terminate the drawing app", "quit the browser", "close the file browser",
    ],
    "cancel": [
        "cancel", "cancel that", "nevermind", "never mind",
        "stop that", "forget it", "abort", "that's enough",
        "cancel everything", "stop", "nevermind that was wrong",
        "cancel the current operation", "forget what i said", "abort that",
        "cancel please", "never mind forget it", "stop everything",
        "cancel the last thing", "abort the test run", "forget it all",
        "cancel cancel", "no wait cancel that", "stop stop",
        "nevermind cancel", "cancel the whole thing", "forget that",
        "abort the operation", "cancel it", "stop that now",
        "cancel the command", "never mind", "forget the last command",
        "cancel immediately", "abort everything", "stop all of it",
        "cancel that request", "no cancel", "cancel please now",
        "forget my last request", "abort now", "stop the current thing",
        "cancel that command", "nevermind cancel everything",
        "undo that", "take that back", "scratch that",
        "cancel the open url", "no wait", "leave it",
    ],
    "unknown": [
        "the mitochondria is the powerhouse of the cell",
        "what is the capital of australia",
        "tell me a joke about penguins",
        "how do i bake sourdough bread",
        "what was the score in the match last night",
        "explain quantum entanglement to me",
        "the shop on the corner closes at six",
        "i had a sandwich earlier and it was fine",
        "my wife is called sarah and she likes gardening",
        "what is the weather in copenhagen tomorrow",
        "write me a poem about the sea",
        "how tall is the eiffel tower",
        "who painted the mona lisa",
        "translate good morning into french",
        "what is the boiling point of water",
        "should i invest in index funds",
        "the train leaves at half past eight",
        "my favourite colour is a difficult question",
        "what does the fox say",
        "how long does it take to boil an egg",
        "give me a recipe for pancakes",
        "what is the difference between rust and c",
        "my cat knocked over the plant again",
        "the meeting has been moved to thursday",
        "it is raining quite hard outside",
        "i finished reading that book last night",
        "what are the rules of cricket",
        "how do i tie a bowline knot",
        "the bus was late this morning",
        "she plays the cello and the piano",
        "what is the smallest prime number",
        "remind me to water the plants",
        "i think i left the oven on",
        "what is the airspeed velocity of an unladen swallow",
        "the library closes at eight on fridays",
        "my keyboard is making a strange noise",
        "could you pass me the salt",
        "it has been a long week",
        "i am reading a book about distributed systems",
        "the wifi keeps dropping out",
        "what do you think about this code",
        "summarise the news for me",
        "play some jazz music",
        "what is two plus two",
        "tell me about the history of this building",
        "i would like to order a coffee",
        "the traffic was terrible this morning",
        "my phone battery is nearly dead",
        "how do i change the wallpaper",
        "what does the acronym api stand for",
        "somebody left a package at the door",
        "the plants need watering",
        "i finished the report last night",
        "what are the odds of rain this weekend",
        "that film was really well made",
    ],
}


def example_id(intent: str, text: str) -> str:
    """Stable id, matching the id scheme `Example::new` uses in Rust.

    AARON derives the train/validation/test split from this id, so it must be
    reproducible across the Python and Rust implementations.
    """
    # FNV-1a 64, exactly as implemented in `aaron_core::text::fnv1a64`.
    h = 0xCBF29CE484222325
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{intent}-{h & 0xFFFFFFFF:08x}"


def normalise_key(text: str) -> str:
    """The same normalisation the model applies before hashing.

    Two utterances that normalise identically are indistinguishable to the
    classifier, so the dataset must not contain both.
    """
    out = []
    for ch in text.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in "'\u2019":
            continue
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def main() -> int:
    out = []
    seen = set()
    dropped = 0
    for intent, utterances in DATA.items():
        for text in utterances:
            key = normalise_key(text)
            if key in seen:
                dropped += 1
                continue
            seen.add(key)
            out.append(
                {
                    "id": example_id(intent, text),
                    "text": text,
                    "intent": intent,
                    "source": "template",
                    "notes": "authored in training/generate_dataset.py",
                }
            )

    out.sort(key=lambda e: (e["intent"], e["text"]))
    json.dump(
        {"format": 1, "version": "v1", "count": len(out), "dropped_duplicates": dropped},
        sys.stderr,
    )
    for e in out:
        sys.stdout.write(json.dumps(e, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())