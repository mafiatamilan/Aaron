//! The deterministic router is the production path, so it gets its own
//! regression suite.
//!
//! Two properties matter, and they pull in opposite directions:
//!
//! 1. **The router must reproduce the shipped dataset.** The dataset is the
//!    project's specification of what AARON should understand, and the neural
//!    model was trained on it. The router used to agree with it on 44.6 % of
//!    the rows, with `read_file` at 4 of 148.
//! 2. **The router must not turn ordinary English into tool calls.** It used
//!    to match substrings, so "hi" inside "which" made a greeting, "kill"
//!    inside "killer" selected `stop_application`, and "http" inside a
//!    sentence about logs selected `open_url`.
//!
//! The second is the safety property, so it is asserted first and in full:
//! every ordinary sentence in the corpus must route to something that
//! dispatches nothing.

use aaron_brain::dataset::Dataset;
use aaron_brain::heuristic::HeuristicClassifier;
use aaron_core::intent::Intent;

/// Ordinary English that must not reach a tool.
///
/// None of these asks AARON to do anything. Every one must route to
/// `unknown`, `greeting` or `help`.
const NEGATIVES: &[&str] = &[
    // Ordinary questions.
    "which file is the largest",
    "they told me the system is down",
    "the killer feature is broken",
    "http logs are noisy",
    "launchpad will not start",
    "how are things",
    "how are you",
    "how are you doing",
    "what is the weather in copenhagen",
    "who won the world cup in 1998",
    "why is the sky blue",
    "when did the second world war end",
    "how far is it to the moon",
    "what is the capital of australia",
    "who wrote hamlet",
    "how do i boil an egg",
    "what time is it",
    "where am i",
    "is it going to rain tomorrow",
    "how much does a kilogram of rice cost",
    "what is the speed of light",
    "which programming language should i learn",
    "how do photosynthesis plants work",
    "who is my boss",
    "what is the meaning of life",
    "can you explain recursion",
    "do you know what time it is",
    "is the deployment ready",
    // Statements.
    "the bus was late this morning",
    "it is raining outside",
    "she left the office at noon",
    "the meeting starts at nine",
    "he bought a new bicycle",
    "that was a long journey",
    "they finished the migration last week",
    "the package arrived yesterday",
    "we should leave before six",
    "the report was long",
    "i finished reading that book last night",
    "the mitochondria is the powerhouse of the cell",
    "my name is david",
    "i live in berlin",
    // Sentences that mention a tool word in passing.
    "the launch was delayed",
    "the help was excellent",
    "this is a high priority task",
    "they have a machine learning team",
    "which machine should i buy",
    "i found a great recipe online",
    "the run took three hours",
    "my test suite is slow",
    "the app crashed again",
    "i closed the window",
    "the process is running",
    "i saved the document",
    "show me how to cook rice",
    "tell me a joke",
    "what do you think about the weather",
    "give me some advice",
    "thanks for your help",
    "nice to meet you",
    "sorry, that was my mistake",
    "good luck with the release",
    "happy birthday",
    "see you tomorrow",
    "i am fine",
    "no problem",
    "of course",
    "sounds good",
    "let me think about it",
];

/// The ones that are genuinely ambiguous, and what they are allowed to be.
///
/// "please kill the lights when you leave" is addressed to a person, not to
/// AARON, and "i need to read the newspaper" asks for something AARON cannot
/// do — but both read as commands, and the tool layer will refuse them
/// truthfully ("not a registered application", "no such file"). Refusing to
/// recognise them would also refuse the real commands, so they are allowed
/// through and pinned here so a change to the router is noticed.
const AMBIGUOUS: &[(&str, Intent)] = &[
    (
        "please kill the lights when you leave",
        Intent::StopApplication,
    ),
    ("i need to read the newspaper", Intent::ReadFile),
    ("what did i tell you yesterday", Intent::Recall),
    ("are the tests passing", Intent::RunTests),
];

/// The commands the 1.0 brief names explicitly, plus the ones the shipped
/// dataset uses. Each must be recognised.
const COMMANDS: &[(&str, Intent)] = &[
    ("recall my name", Intent::Recall),
    ("remember my name is david", Intent::Remember),
    ("check system status", Intent::SystemStatus),
    ("show git status", Intent::GitStatus),
    ("run the tests", Intent::RunTests),
    ("read main.rs", Intent::ReadFile),
    ("read the file main.rs", Intent::ReadFile),
    ("search for the file named main.rs", Intent::SearchFiles),
    ("find files ending in .md", Intent::SearchFiles),
    ("launch the terminal", Intent::StartApplication),
    ("open the browser", Intent::StartApplication),
    ("kill spotify", Intent::StopApplication),
    ("open https://example.com", Intent::OpenUrl),
    ("cancel", Intent::Cancel),
    ("what can you do", Intent::Help),
    ("hello", Intent::Greeting),
    ("good morning", Intent::Greeting),
];

fn router() -> HeuristicClassifier {
    HeuristicClassifier::new(0.60)
}

#[test]
fn ordinary_english_never_reaches_a_tool() {
    let router = router();
    let mut offenders: Vec<String> = Vec::new();
    for text in NEGATIVES {
        let (intent, score) = router.score(text);
        if intent.requests_action() {
            offenders.push(format!("{text:?} -> {intent} @ {score:.2}"));
        }
    }
    assert!(
        offenders.is_empty(),
        "{} ordinary sentence(s) reached a tool:\n  {}",
        offenders.len(),
        offenders.join("\n  ")
    );
}

#[test]
fn the_ambiguous_ones_stay_where_they_are() {
    // Pinned so that a change to the router is noticed: these read as
    // commands, and the tool layer refuses them truthfully.
    let router = router();
    for (text, expected) in AMBIGUOUS {
        let (intent, _) = router.score(text);
        assert_eq!(
            intent, *expected,
            "{text:?} changed intent; the tool layer's refusal depends on which tool is asked"
        );
    }
}

#[test]
fn the_commands_are_recognised() {
    let router = router();
    let mut misses: Vec<String> = Vec::new();
    for (text, expected) in COMMANDS {
        let (intent, score) = router.score(text);
        if intent != *expected {
            misses.push(format!(
                "{text:?} -> {intent} @ {score:.2}, expected {expected}"
            ));
        }
    }
    assert!(
        misses.is_empty(),
        "{} command(s) were not recognised:\n  {}",
        misses.len(),
        misses.join("\n  ")
    );
}

#[test]
fn the_router_reproduces_the_shipped_dataset() {
    // The dataset is the specification. The neural model was trained on it and
    // reaches a macro-F1 of about 0.74; the router, which is what actually
    // ships, must not be far behind it.
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../training/dataset/current.jsonl"
    );
    let Ok(dataset) = Dataset::load(path) else {
        // The dataset is not present in every build environment. The rest of
        // this file still holds.
        return;
    };
    let router = router();
    let mut agree = 0usize;
    for example in &dataset.examples {
        let (intent, _) = router.score(&example.text);
        if intent == example.intent {
            agree += 1;
        }
    }
    let ratio = agree as f64 / dataset.len() as f64;
    assert!(
        ratio >= 0.80,
        "the router agrees with the dataset on {:.1}% of {} rows; it must be at least 80%",
        ratio * 100.0,
        dataset.len()
    );
}
