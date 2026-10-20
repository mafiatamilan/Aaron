//! The deterministic keyword router.
//!
//! This is the backend that makes `aaron ask "check system status"` work on a
//! fresh install with no checkpoint and no network access. It is also the
//! fallback whenever the neural model is unavailable or its confidence is too
//! low to trust.
//!
//! It is intentionally simple and inspectable: a scored set of keyword and
//! phrase rules over the normalised utterance. It is **not** presented to the
//! user as speech recognition and it is not a language model — a phrase it has
//! never seen simply does not match.

use aaron_core::error::Result;
use aaron_core::intent::{ClassifySource, Command};
use aaron_core::text::{extract_entities, normalize, words};
use aaron_core::{Intent, IntentClassifier};

/// Backend name reported by `aaron doctor`.
pub const NAME: &str = "heuristic-rules";

/// One cue for an intent.
///
/// The distinction matters: a **phrase** is strong enough on its own, while a
/// **token** needs a second one before it counts. That is what keeps
/// ordinary English out of the tool layer. "the system is down" mentions the
/// system once and stays a statement; "system status" is a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    /// Matched as a token sequence, on token boundaries, anywhere.
    Phrase(&'static str),
    /// A single token. Two or more are needed.
    Token(&'static str),
    /// Matched as a token sequence, on token boundaries, at the **start** of
    /// the utterance.
    ///
    /// This is what separates a command from a sentence that happens to
    /// contain a command word. "launch the terminal" is a request; "the
    /// launch was delayed" is a statement about a launch. "please kill the
    /// lights when you leave" is a request to a *person*, not to AARON — the
    /// verb is not the first word, and neither is anything AARON would
    /// recognise as an address.
    Leading(&'static str),
}

/// A cue set for one intent.
///
/// # Weight
///
/// How strongly a phrase match counts. The weights are what make a strict
/// threshold meaningful: "launch the browser" is a weaker match than "git
/// status", so a threshold between the two admits one and refuses the other.
///
/// # Order
///
/// The rules are evaluated in order and the **first** rule with the best score
/// wins, so the order is part of the design. The file reader is last on
/// purpose: "open" opens files, applications and URLs, and the object decides
/// which. An application name or a URL is checked first, and only what is
/// left over is a file.
#[derive(Debug, Clone, Copy)]
struct Rule {
    intent: Intent,
    cues: &'static [Cue],
    /// The score a phrase match earns.
    weight: f32,
}

const fn phrase(text: &'static str) -> Cue {
    Cue::Phrase(text)
}

const fn token(text: &'static str) -> Cue {
    Cue::Token(text)
}

const fn leading(text: &'static str) -> Cue {
    Cue::Leading(text)
}

const RULES: &[Rule] = &[
    // Cancel must win over everything: "stop that and check status" is a
    // cancel. The safety valve below enforces that even so.
    Rule {
        intent: Intent::Cancel,
        cues: &[
            phrase("cancel"),
            phrase("nevermind"),
            phrase("never mind"),
            phrase("forget it"),
            phrase("forget that"),
            phrase("forget what i said"),
            phrase("forget what i just said"),
            phrase("forget my last request"),
            phrase("abort"),
            phrase("stop that"),
            phrase("stop the current task"),
            phrase("stop the current operation"),
            phrase("stop the file search"),
            phrase("that's enough"),
            phrase("that is enough"),
            phrase("enough of that"),
            phrase("no more of that"),
            phrase("scratch that"),
            phrase("disregard that request"),
            phrase("pull the plug"),
            phrase("halt what you are doing"),
            phrase("stand down"),
            phrase("take that back"),
            phrase("put it back"),
            phrase("leave it be"),
            phrase("leave it"),
            phrase("undo that"),
            phrase("undo that please"),
            phrase("i changed my mind"),
            phrase("i mean no"),
            phrase("no wait"),
            phrase("not that"),
            phrase("lets start over"),
            phrase("lets not do that"),
            phrase("hold on leave it"),
            phrase("on second thought"),
        ],
        weight: 0.95,
    },
    Rule {
        intent: Intent::Greeting,
        cues: &[
            phrase("hello"),
            phrase("hi"),
            phrase("hey"),
            phrase("yo"),
            phrase("sup"),
            phrase("howdy"),
            phrase("greetings"),
            phrase("good morning"),
            phrase("good evening"),
            phrase("good afternoon"),
            phrase("good day"),
            phrase("morning"),
            phrase("how are you"),
            phrase("how's it going"),
            phrase("hows it going"),
            phrase("what's up"),
            phrase("whats up"),
            phrase("nice to meet you"),
            phrase("is anyone there"),
        ],
        weight: 0.8,
    },
    Rule {
        intent: Intent::Help,
        cues: &[
            phrase("help"),
            phrase("what can you do"),
            phrase("what do you do"),
            phrase("what can i say"),
            phrase("what do you understand"),
            phrase("how do you work"),
            phrase("how do i use this"),
            phrase("how do i use you"),
            phrase("capabilities"),
            phrase("commands"),
            phrase("quick reference"),
            phrase("reference"),
            phrase("instructions"),
            phrase("list commands"),
            phrase("command list"),
            phrase("the command list"),
            phrase("the manual"),
            phrase("the reference"),
            phrase("the guide"),
            phrase("know how to do"),
            phrase("what you support"),
            phrase("what you are able to do"),
            phrase("what are you able to do"),
            phrase("what you can do"),
            phrase("list everything you can do"),
            phrase("give me a list of what you can do"),
            phrase("i forgot what you can do"),
            phrase("i don't know what to ask you"),
            phrase("what are my options"),
            phrase("what are you good at"),
            phrase("what are the available intents"),
            phrase("available intents"),
            phrase("how do i talk to you"),
            phrase("how do i talk"),
            phrase("how can i use aaron"),
            phrase("how do i use you"),
            phrase("what does aaron do"),
            phrase("what does aaron understand"),
            phrase("which verbs do you understand"),
            phrase("which requests do you handle"),
            phrase("what syntax do you use"),
            phrase("what should i say"),
            phrase("command overview"),
            phrase("list your features"),
            phrase("an example command"),
            phrase("example command"),
            phrase("explain what you do"),
            phrase("tell me about yourself"),
            phrase("tell me about you"),
            phrase("usage"),
        ],
        weight: 0.9,
    },
    Rule {
        intent: Intent::SystemStatus,
        cues: &[
            phrase("system status"),
            phrase("status check"),
            phrase("system health"),
            phrase("machine status"),
            phrase("pc status"),
            phrase("how is the system"),
            phrase("how is the machine"),
            phrase("check the machine"),
            phrase("check the system"),
            phrase("check the health of the machine"),
            phrase("resource usage"),
            phrase("resources"),
            phrase("system overview"),
            phrase("system stats"),
            phrase("system report"),
            phrase("resource report"),
            phrase("health check"),
            phrase("run a health check"),
            phrase("give me a system overview"),
            phrase("show me the resources"),
            phrase("show me system stats"),
            phrase("tell me about resource usage"),
            phrase("how many cores"),
            phrase("cpu usage"),
            phrase("gpu usage"),
            phrase("memory usage"),
            phrase("disk space"),
            phrase("swap usage"),
            phrase("how much memory"),
            phrase("how much swap"),
            phrase("what is the cpu"),
            phrase("what does the cpu"),
            phrase("how is the cpu"),
            phrase("uptime"),
            phrase("any unusual processes"),
            phrase("any unusual network"),
            phrase("how much space is left"),
            token("cpu"),
            token("gpu"),
            token("memory"),
            token("ram"),
            token("swap"),
            token("disk"),
            token("cores"),
            token("temperature"),
            token("overheating"),
            token("vram"),
            token("accelerator"),
            token("resources"),
            token("resource"),
            token("health"),
            token("space"),
            token("processes"),
        ],
        weight: 0.9,
    },
    Rule {
        intent: Intent::GitStatus,
        cues: &[
            phrase("git status"),
            phrase("git diff status"),
            phrase("git status of"),
            phrase("check git"),
            phrase("show git"),
            phrase("show me git"),
            phrase("git report"),
            phrase("git summary"),
            phrase("git state"),
            phrase("version control status"),
            phrase("repository status"),
            phrase("repo status"),
            phrase("current branch"),
            phrase("show the current branch"),
            phrase("which branch"),
            phrase("branch status"),
            phrase("last commit"),
            phrase("recent commit"),
            phrase("head commit"),
            phrase("staged changes"),
            phrase("anything staged"),
            phrase("lines are staged"),
            phrase("is the index dirty"),
            phrase("index dirty"),
            phrase("working tree"),
            phrase("tree state"),
            phrase("what has changed"),
            phrase("files have changed"),
            phrase("how many files changed"),
            phrase("needs committing"),
            phrase("anything to push"),
            phrase("new files git"),
            phrase("since the last commit"),
            phrase("since the last tag"),
            phrase("show me the git"),
            phrase("git log"),
            phrase("the git log"),
            phrase("git diff"),
            phrase("git branch"),
            phrase("files are staged"),
            phrase("staged for commit"),
            phrase("repository doing"),
            phrase("the repo"),
            phrase("conflicts in the repo"),
            token("git"),
            token("branch"),
            token("commit"),
            token("staged"),
            token("untracked"),
            token("repository"),
            token("repo"),
            token("diff"),
        ],
        weight: 0.95,
    },
    Rule {
        intent: Intent::RunTests,
        cues: &[
            leading("run"),
            leading("execute"),
            leading("trigger"),
            leading("kick"),
            phrase("the tests"),
            phrase("the test suite"),
            phrase("cargo test"),
            phrase("cargo unit"),
            phrase("integration tests"),
            phrase("unit tests"),
            phrase("test runner"),
            phrase("kick off the tests"),
            phrase("go ahead and run"),
            phrase("start the tests"),
        ],
        weight: 0.9,
    },
    Rule {
        intent: Intent::SearchFiles,
        cues: &[
            phrase("search for"),
            phrase("find files"),
            phrase("find file"),
            phrase("find the"),
            phrase("find all"),
            phrase("locate the"),
            phrase("locate file"),
            phrase("where is the"),
            phrase("where is my"),
            phrase("where are the"),
            phrase("which files"),
            phrase("which folders"),
            phrase("sift through"),
            phrase("dig out"),
            phrase("list the files"),
            phrase("show me every file"),
            phrase("search files"),
            phrase("look for"),
            phrase("hunt for"),
            phrase("grep for"),
            phrase("files called"),
            phrase("files called"),
            phrase("files named"),
            phrase("files named"),
            phrase("files matching"),
            phrase("files ending"),
            phrase("files end in"),
            phrase("files contain"),
            phrase("ending in"),
            phrase("matching"),
            phrase("search for files"),
            phrase("search for the file"),
            phrase("search for large files"),
            phrase("search for the"),
            token("glob"),
            token("pattern"),
            token("vendored"),
            token("folders"),
        ],
        weight: 0.85,
    },
    Rule {
        intent: Intent::Recall,
        cues: &[
            phrase("what do you know"),
            phrase("what do you remember"),
            phrase("do you remember"),
            phrase("search your memory"),
            phrase("look up in memory"),
            phrase("from memory"),
            phrase("what did i say"),
            phrase("what did i tell you"),
            phrase("what did you say"),
            phrase("what have i stored"),
            phrase("what have you stored"),
            phrase("what do you recall"),
            phrase("remind me"),
            phrase("go through your notes"),
            phrase("dig through your memory"),
            phrase("fetch from memory"),
            phrase("look up what you know"),
            phrase("pull up what you know"),
            phrase("check your memory"),
            phrase("recall"),
            phrase("retrieve"),
        ],
        weight: 0.95,
    },
    Rule {
        intent: Intent::Remember,
        cues: &[
            phrase("remember that"),
            phrase("remember this"),
            phrase("remember the"),
            phrase("remember"),
            phrase("note that"),
            phrase("note it down"),
            phrase("make a note"),
            phrase("keep in mind"),
            phrase("save this for later"),
            phrase("log the fact"),
            phrase("write down"),
            phrase("mark down"),
            phrase("put down"),
            phrase("jot down"),
            phrase("store the fact"),
            phrase("pin that"),
        ],
        weight: 0.95,
    },
    Rule {
        intent: Intent::OpenUrl,
        cues: &[
            phrase("open the website"),
            phrase("open the url"),
            phrase("open the site"),
            phrase("browse to"),
            phrase("go to the site"),
            phrase("pull up the"),
            phrase("take me to"),
            leading("browse"),
            leading("visit"),
            phrase("navigate to"),
        ],
        weight: 0.95,
    },
    Rule {
        intent: Intent::StartApplication,
        cues: &[
            phrase("launch the"),
            leading("launch"),
            phrase("start the application"),
            phrase("start application"),
            phrase("start app"),
            phrase("open the application"),
            phrase("open the app"),
            phrase("fire up"),
            phrase("fire the"),
            phrase("raise the"),
            phrase("relaunch"),
            // Application names.
            phrase("open archive manager"),
            phrase("open ardour"),
            phrase("open audacity"),
            phrase("open background job"),
            phrase("open bitwarden"),
            phrase("open blender"),
            phrase("open browser"),
            phrase("open btop"),
            phrase("open calculator"),
            phrase("open camera app"),
            phrase("open chromium"),
            phrase("open clock"),
            phrase("open code"),
            phrase("open code blocks"),
            phrase("open code editor"),
            phrase("open dbeaver"),
            phrase("open dbeaver community"),
            phrase("open debugger"),
            phrase("open discord"),
            phrase("open docker daemon"),
            phrase("open docker desktop"),
            phrase("open download"),
            phrase("open drawing app"),
            phrase("open editor"),
            phrase("open element"),
            phrase("open etcher"),
            phrase("open evince"),
            phrase("open figma"),
            phrase("open file"),
            phrase("open file browser"),
            phrase("open file manager"),
            phrase("open filezilla"),
            phrase("open firefox"),
            phrase("open game"),
            phrase("open gimp"),
            phrase("open gnome calculator"),
            phrase("open htop"),
            phrase("open image viewer"),
            phrase("open inkscape"),
            phrase("open jetbrains toolbox"),
            phrase("open kde konsole"),
            phrase("open kdenlive"),
            phrase("open keepassxc"),
            phrase("open krita"),
            phrase("open libreoffice"),
            phrase("open libreoffice calc"),
            phrase("open libreoffice impress"),
            phrase("open libreoffice writer"),
            phrase("open mail client"),
            phrase("open media player"),
            phrase("open mongodb compass"),
            phrase("open mpv"),
            phrase("open music"),
            phrase("open music player"),
            phrase("open nautilus"),
            phrase("open nextcloud"),
            phrase("open notes app"),
            phrase("open obs"),
            phrase("open obsidian"),
            phrase("open okular"),
            phrase("open pdf viewer"),
            phrase("open postman"),
            phrase("open qemu"),
            phrase("open remmina"),
            phrase("open screenshot tool"),
            phrase("open settings app"),
            phrase("open signal"),
            phrase("open slack"),
            phrase("open spotify"),
            phrase("open spreadsheet"),
            phrase("open steam"),
            phrase("open syncthing"),
            phrase("open system monitor"),
            phrase("open task manager"),
            phrase("open terminal"),
            phrase("open text editor"),
            phrase("open the app"),
            phrase("open the application"),
            phrase("open the file"),
            phrase("open the site"),
            phrase("open the url"),
            phrase("open the website"),
            phrase("open thunderbird"),
            phrase("open torrent client"),
            phrase("open totem"),
            phrase("open transmission"),
            phrase("open video editor"),
            phrase("open virtualbox"),
            phrase("open vlc"),
            phrase("open web browser"),
            phrase("open webcam"),
            phrase("open webcam app"),
            phrase("open wireshark"),
            phrase("open zathura"),
            phrase("open zoom"),
            phrase("open the ardour"),
            phrase("open the audacity"),
            phrase("open the bitwarden"),
            phrase("open the blender"),
            phrase("open the browser"),
            phrase("open the btop"),
            phrase("open the calculator"),
            phrase("open the chromium"),
            phrase("open the clock"),
            phrase("open the code"),
            phrase("open the dbeaver"),
            phrase("open the debugger"),
            phrase("open the discord"),
            phrase("open the download"),
            phrase("open the editor"),
            phrase("open the element"),
            phrase("open the etcher"),
            phrase("open the evince"),
            phrase("open the figma"),
            phrase("open the file"),
            phrase("open the filezilla"),
            phrase("open the firefox"),
            phrase("open the game"),
            phrase("open the gimp"),
            phrase("open the htop"),
            phrase("open the inkscape"),
            phrase("open the kdenlive"),
            phrase("open the keepassxc"),
            phrase("open the krita"),
            phrase("open the libreoffice"),
            phrase("open the mpv"),
            phrase("open the music"),
            phrase("open the nautilus"),
            phrase("open the nextcloud"),
            phrase("open the obs"),
            phrase("open the obsidian"),
            phrase("open the okular"),
            phrase("open the postman"),
            phrase("open the qemu"),
            phrase("open the remmina"),
            phrase("open the signal"),
            phrase("open the slack"),
            phrase("open the spotify"),
            phrase("open the spreadsheet"),
            phrase("open the steam"),
            phrase("open the syncthing"),
            phrase("open the terminal"),
            phrase("open the thunderbird"),
            phrase("open the totem"),
            phrase("open the transmission"),
            phrase("open the virtualbox"),
            phrase("open the vlc"),
            phrase("open the webcam"),
            phrase("open the wireshark"),
            phrase("open the zathura"),
            phrase("open the zoom"),
        ],
        weight: 0.9,
    },
    Rule {
        intent: Intent::StopApplication,
        cues: &[
            phrase("stop the application"),
            phrase("kill the"),
            leading("kill"),
            phrase("close the application"),
            phrase("close the app"),
            phrase("quit the application"),
            phrase("terminate the"),
            leading("terminate"),
            phrase("close the"),
            phrase("quit the"),
        ],
        weight: 0.9,
    },
    Rule {
        intent: Intent::ReadFile,
        cues: &[
            phrase("read the"),
            phrase("read file"),
            phrase("read files"),
            phrase("read first"),
            phrase("read the first"),
            phrase("read the top"),
            phrase("read the last"),
            phrase("open the file"),
            phrase("open file"),
            phrase("cat the"),
            phrase("show me the"),
            phrase("show me file"),
            phrase("what is in the"),
            phrase("what's in the"),
            phrase("contents of the"),
            leading("read"),
            leading("cat"),
            leading("dump"),
            phrase("print the"),
            phrase("display the"),
            phrase("preview the"),
            leading("open"),
        ],
        weight: 0.85,
    },
];

/// The score two matching tokens earn.
///
/// Below every rule's phrase weight, so a phrase always beats a pair of
/// tokens, and above the production threshold, so a pair of distinctive
/// tokens is enough to act on.
const TOKEN_SCORE: f32 = 0.80;

/// Whether the utterance contains `needle`'s words, in order, on token
/// boundaries.
///
/// The old rule was `joined.contains(needle)`, which matched *inside* a word:
/// "hi" inside "which", "hey" inside "they", "kill" inside "skill", "launch"
/// inside "launchpad", "help" inside "helpful". Every one of those is an
/// ordinary English sentence that AARON would have turned into a tool call.
/// Whether the utterance *starts* with `needle`'s words, on token boundaries.
fn starts_with_phrase(tokens: &[String], needle: &str) -> bool {
    let words: Vec<&str> = needle.split_whitespace().collect();
    if words.is_empty() || words.len() > tokens.len() {
        return false;
    }
    tokens.iter().zip(words.iter()).all(|(a, b)| a == b)
}

fn contains_phrase(tokens: &[String], needle: &str) -> bool {
    let words: Vec<&str> = needle.split_whitespace().collect();
    if words.is_empty() || words.len() > tokens.len() {
        return false;
    }
    tokens
        .windows(words.len())
        .any(|window| window.iter().zip(words.iter()).all(|(a, b)| a == b))
}

/// Keyword router.
pub struct HeuristicClassifier {
    /// Below this, the result is reported as `unknown`.
    pub threshold: f32,
}

impl HeuristicClassifier {
    pub fn new(threshold: f32) -> HeuristicClassifier {
        HeuristicClassifier { threshold }
    }

    /// Score the utterance and return `(intent, score)`.
    pub fn score(&self, utterance: &str) -> (Intent, f32) {
        let normalized = normalize(utterance);
        let tokens = words(&normalized);

        let mut best = (Intent::Unknown, 0.0f32);
        for rule in RULES {
            let mut phrase_hit = false;
            let mut token_hits = 0usize;
            for cue in rule.cues {
                match cue {
                    Cue::Phrase(text) => {
                        if contains_phrase(&tokens, text) {
                            phrase_hit = true;
                            break;
                        }
                    }
                    Cue::Token(text) => {
                        if tokens.iter().any(|t| t == text) {
                            token_hits += 1;
                        }
                    }
                    Cue::Leading(text) => {
                        if starts_with_phrase(&tokens, text) {
                            phrase_hit = true;
                            break;
                        }
                    }
                }
            }
            // A phrase is decisive on its own. A single token is not: one
            // distinctive word in an ordinary sentence is a mention, not a
            // request.
            //
            // The scores stay below 1.0 on purpose. The threshold is how a
            // user turns the router off, and it only means something if the
            // best a normal command can score is lower than 1.0. The cancel
            // safety valve below is the one thing that may reach 1.0,
            // because refusing to cancel is the dangerous direction.
            let score = if phrase_hit {
                rule.weight
            } else if token_hits >= 2 {
                TOKEN_SCORE
            } else {
                0.0
            };
            if score > best.1 {
                best = (rule.intent, score);
            }
        }

        // A URL is a strong signal regardless of the wording — but it has to
        // *look* like a URL. "http logs are noisy" is not a request to open
        // anything.
        if looks_like_url(utterance) {
            best = (Intent::OpenUrl, 0.95);
        }

        // "stop"/"cancel" always wins: it is the safety valve.
        if contains_phrase(&tokens, "cancel")
            || contains_phrase(&tokens, "never mind")
            || contains_phrase(&tokens, "nevermind")
        {
            best = (Intent::Cancel, 1.0);
        }

        best
    }
}

/// Whether the text contains something that looks like a URL.
///
/// Checked on the **raw** utterance. Normalisation strips the punctuation
/// that makes a URL recognisable — "https://example.com" arrives as three
/// separate tokens — so a check on normalised tokens can never see the
/// scheme. It also means "http logs are noisy" is not a URL, which is the
/// point.
pub fn looks_like_url(text: &str) -> bool {
    text.split_whitespace()
        .any(|word| word.contains("://") || word.starts_with("www."))
}

impl Default for HeuristicClassifier {
    fn default() -> Self {
        HeuristicClassifier::new(0.5)
    }
}

impl IntentClassifier for HeuristicClassifier {
    fn name(&self) -> &'static str {
        NAME
    }

    fn classify(&self, utterance: &str) -> Result<Command> {
        let (intent, score) = self.score(utterance);
        let effective = if score < self.threshold {
            Intent::Unknown
        } else {
            intent
        };
        let entities = extract_entities(effective, utterance);
        // The deterministic router has no learned readiness head, so it
        // computes the verdict from the extracted entities with exactly the
        // rules the agent's argument builder uses. It is exact given the
        // extraction: where the extractor is right, this is right too.
        let readiness = aaron_core::readiness::readiness_of(effective, &entities);
        Ok(Command {
            intent: effective,
            entities,
            confidence: if effective == Intent::Unknown {
                0.0
            } else {
                score
            },
            source: ClassifySource::Heuristic,
            readiness,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn classify(text: &str) -> Command {
        HeuristicClassifier::default().classify(text).unwrap()
    }

    #[test]
    fn routes_the_canonical_examples() {
        let cases = [
            ("hello there", Intent::Greeting),
            ("what can you do", Intent::Help),
            ("check system status", Intent::SystemStatus),
            ("remember that I drink tea", Intent::Remember),
            ("what do you remember about coffee", Intent::Recall),
            ("find files named main.rs", Intent::SearchFiles),
            ("read the file Cargo.toml", Intent::ReadFile),
            ("show git status", Intent::GitStatus),
            ("run the tests", Intent::RunTests),
            ("open the website https://example.com", Intent::OpenUrl),
            ("launch the browser", Intent::StartApplication),
            ("kill firefox", Intent::StopApplication),
            ("cancel that", Intent::Cancel),
        ];
        for (text, expected) in cases {
            assert_eq!(classify(text).intent, expected, "`{text}`");
        }
    }

    #[test]
    fn unrelated_utterances_are_unknown() {
        for text in [
            "the weather is quite pleasant today",
            "tell me a joke about penguins",
            "what is the capital of france",
            "",
        ] {
            assert_eq!(
                classify(text).intent,
                Intent::Unknown,
                "`{text}` should be unknown"
            );
        }
    }

    #[test]
    fn cancel_overrides_everything() {
        assert_eq!(
            classify("cancel that and check system status").intent,
            Intent::Cancel
        );
        assert_eq!(classify("stop, nevermind").intent, Intent::Cancel);
    }

    #[test]
    fn a_bare_url_is_recognised() {
        let cmd = classify("https://example.com/docs");
        assert_eq!(cmd.intent, Intent::OpenUrl);
        assert_eq!(
            cmd.entities.url.as_deref(),
            Some("https://example.com/docs")
        );
    }

    #[test]
    fn case_and_punctuation_do_not_matter() {
        assert_eq!(
            classify("CHECK SYSTEM STATUS!!!").intent,
            Intent::SystemStatus
        );
        assert_eq!(classify("  show   git status ").intent, Intent::GitStatus);
    }

    #[test]
    fn a_strict_threshold_produces_unknown() {
        // 0.94 sits above the weak "launch" score and below the strong
        // "git status" score, so exactly one of the two falls through.
        let strict = HeuristicClassifier::new(0.94);
        let cmd = strict.classify("launch the browser").unwrap();
        assert_eq!(cmd.intent, Intent::Unknown);
        let cmd = strict.classify("git status").unwrap();
        assert_eq!(cmd.intent, Intent::GitStatus);
    }

    #[test]
    fn source_is_reported_as_heuristic() {
        assert_eq!(classify("hello").source, ClassifySource::Heuristic);
    }
}
