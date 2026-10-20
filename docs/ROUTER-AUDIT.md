# Router and tool-layer audit — AARON 1.0

**Scope.** The production router is the deterministic keyword router in
`crates/aaron-brain/src/heuristic.rs`. No trained checkpoint is installed in
`models/brain/` (only `.gitkeep`), so `brain.backend = "auto"` falls back to this
router on every fresh install (`classifier.rs:322-336`). The agent that
extracts entities, builds arguments and dispatches is
`crates/aaron-agent/src/agent.rs`. All live results below were produced by the
real release binary:

```
AARON_HOME=/tmp/opencode/aaron-audit-state ./target/release/aaron \
    --text-only --yes --json ask "<utterance>"
```

(`--yes` is used only so confirmation-gated tools do not block on stdin; it is
not an authorisation bypass — see §8.) The default config threshold in effect is
`config/default.toml:49` (`confidence_threshold = 0.60`), not the
`HeuristicClassifier::new(0.5)` default in `heuristic.rs:247`.

## Verdict: one paragraph

The router is a scored substring matcher over 14 rules (`heuristic.rs:31-192`)
with a 0.60 production threshold (`config/default.toml:49`). Several rules are
bare substrings of ordinary English words (`hi`, `hey`, `help`, `remember`,
`launch`, `kill`, `http`, "read the first", "find file"), so roughly one-third
of ordinary questions and statements are mis-routed to actionable intents, and
four of them actually reach tool dispatch (one executes a real tool — `system_status`
— on the plain statement "how are things"). At the same time, genuine commands
the 1.0 brief says must be preserved ("read main.rs", "search for the file named
main.rs") are not recognised at all. The tool-layer gates (argument validator,
policy engine, confirmation gate, loop guard) are correctly designed and do
stop a confident prediction from running a tool (`hard_negatives`: 24/24 pass),
but they cannot fix a router that hands them the wrong intent. The router is
not yet safe to ship as the 1.0 production path without targeted rule narrowing
and per-case regression fixes.

## How the router decides (the rules, precisely)

`Rule { required, alternatives, weight }` (`heuristic.rs:22-29`). `score()`
(`heuristic.rs:206-242`) normalises the utterance (`normalize`, `text.rs:23`),
joins tokens with spaces, and for each rule sets `s = weight` when all
`required` tokens are substrings of the joined string, then `s = max(s, weight)`
for any matching `alternative` substring (`joined.contains(alt)`,
`heuristic.rs:217-219`). The highest-scoring rule wins (`heuristic.rs:222`).
Two post-hoc safety valves then fire: any token beginning with `http` forces
`OpenUrl` at 1.0 (`heuristic.rs:228-230`), and `cancel`/`never mind`/`nevermind`
anywhere forces `Cancel` at 1.0 (`heuristic.rs:233-238`). Finally
`classify()` (`heuristic.rs:255-279`) coerces the intent to `Unknown` when
`score < threshold` (`heuristic.rs:257-261`); with the shipped config the
threshold is **0.60** (`config/default.toml:49`).

| # | intent | required tokens (all) | alternatives (any one) | weight | line |
|---|--------|----------------------|------------------------|--------|------|
| 1 | Cancel | — | cancel, nevermind, nevermind, forget it, abort, stop that, that's enough | 1.0 | 33-46 |
| 2 | Greeting | — | hello, hi, hey, good morning, good evening, greetings, howdy | 0.8 | 48-60 |
| 3 | Help | — | help, what can you do, capabilities, how do you work, what do you do, commands | 0.9 | 62-73 |
| 4 | SystemStatus | — | system status, check system, how is the system, machine status, how are things, are you ok, system health, check the machine, pc status, resource usage | 0.9 | 75-90 |
| 5 | Recall | recall | what do you know, what do you remember, do you remember, search your memory, look up in memory, from memory | 1.0 | 92-103 |
| 6 | Remember | remember, remember | remember that, make a note, note that, keep in mind | 1.0 | 105-109 |
| 7 | SearchFiles | — | find file, find files, search for files, locate file, where is the file, search files | 0.85 | 111-122 |
| 8 | ReadFile | — | read file, open file, cat the file, show me the file, read the file, read the first | 0.85 | 124-135 |
| 9 | GitStatus | — | git status, check git, repository status, show git, diff status | 0.95 | 137-147 |
| 10 | RunTests | — | run the tests, run tests, run test, test suite, run the test, run cargo test | 0.9 | 149-160 |
| 11 | OpenUrl | http | open the website, browse to, go to the site | 1.0 | 162-166 |
| 12 | StartApplication | — | launch, start the application, start application, start app, open the application, fire up | 0.7 | 168-179 |
| 13 | StopApplication | — | stop the application, kill, close the application, quit the application, terminate | 0.8 | 181-191 |

Note two implementation details that matter for the audit:
- Rule 6's `required` is `["remember", "remember"]` (`heuristic.rs:106`) — a
  duplicated entry, so the effective condition is just that `remember` appears
  anywhere.
- Rule 13 (`StopApplication`) has `stop the application` and `kill`, but **not**
  bare `stop`. Bare `stop` therefore does not match (`a bus stop` → `unknown`,
  verified live). Only `cancel`/`never mind`/`nevermind` trigger the separate
  Cancel safety valve (`heuristic.rs:233-238`).

### Every broad-substring rule (matches a word inside an ordinary question)

| rule (`heuristic.rs` line) | trigger | ordinary words it swallows |
|---|---|---|
| Greeting `hi` (53) | `joined.contains("hi")` | this, high, machine, him, hint, ohio |
| Greeting `hey` (54) | `joined.contains("hey")` | they |
| Help `help` (65) | `joined.contains("help")` | helping, helpless, helper ("can't help it") |
| Remember `remember` required (106) | `joined.contains("remember")` | "i don't remember", "remember when" |
| SearchFiles `find file` (114) | `joined.contains("find file")` | "find a file", "find file contents" |
| ReadFile `read file` (127) | `joined.contains("read file")` | "read file contents" |
| ReadFile `read the first` (133) | `joined.contains("read the first")` | "read the first chapter" |
| StartApplication `launch` (171) | `joined.contains("launch")` | **single word**: "launch a complaint", "launch into" |
| StopApplication `kill` (185) | `joined.contains("kill")` | skill, skilled, killing, skillful |
| OpenUrl required `http` (163) | token starts with `http` | "what does http mean", "check http status" |
| Cancel safety valve `cancel` (233) | `joined.contains("cancel")` | any utterance containing cancel |

### Mis-routed ordinary questions (live, production threshold 0.60)

37 ordinary questions/statements were run; 12 came back with an actionable
intent instead of `unknown`. Four of those actually reached tool dispatch.

| utterance | intent AARON gave | what AARON did | should be |
|---|---|---|---|
| how are things | system_status | **Executed `system_status`** and printed real CPU/memory/GPU status | unknown (statement/greeting) |
| the bus was late this morning | greeting | Said "Hello. I am AARON." | unknown |
| what do you think of that | greeting | Said "Hello. I am AARON." | unknown |
| this is a test of the system | greeting | Said "Hello. I am AARON." | unknown |
| how high is the mountain | greeting | Said "Hello. I am AARON." | unknown |
| they said it would rain | greeting | Said "Hello. I am AARON." | unknown |
| i can't help it | help | Printed the help menu | unknown (statement) |
| i don't remember her name | remember | Built `remember` args, then **policy-denied** (no policy entry) | unknown (statement) |
| he has a skill with computers | stop_application | Built `stop_application("he has a skill with computers")`, then **policy-denied** | unknown |
| launch a new project | start_application | Built `start_application("a new project")`, then **policy-denied** | unknown |
| read the first chapter | read_file | Built `read_file("the first chapter")`, then **sandbox-failed** (no allowed roots) | unknown (statement) |
| what does http mean | open_url | Asked "which URL to open" (needs_clarification) | unknown |

Correctly returned `unknown` (25 of 37): "how are you", "how are you doing",
"yo", "morning", "the weather is quite pleasant today", "tell me a joke about
penguins", "what is the capital of france", "my cat knocked over the plant
again", "i finished reading that book last night", "it has been a long week",
"how do i bake sourdough bread", "do you like music", "can you hear me", "are
you there", "i wonder if it will rain tomorrow", "a bus stop", "i need to find
a file", plus the genuine help requests "what can you do" / "how do you work"
(→ Help, correct).

Root cause of every mis-route above: a broad `joined.contains(alt)` substring
rule matching inside an ordinary word (hi/hey/help/remember/launch/kill/"read
the first"/http), per `heuristic.rs:217-219`.

## Unsupported requests

Requests AARON cannot fulfil (no intent rule matches) → `unknown` →
`answer_without_tool` (`agent.rs:413-436`) replies:

> "I did not understand that (heuristic 0% confidence). Try `aaron help` for
>  examples, or correct me and I will remember it."

| utterance | intent | refused? | message | useful? |
|---|---|---|---|---|
| book me a flight to paris | unknown | yes (no tool) | "I did not understand that …" | No — does not list what AARON can do |
| send an email to david@example.com | unknown | yes | same | No |
| what's the weather in london | unknown | yes | same | No |
| order a pizza | unknown | yes | same | No |
| call my mom | unknown | yes | same | No |
| set a timer for ten minutes | unknown | yes | same | No |

None is executed (no tool reaches the policy engine). The refusal is honest but
**not useful**: it never enumerates AARON's capabilities; the user must already
know to say `aaron help` (`help_text`, `agent.rs:964-986`, lists the real
commands). For 1.0 the unknown reply should name the supported commands or at
least point at `aaron help` more concretely.

## Greetings

| phrase | intent | greeted? | notes |
|---|---|---|---|
| hey | greeting | yes | `Hello. I am AARON.` |
| hi there | greeting | yes | `hi` matches (heuristic.rs:53) |
| hello aaron | greeting | yes | `hello` matches |
| good morning | greeting | yes | phrase rule |
| good evening | greeting | yes | phrase rule |
| greetings | greeting | yes | phrase rule |
| how are you | **unknown** | **NO** | falls to "I did not understand that …" |
| good morning (covered above) | — | yes | |
| morning | **unknown** | **NO** | not in the rule list |
| yo | **unknown** | **NO** | not in the rule list |

For a voice assistant, "how are you" and "good morning" are greetings users
will say; "how are you" and "morning"/"yo" fall through to "I did not
understand". The Greeting rule list (`heuristic.rs:50-58`) is missing "how are
you", "how are you doing", "morning", "yo", "hi aaron" etc.

## Commands that must be preserved: verified or broken

Run against the default config (empty `allowed_roots`, so filesystem tools are
sandbox-denied; `read_file`/`open_url`/`start_application`/`stop_application`
/`run_tests`/`remember` have no enabling policy entry and are deny-by-default,
`policy.rs:315-320`).

| utterance | intent | outcome | truthful about what happened? |
|---|---|---|---|
| recall my name | recall | executed; "From memory: Nothing in memory matches 'my name'." | Yes (memory empty) |
| recall my birthday | recall | executed; "Nothing in memory matches 'my birthday'." | Yes |
| remember my name is David | remember | **denied**: `tool \`remember\` has no entry in permissions.toml` | Yes — honestly reports the policy block |
| search for the file named main.rs | **unknown** | "I did not understand that …" | **No — a real command the brief says to preserve is not recognised** |
| read main.rs | **unknown** | "I did not understand that …" | **No — same gap** |
| check system status | system_status | executed; printed real system status | Yes |
| show git status | git_status | **sandbox-failed**: `no filesystem roots are configured` | Yes — honest about the sandbox block |
| run the tests | run_tests | **denied**: `tool \`run_tests\` has no entry …` | Yes — honest policy block |
| launch the terminal | start_application | **denied**: no policy entry | Yes — honest policy block |
| kill the browser | stop_application | **denied**: no policy entry | Yes — honest policy block |
| open https://example.com | open_url | **denied**: `network access is disabled (allow_network = false)` | Yes — honest, and correct (network off) |
| cancel | cancel | answered; "Cancelled. Nothing else will run." | Yes |
| what do you remember about my dog | recall | executed; "Nothing in memory matches 'my dog'." | Yes |

Two preservation gaps: `read main.rs` and `search for the file named main.rs`
match no ReadFile/SearchFiles alternative (`heuristic.rs:124-135` for ReadFile,
`heuristic.rs:111-122` for SearchFiles), so they fall to `unknown`. The router's ReadFile triggers require "read file"/"read the
file"/"open file" etc.; "read <path>" is not covered. SearchFiles triggers
require "find file"/"search for files"/"where is the file" etc.; "search for
the file named …" is not covered.

## Missing-argument clarification: what the user is asked

| utterance | intent | readiness | what AARON asks / does | concise? names the gap? | placeholder dispatched? |
|---|---|---|---|---|---|
| read the file | read_file | needs_clarification | `I understood \`read_file\` but not which file to read. Try naming it explicitly.` | Yes | No |
| launch the app | start_application | needs_clarification | `… but not which application. Try naming it explicitly.` | Yes | No |
| search for the file | **unknown** | undetermined | `I did not understand that …` | N/A | No (but misses the obvious search intent) |
| remember that | remember | needs_clarification | `I understood \`remember\` but not what to remember. Try naming it explicitly.` | Yes | No |

The clarification path is `agent.rs:244-267` and `agent.rs:478-480`; the
refusal messages come from `need`/`need_referent`/`need_content` in
`build_arguments` (`agent.rs:775-828`). The historical placeholder-dispatch bug
(the tool ran with `"the file"`) is fixed: `is_generic_referent`
(`text.rs:121-147`) rejects `"the file"`/`"the app"`/`"it"` before dispatch,
and `is_contentless` (`text.rs:231-238`) rejects `"that"`. So **no case
dispatches a placeholder argument**; each incomplete request is refused with a
message that names what is missing. The one gap is "search for the file", which
matches no rule and returns `unknown` instead of a search clarification.

## References ("it") from conversation context

There is **no "it" resolution from conversation context** in either the router
or the agent:

- The router is stateless per utterance (`heuristic.rs:206` — pure function of
  the string; no context parameter).
- The agent's `ConversationContext` ring is only **written** (`agent.rs:349`,
  `self.context.push(...)`); it is never read to resolve a pronoun. Its only
  documented purpose is resolving "run it again" (`agent.rs:169-170` comment),
  and even that consumption is not wired into classification.
- "it" is a bare pronoun (`text.rs:76-96`, `BARE_PRONOUNS`). As an argument it
  fails `is_generic_referent` (`text.rs:127`) and `is_contentless`
  (`text.rs:231`), so the argument builder refuses with "not which X / not what
  to remember" (`agent.rs:806-814`, `820-828`) — it **asks, never guesses**.

Live results:

| utterance (after "remember my dog is a border collie named Max") | intent | outcome |
|---|---|---|
| what about it | unknown | "I did not understand that …" (no recall of the dog fact) |
| read it | unknown | "I did not understand that …" |
| open it | unknown | "I did not understand that …" |
| launch it | start_application | needs_clarification → "not which application" (asks) |

So the referent is not resolved at all — not "only when unambiguous", but never.
"it" never causes a guess; it produces `unknown` or a clarification refusal.
This satisfies "do not guess" but fails "resolve references such as 'it' from
conversation context when the referent is unambiguous".

## High confidence is not authorisation

Confidence is never consulted at dispatch time. In `agent::turn()`
(`agent.rs:371-438`) the classified `Command` is unpacked (`intent`,
`entities`, `confidence`, `source`, `readiness`) but `confidence` is only stored
in the response (`agent.rs:653`) and used in the `unknown` reply wording
(`agent.rs:429-433`, `confidence * 100`). No gate compares it against anything.
The chain that can still refuse a confident prediction:

| gate | exact lines | what it refuses |
|---|---|---|
| Readiness one-way valve | `agent.rs:478-480`, `readiness.rs:220-239` | A request whose required argument is missing or a generic referent/contentless — turns "proceed" into "ask"; can never turn "refuse" into "proceed" |
| Argument validator | `agent.rs:766-902` (`build_arguments`); `need` `775-790`, `need_referent` `806-814`, `need_content` `820-828`; `entities.validate` called at `agent.rs:188` | Missing argument ("which file to read"), generic referent ("the file"), contentless ("that"), oversized/NUL entity, non-http URL — becomes a `Denied` refusal, never a dispatch |
| Loop guard | `agent.rs:489-497` (`guard.admit`) | Repeated identical actions beyond the limit → `Blocked` |
| Policy engine | `policy.rs:290-365` (`evaluate`); facts come from the **registry** (`agent.rs:506-512`), never the classifier | Root refusal (`292-296`), permanently-denied categories (`299-305`), network master switch (`308-312`), deny-by-default missing entry (`315-320`), explicit `deny` (`338-347`) |
| Confirmation gate | `agent.rs:530-574` | `NeedsConfirmation` → user confirmer (`541-544`); `NonInteractive` declines (`546-559`); decline → `Declined`, "Cancelled — I did not run it" (`560-572`) |

Proof: the `hard_negatives` suite (24/24 pass) asserts exactly this — e.g.
`a_low_confidence_prediction_does_not_run_a_tool` (`hard_negatives.rs:350`)
and `the_threshold_is_a_refusal_not_a_guess` (`hard_negatives.rs:390`) show a
classifier cannot force a tool; `the_policy_engine_is_not_reachable_from_the_classifier`
(`hard_negatives.rs:154`) and `a_deny_decision_is_never_overridden_by_a_permissive_classifier`
(`hard_negatives.rs:615`) show the policy engine ignores classifier output.
So a confident router decision **cannot by itself cause a tool to run** — but it
can cause the wrong tool to be *offered*, which is what the router mis-routes
above produce.

## The production router's regression results (the 42 cases)

First, the commanded `learning evaluate` of the 0.5 baseline **neural**
checkpoint was run:

```
AARON_HOME=/tmp/opencode/aaron-05/state XDG_CONFIG_HOME=/tmp/opencode/aaron-05/xdg \
cargo run --release -p aaron-app -- learning evaluate \
  --candidate /tmp/opencode/aaron-05/state/data/checkpoints/candidate-seed2690307353-e100.ckpt --split test
```

Result: `regression suite: 32/42 passed`; the 10 neural failures were:
`are you ok`→help, `are the systems nominal`→stop_application,
`any problems with the machine`→unknown, `do i have uncommitted changes`→unknown,
`nevermind`→unknown, `cancel the open url`→unknown,
`the mitochondria is the powerhouse of the cell`→stop_application,
`i finished reading that book last night`→git_status, `hello`→unknown,
`find *.py`→unknown. The candidate was REJECTED (macro-F1 0.7351 < 0.92,
accuracy 0.7333 < 0.93, false-accept 0.429 > 0.05, safety benchmark not scored).

That is the neural model. The **production path that ships** is the deterministic
router, run here against all 42 utterances with `aaron --text-only --json ask`
(threshold 0.60, `config/default.toml:49`). Expected vs actual intent:

| # | utterance | expected | router gave | PASS/FAIL |
|---|---|---|---|---|
| R-01 | are you ok | system_status | system_status | PASS |
| R-02 | how is the system | system_status | system_status | PASS |
| R-03 | system status | system_status | system_status | PASS |
| R-04 | are the systems nominal | system_status | **unknown** | FAIL |
| R-05 | any problems with the machine | system_status | **greeting** | FAIL (broad `hi` in "machine") |
| R-06 | do i have uncommitted changes | git_status | **unknown** | FAIL |
| R-07 | is the repository clean | git_status | **unknown** | FAIL |
| R-08 | remember that i drink tea | remember | remember | PASS* |
| R-09 | what do you remember about tea | recall | recall | PASS* |
| R-10 | do you remember my dog | recall | recall | PASS* |
| R-11 | make a note that i use a standing desk | remember | remember | PASS* |
| R-12 | search your memory for my address | recall | recall | PASS* |
| R-13 | open https://example.com | open_url | open_url | PASS** |
| R-14 | open the browser | start_application | **unknown** | FAIL |
| R-15 | launch the terminal | start_application | start_application | PASS** |
| R-16 | find files named main.rs | search_files | search_files | PASS** |
| R-17 | read the file cargo.toml | read_file | read_file | PASS** |
| R-18 | open readme.md | read_file | **unknown** | FAIL |
| R-19 | where is the dockerfile | search_files | **unknown** | FAIL |
| R-20 | cancel that | cancel | cancel | PASS |
| R-21 | stop that | cancel | cancel | PASS |
| R-22 | nevermind | cancel | cancel | PASS |
| R-23 | cancel the open url | cancel | cancel | PASS |
| R-24 | abort the test run | cancel | cancel | PASS |
| R-25 | the mitochondria is the powerhouse of the cell | unknown | unknown | PASS |
| R-26 | what is the capital of australia | unknown | unknown | PASS |
| R-27 | tell me a joke about penguins | unknown | unknown | PASS |
| R-28 | my cat knocked over the plant again | unknown | unknown | PASS |
| R-29 | the bus was late this morning | unknown | **greeting** | FAIL (broad `hi` in "this") |
| R-30 | i finished reading that book last night | unknown | unknown | PASS |
| R-31 | it has been a long week | unknown | unknown | PASS |
| R-32 | how do i bake sourdough bread | unknown | unknown | PASS |
| R-33 | hello | greeting | greeting | PASS |
| R-34 | good morning | greeting | greeting | PASS |
| R-35 | what can you do | help | help | PASS |
| R-36 | how do you work | help | help | PASS |
| R-37 | read the first 20 lines of dockerfile | read_file | read_file | PASS** |
| R-38 | show me the file main.rs | read_file | read_file | PASS** |
| R-39 | search for *.rs in the project | search_files | **unknown** | FAIL |
| R-40 | find *.py | search_files | **unknown** | FAIL |
| R-41 | run cargo test | run_tests | run_tests | PASS** |
| R-42 | run the test suite | run_tests | run_tests | PASS** |

\* Intent correct; execution blocked by policy (`remember`/`run_tests` have no
enabling policy entry) or sandbox (filesystem tools: no `allowed_roots`) — those
are configuration/policy outcomes, not router intent errors.
\*\* Intent correct; execution blocked by policy/sandbox as above.

**Router tally: 32/42 passed, 10 FAIL.** The router's failures are the ones
that matter for 1.0 because the router is what ships. Notably the router's 10
failures are *different* from the neural baseline's 10 failures, so neither
covers the other. False positives (actionable intent on an `unknown`-expected
case): R-05 and R-29, both from the broad `hi` substring rule. False negatives
(command expected, `unknown` got): R-04, R-06, R-07, R-14, R-18, R-19, R-39,
R-40.

## Broad rules that create false positives

| rule | matches | ordinary sentence that breaks it | suggested narrowing |
|---|---|---|---|
| Greeting `hi`, `heuristic.rs:53` | this, high, machine, him, hint, ohio | "the bus was late **this** morning" → greeting (R-29); "**high** is the mountain" → greeting; "**machine** status" → greeting (R-05) | Drop bare `hi`/`hey`; keep only multi-token greetings ("hello", "good morning", "greetings", "hi there" as a bigram) |
| Greeting `hey`, `heuristic.rs:54` | they | "they said it would rain" → greeting | Same as above |
| Help `help`, `heuristic.rs:65` | helping, helpless | "i can't help it" → help menu | Require "help" as a lead word or a phrase ("help me", "what can you do") |
| Remember `remember`, `heuristic.rs:106` | any "remember" | "i don't remember her name" → remember tool (dispatched, policy-denied) | Require an imperative/frame: "remember that …", "remember my …", "do you remember …" |
| ReadFile `read the first`, `heuristic.rs:133` | … | "read the first chapter" → read_file (dispatched, sandbox-failed) | Require a file-shaped token after "first" or keep only "read the first N lines of <file>" |
| StartApplication `launch`, `heuristic.rs:171` | single word | "launch a new project" → start_application (dispatched, policy-denied) | Require an application noun after "launch" ("launch the …", "launch <app>") |
| StopApplication `kill`, `heuristic.rs:185` | skill, skilled, killing | "he has a skill with computers" → stop_application (dispatched, policy-denied) | Require word boundaries / an application noun ("kill <app>", "stop the application") |
| OpenUrl required `http`, `heuristic.rs:163` | any `http…` token | "what does http mean" → open_url (asks for URL) | Require a full URL token, not a bare scheme word |

Suggested principle (matches the 1.0 brief): "Do not introduce broad substring
rules that create new false positives" — fix each regression case
individually with targeted, word-boundary or phrase rules rather than widening
existing alternatives.

## Defects, most severe first

| id | severity | what | file:line | how to confirm | suggested fix |
|----|----------|------|-----------|----------------|---------------|
| D1 | critical | Ordinary statement "how are things" matches `how are things` SystemStatus rule and **executes a real tool** | `heuristic.rs:82` ("how are things"), `agent.rs:585` (dispatch) | `aaron ask "how are things"` → executed system_status | Remove/narrow the phrase or gate SystemStatus behind confirmation; add regression test |
| D2 | high | Broad `hi`/`hey` greetings match inside `this`/`high`/`machine`/`they` → ordinary statements greeted | `heuristic.rs:53-54` | `aaron ask "the bus was late this morning"` → greeting | Replace bare `hi`/`hey` with bigram/phrase rules |
| D3 | high | Broad `help`/`remember`/`launch`/`kill`/`read the first`/`http` match inside ordinary sentences → wrong actionable intent, some dispatched then policy/sandbox-denied | `heuristic.rs:65,106,127,133,163,171,185` | `aaron ask "i can't help it"` → help; `aaron ask "launch a new project"` → start_application | Word-boundary/phrase narrowing per case |
| D4 | high | Genuine commands not recognised: "read main.rs", "search for the file named main.rs" → `unknown` | `heuristic.rs:124-135`, `heuristic.rs:111-122` | `aaron ask "read main.rs"` → unknown | Add "read <path>" / "search for the file named …" triggers |
| D5 | high | Greetings "how are you", "morning", "yo" → `unknown` / "I did not understand" | `heuristic.rs:50-58` | `aaron ask "how are you"` → unknown | Add to Greeting alternatives |
| D6 | medium | No "it" reference resolution from conversation context; context ring only written, never read | `agent.rs:169-170,349` | Two-turn: remember dog fact, then "what about it" → unknown | Implement unambiguous anaphora or document the limitation; add tests |
| D7 | medium | Unknown reply does not list what AARON can do | `agent.rs:426-435` | `aaron ask "book me a flight"` → "I did not understand …" (no capability list) | Name supported commands or point to `aaron help` |
| D8 | low | `Remember` rule `required: ["remember", "remember"]` is a duplicated entry | `heuristic.rs:106` | read source | Remove duplicate |
| D9 | low | Production threshold (0.60, `config/default.toml:49`) differs from `HeuristicClassifier::new(0.5)` default (`heuristic.rs:247`) — potential confusion | `config/default.toml:49`, `heuristic.rs:247` | compare | Document or unify |

## Missing tests that matter for 1.0

- A hard-negative/false-positive suite in `crates/aaron-app/tests/` (alongside
  `hard_negatives.rs`) asserting ordinary sentences stay `unknown`:
  "how are you", "yo", "morning", "the bus was late this morning", "how high is
  the mountain", "they said it would rain", "i can't help it", "i don't
  remember her name", "he has a skill with computers", "launch a new project",
  "read the first chapter", "what does http mean", "a bus stop".
- Regression tests for each of the 10 router regression failures (R-04, R-05,
  R-06, R-07, R-14, R-18, R-19, R-29, R-39, R-40) asserting expected intent.
- Preservation tests for "read main.rs" and "search for the file named main.rs"
  asserting ReadFile/SearchFiles (the 1.0 brief lists these explicitly).
- Clarification tests asserting "read the file", "launch the app",
  "remember that" return `NeedsClarification` with a message naming the missing
  argument, and that no placeholder argument is dispatched.
- Reference-resolution tests for "it" (asks, never guesses) once any anaphora
  is added; until then, a test asserting "read it" / "open it" do not dispatch
  a placeholder.
- A test that `aaron ask` on an `unknown` intent replies with a capability
  listing (or at least names `aaron help`), and that no tool runs for any
  `unknown`/`greeting`/`help` intent (`tool_for_intent` returns `None`,
  `agent.rs:757`).
