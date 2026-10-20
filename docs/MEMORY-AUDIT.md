# Memory audit — AARON 1.0

*Audit date: 2026-10-20. Scope: `crates/aaron-memory` (SQLite store), its callers
(`crates/aaron-agent/src/session.rs`, `crates/aaron-agent/src/agent.rs`,
`crates/aaron-tools/src/builtin/memory.rs`, `crates/aaron-app/src/commands.rs`,
`crates/aaron-app/src/context.rs`, `crates/aaron-learning`), the `[memory]` and
`[learning]` config, and `training/evaluation/regression.jsonl`. Method: full source
read plus black-box runs of the built `target/debug/aaron` against throwaway
installations under `/tmp/opencode` (state isolated with `AARON_HOME`/`HOME`;
the repository's config was used read-only). No repository file other than this
document was created or modified.*

## Verdict: one paragraph

The store itself is genuinely good: an explicit, transactional, private-by-default
SQLite layer with forward-only migrations, honest errors, bounded retrieval, and
persistence that I proved across separate processes with the real CLI. But the 1.0
memory requirement is only **partially met**. Three things the config and docs
promise do not exist or do not work: `memory.backup_on_start = true` is parsed and
never read (no backup ever runs on start), there is **no recovery path at all** —
a corrupt database makes every command including `aaron memory backup` and even
`aaron doctor` fail at startup, and `aaron memory restore` does not exist — and
`learning.redact_patterns` are applied only to the dataset file, never to memory
writes, so a user can store a secret verbatim in `memories` with `redacted = 0`.
On top of that, the shipped `config/permissions.toml` has `remember` commented out,
so the advertised "remember that … (asks first)" flow is *denied* on a fresh install
rather than asked; retention (`retention_days`, `max_entries`) only ever runs inside
the `remember` tool, so it is effectively inert for CLI-added and historical rows;
there is no reachable update path (the `update` API exists but no command or intent
calls it); duplicates are appended without dedup, documentation or tests; and not
one test covers corruption, an unwritable database, backup/restore, or redaction of
a memory write. The durable-vs-transient split is documented in four places and is
mostly enforced (the conversation ring is RAM-only and never even read back in
production), except that every executed tool turn implicitly writes its rendered
reply — file contents, recall results, test output — into `task_history` with no
confirmation, which contradicts the store's own "Nothing is stored implicitly"
design rule.

## What is implemented, what is not

| capability | status | evidence |
|---|---|---|
| Store & retrieve facts/preferences (CLI + tool) | **VERIFIED** | `crates/aaron-memory/src/store.rs:370-407,583-672`; `crates/aaron-tools/src/builtin/memory.rs:93,122`; CLI runs below (`memory add` → `memory list` → `memory search`) |
| Persistence across restarts (separate processes) | **VERIFIED** | My runs below; `crates/aaron-app/tests/cli.rs:234-255` (`memory_round_trips_across_processes`, passes); `crates/aaron-memory/src/store.rs:968-987` (`persists_across_restarts`, passes) |
| Database path resolution | **VERIFIED but misreported** | Resolved against the runtime state root, not the project root: `crates/aaron-app/src/context.rs:187-191` + `crates/aaron-core/src/paths.rs:42-53`; proved by run (file appeared at `$HOME/.local/state/aaron/data/memory.db`, repo `data/` stayed empty). But `aaron doctor` prints the **unresolved** config string `data/memory.db` (`crates/aaron-app/src/commands.rs:224`) |
| User confirmation before a durable fact is written | **PARTIALLY VERIFIED** | `remember` spec has `requires_confirmation: true` (`crates/aaron-tools/src/builtin/memory.rs:25`) and policy honours it (`crates/aaron-security/src/policy.rs:349-355`); proved live: non-TTY → `declined`, `--yes` → row written. **But** the shipped `config/permissions.toml:62-64` has `remember` commented out → deny-by-default, so the voice/typed flow is *refused* on a fresh install; and `aaron memory add` writes with no confirmation by design (`crates/aaron-app/src/commands.rs:711-719`) |
| Classifier/model output alone causing a durable write | **VERIFIED (it happens)** | `record_task` writes `task_history` after every executed or failed tool with the classifier's intent and the rendered reply, no confirmation: `crates/aaron-agent/src/agent.rs:612,644,665-678` → `crates/aaron-memory/src/store.rs:467-491`. Proved live: rows 1-4 in `task_history` after plain `aaron ask` |
| Raw transcript reaching durable storage | **VERIFIED (it does not, in production)** | The transcript reaches SQLite only via a *confirmed* `remember` or via `feedback` — and `record_correction` (`crates/aaron-agent/src/agent.rs:720-734`) has **no production caller** (grep: only agent test + `tests/security.rs:482`). `session.rs` itself performs no memory writes (`crates/aaron-agent/src/session.rs:80-124`; it only calls `agent.handle_text` at :117) |
| Durable facts vs transient ring, documented | **VERIFIED** | `crates/aaron-memory/src/lib.rs:3-5`; `store.rs:5-13`; `crates/aaron-resources/src/context.rs:1-13`; `crates/aaron-agent/src/agent.rs:168-171`; `docs/security.md:310-313` |
| Durable facts vs transient ring, enforced | **PARTIALLY VERIFIED** | Ring is RAM-only: `ConversationContext::render()` has **zero production callers** (grep over `crates/`), `Agent::context()` is called only from agent tests, nothing serialises `Turn`/`Summary` → no ring→disk path exists. Ring→model does not exist either (the ring is write-only in production). Disk→ring happens only through replies (`agent.rs:349-354`). **But** `task_history.detail` stores rendered tool output (up to 400 chars, `store.rs:485`) without confirmation — transient/durable boundary broken in that direction (`agent.rs:644`) |
| Duplicates | **PARTIALLY VERIFIED** | Same text added twice → two rows (`#3`, `#4`), no unique index (`schema.rs:18-32`), no dedup in `add` (`store.rs:370-407`), never documented, never tested. Behaviour is at least *consistent* (always append; `search` returns both; `delete` removes one) |
| Update | **NOT IMPLEMENTED** (user-reachable) | `MemoryStore::update` exists (`store.rs:417-430`) but has **no caller** anywhere outside its own unit test (grep `\.update(` across `crates/`); `aaron memory update` → `error: unrecognized subcommand 'update'`; no intent maps to it (`agent.rs:744-759`). Also `update` skips the empty-value and byte-budget checks that `add` enforces (`store.rs:378-388` vs `417-430`) |
| Deletion | **VERIFIED (one entry by id)** | `crates/aaron-app/src/commands.rs:689-709` + `main.rs:248-252`: exactly one row, by id, `--yes` required (proved live: without `--yes` exit 1, with it `Deleted #5.`). No delete-all or delete-by-tag exists; `MemoryStore::clear_kind` (`store.rs:819-829`) has no caller and its doc comment names a command that does not exist (`aaron memory clear --kind`) |
| Retrieval, ranking, bounded results | **VERIFIED** | Weighted, explainable scoring `store.rs:863-885`; `max_retrieved` cap enforced at `store.rs:661` — proved live: 12 matching rows, `--limit 20` returns exactly 10; wildcard injection refused (`store.rs:1046-1053` test) |
| Clear behaviour when a memory is absent | **PARTIALLY VERIFIED** | Recall trace: classifier → `Intent::Recall` (`agent.rs:755`) → `build_arguments` (`agent.rs:894-897`) → policy `allow` (`config/permissions.toml:42-46`) → `RecallTool` (`memory.rs:112-151`) → reply. Not-found is distinct and honest: `From memory:\nNothing in memory matches "dragons".` (live run; wording at `memory.rs:124-125`), prefixed by `From memory:` at `agent.rs:946`; a "found but empty" reply is impossible because `add` refuses empty values (`store.rs:379-381`, proved live). **Caveat:** search ORs all query terms and has no stop-word filter (`store.rs:645`, `crates/aaron-core/src/text.rs:280-286`), so a nonsense query can report a *false* "found" — `memory search "quokka the"` returned a real row (live) |
| Secrets not stored unnecessarily | **NOT IMPLEMENTED** | `learning.redact_patterns` (`config/default.toml:81-88`) are read only by the learning approval path (`crates/aaron-learning/src/lib.rs:283-286`); no memory write path redacts anything: `store.rs:402` hardcodes `redacted = false`, `memory.rs:93` and `commands.rs:716` pass text verbatim. Proved live: `my password is hunter2xyz` stored with `redacted = 0` via CLI, `my Api Key is Sk-1234567890` via the confirmed `remember` tool. Redaction, where it exists, runs **before the dataset write but after the memory write** — and approval then inserts the **raw** utterance into `memories` (`learning/lib.rs:309`) |
| Corruption reported honestly | **PARTIALLY VERIFIED** | Three corruption shapes all produced clear errors and left the file byte-identical (md5 unchanged): `file is not a database`, `malformed database schema`, `database disk image is malformed`, exit 1 (live runs below). Nothing is silently discarded or recreated. **But** there is no integrity check anywhere, no test covers it, and the failure bricks the whole CLI including `doctor` |
| Unavailable storage | **PARTIALLY VERIFIED** | Live: unreadable file → `storage error: unable to open database file: …` exit 1; read-only data dir → `attempt to write a readonly database` exit 1; parent path blocked by a file → `i/o error … File exists` exit 1. Honest, but `remediation()` has **no arm for `Error::Storage`** (`crates/aaron-app/src/context.rs:462-480`), so the user gets no hint; disk-full was not simulated (see Risks) |
| Backup | **PARTIALLY VERIFIED** | `aaron memory backup` works once (`VACUUM INTO`, `store.rs:807-817`) and the snapshot contains **all** tables (proved: `memories 4`, `task_history 4`, `schema_version 2`). Store-level snapshot test passes (`store.rs:1137-1152`). **But** a second run to the same destination fails (`output file already exists`), there is **no `restore` command** (`unrecognized subcommand 'restore'`, exit 2), and backup fails exactly when needed (corrupt DB) because the store is opened before the backup |
| `backup_on_start` | **NOT IMPLEMENTED** | Defined only: `crates/aaron-core/src/config.rs:755,766`, `config/default.toml:71`. Whole-repo grep finds no reader. Proved live: after several commands with `backup_on_start = true`, `data/` contained only `memory.db` — no `memory-backup.db` |
| Retention (`retention_days`, `max_entries`) | **PARTIALLY VERIFIED** | Wired from config (`context.rs:192-195`) and correct in the store (`store.rs:767-804`, tested at `store.rs:1070-1134`). **But** the only production call to `prune()` is inside the `remember` tool (`memory.rs:94`; the other caller is the stress command's throwaway DB, `resources_cmd.rs:254`). CLI `memory add` does not prune, and nothing prunes at start-up, so retention never runs for CLI rows or historical data |
| Connection lifecycle | **VERIFIED (eager, single connection, not pooled)** | Not lazy: `AppContext::load` opens + migrates the DB for *every* command (`context.rs:200-208`) — proved live: a fresh `aaron tools list` created `data/memory.db`. One `Connection` per process held in `Arc<Mutex<…>>` (`store.rs:148-154`, `context.rs:110`), never reopened per operation. `open_with_limits` (which opens and immediately drops a second connection, `store.rs:178-190`) has **zero callers** |
| Test coverage of the above | **PARTIALLY VERIFIED** | See "Missing tests". Present: 21 store tests (pass), 4 tool tests, agent round-trip, 2 CLI tests (pass), 2 injection tests, 1 dataset-redaction test. Absent: corruption, unwritable/full storage, backup CLI/restore, duplicate semantics, memory-write redaction, remember-confirmation gate, retention-from-config, task-history privacy |

### Q1 — every write path into the durable store

| # | what writes | caller | confirmation required? |
|---|---|---|---|
| 1 | `memories` row, `source = "user"` | `RememberTool` (`crates/aaron-tools/src/builtin/memory.rs:93`) via `Intent::Remember` (`agent.rs:754`), typed (`aaron ask`/`run`) or spoken (`session.rs:117`) | **Yes** — `requires_confirmation: true` (`memory.rs:25`) + policy (`policy.rs:349-355`); non-interactive sessions decline (`agent.rs:546-559`). **Except** `--yes`/`AutoConfirm`, and on the *shipped* permissions file the tool is denied outright (`config/permissions.toml:62-64`) |
| 2 | `memories` row, `source = "cli"` | `aaron memory add` (`crates/aaron-app/src/commands.rs:716`) | No confirmation beyond typing the command. Note the asymmetry: `memory delete` demands `--yes`, `memory add` does not |
| 3 | `memories` row, `source = "learning"` | `aaron learning approve` → `crates/aaron-learning/src/lib.rs:305-312` | Yes — explicit CLI review action (but writes the **raw** utterance, defect M-08) |
| 4 | `task_history` row | `Agent::record_task` after every executed/failed tool (`agent.rs:612,644,665-678`) | **No** — fires on any classified utterance that reaches an allowed tool (proved live with plain `aaron ask`) |
| 5 | `feedback` row | `Agent::record_correction` (`agent.rs:732-733`) | Would be an explicit user correction — but **no production caller exists** (grep), so the queue is unreachable and `aaron learning pending` can only ever print "No corrections waiting" (asserted by `tests/cli.rs:540`) |
| 6 | `memories` row, `source = "stress"` | `aaron resources stress` (`resources_cmd.rs:246`) | n/a — deliberately written to a **throwaway** temp DB and deleted afterwards (`resources_cmd.rs:241-258`); this is correct and worth praising |
| 7 | dead API | `add_shareable`, `update`, `set_pinned`, `set_private`, `clear_kind`, `record_audit_meta`, `open_session`, `close_session` | no production callers at all (grep over `crates/`), only unit tests |

So: **writing a fact requires confirmation** (when it is possible at all), but a
*recognised-speech* utterance can still cause a durable SQLite write without one,
through `task_history` (row 4), and `aaron memory add` bypasses confirmation by
construction (row 2, explicit CLI action). No path lets a model/classifier output
*alone* create a `memories` fact — facts only come from the `remember` tool, the CLI,
or an explicit learning approval — which is the important safety property and it
holds.

## Defects, most severe first

| id | severity | what | file:line | how to confirm | suggested fix |
|---|---|---|---|---|---|
| M-01 | **High** | `backup_on_start = true` is parsed and never read; no backup ever runs on start-up. Config promises a behaviour that does not exist. | `crates/aaron-core/src/config.rs:755,766`; `config/default.toml:71`; grep `backup_on_start` → 3 hits, all definitions | `AARON_HOME=/tmp/x ./target/debug/aaron --config config/default.toml memory add "t"; ls /tmp/x/data/` → no `memory-backup.db` | Implement it in `AppContext::load` (non-fatal on failure, rotated destination), or delete the key |
| M-02 | **High** | No recovery path for a corrupt/unavailable database. The store is opened eagerly in `load`, so **every** command — including `aaron memory backup` (the recovery tool) and `aaron doctor` (the diagnostic tool) — fails with the raw SQLite error; `aaron memory restore` does not exist; `remediation()` gives no hint for storage errors. | `crates/aaron-app/src/context.rs:200-208,462-480`; `crates/aaron-app/src/commands.rs:721-731`; `main.rs:230-268` | `printf 'garbage' > $AARON_HOME/data/memory.db; aaron memory backup` → `storage error: file is not a database`, exit 1; `aaron doctor` → same; `aaron memory restore x` → `unrecognized subcommand`, exit 2 | Open lazily per command so `backup`/`doctor` survive; add `memory restore <file>`; add a storage-error hint ("your database at <path> is corrupt; copy a backup over it") |
| M-03 | **High** | Secrets are stored verbatim in durable memory. `learning.redact_patterns` are applied only to the dataset copy; no memory write path redacts; the `redacted` column is hardcoded `false`. A user's password/API key lands in `memories` in clear text. | `crates/aaron-memory/src/store.rs:393-402`; `crates/aaron-tools/src/builtin/memory.rs:93`; `crates/aaron-app/src/commands.rs:716`; patterns used only at `crates/aaron-learning/src/lib.rs:283-286` | `aaron --yes memory add "my password is hunter2xyz"; sqlite3 $AARON_HOME/data/memory.db 'select redacted,value from memories where value like "%hunter2%"'` → `0\|my password is hunter2xyz` (ran: exactly that) | Redact in `MemoryStore::add` (or before it in both callers) using `learning.redact_patterns` + `text::redact_secrets`, and set `redacted = 1`; refuse or warn on patterns like `api_key` |
| M-04 | **High** | On a fresh install the advertised voice/typed `remember` flow is **denied**, not asked: `[tools.remember]` is commented out in the shipped permissions file, and deny-by-default refuses it. `aaron help` still promises "remember that \<fact\> — store a fact (asks first)". | `config/permissions.toml:62-64`; `crates/aaron-agent/src/agent.rs:975`; policy `crates/aaron-security/src/policy.rs:314-320` | `aaron --text-only --json ask "remember that I drink tea"` → `"outcome":"denied"`, `tool \`remember\` has no entry in permissions.toml; deny-by-default applies` (ran: exactly that) | Ship `[tools.remember] effect = "ask"` enabled (the file's own comment says "ask is the right setting for anything irreversible"), or change the help text to say memory writes are disabled until enabled |
| M-05 | **Medium** | Every executed tool turn writes its **rendered reply** (file contents, recall results, test output) into `task_history` with no confirmation and no retention — contradicting the store's own "Nothing is stored implicitly / Transcripts are never written" rule and `docs/security.md:312`. `task_history` is never pruned and is invisible to `doctor`'s row count. | `crates/aaron-agent/src/agent.rs:643-644,665-678`; `crates/aaron-memory/src/store.rs:467-491,767-804,743-754`; claim `store.rs:5-6` | `aaron --text-only ask "check system status"; sqlite3 … 'select detail from task_history'` → reply text present (ran: rows 1-4) | Store intent/tool/outcome/latency only, or gate detail behind an explicit flag; apply retention to `task_history`; count it in `stats()` |
| M-06 | **Medium** | Learning approval sanitises the utterance for the **dataset** but then inserts the **raw** `item.utterance` into `memories` (`"<raw> -> <corrected>"`), so a redacted secret reappears in durable storage. The `feedback` table likewise stores the raw utterance, keeping only a boolean flag (`record_correction` computes `redacted` then stores the original). Production impact is currently limited because `record_correction` has no caller (M-09). | `crates/aaron-learning/src/lib.rs:283-286` (sanitised `text`) vs `:305-312` (raw `item.utterance` at :309); `crates/aaron-agent/src/agent.rs:732-733`; `crates/aaron-memory/src/store.rs:494-509` | Existing test `a_correction_containing_a_secret_is_redacted_before_storing` (`learning/lib.rs:965-989`) asserts the **dataset** only; run `sqlite3 … 'select value from memories where kind="feedback"'` after an approval to see the raw text | Write the sanitised `text` into `memories` (and into `feedback` at insert time); keep the raw form only if a human-review requirement is documented |
| M-07 | **Medium** | Retention is effectively inert: `prune()` runs only inside the `remember` tool. CLI-added rows, learning-added rows and all historical rows are never aged out — `retention_days`/`max_entries` from config almost never act. | only production caller `crates/aaron-tools/src/builtin/memory.rs:94` (other: `resources_cmd.rs:254`, throwaway DB); wiring `crates/aaron-app/src/context.rs:192-195` | Add rows via `aaron memory add`, backdate `created_at` with `sqlite3`, run `aaron memory list` → rows survive; only a later `remember` tool call prunes | Prune at start-up (cheap: one indexed DELETE in a transaction) and/or in `memory_add` |
| M-08 | **Medium** | The correction workflow the assistant advertises ("correct me and I will remember it") is unreachable: `record_correction` has no production caller — no CLI subcommand, no tool, no intent. The `feedback` table can never be populated, so `aaron learning pending/approve/reject` operate on an always-empty queue. | `crates/aaron-agent/src/agent.rs:720-734` (callers: agent test :1318, `tests/security.rs:482` only); claim text `agent.rs:429-430`; `main.rs:270-284` has no correction entry | `grep -rn "record_correction(" crates --include='*.rs'` → definition + 2 test sites; `aaron learning pending` → "No corrections waiting for review." | Add the `aaron correct "<utterance>" <intent>` subcommand (or an intent/tool) that calls `record_correction`, and an integration test that queues then approves |
| M-09 | **Medium** | `aaron memory list` and `aaron memory search` disagree about which rows exist: `list` (no `--kind`) filters to `user_visible` kinds, `search` has **no kind filter**, and `memory add --kind …` accepts *any* kind. Result: rows are searchable and deletable by id but invisible in the default listing — and non-visible kinds also escape retention. | `store.rs:599-607` vs `store.rs:643-647`; `commands.rs:712-716`; prune kinds `store.rs:775-777` | Ran: `memory add "hidden kind probe row" --kind feedback` → `memory list` shows nothing; `memory search hidden` → `#5 feedback hidden kind probe row` | Either restrict `memory add --kind` to user-visible kinds (and say so), or make `list` show every row, or filter `search` the same way `list` does — and make retention cover whatever the CLI can create |
| M-10 | **Medium** | `aaron doctor` reports `database  data/memory.db` — the raw config string — while the file actually lives in the runtime state root. With a source checkout, `config/default.toml` sits beside an empty `data/.gitkeep`, and the doctor line, both point users at `./data/memory.db`, which is never used. | `crates/aaron-app/src/commands.rs:224`; resolution `context.rs:187-191`; `paths.rs:42-53` | `aaron doctor \| grep -A3 ^memory` → `database data/memory.db`; `find $AARON_HOME -name memory.db` → `$AARON_HOME/data/memory.db` (ran: repo `data/` had only `.gitkeep`) | Print `store.path()` (the resolved path), and document the state-dir resolution in `config/default.toml` next to the key |
| M-11 | **Medium** | No reachable update path, and the dormant API is weaker than `add`: `update` skips the empty-value refusal and the `max_value_bytes` check. Nothing can pin/unpin (`set_pinned` uncalled) either, so the "(pinned)" marker in listings is unreachable and `pinned` rows can never exist. | `store.rs:417-430` (vs `370-388`), `store.rs:442-454`; no CLI: `main.rs:230-268`; no intent: `agent.rs:744-759`; grep `set_pinned\|\.update(` → tests only | `aaron memory update 1 x` → `unrecognized subcommand` (ran) | Either expose `aaron memory update <id> <text>` (with the same validation as `add`) and drop `clear_kind`'s phantom command reference, or delete the unused API so the surface matches reality |
| M-12 | **Medium** | `aaron memory backup` is not repeatable: `VACUUM INTO` refuses an existing destination, so the second run (the routine case) fails with an error; no rotation, no overwrite. | `store.rs:807-817`; default dest `commands.rs:726` | Ran twice: 1st `Backed up memory to …`, 2nd `storage error: backup to … failed: output file already exists` | Remove/rotate the destination (timestamped files, keep N) or refuse *before* opening with a clear message |
| M-13 | **Low** | Recall false positives undermine "clear behaviour when absent": search ORs all query terms (`text::words` keeps stop-words), so a nonsense query containing a common word reports hits. | `store.rs:643-647` (`clauses.join(" OR ")`); `crates/aaron-core/src/text.rs:280-286` | Ran: `aaron memory search "quokka the"` → returned `#1 the audit persistence probe fact`; control `"quokka zzyzx"` → `Nothing in memory matches` | Require *all* terms (AND) or drop stop-words; keep OR only as a documented fallback |
| M-14 | **Low** | `aaron memory search --limit 20` silently returns at most `max_retrieved_memories` (10) rows; the cap is correct resource behaviour but the CLI never says results were clipped, while `memory list --limit` is not capped the same way. | `store.rs:661`; `config/default.toml:164`; `commands.rs:670-687` | Ran: 12 matching rows, `memory search captest --limit 20` → exactly 10 lines | Print `[showing 10 of 20 requested; resources.memory.max_retrieved_memories = 10]` |
| M-15 | **Low** | `clear_kind`'s doc comment documents a command that does not exist (`aaron memory clear --kind`); `help_text` also promises a correction flow that is unreachable (M-08). Doc/API drift misleads maintainers and users. | `store.rs:819`; `agent.rs:975,429-430`; `main.rs:230-268` | `aaron memory clear --kind fact` → `unrecognized subcommand` | Fix the comment or ship the command |
| M-16 | **Low** | `MemoryStore::checkpoint()` (the WAL truncation `docs/resources.md:219` cites as the reason WAL growth is bounded) has **no production caller**; only SQLite's default auto-checkpoint runs. `on_disk_bytes` is reported but never acted on. | `store.rs:208-213`; `docs/resources.md:219`; grep callers → none | `grep -rn "checkpoint()" crates --include='*.rs'` (memory checkpoint) → test only | Call `checkpoint()` on start-up or after `prune`, or correct the doc |
| M-17 | **Low** | `RememberTool` does `add` then `prune?` in sequence: if `prune` fails (disk full, lock), the user is told "That did not work" although the fact **was** stored. Conversely `learning`'s `let _ = store.add(...)` swallows write errors silently. | `memory.rs:93-94` + error render `agent.rs:610-618`; `learning/lib.rs:306` | Simulate a failing prune (read-only mid-run) or read the code path | Report the write as stored and log the prune failure separately; surface learning-write failures |
| M-18 | **Low** | Dead schema/API surface: `sessions` and `audit_meta` tables are written only by never-called methods; `open_with_limits` (which opens and drops a pointless second connection) has no callers. Harmless today, but the schema promises features that do not exist. | `store.rs:178-190,527-562`; `schema.rs:34-39,72-79` | grep `open_session\|record_audit_meta\|open_with_limits` → store + tests only | Remove, or wire up and test |

## Persistence evidence (the commands I ran)

All runs used `./target/debug/aaron` (fresh `cargo build --bin aaron`, 0 errors)
with state isolated under `/tmp/opencode`.

**1. Write in one process, read in two separate processes (Q3).**

```
$ export AARON_HOME=/tmp/opencode/mem-audit-home
$ ./target/debug/aaron --config /home/david/proj/Aaron/config/default.toml \
      memory add "the audit persistence probe fact" --key auditprobe
Stored memory #1.
$ ./target/debug/aaron --config …/config/default.toml memory list
#1     fact            auditprobe           the audit persistence probe fact
$ ./target/debug/aaron --config …/config/default.toml memory search "persistence"
#1     fact            the audit persistence probe fact
$ find $AARON_HOME -name '*.db*'
/tmp/opencode/mem-audit-home/data/memory.db   (57344 bytes)
```

Each command is its own OS process; the row survived. The integration test
`crates/aaron-app/tests/cli.rs:234-255` asserts the same thing and passes
(`cargo test -p aaron-app --test cli memory` → `2 passed`).

**2. Where the relative config path actually lands (Q3).**

```
$ unset AARON_HOME XDG_STATE_HOME; export HOME=/tmp/opencode/mem-audit-home2
$ ./target/debug/aaron --config /home/david/proj/Aaron/config/default.toml \
      memory add "landed in the default state dir"
Stored memory #1.
$ find $HOME -name memory.db
/tmp/opencode/mem-audit-home2/.local/state/aaron/data/memory.db
$ ls -la /home/david/proj/Aaron/data/
total 8   …  .gitkeep 0 bytes
```

So `database_path = "data/memory.db"` (`config/default.toml:68`) is resolved against
the **runtime state directory** (`$AARON_HOME` → `$XDG_STATE_HOME/aaron` →
`~/.local/state/aaron`, `crates/aaron-core/src/paths.rs:42-53`, applied at
`context.rs:187-191`), **not** the project root and not the CWD. Whether a user
would expect that: partially no. The "one runtime directory" design is documented in
code (`paths.rs:3-9`) and `docs/troubleshooting.md:491` shows `~/.local/state/aaron`
for the audit log, but a source checkout ships an empty `data/` directory, the config
key is a bare relative path, and `aaron doctor` prints `database  data/memory.db`
unexpanded — three signals that say `./data/memory.db`. The behaviour is defensible;
the reporting is misleading (M-10).

**3. Confirmation on the remember path (Q1).**

```
# shipped permissions (remember commented out):
$ aaron --config config/default.toml --text-only --json ask "remember that I drink tea"
{"intent":"remember","outcome":"denied","tool":"remember",
 "denied_reason":"tool `remember` has no entry in permissions.toml; deny-by-default applies", …}

# with [tools.remember] effect = "ask", no --yes, no TTY:
{"intent":"remember","outcome":"declined","reply":"Cancelled — I did not run it.", …}

# with --yes:
{"intent":"remember","outcome":"executed","reply":"Remembered (id 2): i Drink Tea", …}
$ aaron memory list
#2     fact            -                    i Drink Tea
```

So: an utterance **cannot** write a fact without confirmation in an interactive or
non-interactive session; `--yes` (an explicit user flag) is the only bypass; and the
shipped config refuses the flow entirely.

**4. Absence (Q5).**

```
$ aaron --config … --text-only ask "what do you remember about dragons"
From memory:
Nothing in memory matches "dragons".
```
Trace: `agent.rs:384` classify → `Intent::Recall` → `agent.rs:894-897` build `query`
→ `policy` allow (`permissions.toml:42-46`) → `memory.rs:112-151` →
`memory.rs:124-125` not-found wording → `agent.rs:946` prefix. The not-found reply is
distinct from a hit (`#id key: value [tags]`); a "found but empty" reply cannot occur
because `add` refuses whitespace (`store.rs:379-381`) — reproduced:
`aaron memory add "   "` → `storage error: refusing to store an empty memory`, exit 1.
Caveat: false positives from OR/stop-word matching (M-13).

**5. Duplicates, delete, limits (Q4).**

```
$ aaron memory add "duplicate probe fact"   → Stored memory #3.
$ aaron memory add "duplicate probe fact"   → Stored memory #4.   # no dedup
$ aaron memory list
#4  fact  -  duplicate probe fact
#3  fact  -  duplicate probe fact
…
$ aaron memory delete 5                      → "Re-run with --yes to confirm."  exit=1
$ aaron --yes memory delete 5                → "Deleted #5."                    exit=0
$ aaron --yes memory delete 999              → "No memory with id 999."         exit=1
$ aaron memory add "hidden kind probe row" --kind feedback
$ aaron memory list            → (row absent)
$ aaron memory search hidden   → #5  feedback  hidden kind probe row     # M-09
$ aaron memory search captest --limit 20   → 10 of 12 rows               # M-14
$ aaron memory update 1 x      → error: unrecognized subcommand          # M-12/M-11
$ aaron memory restore x       → error: unrecognized subcommand          # M-02
```

**6. Secrets (Q6).**

```
$ aaron --yes memory add "my password is hunter2xyz"
$ sqlite3 $AARON_HOME/data/memory.db "select id,kind,redacted,value from memories where value like '%hunter2%'"
6|fact|0|my password is hunter2xyz
$ aaron --yes --text-only ask "remember that my api key is sk-1234567890"
$ sqlite3 … "select id,kind,redacted,source,value from memories order by id desc limit 1"
7|fact|0|user|my Api Key is Sk-1234567890
```
Verbatim, `redacted = 0`, through both write paths. `learning.redact_patterns` never
run on these writes; they run only inside `aaron learning approve`
(`learning/lib.rs:283-286`) and only for the dataset file.

**7. Corruption and unavailability (Q7).**

```
# (a) garbage bytes where the DB should be
$ printf 'this is not a sqlite database at all…' > $AARON_HOME/data/memory.db
$ aaron memory list
aaron: storage error: file is not a database            exit=1
$ ls -la $AARON_HOME/data/memory.db     → still 61 bytes of the original garbage
                                          (nothing recreated, nothing discarded)

# (b) valid header, corrupted schema page  (bytes inside the error are invalid UTF-8)
aaron: storage error: malformed database schema (on�CREATE�TABL)                        exit=1

# (c) corrupted b-tree pages
aaron: storage error: database disk image is malformed                                    exit=1
# md5 of the file before and after both attempts: identical
ed3b5ec38e9d115dc7d6f5f2dccb496f  (unchanged)

# (d) unreadable file (chmod 000)
aaron: storage error: unable to open database file: …/data/memory.db                      exit=1

# (e) read-only data directory
aaron: storage error: attempt to write a readonly database                                exit=1

# (f) state root blocked by a regular file
aaron: i/o error at …/data: File exists (os error 17)                                     exit=1

# (g) the recovery tool against the corrupt DB
$ aaron memory backup
aaron: storage error: file is not a database            exit=1     ← cannot back up
$ aaron doctor
aaron: storage error: file is not a database            exit=1     ← cannot diagnose
```

So: corruption is reported **honestly and non-destructively** (this is genuinely
correct), but there is no recovery path, no hint, and no test (M-02).

**8. Backup contents and `backup_on_start` (Q7/Q8).**

```
$ aaron memory backup
Backed up memory to /tmp/opencode/mem-audit-home/data/memory-backup.db      exit=0
$ aaron memory backup          # again, same destination
aaron: storage error: backup to …/memory-backup.db failed: output file already exists   exit=1
$ sqlite3 data/memory-backup.db "select count(*) from memories; select count(*) from task_history; select max(version) from schema_version;"
4
4
2
$ rm data/memory-backup.db; aaron --text-only ask "check system status" >/dev/null; ls data/
memory.db                     # backup_on_start = true → no backup created   (M-01)
```

**9. Eager open (Q8).**

```
$ AARON_HOME=/tmp/opencode/mem-audit-eager aaron --config config/default.toml tools list
$ find /tmp/opencode/mem-audit-eager -name memory.db
/tmp/opencode/mem-audit-eager/data/memory.db        # a read-only command created the DB
```
The store is opened (and migrated) once per process in `AppContext::load`
(`context.rs:200-208`) — not lazy, not pooled, never reopened per operation. That is
cheap (migrations are idempotent and indexed by `schema_version`), but it is why a
corrupt DB bricks every command (M-02).

**10. Tests actually run.**

```
$ cargo test -p aaron-memory                     → 21 passed; 0 failed
$ cargo test -p aaron-app --test cli memory      → 2 passed; 0 failed
```

## Missing tests that matter for 1.0

Ranked by 1.0 risk. **All of them can be written without a microphone and without a
model** (`--text-only` heuristic router, `MemoryStore::open_in_memory`, and the
existing `Install` harness in `crates/aaron-app/tests/cli.rs:22-52`).

1. **Corrupt database is reported, not discarded** — write garbage / flip pages in a
   temp `memory.db`, assert `aaron memory list` exits 1 with a storage error, the
   file is byte-identical afterwards, and no fresh DB is created. (Q7; today only my
   manual runs cover this.) *No mic/model.*
2. **Recovery works** — `memory backup` → destroy the live file → restore (today
   there is no restore command, so this test would first expose M-02) → data intact
   in a new process. Also: second consecutive `memory backup` must succeed (M-12).
   *No mic/model.*
3. **Unwritable/unreadable storage** — `chmod 000` file, read-only directory: assert
   honest error + non-zero exit, and that `doctor`/`backup` still behave sensibly.
   *No mic/model.*
4. **Redaction of memory writes** — `aaron memory add "password=hunter2"` and a
   confirmed `remember` with a secret: assert the stored value is redacted (or at
   minimum `redacted = 1`). The existing secret test
   (`crates/aaron-learning/src/lib.rs:965`) asserts only the dataset file. *No mic/model.*
5. **`remember` needs confirmation end-to-end** — with an `AlwaysDecline` confirmer
   assert no row is created; with `AutoConfirm` assert exactly one; with the *shipped*
   `permissions.toml` assert the documented behaviour (today: denial — the test should
   pin whichever behaviour is intended, M-04). The current confirmation tests use
   `system_status`, never `remember`. *No mic/model.*
6. **Duplicate policy** — add the same fact twice, assert the chosen behaviour
   (two rows is fine) *and* that `recall` reports it coherently and `delete <id>`
   removes exactly one. Currently no test at all. *No mic/model.*
7. **Retention actually runs from config** — insert a backdated row, run a real
   command with `retention_days` set, assert the row is gone (today it would fail:
   M-07, prune only runs inside the `remember` tool). *No mic/model.*
8. **`task_history` privacy** — execute `read_file`/`recall`, then assert the file
   contents / remembered values are **absent** from `task_history.detail` (or that the
   table obeys retention). This is the direct test for the "Nothing is stored
   implicitly" claim (M-05). *No mic/model.*
9. **Recall absence wording at the agent level** — absent query → reply contains
   `Nothing in memory matches`; a found fact → reply does not; plus a negative test
   that a stop-word query (`"quokka the"`) returns nothing (M-13). Tool-level wording
   is tested (`memory.rs:216`), the agent-level reply is not. *No mic/model.*
10. **`doctor` reports the resolved database path** — assert the printed path equals
    the file that actually exists (M-10). *No mic/model.*
11. **`backup_on_start`** — once implemented, assert a backup file exists after any
    command start; as written today this test would fail immediately, which is the
    point (M-01). *No mic/model.*
12. **List/search consistency** — every kind creatable via `memory add --kind` must
    appear in default `memory list` (or creation must be refused) (M-09).
    *No mic/model.*

Covered already, so no new test needed: cross-process persistence (`tests/cli.rs:235`),
delete confirmation (`tests/cli.rs:258`), private-by-default export exclusion
(`store.rs:1056`, `memory.rs:225`), empty-value refusal (`store.rs:990`), retrieval
cap (`store.rs:1194`), schema forward-compat refusal (`store.rs:948`), retention
scoping (`store.rs:1070,1100,1116`), snapshot readability (`store.rs:1137`), memory
content cannot instruct the agent (`tests/security.rs:479`,
`tests/hard_negatives.rs:282`).

## The 42-case regression suite (Q10)

`training/evaluation/regression.jsonl` has 42 cases, of which **5 are memory-related**:

```
remember-7b225c5c  "remember that i drink tea"                  intent remember
remember-4c24836b  "make a note that i use a standing desk"     intent remember
recall-e9de6b21    "what do you remember about tea"             intent recall
recall-e70c41ca    "do you remember my dog"                     intent recall
recall-30d8cdf2    "search your memory for my address"          intent recall
```

They cover **routing only**: scoring is `predicted == expected && confidence >=
min_regression_confidence` (`crates/aaron-learning/src/lib.rs:417-427`; loader
`crates/aaron-app/src/commands.rs:1464-1467`). Nothing in the suite writes, reads,
persists, deletes or fails to find a memory; no duplicate, no absence case, no
corruption case, no secret case. So the answer to Q10 is: it covers memory *intent
classification* (5/42 labels, used as a promotion gate) and **zero** memory
*behaviour*. `crates/aaron-app/tests/review_verification.rs:202` additionally guards
that the regression file does not leak into the training split — that is about the
suite's integrity, not memory.

## Risks and unknowns

- **Disk full was not simulated** (would need a small loopback/tmpfs mount or quota).
  By code inspection a failed write surfaces: CLI → `storage error`, exit 1 (same path
  as the read-only case I did run); `remember` tool → "That did not work: …"; but
  `record_task` failures are only `tracing::warn!` (`agent.rs:674-675`) and learning's
  `let _ = store.add(...)` discards them (`learning/lib.rs:306`) — silent data loss is
  possible there (M-17).
- **Multi-process concurrency** was not stress-tested. Design is sound (WAL,
  `busy_timeout(5s)`, `store.rs:285-292`) but there is no test with two processes
  writing simultaneously; the 5-second busy wait means a stuck peer stalls any CLI
  command for 5s before erroring.
- **Voice path not exercised with real audio/model.** The confirmation prompt on the
  spoken `remember` path is written to **stderr** (`context.rs:75-77`), not spoken —
  a voice user may not perceive the ask at all. Under `systemd/` (no TTY) the confirmer
  is `NonInteractive` and always declines (`context.rs:44-54`, `agent.rs:83-94`), so a
  service-run AARON can *never* store a fact. Intended? Worth stating in the docs.
- **`aaron learning approve` was not run end-to-end** because `dataset_path` points at
  the repository file `training/dataset/current.jsonl` (`config/default.toml:77`);
  running it would have modified a document I may not touch. The raw-utterance finding
  (M-06) is therefore source-proven (`learning/lib.rs:305-312`) plus test-proven for
  the dataset half (`learning/lib.rs:965`), not reproduced live.
- **Is `task_history` meant to be user-visible memory or an operational log?** This
  decides whether M-05 is a defect or a documentation fix. It is currently excluded
  from `list_all` (invisible), included in `search` only if a row was created with
  that `kind`, never pruned, and not counted by `doctor` (`store.rs:743-754`).
  The same question applies to `sessions`/`audit_meta`, which nothing ever writes.
- **`max_value_chars` interaction** is coherent but subtle: config 4000, further
  clamped to `max_memory_content_bytes / 4` (`context.rs:201-206`), while the byte
  budget is enforced separately in `add` (`store.rs:382-388`) and *not* in `update`
  (`store.rs:417-430`).
- **Other sessions were editing this workspace during the audit** (`heuristic.rs`,
  `tests/router.rs`, `aaron-audio/*` and several docs changed while I worked — none of
  them by me; I wrote only this file). Every line reference in this report was
  re-verified against the files *after* those edits landed.
- **Interaction with `docs/DEEPSEEK-1.0-AUDIT.md`**: its D-06/D-07/D-08/D-22 memory
  findings are confirmed by independent reproduction here (M-05, M-06/M-03, M-07,
  and the `doctor` row count). This audit adds M-01 (`backup_on_start`), M-02
  (no recovery path), M-04 (shipped config denies `remember`), M-08 (correction flow
  unreachable), M-09 (list/search disagreement), M-10 (path misreport), M-11
  (update/pin unreachable), M-12 (backup not repeatable) and M-13/M-14 (search
  semantics).
