# Readiness corpus — generation specification

You are generating training data for **AARON 0.5**, an offline voice
assistant. The data teaches a classifier one thing:

> **Is this request specific enough to act on?**

The project is at `/home/david/proj/Aaron`. Read
`docs/release-0.5-report.md` §4.2 for why this corpus exists and
`crates/aaron-core/src/readiness.rs` for the deterministic rule it is
compared against.

**You are generating data, not writing code.** Do not modify any file except
the output file named in your task.

---

## The label

For a request that names a particular object:

* **incomplete** → `needs_clarification`
* **complete** → `ready`

The test is: **does the utterance identify the thing the action would act
on?**

* "read the file" — a *category*, not a thing. **incomplete**
* "read the readme" — one specific file. **complete**
* "read main.rs" — one specific file. **complete**
* "launch the app" — a category. **incomplete**
* "launch the terminal" — one specific application. **complete**
* "remember that" — nothing was said. **incomplete**
* "remember my birthday is in June" — a fact. **complete**
* "open the plugin" — a category; there are many plugins. **incomplete**
* "open vscode" — one specific application. **complete**

Two rules that are easy to get wrong:

1. **A common noun that names a *kind* of thing is incomplete**, even when it
   is not on any closed list: "the podcast", "the widget", "the theme",
   "the receipt". A noun that names *one* recognisable thing is complete:
   "the readme", "the terminal", "the browser", "the changelog".
2. **A free-text argument is content, not a reference.** "recall my name"
   names the topic the user wants. It is **complete**. "recall it" and
   "recall that" say nothing and are **incomplete**.

---

## The output format

One JSON object per line, no comments, no blank lines, no trailing commas:

```json
{"text": "read the pattern", "kind": "incomplete", "family": "category_outside_list"}
{"text": "read the pattern notes.txt", "kind": "complete", "family": "category_outside_list"}
```

* `text` — the utterance, lowercase except where a proper name is needed
  (`vscode`, `firefox`, `obsidian`). Two to six words.
* `kind` — `incomplete` or `complete`.
* `family` — one of the families below, for your slice.

**Pairs.** Every `incomplete` row must have a `complete` row in the same
family and the same intent, and the two must differ **only** in the presence
of the argument. Put the incomplete row first.

**No bare pronouns.** `it`, `that`, `them`, `this` are excluded: a pronoun is
only resolvable from the conversation history, which this classifier does not
see, and release 0.6 takes that up separately.

**No duplicates.** Every `text` must be unique within your output.

**No near-duplicates of the shipped data.** The generator filters your rows
against the shipped dataset with a word-3-gram containment test, so a row that
merely rewords an existing example is discarded. Prefer genuinely new
vocabulary: new categories, new application names, new file names.

---

## The families

| family | shape | example incomplete | example complete |
|---|---|---|---|
| `verb_only` | the verb, nothing after it | `cat` | `cat the changelog` |
| `bare_noun` | determiner + a category noun | `read the invoice` | `read the invoice item` |
| `quantity` | determiner + a quantity | `read the first 20 lines` | `read the first 20 lines of main.rs` |
| `dangling_qualifier` | determiner + noun + a qualifier with nothing after it | `read the file called` | `read the file called main.rs` |
| `category_outside_list` | determiner + a category noun that is not a well-known single thing | `open the theme` | `open the theme dusk.toml` |
| `contentless` | a free-text argument that says nothing | `remember that` | `remember that the boiler is broken` |

---

## The slices

You are given one slice. Produce **40 pairs** (80 rows) for it.

### Slice A — `read_file`

The action reads a file. The argument is a path.

Families: `verb_only`, `bare_noun`, `quantity`, `dangling_qualifier`,
`category_outside_list`.

### Slice B — `search_files`

The action searches for files. The argument is a pattern.

Families: `verb_only`, `bare_noun`, `quantity`, `dangling_qualifier`,
`category_outside_list`.

### Slice C — `open_url`

The action opens a URL. The argument is a URL.

Families: `verb_only`, `bare_noun`, `category_outside_list`.

For the complete rows, give a real `https://` URL. For the incomplete rows,
the utterance names *a kind of page* without a URL.

### Slice D — `start_application`

The action launches a registered application. The argument is an application
name.

Families: `verb_only`, `bare_noun`, `category_outside_list`.

Application names that exist on a typical Linux desktop are good complete
rows: `code`, `firefox`, `thunar`, `okular`, `obsidian`, `signal`, `gimp`,
`libreoffice`, `spotify`, `discord`, `zoom`, `kdenlive`, `inkscape`,
`blender`, `keepassxc`, `veracrypt`, `pavucontrol`.

Categories that name a *kind* of application are good incomplete rows: `the
editor`, `the music player`, `the file manager`, `the chat client`, `the
image viewer`, `the pdf reader`, `the mail client`, `the debugger`.

### Slice E — `stop_application`

The action stops a running application. The argument is an application name.

Families: `verb_only`, `bare_noun`, `category_outside_list`.

### Slice F — `remember`

The action stores a fact. The argument is free text.

Families: `verb_only`, `contentless`.

### Slice G — `recall`

The action searches long-term memory. The argument is free text.

Families: `verb_only`, `contentless`.

---

## Rules the generator must not break

1. **The incomplete row must not name a particular object.** If the reader
   could act on it, it is not incomplete.
2. **The complete row must name something.** If the reader still cannot act,
   it is not complete.
3. **The pair must be recognisably the same request**, differing in the
   argument. "read the file" / "read the readme" is a pair; "read the file" /
   "launch the terminal" is not.
4. **No safety-relevant content.** These are classifier inputs. Keep them
   ordinary and benign.
5. **Stay in your slice.** Do not produce rows for other intents.

---

## Output

Write only the JSON lines to the output path given in your task. Then reply
with a two-line summary: the number of pairs written, and any row you were
unsure about.
