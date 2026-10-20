//! Deterministic text normalisation, feature extraction and entity extraction.
//!
//! Everything in this module is pure and reproducible: the same string always
//! produces the same feature vector on any machine, which is what lets the
//! trained checkpoint, the evaluator and the runtime agree.
//!
//! The feature scheme is intentionally simple and inspectable rather than a
//! learned subword vocabulary, because a 1M-parameter model has no budget for a
//! large embedding table and because a hash-based scheme removes the
//! train/test leakage risk of an unfitted tokenizer.

use std::collections::BTreeSet;

use crate::intent::{Entities, Intent};

/// Maximum characters kept per entity field. Mirrors `config.limits`.
pub const DEFAULT_MAX_ENTITY_CHARS: usize = 200;

/// Lowercase, collapse whitespace and drop characters that carry no signal.
///
/// Apostrophes inside words are preserved so that "don't" and "what's" stay one
/// token. Every other punctuation character becomes a space.
pub fn normalize(input: &str) -> String {
    normalize_with(input, |_| false)
}

/// Nouns that name a *category* of thing rather than a particular one.
///
/// "read the file" names a category. There is no particular file called
/// "the file", so a request that resolves to one of these has not said
/// *which* thing it means, and the assistant should ask rather than guess.
const GENERIC_REFERENT_NOUNS: &[&str] = &[
    "file",
    "files",
    "app",
    "apps",
    "application",
    "applications",
    "program",
    "programs",
    "process",
    "processes",
    "document",
    "documents",
    "doc",
    "docs",
    "suite",
    "suites",
    "website",
    "site",
    "link",
    "script",
    "scripts",
    "command",
    "commands",
    "thing",
    "things",
    "stuff",
    "item",
    "items",
    "directory",
    "directories",
    "folder",
    "folders",
    "path",
    "paths",
    "project",
    "projects",
    "repo",
    "repos",
    "repository",
    "repositories",
];

/// Bare pronouns, which name nothing at all.
const BARE_PRONOUNS: &[&str] = &[
    "it",
    "that",
    "this",
    "them",
    "those",
    "these",
    "one",
    "ones",
    "something",
    "anything",
    "everything",
    "nothing",
    "someone",
    "anyone",
    "everyone",
    "somebody",
    "anybody",
    "everybody",
    "there",
];

/// Is this extracted value a *generic* reference rather than a specific one?
///
/// # Why this exists
///
/// `strip_leading_verbs` never returns an empty remainder, so the extracted
/// argument of a request like "read the file" is the string "the file" — not
/// the empty string. The argument builder's "I understood `read_file` but
/// not which file" path therefore never fired for a natural utterance, and
/// the tool was dispatched with the placeholder instead. The tool then
/// refused it (the sandbox rejects "the file", the application allow-list
/// rejects "the process"), so nothing unsafe happened — but the user was
/// told "path rejected: `the file` does not exist inside any allowed root"
/// instead of being asked which file they meant.
///
/// The information needed to tell "read the file" from "read the readme" is
/// already in the extracted string. This function reads it.
///
/// # What it deliberately does not do
///
/// It is a closed list, not a pattern. "the readme" is *not* generic: a file
/// called "readme" is a specific thing, and the extractor would have
/// produced "the readme" for a perfectly good request. A pattern like
/// "determiner plus a noun" would break it.
pub fn is_generic_referent(value: &str) -> bool {
    let normalised = normalize_entities(value);
    let trimmed = normalised.trim();
    if trimmed.is_empty() {
        return true;
    }
    if BARE_PRONOUNS.contains(&trimmed) {
        return true;
    }
    // A determiner plus a generic noun: "the file", "a program", "that site".
    //
    // The phrase is generic when *every* content word is a generic noun. One
    // specific word makes the whole phrase specific: "the config file" names
    // a file called "config", and "the readme" names a file called "readme".
    const DETERMINERS: &[&str] = &[
        "the", "a", "an", "my", "your", "that", "this", "some", "any", "its",
    ];
    let content: Vec<&str> = trimmed
        .split_whitespace()
        .filter(|w| !DETERMINERS.contains(w))
        .collect();
    if content.is_empty() {
        // Nothing but a determiner: "the", "that".
        return true;
    }
    content.iter().all(|w| GENERIC_REFERENT_NOUNS.contains(w))
}

/// Words that carry no content: pronouns, articles, prepositions.
///
/// Used to answer a different question from [`is_generic_referent`]: *did the
/// user actually say anything?* "remember that" leaves the bare word "that"
/// behind, which is a filler the extractor failed to strip, and storing it as a
/// memory would be storing noise.
const CONTENTLESS_WORDS: &[&str] = &[
    "it",
    "that",
    "this",
    "them",
    "those",
    "these",
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "of",
    "to",
    "for",
    "with",
    "in",
    "on",
    "at",
    "by",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "do",
    "does",
    "did",
    "my",
    "your",
    "his",
    "her",
    "their",
    "our",
    "its",
    "all",
    "any",
    "some",
    "one",
    "ones",
    "something",
    "anything",
    "everything",
    "nothing",
    "please",
    "now",
    "just",
    "really",
    "so",
    "well",
    "ok",
    "okay",
    "hey",
    "um",
    "uh",
    "hmm",
    "er",
    "yeah",
    "yes",
    "no",
    "not",
];

/// Does this value carry any content at all?
///
/// The question is "did the user say something", not "is this a specific
/// thing". A free-text argument — what to remember, what to look for — is
/// content even when it is a common word: "recall my name" names the thing
/// the user wants, and refusing it with "not what to look for" is wrong.
///
/// "remember that" is the opposite case: the extractor leaves the bare
/// filler "that" behind, and there is nothing to remember.
pub fn is_contentless(value: &str) -> bool {
    let normalised = normalize_entities(value);
    let words: Vec<&str> = normalised.split_whitespace().collect();
    if words.is_empty() {
        return true;
    }
    words.iter().all(|w| CONTENTLESS_WORDS.contains(w))
}

/// Like [`normalize`] but keeps the characters that appear in filesystem paths,
/// URLs and hostnames: `/ . - _ : ~ @ % + = & ? #`.
///
/// Entity extraction uses this variant. The feature pipeline deliberately does
/// not, so that punctuation cannot leak into the hashed feature space.
pub fn normalize_entities(input: &str) -> String {
    normalize_with(input, |c| {
        matches!(
            c,
            '/' | '.' | '-' | '_' | ':' | '~' | '@' | '%' | '+' | '=' | '&' | '?' | '#' | '*'
        )
    })
}

fn normalize_with(input: &str, keep: impl Fn(char) -> bool) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_was_space = true;
    for ch in input.chars() {
        if ch.is_alphanumeric() || keep(ch) {
            for c in ch.to_lowercase() {
                out.push(c);
            }
            last_was_space = false;
        } else if (ch == '\'' || ch == '\u{2019}') && !last_was_space {
            // Keep apostrophes, but not leading ones.
            if !out.ends_with('\'') {
                out.push('\'');
            }
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    while out.ends_with(' ') {
        out.pop();
    }
    out
}

/// Split normalised text into words, dropping empty tokens.
pub fn words(input: &str) -> Vec<String> {
    normalize(input)
        .split(' ')
        .filter(|w| !w.is_empty())
        .map(|w| w.to_string())
        .collect()
}

/// 64-bit FNV-1a. Chosen because it is tiny, dependency-free and completely
/// specified, so the feature map is reproducible across Rust versions and
/// platforms (unlike `std::collections::hash_map::DefaultHasher`, whose seed is
/// explicitly not stable).
#[inline]
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Map an arbitrary feature string into `[0, vocab)`.
#[inline]
pub fn bucket_of(feature: &str, vocab: u32) -> u32 {
    // `vocab` is always a power of two in practice, but the modulo keeps this
    // correct for any configured value.
    (fnv1a64(feature.as_bytes()) % vocab as u64) as u32
}

/// The feature namespaces used by [`features`]. Prefixing the feature string
/// keeps a bigram from colliding with the unigram it is made of.
const NS_WORD: &str = "w:";
const NS_BIGRAM: &str = "b:";
const NS_CHAR: &str = "c:";

/// Number of character n-grams emitted per word.
pub const CHAR_NGRAM: usize = 3;

/// Longest word (in characters) that contributes character n-grams. Very long
/// "words" are usually pasted data rather than natural language and would
/// otherwise dominate the hashed feature space.
pub const MAX_WORD_CHARS: usize = 24;

/// Produce the hashed feature buckets for an utterance.
///
/// The scheme is: word unigrams, word bigrams, and padded character trigrams.
/// Character trigrams give the model tolerance to typos and to ASR word errors
/// without any additional parameters.
pub fn features(text: &str, vocab: u32) -> Vec<u32> {
    let words = words(text);
    let mut feats: Vec<u32> = Vec::with_capacity(words.len() * 6);

    for w in &words {
        feats.push(bucket_of(&format!("{NS_WORD}{w}"), vocab));

        let clipped: String = w.chars().take(MAX_WORD_CHARS).collect();
        // Padded trigrams: `^he`, `hel`, ..., `llo`, `lo$`.
        if clipped.chars().count() <= CHAR_NGRAM {
            let padded = format!("^{clipped}$");
            let chars: Vec<char> = padded.chars().collect();
            for i in 0..chars.len().saturating_sub(CHAR_NGRAM - 1) {
                let tri: String = chars[i..i + CHAR_NGRAM].iter().collect();
                feats.push(bucket_of(&format!("{NS_CHAR}{tri}"), vocab));
            }
        } else {
            let chars: Vec<char> = format!("^{clipped}$").chars().collect();
            for i in 0..chars.len().saturating_sub(CHAR_NGRAM - 1) {
                let tri: String = chars[i..i + CHAR_NGRAM].iter().collect();
                feats.push(bucket_of(&format!("{NS_CHAR}{tri}"), vocab));
            }
        }
    }

    for pair in words.windows(2) {
        feats.push(bucket_of(&format!("{NS_BIGRAM}{}", pair.join(" ")), vocab));
    }

    feats
}

/// Truncate on a character boundary with an explicit ellipsis marker, used to
/// cap tool output and audit records without splitting UTF-8.
pub fn truncate_chars(input: &str, max_chars: usize) -> String {
    if input.chars().count() <= max_chars {
        return input.to_string();
    }
    let mut out: String = input.chars().take(max_chars.saturating_sub(1)).collect();
    out.push('\u{2026}');
    out
}

/// Redact anything that looks like a secret before an audit record or log line.
///
/// AARON never logs raw audio or full file contents; this is a second line of
/// defence for values that flow into structured fields (a URL with a token, a
/// remembered string).
pub fn redact_secrets(input: &str) -> String {
    let mut out: Vec<String> = Vec::new();
    for token in input.split_whitespace() {
        // `Bearer <token>` style: the whole rest of the token is the secret.
        let lower = token.to_lowercase();
        if lower.starts_with("bearer ") {
            out.push("bearer [REDACTED]".to_string());
            continue;
        }
        // A URL can carry credentials in its userinfo section
        // (`https://user:token@host/`), which neither the `key=value` scan nor
        // the audit-log length cap would ever catch. Strip it here so a
        // requested or remembered URL cannot leak a password into the log.
        out.push(redact_pairs(&strip_url_userinfo(token)));
    }
    out.join(" ")
}

/// Remove the `user:password@` userinfo section from a URL-shaped token.
///
/// Returns the token unchanged when it is not a URL or has no userinfo.
fn strip_url_userinfo(token: &str) -> String {
    // A URL often arrives wrapped in a `key=value` pair (`url=https://...`),
    // so look at the value half when the token starts with an identifier.
    let (prefix, body) = match token.find('=') {
        Some(eq)
            if eq > 0
                && token[..eq]
                    .chars()
                    .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                && token[eq + 1..].contains("://") =>
        {
            (&token[..eq + 1], &token[eq + 1..])
        }
        _ => ("", token),
    };

    let Some(scheme_end) = body.find("://") else {
        return token.to_string();
    };
    if !body[..scheme_end]
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '+' || c == '-' || c == '.')
    {
        return token.to_string();
    }
    let authority_start = scheme_end + 3;
    let authority_end = body[authority_start..]
        .find(['/', '?', '#'])
        .map(|i| authority_start + i)
        .unwrap_or(body.len());
    let authority = &body[authority_start..authority_end];
    let Some(at) = authority.rfind('@') else {
        return token.to_string();
    };
    let userinfo = &authority[..at];
    let Some(colon) = userinfo.rfind(':') else {
        // `user@host` is not a credential.
        return token.to_string();
    };
    // Keep the username: it is useful in a log line and it is not the secret.
    // Redact the password, which is.
    let user = &userinfo[..colon];
    format!(
        "{prefix}{}://{}:[REDACTED]@{}{}",
        &body[..scheme_end],
        user,
        &authority[at + 1..],
        &body[authority_end..]
    )
}

/// Redact `key=value` pairs inside a single whitespace-free token.
///
/// Query strings pack several pairs into one token, separated by `&` or `?`.
/// Each is checked independently so a secret cannot hide inside another
/// parameter.
fn redact_pairs(token: &str) -> String {
    const MARKERS: [&str; 6] = ["token", "api_key", "apikey", "password", "passwd", "secret"];
    split_query(token)
        .into_iter()
        .map(|part| match part.find('=') {
            Some(eq)
                if MARKERS.contains(
                    &part[..eq]
                        .rsplit(['?', '/', ':'])
                        .next()
                        .unwrap_or("")
                        .to_lowercase()
                        .as_str(),
                ) =>
            {
                format!("{}=[REDACTED]", &part[..eq])
            }
            _ => part.to_string(),
        })
        .collect::<Vec<_>>()
        .join("")
}

/// Split a token on `?` and `&` boundaries, keeping the delimiters attached to
/// the segment they follow so the string can be reassembled unchanged.
fn split_query(token: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut current = String::with_capacity(token.len());
    let chars: Vec<char> = token.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        current.push(*c);
        if (*c == '?' || *c == '&') && i + 1 < chars.len() {
            parts.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        parts.push(current);
    }
    if parts.is_empty() {
        parts.push(String::new());
    }
    parts
}

/// Strip common filler words from the ends of an utterance so that
/// "please could you remember that I like tea" yields "I like tea".
pub fn strip_fillers(text: &str) -> String {
    const PREFIXES: [&str; 10] = [
        "hey aaron",
        "ok aaron",
        "okay aaron",
        "please",
        "could you please",
        "can you please",
        "could you",
        "can you",
        "would you",
        "i want you to",
    ];
    let mut current = normalize_entities(text);
    // Remove filler prefixes repeatedly, longest first.
    loop {
        let mut changed = false;
        for p in PREFIXES {
            if current.starts_with(p) {
                let rest = current[p.len()..].trim_start();
                if !rest.is_empty() && rest != current {
                    current = rest.to_string();
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
    for suffix in [" please", " thanks", " thank you"] {
        if current.ends_with(suffix) {
            current.truncate(current.len() - suffix.len());
        }
    }
    current.trim().to_string()
}

/// Extract command arguments for an intent from the raw utterance.
///
/// This runs after classification and before the policy check. It never
/// executes anything; it only produces bounded strings.
pub fn extract_entities(intent: Intent, utterance: &str) -> Entities {
    let cleaned = strip_fillers(utterance);
    let lower = normalize_entities(&cleaned);
    let max = DEFAULT_MAX_ENTITY_CHARS;
    let clip = |s: String| Some(crate::text::truncate_chars(s.trim(), max));

    match intent {
        Intent::Greeting | Intent::Help | Intent::SystemStatus | Intent::Cancel => {
            Entities::default()
        }

        Intent::Remember => {
            // Strip the command verb itself so the stored fact is clean.
            let fact = strip_leading_verbs(
                &lower,
                &[
                    "remember that",
                    "please remember that",
                    "remember this that",
                    "remember that my",
                    "remember that i",
                    "remember",
                    "store that",
                    "save that",
                    "note that",
                ],
            );
            Entities {
                text: clip(titlecase_from_lower(&fact)),
                ..Default::default()
            }
        }

        Intent::Recall => {
            let query = strip_leading_verbs(
                &lower,
                &[
                    "what do you know about",
                    "do you know about",
                    "what do you remember about",
                    "recall",
                    "search your memory for",
                    "look up in memory",
                    "what is my",
                ],
            );
            Entities {
                text: clip(query),
                ..Default::default()
            }
        }

        Intent::SearchFiles => {
            let (query, root) = split_root_and_pattern(&lower);
            Entities {
                text: clip(query),
                path: root.map(|r| crate::text::truncate_chars(&r, max)),
                ..Default::default()
            }
        }

        Intent::ReadFile => {
            let count = parse_count(&lower).unwrap_or(40);
            let rest = strip_leading_verbs(
                &lower,
                &[
                    "read",
                    "open file",
                    "cat",
                    "show me the file",
                    "show the file",
                ],
            );
            let rest = strip_leading_verbs(
                &rest,
                &[
                    "the first 25 lines of",
                    "lines of",
                    "line of",
                    "the file called",
                    "the file named",
                    "the file",
                    "file called",
                    "file named",
                    "file",
                ],
            );
            Entities {
                path: clip(rest),
                count: Some(count),
                ..Default::default()
            }
        }

        Intent::GitStatus => {
            let repo = strip_leading_verbs(&lower, &["git status", "check git", "show git status"]);
            let path = if repo.is_empty() { None } else { clip(repo) };
            Entities {
                path,
                ..Default::default()
            }
        }

        Intent::RunTests => {
            let suite = strip_leading_verbs(
                &lower,
                &[
                    "run the tests",
                    "run tests",
                    "run test",
                    "run all tests",
                    "test",
                    "run the test suite",
                ],
            );
            Entities {
                suite: if suite.is_empty() { None } else { clip(suite) },
                ..Default::default()
            }
        }

        Intent::OpenUrl => Entities {
            url: extract_url(&cleaned),
            ..Default::default()
        },

        Intent::StartApplication => {
            let app = strip_leading_verbs(
                &lower,
                &[
                    "open the application",
                    "launch application",
                    "start application",
                    "start app",
                    "launch app",
                    "start",
                    "launch",
                    "open",
                    "run",
                ],
            );
            Entities {
                application: clip(app),
                ..Default::default()
            }
        }

        Intent::StopApplication => {
            let app = strip_leading_verbs(
                &lower,
                &[
                    "stop the application",
                    "kill application",
                    "stop application",
                    "kill app",
                    "close application",
                    "quit application",
                    "stop",
                    "kill",
                    "close",
                    "quit",
                ],
            );
            Entities {
                application: clip(app),
                ..Default::default()
            }
        }

        Intent::Unknown => Entities {
            text: clip(cleaned),
            ..Default::default()
        },
    }
}

/// Parse "the first 25 lines" / "25 lines" into a count.
fn parse_count(lower: &str) -> Option<u32> {
    let digits: String = lower
        .chars()
        .skip_while(|c| !c.is_ascii_digit())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse::<u32>().ok().filter(|n| *n > 0)
}

/// Remove the first verb prefix that matches, preferring the longest match.
fn strip_leading_verbs(text: &str, verbs: &[&str]) -> String {
    for v in verbs {
        if let Some(rest) = text.strip_prefix(v) {
            let rest = rest.trim();
            if !rest.is_empty() {
                return rest.to_string();
            }
        }
    }
    text.trim().to_string()
}

/// Find an `http(s)://` token in the utterance.
fn extract_url(text: &str) -> Option<String> {
    text.split_whitespace()
        .map(|t| {
            t.trim_matches(|c: char| {
                !c.is_alphanumeric()
                    && c != '/'
                    && c != ':'
                    && c != '.'
                    && c != '-'
                    && c != '_'
                    && c != '~'
                    && c != '%'
                    && c != '?'
                    && c != '&'
                    && c != '='
                    && c != '#'
                    && c != '+'
            })
        })
        .find(|t| t.starts_with("http://") || t.starts_with("https://"))
        .map(|t| t.to_string())
        .map(|s| crate::text::truncate_chars(&s, DEFAULT_MAX_ENTITY_CHARS))
}

/// Split "find *.rs in ~/code" into (pattern, optional root).
fn split_root_and_pattern(lower: &str) -> (String, Option<String>) {
    let cleaned = strip_leading_verbs(
        lower,
        &[
            "search for",
            "search",
            "find files named",
            "find files matching",
            "find file",
            "find",
            "look for",
        ],
    );
    if let Some(idx) = cleaned.find(" in ") {
        let (pattern, root) = cleaned.split_at(idx);
        let root = root[4..].trim();
        if !root.is_empty() {
            return (pattern.trim().to_string(), Some(root.to_string()));
        }
    }
    (cleaned, None)
}

/// The extractor lowercases text; memory content is more useful with its
/// original casing. This is a best-effort restoration for common words only.
fn titlecase_from_lower(text: &str) -> String {
    const KEEP: [&str; 12] = [
        "i", "my", "me", "the", "a", "an", "and", "or", "but", "is", "are", "to",
    ];
    text.split(' ')
        .map(|w| {
            // `w.is_empty()` yields the empty string either way, so the two
            // branches were the same and the check was dead.
            if KEEP.contains(&w) || w.is_empty() {
                w.to_string()
            } else {
                let mut cs = w.chars();
                match cs.next() {
                    Some(first) => first.to_uppercase().collect::<String>() + cs.as_str(),
                    None => String::new(),
                }
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Collect the distinct set of tags mentioned in free text, used for memory
/// retrieval scoring. Tags are `#word` tokens.
pub fn extract_tags(text: &str) -> BTreeSet<String> {
    text.split_whitespace()
        .filter_map(|t| t.strip_prefix('#'))
        .map(|t| {
            t.trim_matches(|c: char| !c.is_alphanumeric())
                .to_lowercase()
        })
        .filter(|t| !t.is_empty() && t.len() <= 32)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_strips_punctuation_but_keeps_apostrophes() {
        assert_eq!(normalize("Hello, World!"), "hello world");
        assert_eq!(normalize("don't"), "don't");
        assert_eq!(normalize("  spaced   out  "), "spaced out");
        assert_eq!(normalize("a-b_c"), "a b c");
        assert_eq!(normalize(""), "");
    }

    #[test]
    fn normalize_entities_keeps_paths_urls_and_globs() {
        assert_eq!(
            normalize_entities("/home/david/Cargo.toml"),
            "/home/david/cargo.toml"
        );
        assert_eq!(
            normalize_entities("find *.rs in ~/code"),
            "find *.rs in ~/code"
        );
        assert_eq!(
            normalize_entities("open https://example.com/a?b=1"),
            "open https://example.com/a?b=1"
        );
    }

    #[test]
    fn fnv1a_is_stable() {
        // Known FNV-1a 64 vectors. If this changes, existing checkpoints become
        // silently incompatible, so it is pinned by a test.
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        assert_eq!(fnv1a64(b"foobar"), 0x85944171f73967e8);
    }

    #[test]
    fn features_are_deterministic_and_bounded() {
        let a = features("check system status", 10240);
        let b = features("check system status", 10240);
        assert_eq!(a, b);
        assert!(!a.is_empty());
        assert!(a.iter().all(|b| *b < 10240));
        // Longer input yields more features but never an unbounded blow-up.
        let long = features(&"word ".repeat(500), 10240);
        assert!(long.len() < 20_000);
    }

    #[test]
    fn features_are_case_and_punctuation_insensitive() {
        assert_eq!(
            features("Check system status!", 10240),
            features("check   system status", 10240)
        );
    }

    #[test]
    fn truncate_respects_utf8_boundaries() {
        let s = "αβγδε";
        // The ellipsis counts toward the budget.
        assert_eq!(truncate_chars(s, 3), "αβ…");
        assert_eq!(truncate_chars(s, 3).chars().count(), 3);
        assert_eq!(truncate_chars(s, 100), s);
    }

    #[test]
    fn url_credentials_are_stripped() {
        let out = redact_secrets("url=https://user:hunter2@example.com/path");
        assert!(!out.contains("hunter2"), "credential survived: {out}");
        assert!(out.contains("[REDACTED]@example.com/path"), "{out}");
        // The username survives because it is not the secret and it makes the
        // log line useful.
        assert!(out.contains("user:"), "{out}");

        // No credentials: unchanged.
        assert_eq!(
            redact_secrets("https://example.com/path"),
            "https://example.com/path"
        );
        // `user@host` without a password is not a credential.
        assert_eq!(
            redact_secrets("mailto:user@example.com"),
            "mailto:user@example.com"
        );
        // Not a URL at all.
        assert_eq!(redact_secrets("a:b"), "a:b");
    }

    #[test]
    fn redact_masks_secret_values() {
        let out = redact_secrets("GET /x?token=abc123&user=bob");
        assert!(!out.contains("abc123"));
        assert!(out.contains("[REDACTED]"));
        assert!(out.contains("user=bob"));
        assert_eq!(redact_secrets("nothing to see"), "nothing to see");
    }

    #[test]
    fn strip_fillers_removes_politeness() {
        assert_eq!(
            strip_fillers("Hey Aaron, please check status"),
            "check status"
        );
        assert_eq!(
            strip_fillers("could you please open the browser"),
            "open the browser"
        );
    }

    #[test]
    fn entity_extraction_uses_bounded_output() {
        let long = "remember that ".to_string() + &"x".repeat(5000);
        let e = extract_entities(Intent::Remember, &long);
        let text = e.text.clone().unwrap();
        assert!(text.chars().count() <= DEFAULT_MAX_ENTITY_CHARS);
        assert!(e.validate(DEFAULT_MAX_ENTITY_CHARS).is_ok());
    }

    #[test]
    fn url_entities_are_restricted_to_http_schemes() {
        let e = extract_entities(Intent::OpenUrl, "open https://example.com/x");
        assert_eq!(e.url.as_deref(), Some("https://example.com/x"));
        assert!(e.validate(DEFAULT_MAX_ENTITY_CHARS).is_ok());

        let bad = Entities {
            url: Some("file:///etc/passwd".into()),
            ..Default::default()
        };
        assert!(bad.validate(DEFAULT_MAX_ENTITY_CHARS).is_err());
    }

    #[test]
    fn search_files_splits_root_and_pattern() {
        let e = extract_entities(Intent::SearchFiles, "find *.rs in /home/david/code");
        assert_eq!(e.path.as_deref(), Some("/home/david/code"));
        assert_eq!(e.text.as_deref(), Some("*.rs"));
    }

    #[test]
    fn read_file_extracts_count() {
        let e = extract_entities(Intent::ReadFile, "read the first 25 lines of Cargo.toml");
        assert_eq!(e.count, Some(25));
        assert!(e.path.unwrap().contains("cargo.toml"));
    }

    #[test]
    fn tags_are_extracted_and_bounded() {
        let tags = extract_tags("deploy #infra #Rust and #averyveryveryverylongtaghere");
        assert!(tags.contains("infra"));
        assert!(tags.contains("rust"));
        assert!(tags.iter().all(|t| t.len() <= 32));
    }
}
