//! Wake-phrase detection.
//!
//! # What AARON actually does
//!
//! AARON ships **no wake-word neural model**. Two strategies are provided:
//!
//! * [`TranscriptMatcher`] — the default. The wake phrase is matched against
//!   the ASR transcript of a short utterance, using token overlap and edit
//!   distance so that "hey aron" and "hey, aaron?" still wake.
//! * [`EnergyDetector`] — a fallback for installs with no ASR at all. It
//!   triggers on a sustained loud burst and is honest about its false-activation
//!   rate in `aaron doctor` and `aaron benchmark`.
//!
//! A production-quality wake-word model (openWakeWord, Porcupine, …) is a
//! separate download with its own licence and its own parameter count. AARON's
//! 1,000,000-parameter budget does **not** cover it, and it is not installed by
//! default. See `docs/model-training.md`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use aaron_core::config::{WakeConfig, WakeStrategy};
use aaron_core::error::{Error, Result};
use aaron_core::text::{normalize, words};
use aaron_core::WakeDetector;

/// Outcome of checking one utterance for the wake phrase.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WakeMatch {
    /// Similarity is at or above the configured threshold.
    Woke { similarity: f32 },
    /// Speech was present but the phrase did not match.
    NoMatch { best_similarity: f32 },
    /// Nothing was heard.
    Silence,
}

/// Similarity between a wake phrase and a heard utterance, in `[0, 1]`.
///
/// The wake phrase is almost always a *prefix* of what the recogniser returns
/// ("hey aaron check system status"), and the recogniser frequently mishears
/// one word ("hey aron"). A whole-string comparison scores both of those badly.
/// So the score is the best similarity over sliding windows of roughly the
/// phrase's length, measured at character level, with a small bonus for the
/// phrase's words appearing anywhere at all.
///
/// Deterministic, allocation-light and free of any model, which is what makes
/// false activations measurable rather than mysterious.
pub fn phrase_similarity(phrase: &str, utterance: &str) -> f32 {
    let a: Vec<String> = words(phrase);
    let b: Vec<String> = words(utterance);
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let a_joined = a.join(" ");
    let a_chars = a_joined.chars().count();

    let lo = a.len().saturating_sub(1).max(1);
    let hi = a.len() + 1;
    // The phrase has to be at the **start** of the utterance, or just after a
    // short run of fillers.
    //
    // It used to be matched anywhere, with a bonus for every phrase word
    // found anywhere in the utterance. That made an ordinary sentence
    // containing the wake words a wake phrase: "they said hey aaron was here"
    // scored 1.0000 against a 0.72 threshold, and the assistant would have
    // acted on the rest of the sentence.
    //
    // A wake word is an *address*, and an address comes first.
    let mut best = 0.0f32;
    for start in leading_positions(&b) {
        // A recogniser runs words together. "aaron check" comes back as the
        // single token "aaroncheck", and a character-level comparison
        // cannot see the wake phrase in that — the assistant would sit
        // there waiting for a word it never heard.
        //
        // So the phrase's words are also compared against the *starts* of
        // the transcript's words. It is still anchored at the start of the
        // utterance, so an ordinary sentence that merely mentions the name
        // does not wake it.
        if token_prefix_match(&a, &b[start..]) {
            best = best.max(0.9);
        }
        for window_len in lo..=hi {
            if start + window_len > b.len() {
                continue;
            }
            let candidate = b[start..start + window_len].join(" ");
            let c_chars = candidate.chars().count();
            let distance = levenshtein_chars(&a_joined, &candidate);
            let sim = 1.0 - (distance as f32 / a_chars.max(c_chars).max(1) as f32).min(1.0);
            if sim > best {
                best = sim;
            }
        }
    }
    best.clamp(0.0, 1.0)
}

/// Whether the phrase's words are the starts of the transcript's words.
fn token_prefix_match(phrase: &[String], tokens: &[String]) -> bool {
    if phrase.is_empty() || phrase.len() > tokens.len() {
        return false;
    }
    phrase
        .iter()
        .zip(tokens.iter())
        .all(|(p, t)| t.starts_with(p.as_str()))
}

/// The positions at which a wake phrase may begin.
///
/// The first word, and the position after any run of filler words. The
/// filler list is deliberately short: it covers the ways people actually
/// start a request ("ok", "so", "now") and nothing else.
fn leading_positions(tokens: &[String]) -> Vec<usize> {
    let mut positions = vec![0usize];
    let mut index = 0usize;
    while index < tokens.len() && FILLER_WORDS.contains(&tokens[index].as_str()) {
        index += 1;
        if index < tokens.len() {
            positions.push(index);
        }
    }
    positions
}

/// Words that may precede the wake phrase.
const FILLER_WORDS: &[&str] = &[
    "ok", "okay", "so", "now", "um", "uh", "er", "hmm", "please", "just", "right", "well",
];

fn levenshtein_chars(a: &str, b: &str) -> usize {
    let ac: Vec<char> = a.chars().collect();
    let bc: Vec<char> = b.chars().collect();
    levenshtein(&ac, &bc)
}

fn levenshtein<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Default detector: match the wake phrase against an ASR transcript.
pub struct TranscriptMatcher {
    phrase: String,
    threshold: f32,
    /// Number of successful wakes, for `aaron benchmark`.
    activations: Arc<AtomicU64>,
    last_similarity: std::sync::Mutex<f32>,
}

impl TranscriptMatcher {
    pub fn new(config: &WakeConfig) -> TranscriptMatcher {
        TranscriptMatcher {
            phrase: normalize(&config.phrase),
            threshold: config.match_threshold,
            activations: Arc::new(AtomicU64::new(0)),
            last_similarity: std::sync::Mutex::new(0.0),
        }
    }

    /// The configured wake phrase.
    pub fn phrase(&self) -> &str {
        &self.phrase
    }

    /// How many times this detector has fired.
    pub fn activations(&self) -> u64 {
        self.activations.load(Ordering::Relaxed)
    }

    /// Check one transcript.
    pub fn check(&self, transcript: &str) -> WakeMatch {
        let similarity = phrase_similarity(&self.phrase, transcript);
        if let Ok(mut last) = self.last_similarity.lock() {
            *last = similarity;
        }
        if similarity >= self.threshold {
            self.activations.fetch_add(1, Ordering::Relaxed);
            WakeMatch::Woke { similarity }
        } else {
            WakeMatch::NoMatch {
                best_similarity: similarity,
            }
        }
    }

    /// Most recent similarity score.
    pub fn last_similarity(&self) -> f32 {
        self.last_similarity.lock().map(|v| *v).unwrap_or(0.0)
    }
}

impl WakeDetector for TranscriptMatcher {
    fn name(&self) -> String {
        "transcript-keyword".into()
    }

    fn ready(&self) -> Result<()> {
        if self.phrase.is_empty() {
            return Err(Error::Config("wake.phrase must not be empty".into()));
        }
        Ok(())
    }

    fn reset(&self) {
        if let Ok(mut last) = self.last_similarity.lock() {
            *last = 0.0;
        }
    }
}

/// Loud-burst detector, for installs without any ASR backend.
pub struct EnergyDetector {
    /// Consecutive loud frames required before waking.
    required_frames: usize,
    seen: std::sync::atomic::AtomicUsize,
    activations: Arc<AtomicU64>,
}

impl EnergyDetector {
    pub fn new() -> EnergyDetector {
        EnergyDetector {
            required_frames: 3,
            seen: std::sync::atomic::AtomicUsize::new(0),
            activations: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Feed one frame's level in dBFS.
    pub fn push_frame(&self, dbfs: f32, threshold_dbfs: f32) -> bool {
        if dbfs >= threshold_dbfs {
            let n = self.seen.fetch_add(1, Ordering::Relaxed) + 1;
            if n >= self.required_frames {
                self.activations.fetch_add(1, Ordering::Relaxed);
                return true;
            }
        } else {
            self.seen.store(0, Ordering::Relaxed);
        }
        false
    }

    pub fn activations(&self) -> u64 {
        self.activations.load(Ordering::Relaxed)
    }
}

impl Default for EnergyDetector {
    fn default() -> Self {
        EnergyDetector::new()
    }
}

impl WakeDetector for EnergyDetector {
    fn name(&self) -> String {
        "energy-burst".into()
    }

    fn ready(&self) -> Result<()> {
        Ok(())
    }

    fn reset(&self) {
        self.seen.store(0, Ordering::Relaxed);
    }
}

/// Wake detection switched off: the agent runs in push-to-talk mode.
pub struct AlwaysAwake;

impl WakeDetector for AlwaysAwake {
    fn name(&self) -> String {
        "disabled".into()
    }

    fn ready(&self) -> Result<()> {
        Ok(())
    }

    fn reset(&self) {}
}

/// Build the configured detector.
pub fn build(config: &WakeConfig) -> Arc<dyn WakeDetector> {
    match config.strategy {
        WakeStrategy::Transcript => Arc::new(TranscriptMatcher::new(config)),
        WakeStrategy::Energy => Arc::new(EnergyDetector::new()),
        WakeStrategy::External => Arc::new(AlwaysAwake),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matcher(threshold: f32) -> TranscriptMatcher {
        TranscriptMatcher::new(&WakeConfig {
            phrase: "hey aaron".into(),
            match_threshold: threshold,
            ..Default::default()
        })
    }

    #[test]
    fn similarity_is_one_for_an_exact_phrase() {
        assert!((phrase_similarity("hey aaron", "hey aaron") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_is_case_and_punctuation_insensitive() {
        assert!((phrase_similarity("Hey, Aaron!", "hey aaron") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn similarity_tolerates_a_misheard_word() {
        let s = phrase_similarity("hey aaron", "hey aron");
        assert!(s > 0.6, "expected a tolerant score, got {s}");
        let s = phrase_similarity("hey aaron", "hey aron check the status");
        assert!(
            s > 0.7,
            "a longer utterance should still be similar, got {s}"
        );
    }

    #[test]
    fn similarity_is_low_for_unrelated_text() {
        // Comfortably below the default 0.72 wake threshold.
        let s = phrase_similarity("hey aaron", "the weather is nice today");
        assert!(s < 0.4, "unrelated text scored {s}");
        assert_eq!(phrase_similarity("hey aaron", ""), 0.0);
    }

    #[test]
    fn exact_phrase_wakes() {
        let m = matcher(0.72);
        assert!(matches!(m.check("hey aaron"), WakeMatch::Woke { .. }));
        assert_eq!(m.activations(), 1);
    }

    #[test]
    fn a_misheard_wake_word_still_wakes() {
        let m = matcher(0.6);
        assert!(matches!(m.check("hey aron"), WakeMatch::Woke { .. }));
    }

    #[test]
    fn unrelated_speech_does_not_wake() {
        let m = matcher(0.72);
        assert!(matches!(
            m.check("what is the time"),
            WakeMatch::NoMatch { .. }
        ));
        assert_eq!(m.activations(), 0);
        assert!(m.last_similarity() < 0.72);
    }

    #[test]
    fn a_command_spoken_after_the_wake_phrase_still_wakes() {
        let m = matcher(0.5);
        assert!(matches!(
            m.check("hey aaron check system status"),
            WakeMatch::Woke { .. }
        ));
    }

    #[test]
    fn a_higher_threshold_tightens_matching() {
        let loose = matcher(0.3);
        let strict = matcher(0.95);
        assert!(matches!(loose.check("hey aron"), WakeMatch::Woke { .. }));
        assert!(matches!(
            strict.check("hey aron"),
            WakeMatch::NoMatch { .. }
        ));
    }

    #[test]
    fn energy_detector_needs_a_sustained_burst() {
        let d = EnergyDetector::new();
        assert!(!d.push_frame(-60.0, -42.0));
        assert!(!d.push_frame(-10.0, -42.0), "one loud frame is a click");
        assert!(!d.push_frame(-10.0, -42.0));
        assert!(d.push_frame(-10.0, -42.0));
        assert_eq!(d.activations(), 1);
        d.reset();
        assert!(!d.push_frame(-10.0, -42.0), "reset clears the counter");
    }

    #[test]
    fn silence_resets_the_burst_counter() {
        let d = EnergyDetector::new();
        d.push_frame(-10.0, -42.0);
        d.push_frame(-10.0, -42.0);
        d.push_frame(-80.0, -42.0);
        assert!(!d.push_frame(-10.0, -42.0));
    }

    #[test]
    fn build_returns_the_configured_strategy() {
        let mk = |strategy| {
            build(&WakeConfig {
                strategy,
                ..Default::default()
            })
            .name()
        };
        assert_eq!(mk(WakeStrategy::Transcript), "transcript-keyword");
        assert_eq!(mk(WakeStrategy::Energy), "energy-burst");
        assert_eq!(mk(WakeStrategy::External), "disabled");
    }

    #[test]
    fn an_empty_phrase_is_a_config_error() {
        let m = TranscriptMatcher::new(&WakeConfig {
            phrase: "  ".into(),
            ..Default::default()
        });
        assert!(m.ready().is_err());
    }

    /// Ordinary sentences that contain the wake words but are not addressed to
    /// the assistant.
    ///
    /// Before the phrase had to be at the start of the utterance, the
    /// containment bonus scored these at 1.0000 — the assistant would have
    /// acted on the rest of the sentence.
    #[test]
    fn an_ordinary_sentence_containing_the_wake_words_is_not_a_wake_phrase() {
        let phrase = "hey aaron";
        let threshold = 0.72;
        for sentence in [
            "they said hey aaron was here",
            "hey what did aaron say",
            "i told aaron about hey",
            "the meeting with aaron and hey",
            "aaron said hey to me",
        ] {
            let similarity = phrase_similarity(phrase, sentence);
            assert!(
                similarity < threshold,
                "{sentence:?} scored {similarity:.4}, above the {threshold} threshold"
            );
        }
    }

    /// A sentence that *starts* with the wake phrase is a wake phrase.
    ///
    /// "hey aaron sounds like a name" begins with the address, so the
    /// matcher cannot tell it from a request — and it does not need to: the
    /// remainder, "sounds like a name", is not a command, so the router
    /// refuses it. The text-level matcher is the first filter, not the only
    /// one.
    #[test]
    fn a_sentence_that_starts_with_the_phrase_is_addressed_to_the_assistant() {
        assert!(phrase_similarity("hey aaron", "hey aaron sounds like a name") >= 0.72);
    }

    /// The phrase still has to work in the shapes a recogniser produces.
    #[test]
    fn the_wake_phrase_is_still_recognised() {
        let phrase = "hey aaron";
        for utterance in [
            "hey aaron",
            "hey aaron check system status",
            "hey aaron what is the weather",
            "hey aron check system status",
            "ok hey aaron check system status",
        ] {
            let similarity = phrase_similarity(phrase, utterance);
            assert!(
                similarity >= 0.72,
                "{utterance:?} scored {similarity:.4}, below the threshold"
            );
        }
    }

    #[test]
    fn the_similarity_of_a_measured_transcript_is_known() {
        // What the Phonon engine actually returned for a synthesised
        // "aaron check system status", and for the two-word phrase.
        for (phrase, transcript, expected_above) in [
            ("aaron", "aaron check system status", true),
            ("aaron", "aaroncheck system status", true),
            ("aaron", "a erincheck system status", false),
            ("aaron", "they said aaron was here", false),
            ("hey aaron", "hey aaron check system status", true),
            ("hey aaron", "hey aaroncheck system status", true),
            ("hey aaron", "hey erincheck system status", false),
        ] {
            let similarity = phrase_similarity(phrase, transcript);
            eprintln!("{phrase:?} / {transcript:?} -> {similarity:.4}");
            assert_eq!(
                similarity >= 0.72,
                expected_above,
                "{phrase:?} / {transcript:?} scored {similarity:.4}"
            );
        }
    }

    #[test]
    fn levenshtein_known_values() {
        assert_eq!(levenshtein(&["a", "b"], &["a", "b"]), 0);
        assert_eq!(levenshtein(&["a"], &["b"]), 1);
        assert_eq!(levenshtein(&[], &["a", "b"]), 2);
        assert_eq!(levenshtein(&["a", "b", "c"], &["a", "x", "c"]), 1);
    }
}
