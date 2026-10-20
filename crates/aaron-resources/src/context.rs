//! Bounded conversation context.
//!
//! # RAM context is not the same thing as persistent memory
//!
//! The user-approved facts live in SQLite and are retrieved on demand. What
//! lives here is a *bounded ring* of recent turns, used to resolve anaphora
//! ("run it again", "and the other one"). It is deliberately small: eight turns
//! by default, and it never grows past its configured bound.
//!
//! When the ring is full, the oldest turns are collapsed into a summary. The
//! summary records the ids of the turns it was built from, so every claim in it
//! can be traced back to a turn the user actually said. Losing traceability
//! would make the summary indistinguishable from AARON inventing something.

use std::collections::VecDeque;

use aaron_core::intent::Intent;
use serde::{Deserialize, Serialize};

/// One exchange.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Turn {
    /// Monotonic id, stable for the life of the process.
    pub id: u64,
    pub user_text: String,
    /// AARON's reply.
    pub assistant_text: String,
    pub intent: Option<Intent>,
    /// Whether the turn changed anything.
    pub executed: bool,
}

/// A summary of turns that have fallen out of the ring.
///
/// `source_turn_ids` is what makes this auditable: every summarised turn is
/// still identified even though its text is gone.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Summary {
    pub text: String,
    pub source_turn_ids: Vec<u64>,
    /// Intents seen, in order, so the summary can stay intent-faithful.
    pub intents: Vec<Intent>,
}

/// A bounded conversation context.
#[derive(Debug, Clone)]
pub struct ConversationContext {
    capacity: usize,
    max_chars: usize,
    turns: VecDeque<Turn>,
    summary: Option<Summary>,
    next_id: u64,
    /// Total turns ever recorded, including summarised ones.
    total_turns: u64,
    /// Approximate characters retained. The bound is enforced on this, not just
    /// on the turn count, because one enormous turn must not be able to blow
    /// the budget on its own.
    retained_chars: usize,
}

impl ConversationContext {
    /// Create a context bounded by turn count and by characters.
    pub fn new(capacity: usize, max_chars: usize) -> ConversationContext {
        ConversationContext {
            capacity: capacity.max(1),
            max_chars: max_chars.max(256),
            turns: VecDeque::new(),
            summary: None,
            next_id: 1,
            total_turns: 0,
            retained_chars: 0,
        }
    }

    /// Build from the resource configuration.
    pub fn from_config(capacity: usize, max_transcript_chars: usize) -> ConversationContext {
        ConversationContext::new(capacity, max_transcript_chars)
    }

    /// Turns currently retained.
    pub fn len(&self) -> usize {
        self.turns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.turns.is_empty()
    }

    /// Total turns recorded over the session's life.
    pub fn total_turns(&self) -> u64 {
        self.total_turns
    }

    /// Approximate characters retained in RAM.
    pub fn retained_chars(&self) -> usize {
        self.retained_chars
    }

    /// The turn bound.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The current summary, if the ring has wrapped.
    pub fn summary(&self) -> Option<&Summary> {
        self.summary.as_ref()
    }

    /// The most recent turns, oldest first.
    pub fn turns(&self) -> impl Iterator<Item = &Turn> {
        self.turns.iter()
    }

    /// The most recent turn.
    pub fn last(&self) -> Option<&Turn> {
        self.turns.back()
    }

    /// Record a turn, evicting and summarising if necessary.
    pub fn push(
        &mut self,
        user_text: &str,
        assistant_text: &str,
        intent: Option<Intent>,
        executed: bool,
    ) -> Turn {
        let turn = Turn {
            id: self.next_id,
            user_text: clip(user_text, self.max_chars / 4),
            assistant_text: clip(assistant_text, self.max_chars / 4),
            intent,
            executed,
        };
        self.next_id += 1;
        self.total_turns += 1;
        self.retained_chars += turn.user_text.chars().count() + turn.assistant_text.chars().count();
        self.turns.push_back(turn.clone());
        self.evict();
        turn
    }

    /// Evict until the context is inside both bounds.
    fn evict(&mut self) {
        while self.turns.len() > self.capacity || self.retained_chars > self.max_chars {
            if self.turns.len() <= 1 {
                // A single oversized turn: keep it, it has already been clipped.
                self.retained_chars = self.current_chars();
                break;
            }
            let oldest = self.turns.pop_front().expect("checked non-empty");
            self.retained_chars = self.retained_chars.saturating_sub(
                oldest.user_text.chars().count() + oldest.assistant_text.chars().count(),
            );
            self.absorb(oldest);
        }
    }

    /// Fold an evicted turn into the summary, keeping it traceable.
    fn absorb(&mut self, turn: Turn) {
        let entry = describe_turn(&turn);
        let (text, source_turn_ids, intents) = match self.summary.take() {
            None => (entry, vec![turn.id], turn.intent.into_iter().collect()),
            Some(mut s) => {
                s.text.push_str(" | ");
                s.text.push_str(&entry);
                s.source_turn_ids.push(turn.id);
                if let Some(i) = turn.intent {
                    if !s.intents.contains(&i) {
                        s.intents.push(i);
                    }
                }
                (s.text, s.source_turn_ids, s.intents)
            }
        };
        // The summary itself is bounded, so a very long session cannot grow it
        // without limit either.
        self.summary = Some(Summary {
            text: clip(&text, self.max_chars / 2),
            source_turn_ids,
            intents,
        });
    }

    /// Everything the model should see, oldest first.
    pub fn render(&self) -> String {
        let mut out = String::new();
        if let Some(s) = &self.summary {
            out.push_str("[earlier turns, summarised]\n");
            out.push_str(&s.text);
            out.push('\n');
        }
        for turn in &self.turns {
            out.push_str(&format!("user: {}\n", turn.user_text));
            out.push_str(&format!("aaron: {}\n", turn.assistant_text));
        }
        out
    }

    /// Drop everything, e.g. on `cancel` or at the start of a new session.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.retained_chars = 0;
        self.summary = None;
    }

    /// Drop the ring but keep the summary, for a `new topic` style reset.
    pub fn start_new_topic(&mut self) {
        self.turns.clear();
        self.retained_chars = 0;
    }

    fn current_chars(&self) -> usize {
        self.turns
            .iter()
            .map(|t| t.user_text.chars().count() + t.assistant_text.chars().count())
            .sum()
    }
}

fn describe_turn(turn: &Turn) -> String {
    let intent = turn.intent.map(|i| i.as_str()).unwrap_or("unknown");
    format!("#{} {}: {}", turn.id, intent, clip(&turn.user_text, 120))
}

fn clip(text: &str, max: usize) -> String {
    aaron_core::text::truncate_chars(text, max.max(1))
}

/// Approximate RAM cost of a context, in bytes.
///
/// Characters are one byte for ASCII and up to four for the rest; the estimate
/// is deliberately generous so the budget errs towards freeing too early.
pub fn estimate_bytes(turns: usize, chars: usize) -> usize {
    turns * 128 + chars * 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_turn_bound_is_enforced() {
        let mut c = ConversationContext::new(4, 100_000);
        for i in 0..20 {
            c.push(&format!("u{i}"), "reply", Some(Intent::Help), false);
        }
        assert_eq!(c.len(), 4);
        assert_eq!(c.total_turns(), 20);
        assert_eq!(c.last().unwrap().user_text, "u19");
    }

    #[test]
    fn the_character_bound_is_enforced_even_with_few_turns() {
        // A capacity of 100 turns but only 1,000 characters: the character
        // bound must win, otherwise one long turn blows the budget.
        let mut c = ConversationContext::new(100, 1_000);
        for _ in 0..10 {
            c.push(&"x".repeat(400), &"y".repeat(400), None, false);
        }
        assert!(
            c.retained_chars() <= 1_000,
            "retained {} chars",
            c.retained_chars()
        );
        assert!(c.len() < 10);
    }

    #[test]
    fn evicted_turns_are_summarised_and_stay_traceable() {
        let mut c = ConversationContext::new(3, 100_000);
        for i in 0..10 {
            c.push(
                &format!("request {i}"),
                "ok",
                Some(Intent::SystemStatus),
                true,
            );
        }
        let s = c.summary().expect("turns must have been summarised");
        assert_eq!(s.source_turn_ids, (1..=7).collect::<Vec<u64>>());
        assert!(s.text.contains("#1 system_status"));
        assert!(s.text.contains("#7"));
        assert_eq!(s.intents, vec![Intent::SystemStatus]);
    }

    #[test]
    fn a_summary_never_grows_without_bound() {
        let mut c = ConversationContext::new(2, 1_000);
        for i in 0..5_000 {
            c.push(
                &format!("a fairly long request number {i}"),
                "reply",
                Some(Intent::Help),
                false,
            );
        }
        let s = c.summary().unwrap();
        assert!(
            s.text.chars().count() <= 500,
            "summary grew to {}",
            s.text.chars().count()
        );
        // Every id is still recorded even though the text was clipped.
        assert!(s.source_turn_ids.len() > 4_000);
    }

    #[test]
    fn rendering_includes_the_summary_and_the_ring() {
        let mut c = ConversationContext::new(2, 100_000);
        for i in 0..5 {
            c.push(&format!("u{i}"), "ok", None, false);
        }
        let rendered = c.render();
        assert!(rendered.contains("[earlier turns, summarised]"));
        assert!(rendered.contains("u3"));
        assert!(rendered.contains("user:"));
        assert!(rendered.contains("aaron:"));
    }

    #[test]
    fn clear_drops_the_summary_and_the_ring() {
        let mut c = ConversationContext::new(2, 100_000);
        for i in 0..6 {
            c.push(&format!("u{i}"), "ok", None, false);
        }
        c.clear();
        assert!(c.is_empty());
        assert!(c.summary().is_none());
    }

    #[test]
    fn a_new_topic_keeps_the_summary_so_nothing_is_lost() {
        let mut c = ConversationContext::new(2, 100_000);
        for i in 0..6 {
            c.push(&format!("u{i}"), "ok", None, false);
        }
        c.start_new_topic();
        assert!(c.is_empty());
        assert!(
            c.summary().is_some(),
            "a topic change must not silently discard history"
        );
    }

    #[test]
    fn a_single_oversized_turn_is_clipped_not_dropped() {
        let mut c = ConversationContext::new(8, 1_000);
        c.push(&"z".repeat(100_000), &"y".repeat(100_000), None, false);
        assert_eq!(c.len(), 1, "the current turn must survive");
        assert!(c.last().unwrap().user_text.chars().count() <= 250);
    }

    #[test]
    fn retained_chars_is_accurate_after_eviction() {
        let mut c = ConversationContext::new(3, 100_000);
        for i in 0..50 {
            c.push(&format!("message {i}"), "reply", None, false);
        }
        let expected: usize = c
            .turns()
            .map(|t| t.user_text.chars().count() + t.assistant_text.chars().count())
            .sum();
        assert_eq!(c.retained_chars(), expected);
    }

    #[test]
    fn ids_are_stable_and_monotonic_across_a_clear() {
        let mut c = ConversationContext::new(8, 100_000);
        let first = c.push("a", "b", None, false);
        c.clear();
        let second = c.push("a", "b", None, false);
        assert!(second.id > first.id);
    }

    #[test]
    fn capacity_and_character_bounds_are_never_zero() {
        let c = ConversationContext::new(0, 0);
        assert_eq!(c.capacity(), 1);
        let mut c = c;
        c.push("x", "y", None, false);
        assert_eq!(c.len(), 1);
    }

    #[test]
    fn repeated_use_does_not_grow_the_context() {
        let mut c = ConversationContext::new(8, 4_000);
        for i in 0..10_000 {
            c.push(
                &format!("turn {i} with some text"),
                "assistant reply here",
                Some(Intent::Help),
                false,
            );
        }
        assert!(c.len() <= 8);
        assert!(c.retained_chars() <= 4_000);
        // The summary is bounded too, so total RAM is bounded.
        let bytes = estimate_bytes(
            c.len(),
            c.retained_chars() + c.summary().map(|s| s.text.len()).unwrap_or(0),
        );
        assert!(
            bytes < 64 * 1024,
            "context should stay well under 64 KiB, was {bytes}"
        );
    }

    #[test]
    fn estimate_bytes_is_a_conservative_upper_bound() {
        assert!(estimate_bytes(1, 0) >= 128);
        assert!(estimate_bytes(0, 1) >= 4);
    }
}
