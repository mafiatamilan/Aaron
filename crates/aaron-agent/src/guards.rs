//! Loop and runaway-execution guards.
//!
//! AARON is voice-driven, so a mis-heard phrase could otherwise fire the same
//! tool repeatedly. Two guards apply to every turn:
//!
//! * **Per-turn budget** — at most `max_tool_calls_per_utterance` tools run for
//!   one utterance.
//! * **Repetition guard** — the same `(tool, arguments)` pair may not run more
//!   than `max_repeat_identical_actions` times in a row. This is what stops an
//!   echo of the user's voice from re-triggering `run_tests` forever.

use std::collections::VecDeque;

use aaron_core::error::{Error, Result};

/// Why a repeat was blocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockReason {
    /// The per-turn tool budget is exhausted.
    TurnBudget,
    /// The identical action repeated too many times.
    Repetition { repeats: usize },
}

impl std::fmt::Display for BlockReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BlockReason::TurnBudget => write!(f, "the per-utterance tool limit was reached"),
            BlockReason::Repetition { repeats } => {
                write!(
                    f,
                    "`{repeats}` identical actions in a row; refusing to repeat further"
                )
            }
        }
    }
}

/// A fingerprint of one proposed action.
pub fn fingerprint(tool: &str, args_json: &str) -> String {
    format!("{tool}:{}", aaron_core::text::fnv1a64(args_json.as_bytes()))
}

/// Guards against runaway execution.
#[derive(Debug, Clone)]
pub struct LoopGuard {
    recent: VecDeque<String>,
    max_repeat: usize,
    max_per_turn: usize,
    calls_this_turn: usize,
}

impl LoopGuard {
    pub fn new(max_repeat: usize, max_per_turn: usize) -> LoopGuard {
        LoopGuard {
            recent: VecDeque::new(),
            max_repeat: max_repeat.max(1),
            max_per_turn: max_per_turn.max(1),
            calls_this_turn: 0,
        }
    }

    /// Reset the per-turn counter. Called at the start of every utterance.
    pub fn begin_turn(&mut self) {
        self.calls_this_turn = 0;
    }

    /// Record and authorise one action.
    pub fn admit(&mut self, tool: &str, args_json: &str) -> Result<()> {
        if self.calls_this_turn >= self.max_per_turn {
            return Err(Error::LimitExceeded(BlockReason::TurnBudget.to_string()));
        }
        let fp = fingerprint(tool, args_json);
        let repeats = self.recent.iter().rev().take_while(|p| **p == fp).count() + 1;
        if repeats > self.max_repeat {
            return Err(Error::LimitExceeded(
                BlockReason::Repetition { repeats }.to_string(),
            ));
        }
        self.recent.push_back(fp);
        // Keep a bounded history: only the trailing run matters.
        while self.recent.len() > self.max_repeat * 4 {
            self.recent.pop_front();
        }
        self.calls_this_turn += 1;
        Ok(())
    }

    /// How many identical actions are currently in the trailing run.
    pub fn trailing_repeats(&self, tool: &str, args_json: &str) -> usize {
        let fp = fingerprint(tool, args_json);
        self.recent.iter().rev().take_while(|p| **p == fp).count()
    }

    /// Forget all history, e.g. after an explicit cancel or shutdown.
    pub fn reset(&mut self) {
        self.recent.clear();
        self.begin_turn();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allows_a_single_action_per_turn_by_default() {
        let mut g = LoopGuard::new(2, 1);
        assert!(g.admit("system_status", "{}").is_ok());
        assert!(matches!(
            g.admit("system_status", "{}"),
            Err(Error::LimitExceeded(m)) if m.contains("per-utterance tool limit")
        ));
        g.begin_turn();
        assert!(g.admit("system_status", "{}").is_ok());
    }

    #[test]
    fn blocks_identical_repeats_beyond_the_limit() {
        let mut g = LoopGuard::new(2, 10);
        g.begin_turn();
        assert!(g.admit("run_tests", "{\"suite\":\"cargo\"}").is_ok());
        g.begin_turn();
        assert!(g.admit("run_tests", "{\"suite\":\"cargo\"}").is_ok());
        g.begin_turn();
        let err = g.admit("run_tests", "{\"suite\":\"cargo\"}").unwrap_err();
        assert!(err.to_string().contains("identical actions"), "{err}");
    }

    #[test]
    fn different_arguments_do_not_count_as_repeats() {
        let mut g = LoopGuard::new(2, 10);
        for suite in ["cargo", "cargo-unit", "npm", "cargo-unit"] {
            g.begin_turn();
            assert!(g
                .admit("run_tests", &format!("{{\"suite\":\"{suite}\"}}"))
                .is_ok());
        }
    }

    #[test]
    fn an_intervening_action_breaks_the_run() {
        let mut g = LoopGuard::new(2, 10);
        g.begin_turn();
        g.admit("run_tests", "{}").unwrap();
        g.begin_turn();
        g.admit("system_status", "{}").unwrap();
        // The intervening action reset the run, so two more `run_tests` calls
        // are still permitted with a repeat limit of two.
        for expected_repeats in 1..=2 {
            g.begin_turn();
            g.admit("run_tests", "{}").unwrap();
            assert_eq!(g.trailing_repeats("run_tests", "{}"), expected_repeats);
        }
        g.begin_turn();
        let err = g.admit("run_tests", "{}").unwrap_err();
        assert!(err.to_string().contains("identical"), "{err}");
    }

    #[test]
    fn trailing_repeats_is_reported() {
        let mut g = LoopGuard::new(3, 10);
        g.begin_turn();
        g.admit("a", "{}").unwrap();
        g.begin_turn();
        g.admit("a", "{}").unwrap();
        assert_eq!(g.trailing_repeats("a", "{}"), 2);
        assert_eq!(g.trailing_repeats("b", "{}"), 0);
    }

    #[test]
    fn reset_clears_everything() {
        let mut g = LoopGuard::new(1, 10);
        g.begin_turn();
        g.admit("a", "{}").unwrap();
        g.begin_turn();
        assert!(g.admit("a", "{}").is_err());
        g.reset();
        assert!(g.admit("a", "{}").is_ok());
    }

    #[test]
    fn fingerprints_differ_for_different_arguments() {
        assert_ne!(fingerprint("t", "{}"), fingerprint("t", "{\"a\":1}"));
        assert_ne!(fingerprint("a", "{}"), fingerprint("b", "{}"));
        assert_eq!(fingerprint("t", "{}"), fingerprint("t", "{}"));
    }
}
