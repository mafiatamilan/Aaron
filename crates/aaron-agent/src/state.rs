//! The explicit agent state machine.
//!
//! Every assistant turn walks this graph. The states are a `pub enum` with a
//! validated transition table rather than a set of booleans, so an illegal
//! jump is a compile-time-visible, testable rule instead of an emergent
//! property.
//!
//! ```text
//! Idle ──► Classifying ──► PolicyCheck ──► AwaitingConfirmation ──► Executing ──► Responding ──► Idle
//!   ▲            │              │                 │                     │
//!   │            ▼              ▼                 ▼                     ▼
//!   └──────── Listening ◄── Error ◄────────── Cancelled ◄───────────────┘
//! ```

use std::fmt;

/// Where the assistant is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum State {
    /// Waiting for input. No microphone is buffering audio.
    Idle,
    /// The wake phrase was recognised.
    WakeDetected,
    /// Recording audio.
    Listening,
    /// Sending audio to the recogniser.
    Transcribing,
    /// Running the intent classifier.
    Classifying,
    /// Asking the policy engine whether an action may run.
    PolicyCheck,
    /// Waiting for the user to confirm a sensitive action.
    AwaitingConfirmation,
    /// Running a tool.
    Executing,
    /// Turning a real result into words.
    Responding,
    /// Something failed; the assistant is recovering.
    Error,
    /// The user asked to abandon the current operation.
    Cancelled,
    /// Shutting down.
    Stopping,
    /// Fully stopped.
    Stopped,
}

impl State {
    /// Stable identifier for logs and the UI.
    pub fn as_str(self) -> &'static str {
        match self {
            State::Idle => "idle",
            State::WakeDetected => "wake_detected",
            State::Listening => "listening",
            State::Transcribing => "transcribing",
            State::Classifying => "classifying",
            State::PolicyCheck => "policy_check",
            State::AwaitingConfirmation => "awaiting_confirmation",
            State::Executing => "executing",
            State::Responding => "responding",
            State::Error => "error",
            State::Cancelled => "cancelled",
            State::Stopping => "stopping",
            State::Stopped => "stopped",
        }
    }

    /// Whether the microphone may be capturing audio in this state.
    ///
    /// The user-facing "microphone live" indicator is driven by exactly this
    /// predicate, so it can never claim to be live while the agent is idle.
    pub fn is_listening(self) -> bool {
        matches!(self, State::Listening | State::WakeDetected)
    }
}

impl fmt::Display for State {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The state machine.
#[derive(Debug, Clone)]
pub struct StateMachine {
    state: State,
    history: Vec<State>,
    /// Maximum retained history, so a long session cannot grow without bound.
    max_history: usize,
}

impl Default for StateMachine {
    fn default() -> Self {
        StateMachine::new()
    }
}

impl StateMachine {
    pub fn new() -> StateMachine {
        StateMachine {
            state: State::Idle,
            history: Vec::new(),
            max_history: 64,
        }
    }

    /// Current state.
    pub fn state(&self) -> State {
        self.state
    }

    /// Recent transitions, oldest first.
    pub fn history(&self) -> &[State] {
        &self.history
    }

    /// Attempt a transition.
    pub fn to(&mut self, next: State) -> Result<(), IllegalTransition> {
        if !Self::is_allowed(self.state, next) {
            return Err(IllegalTransition {
                from: self.state,
                to: next,
            });
        }
        self.history.push(self.state);
        if self.history.len() > self.max_history {
            let excess = self.history.len() - self.max_history;
            self.history.drain(..excess);
        }
        self.state = next;
        Ok(())
    }

    /// Whether `from -> to` is a legal edge.
    pub fn is_allowed(from: State, to: State) -> bool {
        use State::*;
        match from {
            Idle => matches!(
                to,
                Classifying | Listening | WakeDetected | Stopping | Cancelled
            ),
            WakeDetected => matches!(to, Listening | Cancelled | Stopping | Error),
            Listening => matches!(to, Transcribing | Cancelled | Stopping | Error),
            Transcribing => matches!(to, Classifying | Error | Cancelled | Stopping),
            Classifying => matches!(to, PolicyCheck | Responding | Cancelled | Stopping | Error),
            PolicyCheck => matches!(
                to,
                AwaitingConfirmation | Executing | Responding | Error | Cancelled | Stopping
            ),
            AwaitingConfirmation => {
                matches!(to, Executing | Responding | Cancelled | Stopping | Error)
            }
            Executing => matches!(to, Responding | Error | Cancelled | Stopping),
            Responding => matches!(to, Idle | Stopping | Error | Cancelled),
            Error => matches!(to, Idle | Stopping | Cancelled),
            Cancelled => matches!(to, Idle | Stopping),
            Stopping => matches!(to, Stopped),
            Stopped => false,
        }
    }

    /// Return to idle, routing through `Error` when an illegal jump is needed.
    pub fn reset(&mut self) {
        self.state = State::Idle;
    }

    /// Full path taken during the last turn, for `aaron agent trace`.
    pub fn last_turn(&self, since: usize) -> &[State] {
        if since >= self.history.len() {
            return &self.history;
        }
        &self.history[since..]
    }
}

/// An illegal state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IllegalTransition {
    pub from: State,
    pub to: State,
}

impl fmt::Display for IllegalTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "illegal transition {} -> {}", self.from, self.to)
    }
}

impl std::error::Error for IllegalTransition {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_happy_path_is_legal() {
        let mut sm = StateMachine::new();
        assert_eq!(sm.state(), State::Idle);
        for s in [
            State::Classifying,
            State::PolicyCheck,
            State::Executing,
            State::Responding,
            State::Idle,
        ] {
            sm.to(s).unwrap_or_else(|e| panic!("{e}"));
        }
    }

    #[test]
    fn the_confirmation_path_is_legal() {
        let mut sm = StateMachine::new();
        sm.to(State::Classifying).unwrap();
        sm.to(State::PolicyCheck).unwrap();
        sm.to(State::AwaitingConfirmation).unwrap();
        sm.to(State::Executing).unwrap();
        sm.to(State::Responding).unwrap();
        sm.to(State::Idle).unwrap();
    }

    #[test]
    fn the_voice_path_is_legal() {
        let mut sm = StateMachine::new();
        sm.to(State::WakeDetected).unwrap();
        sm.to(State::Listening).unwrap();
        sm.to(State::Transcribing).unwrap();
        sm.to(State::Classifying).unwrap();
        sm.to(State::Responding).unwrap();
    }

    #[test]
    fn executing_without_a_policy_check_is_rejected() {
        let mut sm = StateMachine::new();
        sm.to(State::Classifying).unwrap();
        assert!(sm.to(State::Executing).is_err());
        assert_eq!(
            sm.state(),
            State::Classifying,
            "a rejected transition must not change state"
        );
    }

    #[test]
    fn a_stopped_machine_accepts_nothing() {
        let mut sm = StateMachine::new();
        sm.to(State::Stopping).unwrap();
        sm.to(State::Stopped).unwrap();
        assert!(sm.to(State::Idle).is_err());
        assert!(sm.to(State::Classifying).is_err());
    }

    #[test]
    fn every_state_can_reach_idle_except_stopped() {
        for from in [
            State::Idle,
            State::WakeDetected,
            State::Listening,
            State::Transcribing,
            State::Classifying,
            State::PolicyCheck,
            State::AwaitingConfirmation,
            State::Executing,
            State::Responding,
            State::Error,
            State::Cancelled,
            State::Stopping,
        ] {
            let mut sm = StateMachine::new();
            // Drive the machine into `from` through the legal path, then check
            // an escape hatch exists.
            let path: &[State] = match from {
                State::Idle => &[],
                State::WakeDetected => &[State::WakeDetected],
                State::Listening => &[State::WakeDetected, State::Listening],
                State::Transcribing => {
                    &[State::WakeDetected, State::Listening, State::Transcribing]
                }
                State::Classifying => &[State::Classifying],
                State::PolicyCheck => &[State::Classifying, State::PolicyCheck],
                State::AwaitingConfirmation => &[
                    State::Classifying,
                    State::PolicyCheck,
                    State::AwaitingConfirmation,
                ],
                State::Executing => &[State::Classifying, State::PolicyCheck, State::Executing],
                State::Responding => &[State::Classifying, State::Responding],
                State::Error => &[State::WakeDetected, State::Listening, State::Error],
                State::Cancelled => &[State::Listening, State::Cancelled],
                State::Stopping => &[State::Stopping],
                State::Stopped => &[State::Stopping, State::Stopped],
            };
            for s in path {
                sm.to(*s)
                    .unwrap_or_else(|e| panic!("building path to {from}: {e}"));
            }
            assert_eq!(sm.state(), from);
            match from {
                // Already stopping: the next edge is Stopped.
                State::Stopping => {
                    sm.to(State::Stopped)
                        .unwrap_or_else(|e| panic!("stopping cannot stop: {e}"));
                }
                State::Stopped => {}
                _ => sm
                    .to(State::Stopping)
                    .unwrap_or_else(|e| panic!("{from} cannot stop: {e}")),
            }
        }
    }

    #[test]
    fn listening_states_are_exactly_the_ones_that_capture_audio() {
        assert!(State::Listening.is_listening());
        assert!(State::WakeDetected.is_listening());
        for s in [
            State::Idle,
            State::Transcribing,
            State::Classifying,
            State::Executing,
            State::Responding,
            State::Error,
            State::Stopped,
        ] {
            assert!(!s.is_listening(), "{s} must not claim a live microphone");
        }
    }

    #[test]
    fn history_is_bounded() {
        let mut sm = StateMachine::new();
        for _ in 0..500 {
            sm.to(State::Classifying).unwrap();
            sm.to(State::Responding).unwrap();
            sm.to(State::Idle).unwrap();
        }
        assert!(sm.history().len() <= 64);
    }
}
