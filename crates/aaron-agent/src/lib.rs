//! The AARON agent: state machine, guards and untrusted-content handling.
//!
//! The agent is the only component that may ask a tool to do something, and it
//! can only do so through a tool that is registered *and* permitted by the
//! policy engine. The intent classifier proposes; the agent disposes.

pub mod agent;
pub mod guards;
pub mod session;
pub mod state;
pub mod untrusted;
pub mod voice_confirm;

pub use agent::{
    build_arguments, describe_action, tool_for_intent, Agent, AgentDeps, AgentResponse,
    AlwaysDecline, AutoConfirm, Confirmer, NonInteractive, Outcome2,
};
pub use guards::{fingerprint, BlockReason, LoopGuard};
pub use session::{SessionOutcome, VoiceSession};
pub use state::{IllegalTransition, State, StateMachine};
pub use untrusted::{frame, scrub, Untrusted};
pub use voice_confirm::VoiceConfirmer;
