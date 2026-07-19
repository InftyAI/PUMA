//! FSM - Finite State Machine for sequence lifecycle
//!
//! This module defines states and events for the sequence state machine.
//! The actual transition logic is implemented in `Scheduler::transition()` and
//! private `Scheduler::transition_*()` methods.
//!
//! # Architecture
//!
//! - `states`: State definitions (Waiting, Prefilling, Decoding, etc.)
//! - `events`: Event types that trigger transitions
//!
//! # Usage
//!
//! States and events are used by the Scheduler:
//!
//! ```rust,ignore
//! use crate::fsm::{Event, SequenceState};
//! use crate::scheduler::Scheduler;
//!
//! let mut scheduler = Scheduler::new(...);
//! let state = SequenceState::Waiting(WaitingState { ... });
//! let event = Event::Schedule { tokens_per_block: 16 };
//!
//! // Scheduler owns BlockManager and performs transitions
//! let new_state = scheduler.transition(state, event)?;
//! ```
//!
//! # Design Rationale
//!
//! FSM logic is **inside Scheduler** rather than a separate `fsm::apply()` because:
//! - Scheduler owns `BlockManager` - no need to pass as parameter
//! - Direct access to scheduler context (tokens_per_block, etc.)
//! - Event-based `transition()` method provides single entry point for logging/metrics
//! - Type-safe internal `transition_*()` methods enforce correct state types
//!
//! See `docs/fsm_architecture.md` for detailed documentation.

pub mod events;
pub mod states;

// Re-export commonly used types
pub use events::Event;
pub use states::*;
