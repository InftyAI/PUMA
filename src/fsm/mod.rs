//! FSM - Finite State Machine for sequence lifecycle
//!
//! This module defines the *data* of the state machine: the states and the
//! events that trigger transitions. The actual transition logic lives in
//! [`SequenceManager`](crate::sequence_manager::SequenceManager) — its private
//! `transition()`/`transition_*()` methods compute the next state and
//! allocate/free blocks along the way.
//!
//! # Architecture
//!
//! - `states`: State definitions (Waiting, Prefilling, Decoding, etc.)
//! - `events`: Event types that trigger transitions
//!
//! # Usage
//!
//! The `Scheduler` (policy) drives the `SequenceManager` (mechanism) with these
//! events; it never touches `BlockManager` or transition logic directly:
//!
//! ```rust,ignore
//! use crate::fsm::Event;
//!
//! // seq_manager: SequenceManager, owned by the scheduler
//! seq_manager.create(seq_id, token_ids, max_tokens)?;
//! let new_state = seq_manager.advance(seq_id, Event::Schedule { tokens_per_block: 16 })?;
//! ```
//!
//! # Design Rationale
//!
//! Transition logic lives **inside `SequenceManager`** rather than a free
//! `fsm::apply()` because:
//! - The manager owns `BlockManager` - no need to pass it as a parameter
//! - Direct access to context (tokens_per_block, etc.)
//! - Event-based `advance()` gives a single entry point for logging/metrics
//! - Type-safe internal `transition_*()` methods enforce correct state types
//!
//! See `docs/fsm_architecture.md` for detailed documentation.

pub mod events;
pub mod states;

// Re-export commonly used types
pub use events::Event;
pub use states::*;
