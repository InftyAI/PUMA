//! Scheduler Events - External API
//!
//! # Two-Level Event Architecture
//!
//! PUMA follows TokenSpeed's pattern with two distinct event levels:
//!
//! ## Level 1: FSM Events (Internal - `crate::fsm::Event`)
//! - Pure state machine transitions
//! - Created by scheduler internally
//! - Includes scheduler resources (BlockManager, allocators)
//! - Examples: Schedule, AppendTokens, Complete
//!
//! ## Level 2: Scheduler Events (External - `SchedulerEvent`)
//! - API boundary for external components
//! - Simple data, no resource pointers
//! - Can be created by clients, GPU, cache systems
//! - Examples: AddRequest, CancelRequest, TokensGenerated
//!
//! ## Why Two Levels?
//!
//! External components (GPU, clients) **cannot create** FSM events because:
//! - FSM events need `&mut BlockManager` (owned by scheduler)
//! - FSM events include policy decisions (scheduler computes)
//! - FSM events track internal state (scheduler maintains)
//!
//! The scheduler translates external events → FSM events, adding
//! the necessary context and resources.

use crate::block_manager::types::*;
use std::collections::HashMap;
use tokio::sync::mpsc;

/// External events that clients/backend send to the scheduler
///
/// Only includes events that come from outside the scheduler:
/// - AddRequest: Client starts a new request
/// - CancelRequest: Client cancels an ongoing request
/// - GetBlocks: Backend queries block IDs for inference
/// - GetStats: Monitoring/observability queries
///
/// Internal operations (token generation, state transitions, OOM handling)
/// are NOT events - the scheduler handles them directly in its run loop.
#[derive(Debug)]
pub enum SchedulerEvent {
    /// Client starts a new inference request
    AddRequest {
        seq_id: SequenceId,
        token_ids: Vec<TokenId>, // Tokenized by LLMEngine
        max_tokens: usize,
        response_tx: tokio::sync::oneshot::Sender<Result<String>>, // Send result back
    },

    /// Client cancels a request (user stop, disconnect, timeout)
    CancelRequest { seq_id: SequenceId },

    /// Backend queries block IDs for a sequence (for KV cache addressing)
    GetBlocks {
        seq_id: SequenceId,
        response: tokio::sync::oneshot::Sender<Result<Vec<BlockId>>>,
    },

    /// Query scheduler statistics
    GetStats {
        response: tokio::sync::oneshot::Sender<SchedulerStats>,
    },
}

#[derive(Debug, Clone)]
pub struct SchedulerStats {
    pub num_sequences: usize,
    pub num_running: usize,
    pub num_waiting: usize,
    pub num_preempted: usize,
    pub block_stats: HashMap<BlockType, BlockStats>,
}

pub type SchedulerEventSender = mpsc::UnboundedSender<SchedulerEvent>;
pub type SchedulerEventReceiver = mpsc::UnboundedReceiver<SchedulerEvent>;

/// Create event channel for scheduler
///
/// TODO: Use bounded channel to prevent OOM when producers outpace the event loop.
/// Capacity should be calculated based on available memory and average event size.
/// Consider: capacity = (available_memory * 0.1) / sizeof(SchedulerEvent)
pub fn create_event_channel() -> (SchedulerEventSender, SchedulerEventReceiver) {
    mpsc::unbounded_channel()
}
