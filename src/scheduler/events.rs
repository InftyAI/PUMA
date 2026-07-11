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
        prompt_tokens: usize,
        max_tokens: usize,
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
