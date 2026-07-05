use crate::block_manager::types::*;
use std::collections::HashMap;
use tokio::sync::mpsc;

/// Events that drive the sequence manager
#[derive(Debug)]
pub enum SequenceEvent {
    // Request lifecycle
    AddRequest {
        seq_id: SequenceId,
        prompt_tokens: usize,
        max_tokens: usize,
    },

    // Token generation
    AppendTokens {
        seq_id: SequenceId,
        num_tokens: usize,
    },

    // Sequence forking (for beam search, parallel sampling)
    ForkSequence {
        parent_id: SequenceId,
        child_id: SequenceId,
    },

    // Completion
    CompleteSequence {
        seq_id: SequenceId,
    },

    // Preemption (when OOM)
    PreemptSequence {
        seq_id: SequenceId,
    },

    // Resume preempted sequence
    ResumeSequence {
        seq_id: SequenceId,
    },

    // Query
    GetBlocks {
        seq_id: SequenceId,
        response: tokio::sync::oneshot::Sender<Result<Vec<BlockId>>>,
    },

    // Stats
    GetStats {
        response: tokio::sync::oneshot::Sender<SequenceManagerStats>,
    },
}

#[derive(Debug, Clone)]
pub struct SequenceManagerStats {
    pub num_sequences: usize,
    pub num_running: usize,
    pub num_waiting: usize,
    pub num_preempted: usize,
    pub block_stats: HashMap<BlockType, BlockStats>,
}

pub type SequenceEventSender = mpsc::UnboundedSender<SequenceEvent>;
pub type SequenceEventReceiver = mpsc::UnboundedReceiver<SequenceEvent>;

pub fn create_event_channel() -> (SequenceEventSender, SequenceEventReceiver) {
    mpsc::unbounded_channel()
}
