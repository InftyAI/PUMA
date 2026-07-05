use crate::block_manager::types::*;

/// FSM States - each state owns its resources
///
/// State transitions:
/// Waiting → Scheduling → Prefilling → Decoding → Finished
///                            ↓           ↓
///                            └→ Preempted ←┘

#[derive(Debug)]
pub enum SequenceState {
    /// Waiting in queue for scheduling
    Waiting(WaitingState),

    /// Being scheduled (allocating resources)
    Scheduling(SchedulingState),

    /// Running prefill phase
    Prefilling(PrefillingState),

    /// Running decode phase (generating tokens)
    Decoding(DecodingState),

    /// Preempted due to OOM
    Preempted(PreemptedState),

    /// Finished generation
    Finished(FinishedState),

    /// Aborted (error or cancelled)
    Aborted(AbortedState),
}

/// Waiting state - no resources allocated yet
#[derive(Debug)]
pub struct WaitingState {
    pub seq_id: SequenceId,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
}

/// Scheduling state - in process of allocating blocks
#[derive(Debug)]
pub struct SchedulingState {
    pub seq_id: SequenceId,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    // Partially allocated blocks (if any)
    pub blocks: Vec<BlockId>,
}

/// Prefilling state - processing prompt
#[derive(Debug)]
pub struct PrefillingState {
    pub seq_id: SequenceId,
    pub blocks: Vec<BlockId>,
    pub tokens_filled: usize,
    pub tokens_total: usize,
    pub max_tokens: usize,
}

/// Decoding state - generating tokens
#[derive(Debug)]
pub struct DecodingState {
    pub seq_id: SequenceId,
    pub blocks: Vec<BlockId>,
    pub num_tokens: usize,
    pub max_tokens: usize,
}

/// Preempted state - blocks freed, waiting to resume
#[derive(Debug)]
pub struct PreemptedState {
    pub seq_id: SequenceId,
    pub num_tokens: usize,
    pub max_tokens: usize,
    // No blocks - they were freed
}

/// Finished state - all resources freed
#[derive(Debug)]
pub struct FinishedState {
    pub seq_id: SequenceId,
    pub finish_reason: FinishReason,
}

/// Aborted state - error or cancellation
#[derive(Debug)]
pub struct AbortedState {
    pub seq_id: SequenceId,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub enum FinishReason {
    /// Reached max_tokens limit
    MaxTokens,
    /// Natural completion (EOS token)
    Stop,
    /// User cancelled
    Cancelled,
}

// Helper methods for SequenceState
impl SequenceState {
    pub fn seq_id(&self) -> SequenceId {
        match self {
            SequenceState::Waiting(s) => s.seq_id,
            SequenceState::Scheduling(s) => s.seq_id,
            SequenceState::Prefilling(s) => s.seq_id,
            SequenceState::Decoding(s) => s.seq_id,
            SequenceState::Preempted(s) => s.seq_id,
            SequenceState::Finished(s) => s.seq_id,
            SequenceState::Aborted(s) => s.seq_id,
        }
    }

    pub fn is_running(&self) -> bool {
        matches!(
            self,
            SequenceState::Prefilling(_) | SequenceState::Decoding(_)
        )
    }

    pub fn is_waiting(&self) -> bool {
        matches!(
            self,
            SequenceState::Waiting(_) | SequenceState::Scheduling(_)
        )
    }

    pub fn is_finished(&self) -> bool {
        matches!(self, SequenceState::Finished(_) | SequenceState::Aborted(_))
    }

    pub fn blocks(&self) -> Option<&[BlockId]> {
        match self {
            SequenceState::Prefilling(s) => Some(&s.blocks),
            SequenceState::Decoding(s) => Some(&s.blocks),
            SequenceState::Scheduling(s) if !s.blocks.is_empty() => Some(&s.blocks),
            _ => None,
        }
    }
}
