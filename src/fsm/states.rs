use crate::block_manager::types::*;
use std::sync::Arc;

/// FSM States - each state owns its resources
///
/// State transitions:
/// Waiting → Scheduling → Prefilling → Decoding → Finished
///                            ↓           ↓
///                            └→ Preempted ←┘

#[derive(Debug, Clone, Default)]
pub enum SequenceState {
    /// Pre-birth placeholder: the default "from" state a new sequence
    /// transitions out of via `Event::Create`. Never persisted in the
    /// scheduler's sequence map.
    #[default]
    Empty,

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
#[derive(Debug, Clone)]
pub struct WaitingState {
    pub seq_id: SequenceId,
    pub token_ids: Arc<Vec<TokenId>>, // Arc for O(1) cloning, GPU gets reference
    pub max_tokens: usize,
}

/// Scheduling state - in process of allocating blocks
#[derive(Debug, Clone)]
pub struct SchedulingState {
    pub seq_id: SequenceId,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
    // Partially allocated blocks (if any)
    pub blocks: Arc<Vec<BlockId>>,
}

/// Prefilling state - processing prompt
#[derive(Debug, Clone)]
pub struct PrefillingState {
    pub seq_id: SequenceId,
    pub token_ids: Arc<Vec<TokenId>>, // Arc for O(1) cloning, needed if preempted
    pub blocks: Arc<Vec<BlockId>>,    // Arc for cheap cloning (O(1))
    pub tokens_filled: usize,
    pub tokens_total: usize,
    pub max_tokens: usize,
}

/// Decoding state - generating tokens
#[derive(Debug, Clone)]
pub struct DecodingState {
    pub seq_id: SequenceId,
    pub token_ids: Arc<Vec<TokenId>>, // Arc for O(1) cloning, needed if preempted
    pub blocks: Arc<Vec<BlockId>>,    // Arc for cheap cloning (O(1))
    pub num_tokens: usize,
    pub max_tokens: usize,
}

/// Preempted state - blocks freed, waiting to resume
#[derive(Debug, Clone)]
pub struct PreemptedState {
    pub seq_id: SequenceId,
    pub token_ids: Arc<Vec<TokenId>>, // Arc for O(1) cloning, needed to resume
    pub num_tokens: usize,            // How many tokens were generated so far
    pub max_tokens: usize,
    // No blocks - they were freed
}

/// Finished state - all resources freed
#[derive(Debug, Clone)]
pub struct FinishedState {
    pub seq_id: SequenceId,
    pub finish_reason: FinishReason,
}

/// Aborted state - error or cancellation
#[derive(Debug, Clone)]
pub struct AbortedState {
    pub seq_id: SequenceId,
    pub reason: String,
}

#[derive(Debug, Clone, PartialEq)]
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
            SequenceState::Empty => panic!("Empty state has no seq_id"),
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
            SequenceState::Prefilling(s) => Some(s.blocks.as_ref()),
            SequenceState::Decoding(s) => Some(s.blocks.as_ref()),
            SequenceState::Scheduling(s) if !s.blocks.is_empty() => Some(s.blocks.as_ref()),
            _ => None,
        }
    }

    pub fn variant_name(&self) -> &'static str {
        match self {
            SequenceState::Empty => "Empty",
            SequenceState::Waiting(_) => "Waiting",
            SequenceState::Scheduling(_) => "Scheduling",
            SequenceState::Prefilling(_) => "Prefilling",
            SequenceState::Decoding(_) => "Decoding",
            SequenceState::Preempted(_) => "Preempted",
            SequenceState::Finished(_) => "Finished",
            SequenceState::Aborted(_) => "Aborted",
        }
    }
}
