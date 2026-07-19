use crate::block_manager::types::*;
use std::sync::Arc;

/// FSM States - each state owns its resources
///
/// State transitions:
/// Waiting → Scheduling → Prefilling → Decoding → Finished
///                            ↓           ↓
///                            └→ Preempted ←┘

#[derive(Debug, Clone)]
pub enum SequenceState {
    /// Pre-birth placeholder: the "from" state a new sequence transitions out
    /// of via `Event::Create`. Carries the `seq_id` so every state variant can
    /// answer `seq_id()`. Never persisted in the scheduler's sequence map.
    Empty(SequenceId),

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
            SequenceState::Empty(seq_id) => *seq_id,
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
            SequenceState::Empty(_) => "Empty",
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

#[cfg(test)]
mod tests {
    use super::*;

    fn waiting(seq_id: u64) -> SequenceState {
        SequenceState::Waiting(WaitingState {
            seq_id: SequenceId(seq_id),
            token_ids: Arc::new(vec![1, 2, 3]),
            max_tokens: 10,
        })
    }

    fn prefilling(seq_id: u64, blocks: Vec<BlockId>) -> SequenceState {
        SequenceState::Prefilling(PrefillingState {
            seq_id: SequenceId(seq_id),
            token_ids: Arc::new(vec![1, 2, 3]),
            blocks: Arc::new(blocks),
            tokens_filled: 0,
            tokens_total: 3,
            max_tokens: 10,
        })
    }

    fn decoding(seq_id: u64, blocks: Vec<BlockId>) -> SequenceState {
        SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(seq_id),
            token_ids: Arc::new(vec![1, 2, 3]),
            blocks: Arc::new(blocks),
            num_tokens: 3,
            max_tokens: 10,
        })
    }

    #[test]
    fn test_seq_id_for_every_variant() {
        assert_eq!(SequenceState::Empty(SequenceId(7)).seq_id(), SequenceId(7));
        assert_eq!(waiting(1).seq_id(), SequenceId(1));
        assert_eq!(prefilling(2, vec![]).seq_id(), SequenceId(2));
        assert_eq!(decoding(3, vec![]).seq_id(), SequenceId(3));

        let scheduling = SequenceState::Scheduling(SchedulingState {
            seq_id: SequenceId(4),
            prompt_tokens: 3,
            max_tokens: 10,
            blocks: Arc::new(vec![]),
        });
        assert_eq!(scheduling.seq_id(), SequenceId(4));

        let preempted = SequenceState::Preempted(PreemptedState {
            seq_id: SequenceId(5),
            token_ids: Arc::new(vec![1]),
            num_tokens: 1,
            max_tokens: 10,
        });
        assert_eq!(preempted.seq_id(), SequenceId(5));

        let finished = SequenceState::Finished(FinishedState {
            seq_id: SequenceId(6),
            finish_reason: FinishReason::Stop,
        });
        assert_eq!(finished.seq_id(), SequenceId(6));

        let aborted = SequenceState::Aborted(AbortedState {
            seq_id: SequenceId(8),
            reason: "boom".to_string(),
        });
        assert_eq!(aborted.seq_id(), SequenceId(8));
    }

    #[test]
    fn test_is_running() {
        assert!(prefilling(1, vec![]).is_running());
        assert!(decoding(1, vec![]).is_running());
        assert!(!waiting(1).is_running());
        assert!(!SequenceState::Empty(SequenceId(1)).is_running());
    }

    #[test]
    fn test_is_waiting() {
        assert!(waiting(1).is_waiting());
        let scheduling = SequenceState::Scheduling(SchedulingState {
            seq_id: SequenceId(1),
            prompt_tokens: 3,
            max_tokens: 10,
            blocks: Arc::new(vec![]),
        });
        assert!(scheduling.is_waiting());
        assert!(!decoding(1, vec![]).is_waiting());
    }

    #[test]
    fn test_is_finished() {
        let finished = SequenceState::Finished(FinishedState {
            seq_id: SequenceId(1),
            finish_reason: FinishReason::MaxTokens,
        });
        let aborted = SequenceState::Aborted(AbortedState {
            seq_id: SequenceId(1),
            reason: "x".to_string(),
        });
        assert!(finished.is_finished());
        assert!(aborted.is_finished());
        assert!(!decoding(1, vec![]).is_finished());
    }

    #[test]
    fn test_blocks_exposed_only_for_resource_holding_states() {
        // Prefilling/Decoding always expose their blocks.
        assert_eq!(
            prefilling(1, vec![BlockId(0), BlockId(1)]).blocks(),
            Some(&[BlockId(0), BlockId(1)][..])
        );
        assert_eq!(
            decoding(1, vec![BlockId(2)]).blocks(),
            Some(&[BlockId(2)][..])
        );

        // Scheduling exposes blocks only when it has partially allocated some.
        let scheduling_with = SequenceState::Scheduling(SchedulingState {
            seq_id: SequenceId(1),
            prompt_tokens: 3,
            max_tokens: 10,
            blocks: Arc::new(vec![BlockId(9)]),
        });
        assert_eq!(scheduling_with.blocks(), Some(&[BlockId(9)][..]));

        let scheduling_empty = SequenceState::Scheduling(SchedulingState {
            seq_id: SequenceId(1),
            prompt_tokens: 3,
            max_tokens: 10,
            blocks: Arc::new(vec![]),
        });
        assert_eq!(scheduling_empty.blocks(), None);

        // States without KV blocks return None.
        assert_eq!(waiting(1).blocks(), None);
        assert_eq!(SequenceState::Empty(SequenceId(1)).blocks(), None);
    }

    #[test]
    fn test_variant_name() {
        assert_eq!(SequenceState::Empty(SequenceId(1)).variant_name(), "Empty");
        assert_eq!(waiting(1).variant_name(), "Waiting");
        assert_eq!(prefilling(1, vec![]).variant_name(), "Prefilling");
        assert_eq!(decoding(1, vec![]).variant_name(), "Decoding");
    }
}
