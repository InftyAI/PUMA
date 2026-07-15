use super::states::FinishReason;
use crate::block_manager::types::SequenceId;

/// FSM Events - pure data that triggers state transitions
///
/// These events represent state machine transitions and are created
/// internally by the scheduler. External components should use
/// SchedulerEvent instead.
#[derive(Debug, Clone)]
pub enum Event {
    /// Allocate blocks and start prefilling
    Schedule { tokens_per_block: usize },

    /// Append tokens to a running sequence
    AppendTokens {
        num_tokens: usize,
        tokens_per_block: usize,
    },

    /// Mark sequence as complete
    Complete { reason: FinishReason },

    /// Preempt a running sequence (free blocks)
    Preempt,

    /// Resume a preempted sequence
    Resume,

    /// Fork a sequence (copy-on-write)
    Fork { child_id: SequenceId },

    /// Abort a sequence with error
    Abort { reason: String },
}
