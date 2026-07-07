
use super::states::*;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use tracing::{debug, warn};

/// FSM Events - transform states with type-safe transitions
///
/// Pattern: Event takes old state, returns new state
/// Invalid transitions return Error::invalid_transition

/// Schedule a waiting sequence (allocate blocks)
pub struct ScheduleEvent<'a> {
    pub block_manager: &'a mut BlockManager,
    pub tokens_per_block: usize,
}

impl<'a> ScheduleEvent<'a> {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        match state {
            SequenceState::Waiting(s) => self.apply_to_waiting(s),
            _ => Err(Error::invalid_transition(
                "ScheduleEvent requires Waiting state",
            )),
        }
    }

    fn apply_to_waiting(self, state: WaitingState) -> Result<SequenceState> {
        let blocks_needed = state.prompt_tokens.div_ceil(self.tokens_per_block);

        let mut blocks = Vec::new();
        for _ in 0..blocks_needed {
            match self.block_manager.allocate() {
                Ok(block_id) => blocks.push(block_id),
                Err(Error::OutOfMemory) => {
                    // OOM during scheduling - free what we allocated
                    for block in blocks {
                        let _ = self.block_manager.free(block);
                    }
                    return Err(Error::OutOfMemory);
                }
                Err(e) => {
                    // Other error - cleanup and propagate
                    for block in blocks {
                        let _ = self.block_manager.free(block);
                    }
                    return Err(e);
                }
            }
        }

        debug!(
            "Scheduled seq {:?}: allocated {} blocks for {} tokens",
            state.seq_id,
            blocks.len(),
            state.prompt_tokens
        );

        Ok(SequenceState::Prefilling(PrefillingState {
            seq_id: state.seq_id,
            blocks,
            tokens_filled: 0,
            tokens_total: state.prompt_tokens,
            max_tokens: state.max_tokens,
        }))
    }
}

/// Append tokens to a running sequence
pub struct AppendTokensEvent<'a> {
    pub num_tokens: usize,
    pub block_manager: &'a mut BlockManager,
    pub tokens_per_block: usize,
}

impl<'a> AppendTokensEvent<'a> {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        match state {
            SequenceState::Prefilling(s) => self.apply_to_prefilling(s),
            SequenceState::Decoding(s) => self.apply_to_decoding(s),
            _ => Err(Error::invalid_transition(
                "AppendTokensEvent requires Prefilling or Decoding state",
            )),
        }
    }

    fn apply_to_prefilling(self, mut state: PrefillingState) -> Result<SequenceState> {
        state.tokens_filled += self.num_tokens;

        debug!(
            "Prefilling seq {:?}: {}/{} tokens",
            state.seq_id, state.tokens_filled, state.tokens_total
        );

        if state.tokens_filled >= state.tokens_total {
            // Transition to decode
            debug!("Seq {:?} transitioning to Decoding", state.seq_id);
            Ok(SequenceState::Decoding(DecodingState {
                seq_id: state.seq_id,
                blocks: state.blocks,
                num_tokens: state.tokens_filled,
                max_tokens: state.max_tokens,
            }))
        } else {
            Ok(SequenceState::Prefilling(state))
        }
    }

    fn apply_to_decoding(self, mut state: DecodingState) -> Result<SequenceState> {
        state.num_tokens += self.num_tokens;

        // Check if finished
        if state.num_tokens >= state.max_tokens {
            // Free blocks
            for block_id in state.blocks {
                if let Err(e) = self.block_manager.free(block_id) {
                    warn!(
                        "Failed to free block {:?} for {:?}: {:?}",
                        block_id, state.seq_id, e
                    );
                }
            }

            debug!(
                "Seq {:?} finished: reached max_tokens ({})",
                state.seq_id, state.max_tokens
            );

            return Ok(SequenceState::Finished(FinishedState {
                seq_id: state.seq_id,
                finish_reason: FinishReason::MaxTokens,
            }));
        }

        // Check if need more blocks
        let blocks_needed = state.num_tokens.div_ceil(self.tokens_per_block);
        let initial_block_count = state.blocks.len();

        while state.blocks.len() < blocks_needed {
            match self.block_manager.allocate() {
                Ok(block_id) => {
                    state.blocks.push(block_id);
                    debug!(
                        "Seq {:?}: allocated block, total blocks: {}",
                        state.seq_id,
                        state.blocks.len()
                    );
                }
                Err(e) => {
                    // Free any blocks we allocated in this call
                    warn!("Failed to allocate block for {:?}: {:?}", state.seq_id, e);
                    for block_id in state.blocks.drain(initial_block_count..) {
                        let _ = self.block_manager.free(block_id);
                    }
                    return Err(e);
                }
            }
        }

        Ok(SequenceState::Decoding(state))
    }
}

/// Preempt a running sequence (free blocks)
pub struct PreemptEvent<'a> {
    pub block_manager: &'a mut BlockManager,
}

impl<'a> PreemptEvent<'a> {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        match state {
            SequenceState::Decoding(s) => self.apply_to_decoding(s),
            SequenceState::Prefilling(s) => self.apply_to_prefilling(s),
            _ => Err(Error::invalid_transition(
                "PreemptEvent requires running state",
            )),
        }
    }

    fn apply_to_decoding(self, state: DecodingState) -> Result<SequenceState> {
        // Free all blocks
        for block_id in state.blocks {
            if let Err(e) = self.block_manager.free(block_id) {
                warn!(
                    "Failed to free block {:?} for {:?}: {:?}",
                    block_id, state.seq_id, e
                );
            }
        }

        debug!("Preempted seq {:?}", state.seq_id);

        Ok(SequenceState::Preempted(PreemptedState {
            seq_id: state.seq_id,
            num_tokens: state.num_tokens,
            max_tokens: state.max_tokens,
        }))
    }

    fn apply_to_prefilling(self, state: PrefillingState) -> Result<SequenceState> {
        // Free all blocks (best effort)
        for block_id in state.blocks {
            if let Err(e) = self.block_manager.free(block_id) {
                warn!(
                    "Failed to free block {:?} for {:?}: {:?}",
                    block_id, state.seq_id, e
                );
            }
        }

        debug!("Preempted seq {:?} during prefill", state.seq_id);

        Ok(SequenceState::Preempted(PreemptedState {
            seq_id: state.seq_id,
            num_tokens: state.tokens_filled,
            max_tokens: state.max_tokens,
        }))
    }
}

/// Resume a preempted sequence
pub struct ResumeEvent;

impl ResumeEvent {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        match state {
            SequenceState::Preempted(s) => self.apply_to_preempted(s),
            _ => Err(Error::invalid_transition(
                "ResumeEvent requires Preempted state",
            )),
        }
    }

    fn apply_to_preempted(self, state: PreemptedState) -> Result<SequenceState> {
        debug!("Resuming seq {:?}", state.seq_id);

        // Transition back to waiting for rescheduling
        Ok(SequenceState::Waiting(WaitingState {
            seq_id: state.seq_id,
            prompt_tokens: state.num_tokens,
            max_tokens: state.max_tokens,
        }))
    }
}

/// Fork a sequence (copy-on-write)
pub struct ForkEvent<'a> {
    pub child_id: SequenceId,
    pub block_manager: &'a mut BlockManager,
}

impl<'a> ForkEvent<'a> {
    pub fn apply(self, state: SequenceState) -> Result<(SequenceState, SequenceState)> {
        match state {
            SequenceState::Decoding(s) => self.apply_to_decoding(s),
            _ => Err(Error::invalid_transition(
                "ForkEvent requires Decoding state",
            )),
        }
    }

    fn apply_to_decoding(self, state: DecodingState) -> Result<(SequenceState, SequenceState)> {
        // Copy-on-write: increment ref counts
        for &block_id in &state.blocks {
            self.block_manager.add_ref(block_id)?;
        }

        let child = DecodingState {
            seq_id: self.child_id,
            blocks: state.blocks.clone(),
            num_tokens: state.num_tokens,
            max_tokens: state.max_tokens,
        };

        debug!("Forked seq {:?} → {:?}", state.seq_id, child.seq_id);

        Ok((
            SequenceState::Decoding(state),
            SequenceState::Decoding(child),
        ))
    }
}

/// Complete a sequence
pub struct CompleteEvent<'a> {
    pub reason: FinishReason,
    pub block_manager: &'a mut BlockManager,
}

impl<'a> CompleteEvent<'a> {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        match state {
            SequenceState::Decoding(s) => self.apply_to_decoding(s),
            SequenceState::Prefilling(s) => self.apply_to_prefilling(s),
            _ => Err(Error::invalid_transition(
                "CompleteEvent requires running state",
            )),
        }
    }

    fn apply_to_decoding(self, state: DecodingState) -> Result<SequenceState> {
        // Free all blocks
        for block_id in state.blocks {
            if let Err(e) = self.block_manager.free(block_id) {
                warn!(
                    "Failed to free block {:?} for {:?}: {:?}",
                    block_id, state.seq_id, e
                );
            }
        }

        debug!("Completed seq {:?}: {:?}", state.seq_id, self.reason);

        Ok(SequenceState::Finished(FinishedState {
            seq_id: state.seq_id,
            finish_reason: self.reason,
        }))
    }

    fn apply_to_prefilling(self, state: PrefillingState) -> Result<SequenceState> {
        // Free all blocks
        for block_id in state.blocks {
            if let Err(e) = self.block_manager.free(block_id) {
                warn!(
                    "Failed to free block {:?} for {:?}: {:?}",
                    block_id, state.seq_id, e
                );
            }
        }

        debug!(
            "Completed seq {:?} during prefill: {:?}",
            state.seq_id, self.reason
        );

        Ok(SequenceState::Finished(FinishedState {
            seq_id: state.seq_id,
            finish_reason: self.reason,
        }))
    }
}

/// Abort a sequence (error or cancellation)
pub struct AbortEvent<'a> {
    pub reason: String,
    pub block_manager: &'a mut BlockManager,
}

impl<'a> AbortEvent<'a> {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        // Can abort from any state
        let seq_id = state.seq_id();

        // Free blocks if any
        if let Some(blocks) = state.blocks() {
            for &block_id in blocks {
                if let Err(e) = self.block_manager.free(block_id) {
                    warn!(
                        "Failed to free block {:?} for {:?}: {:?}",
                        block_id, seq_id, e
                    );
                }
            }
        }

        warn!("Aborted seq {:?}: {}", seq_id, self.reason);

        Ok(SequenceState::Aborted(AbortedState {
            seq_id,
            reason: self.reason,
        }))
    }
}

/// Invalid transition error helper
impl Error {
    pub fn invalid_transition(msg: &'static str) -> Self {
        Error::AllocationError(format!("Invalid state transition: {}", msg))
    }
}
