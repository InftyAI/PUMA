use super::events::Event;
use super::states::*;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use std::sync::Arc;
use tracing::{debug, warn};

/// Apply an event to a state, producing a new state
///
/// Pure function that handles all state transitions.
/// Pattern: (current_state, event) -> new_state
pub fn apply(
    state: SequenceState,
    event: Event,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    match (state, event) {
        // Waiting → Prefilling (Schedule)
        (SequenceState::Waiting(s), Event::Schedule { tokens_per_block }) => {
            apply_schedule(s, block_manager, tokens_per_block)
        }

        // Prefilling → Decoding/Prefilling (AppendTokens)
        (
            SequenceState::Prefilling(s),
            Event::AppendTokens {
                num_tokens,
                tokens_per_block,
            },
        ) => apply_append_tokens_prefilling(s, num_tokens, block_manager, tokens_per_block),

        // Decoding → Decoding/Finished (AppendTokens)
        (
            SequenceState::Decoding(s),
            Event::AppendTokens {
                num_tokens,
                tokens_per_block,
            },
        ) => apply_append_tokens_decoding(s, num_tokens, block_manager, tokens_per_block),

        // Prefilling/Decoding → Finished (Complete)
        (SequenceState::Prefilling(s), Event::Complete { reason }) => {
            apply_complete_prefilling(s, reason, block_manager)
        }
        (SequenceState::Decoding(s), Event::Complete { reason }) => {
            apply_complete_decoding(s, reason, block_manager)
        }

        // Prefilling/Decoding → Preempted (Preempt)
        (SequenceState::Prefilling(s), Event::Preempt) => {
            apply_preempt_prefilling(s, block_manager)
        }
        (SequenceState::Decoding(s), Event::Preempt) => apply_preempt_decoding(s, block_manager),

        // Preempted → Waiting (Resume)
        (SequenceState::Preempted(s), Event::Resume) => apply_resume(s),

        // Decoding → (Decoding, Decoding) (Fork)
        (SequenceState::Decoding(s), Event::Fork { child_id }) => {
            apply_fork(s, child_id, block_manager)
        }

        // Any → Aborted (Abort)
        (state, Event::Abort { reason }) => apply_abort(state, reason, block_manager),

        // Invalid transitions
        (_state, _event) => Err(Error::InvalidTransition("Invalid state transition")),
    }
}

fn state_name(state: &SequenceState) -> &'static str {
    match state {
        SequenceState::Waiting(_) => "Waiting",
        SequenceState::Scheduling(_) => "Scheduling",
        SequenceState::Prefilling(_) => "Prefilling",
        SequenceState::Decoding(_) => "Decoding",
        SequenceState::Preempted(_) => "Preempted",
        SequenceState::Finished(_) => "Finished",
        SequenceState::Aborted(_) => "Aborted",
    }
}

// Helper functions for each transition

fn apply_schedule(
    state: WaitingState,
    block_manager: &mut BlockManager,
    tokens_per_block: usize,
) -> Result<SequenceState> {
    let blocks_needed = state.prompt_tokens.div_ceil(tokens_per_block);

    let mut blocks = Vec::new();
    for _ in 0..blocks_needed {
        match block_manager.allocate() {
            Ok(block_id) => blocks.push(block_id),
            Err(Error::OutOfMemory) => {
                // Cleanup on OOM
                for block in blocks {
                    let _ = block_manager.free(block);
                }
                return Err(Error::OutOfMemory);
            }
            Err(e) => {
                // Other error - cleanup and propagate
                for block in blocks {
                    let _ = block_manager.free(block);
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
        blocks: Arc::new(blocks),
        tokens_filled: 0,
        tokens_total: state.prompt_tokens,
        max_tokens: state.max_tokens,
    }))
}

fn apply_append_tokens_prefilling(
    mut state: PrefillingState,
    num_tokens: usize,
    _block_manager: &mut BlockManager,
    _tokens_per_block: usize,
) -> Result<SequenceState> {
    state.tokens_filled += num_tokens;

    debug!(
        "Prefilling seq {:?}: {}/{} tokens",
        state.seq_id, state.tokens_filled, state.tokens_total
    );

    if state.tokens_filled >= state.tokens_total {
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

fn apply_append_tokens_decoding(
    mut state: DecodingState,
    num_tokens: usize,
    block_manager: &mut BlockManager,
    tokens_per_block: usize,
) -> Result<SequenceState> {
    state.num_tokens += num_tokens;

    // Check if finished
    if state.num_tokens >= state.max_tokens {
        // Free blocks
        for &block_id in state.blocks.iter() {
            if let Err(e) = block_manager.free(block_id) {
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
    let blocks_needed = state.num_tokens.div_ceil(tokens_per_block);
    let initial_block_count = state.blocks.len();

    if state.blocks.len() < blocks_needed {
        // Need to allocate more blocks - use Arc::make_mut for copy-on-write
        let blocks = Arc::make_mut(&mut state.blocks);

        while blocks.len() < blocks_needed {
            match block_manager.allocate() {
                Ok(block_id) => {
                    blocks.push(block_id);
                    debug!(
                        "Seq {:?}: allocated block, total blocks: {}",
                        state.seq_id,
                        blocks.len()
                    );
                }
                Err(e) => {
                    // Free any blocks we allocated in this call
                    warn!("Failed to allocate block for {:?}: {:?}", state.seq_id, e);
                    for block_id in blocks.drain(initial_block_count..) {
                        let _ = block_manager.free(block_id);
                    }
                    return Err(e);
                }
            }
        }
    }

    Ok(SequenceState::Decoding(state))
}

fn apply_complete_prefilling(
    state: PrefillingState,
    reason: FinishReason,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    // Free all blocks
    for &block_id in state.blocks.iter() {
        if let Err(e) = block_manager.free(block_id) {
            warn!(
                "Failed to free block {:?} for {:?}: {:?}",
                block_id, state.seq_id, e
            );
        }
    }

    debug!(
        "Completed seq {:?} during prefill: {:?}",
        state.seq_id, reason
    );

    Ok(SequenceState::Finished(FinishedState {
        seq_id: state.seq_id,
        finish_reason: reason,
    }))
}

fn apply_complete_decoding(
    state: DecodingState,
    reason: FinishReason,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    // Free all blocks
    for &block_id in state.blocks.iter() {
        if let Err(e) = block_manager.free(block_id) {
            warn!(
                "Failed to free block {:?} for {:?}: {:?}",
                block_id, state.seq_id, e
            );
        }
    }

    debug!("Completed seq {:?}: {:?}", state.seq_id, reason);

    Ok(SequenceState::Finished(FinishedState {
        seq_id: state.seq_id,
        finish_reason: reason,
    }))
}

fn apply_preempt_prefilling(
    state: PrefillingState,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    // Free all blocks
    for &block_id in state.blocks.iter() {
        if let Err(e) = block_manager.free(block_id) {
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

fn apply_preempt_decoding(
    state: DecodingState,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    // Free all blocks
    for &block_id in state.blocks.iter() {
        if let Err(e) = block_manager.free(block_id) {
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

fn apply_resume(state: PreemptedState) -> Result<SequenceState> {
    debug!("Resuming seq {:?}", state.seq_id);

    // Transition back to waiting for rescheduling
    Ok(SequenceState::Waiting(WaitingState {
        seq_id: state.seq_id,
        prompt_tokens: state.num_tokens,
        max_tokens: state.max_tokens,
    }))
}

fn apply_fork(
    state: DecodingState,
    child_id: SequenceId,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    // Copy-on-write: increment ref counts
    for &block_id in state.blocks.iter() {
        block_manager.add_ref(block_id)?;
    }

    let child = DecodingState {
        seq_id: child_id,
        blocks: state.blocks.clone(),
        num_tokens: state.num_tokens,
        max_tokens: state.max_tokens,
    };

    debug!("Forked seq {:?} → {:?}", state.seq_id, child.seq_id);

    // Fork returns parent, child needs to be handled separately by caller
    // For now just return error - need to handle multi-state result
    Err(Error::InvalidTransition(
        "Fork not yet supported in unified apply",
    ))
}

fn apply_abort(
    state: SequenceState,
    reason: String,
    block_manager: &mut BlockManager,
) -> Result<SequenceState> {
    let seq_id = state.seq_id();

    // Free blocks if any
    if let Some(blocks) = state.blocks() {
        for &block_id in blocks {
            if let Err(e) = block_manager.free(block_id) {
                warn!(
                    "Failed to free block {:?} for {:?}: {:?}",
                    block_id, seq_id, e
                );
            }
        }
    }

    warn!("Aborted seq {:?}: {}", seq_id, reason);

    Ok(SequenceState::Aborted(AbortedState { seq_id, reason }))
}
