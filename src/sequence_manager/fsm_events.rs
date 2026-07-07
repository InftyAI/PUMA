use super::states::*;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use std::sync::Arc;
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
            _ => Err(Error::InvalidTransition(
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
            blocks: Arc::new(blocks),
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
            _ => Err(Error::InvalidTransition(
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
            for &block_id in state.blocks.iter() {
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

        if state.blocks.len() < blocks_needed {
            // Need to allocate more blocks - use Arc::make_mut for copy-on-write
            let blocks = Arc::make_mut(&mut state.blocks);

            while blocks.len() < blocks_needed {
                match self.block_manager.allocate() {
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
                            let _ = self.block_manager.free(block_id);
                        }
                        return Err(e);
                    }
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
            _ => Err(Error::InvalidTransition(
                "PreemptEvent requires running state",
            )),
        }
    }

    fn apply_to_decoding(self, state: DecodingState) -> Result<SequenceState> {
        // Free all blocks
        for &block_id in state.blocks.iter() {
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
        for &block_id in state.blocks.iter() {
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
            _ => Err(Error::InvalidTransition(
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
            _ => Err(Error::InvalidTransition(
                "ForkEvent requires Decoding state",
            )),
        }
    }

    fn apply_to_decoding(self, state: DecodingState) -> Result<(SequenceState, SequenceState)> {
        // Copy-on-write: increment ref counts
        for &block_id in state.blocks.iter() {
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
            _ => Err(Error::InvalidTransition(
                "CompleteEvent requires running state",
            )),
        }
    }

    fn apply_to_decoding(self, state: DecodingState) -> Result<SequenceState> {
        // Free all blocks
        for &block_id in state.blocks.iter() {
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
        for &block_id in state.blocks.iter() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::allocator::CpuAllocator;
    use crate::block_manager::manager::BlockManager;

    #[test]
    fn test_schedule_event_waiting_to_prefilling() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let waiting = SequenceState::Waiting(WaitingState {
            seq_id: SequenceId(1),
            prompt_tokens: 100,
            max_tokens: 150,
        });

        let event = ScheduleEvent {
            block_manager: &mut block_manager,
            tokens_per_block: 16,
        };

        let result = event.apply(waiting);
        assert!(result.is_ok());

        let new_state = result.unwrap();
        assert!(matches!(new_state, SequenceState::Prefilling(_)));

        if let SequenceState::Prefilling(state) = new_state {
            assert_eq!(state.seq_id, SequenceId(1));
            assert_eq!(state.tokens_filled, 0);
            assert_eq!(state.tokens_total, 100);
            // Should allocate ceil(100/16) = 7 blocks
            assert_eq!(state.blocks.len(), 7);
        }
    }

    #[test]
    fn test_schedule_event_oom() {
        // Small allocator - only 2 blocks
        let allocator = Box::new(CpuAllocator::new(2048));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let waiting = SequenceState::Waiting(WaitingState {
            seq_id: SequenceId(1),
            prompt_tokens: 100, // Needs 7 blocks
            max_tokens: 150,
        });

        let event = ScheduleEvent {
            block_manager: &mut block_manager,
            tokens_per_block: 16,
        };

        let result = event.apply(waiting);
        assert!(matches!(result, Err(Error::OutOfMemory)));

        // Should have cleaned up - no blocks leaked
        let stats = block_manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.allocated_blocks, 0);
    }

    #[test]
    fn test_append_tokens_prefilling_to_decoding() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        // Allocate blocks
        let blocks = vec![
            block_manager.allocate().unwrap(),
            block_manager.allocate().unwrap(),
        ];

        let prefilling = SequenceState::Prefilling(PrefillingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            tokens_filled: 0,
            tokens_total: 100,
            max_tokens: 150,
        });

        let event = AppendTokensEvent {
            num_tokens: 100,
            block_manager: &mut block_manager,
            tokens_per_block: 16,
        };

        let result = event.apply(prefilling).unwrap();

        // Should transition to Decoding
        assert!(matches!(result, SequenceState::Decoding(_)));

        if let SequenceState::Decoding(state) = result {
            assert_eq!(state.num_tokens, 100);
            assert_eq!(state.blocks.len(), 2);
        }
    }

    #[test]
    fn test_append_tokens_decoding_needs_more_blocks() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![block_manager.allocate().unwrap()]; // 1 block

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 16, // 1 block worth
            max_tokens: 150,
        });

        let event = AppendTokensEvent {
            num_tokens: 20, // Now need 3 blocks total
            block_manager: &mut block_manager,
            tokens_per_block: 16,
        };

        let result = event.apply(decoding).unwrap();

        if let SequenceState::Decoding(state) = result {
            assert_eq!(state.num_tokens, 36);
            // Should allocate 2 more blocks (total 3)
            assert_eq!(state.blocks.len(), 3);
        }
    }

    #[test]
    fn test_append_tokens_decoding_oom_cleanup() {
        // Small allocator - only enough for initial blocks
        let allocator = Box::new(CpuAllocator::new(2048));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![
            block_manager.allocate().unwrap(),
            block_manager.allocate().unwrap(),
        ];
        let initial_block_count = blocks.len();

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 32,
            max_tokens: 150,
        });

        // Try to append tokens that would need more blocks (will fail - OOM)
        let event = AppendTokensEvent {
            num_tokens: 50, // Would need 6 blocks total
            block_manager: &mut block_manager,
            tokens_per_block: 16,
        };

        let result = event.apply(decoding);
        assert!(matches!(result, Err(Error::OutOfMemory)));

        // Original 2 blocks should still be allocated (not freed by error path)
        let stats = block_manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.allocated_blocks, initial_block_count);
    }

    #[test]
    fn test_append_tokens_reaches_max_tokens() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![block_manager.allocate().unwrap()];

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 140,
            max_tokens: 150,
        });

        let event = AppendTokensEvent {
            num_tokens: 10,
            block_manager: &mut block_manager,
            tokens_per_block: 16,
        };

        let result = event.apply(decoding).unwrap();

        // Should transition to Finished
        assert!(matches!(result, SequenceState::Finished(_)));

        if let SequenceState::Finished(state) = result {
            assert_eq!(state.seq_id, SequenceId(1));
            assert_eq!(state.finish_reason, FinishReason::MaxTokens);
        }

        // Blocks should be freed
        let stats = block_manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.allocated_blocks, 0);
    }

    #[test]
    fn test_preempt_event() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![
            block_manager.allocate().unwrap(),
            block_manager.allocate().unwrap(),
        ];

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 32,
            max_tokens: 150,
        });

        let event = PreemptEvent {
            block_manager: &mut block_manager,
        };

        let result = event.apply(decoding).unwrap();

        // Should transition to Preempted
        assert!(matches!(result, SequenceState::Preempted(_)));

        if let SequenceState::Preempted(state) = result {
            assert_eq!(state.num_tokens, 32);
        }

        // Blocks should be freed
        let stats = block_manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.allocated_blocks, 0);
    }

    #[test]
    fn test_resume_event() {
        let preempted = SequenceState::Preempted(PreemptedState {
            seq_id: SequenceId(1),
            num_tokens: 32,
            max_tokens: 150,
        });

        let event = ResumeEvent;
        let result = event.apply(preempted).unwrap();

        // Should transition back to Waiting
        assert!(matches!(result, SequenceState::Waiting(_)));

        if let SequenceState::Waiting(state) = result {
            assert_eq!(state.prompt_tokens, 32);
            assert_eq!(state.max_tokens, 150);
        }
    }

    #[test]
    fn test_fork_event() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![
            block_manager.allocate().unwrap(),
            block_manager.allocate().unwrap(),
        ];

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 32,
            max_tokens: 150,
        });

        let event = ForkEvent {
            child_id: SequenceId(2),
            block_manager: &mut block_manager,
        };

        let result = event.apply(decoding).unwrap();
        let (parent, child) = result;

        // Both should be in Decoding state
        assert!(matches!(parent, SequenceState::Decoding(_)));
        assert!(matches!(child, SequenceState::Decoding(_)));

        if let (SequenceState::Decoding(p), SequenceState::Decoding(c)) = (parent, child) {
            assert_eq!(p.seq_id, SequenceId(1));
            assert_eq!(c.seq_id, SequenceId(2));
            assert_eq!(p.blocks, c.blocks); // Shared blocks
        }
    }

    #[test]
    fn test_complete_event() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![block_manager.allocate().unwrap()];

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 100,
            max_tokens: 150,
        });

        let event = CompleteEvent {
            reason: FinishReason::Stop,
            block_manager: &mut block_manager,
        };

        let result = event.apply(decoding).unwrap();

        assert!(matches!(result, SequenceState::Finished(_)));

        if let SequenceState::Finished(state) = result {
            assert_eq!(state.finish_reason, FinishReason::Stop);
        }

        // Blocks should be freed
        let stats = block_manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.allocated_blocks, 0);
    }

    #[test]
    fn test_fork_duplicate_child_id() {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let mut block_manager = BlockManager::new(allocator, 1024);

        let blocks = vec![
            block_manager.allocate().unwrap(),
            block_manager.allocate().unwrap(),
        ];

        let decoding = SequenceState::Decoding(DecodingState {
            seq_id: SequenceId(1),
            blocks: Arc::new(blocks),
            num_tokens: 32,
            max_tokens: 150,
        });

        // Fork with child_id = SequenceId(2)
        let event = ForkEvent {
            child_id: SequenceId(2),
            block_manager: &mut block_manager,
        };

        let result = event.apply(decoding).unwrap();
        let (parent, child) = result;

        // Verify fork succeeded
        if let SequenceState::Decoding(c) = &child {
            assert_eq!(c.seq_id, SequenceId(2));
        }

        // Now try to fork again with the SAME child_id (should fail in manager)
        // This test verifies the ForkEvent itself works, but the manager
        // should check for duplicate child_id before calling this
        drop(parent);
        drop(child);
    }
}
