#[cfg(test)]
mod tests {
    use crate::block_manager::allocator::CpuAllocator;
    use crate::block_manager::manager::BlockManager;
    use crate::block_manager::types::*;
    use crate::sequence_manager::fsm_events::*;
    use crate::sequence_manager::states::*;

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
            blocks,
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
            blocks,
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
            blocks,
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
            blocks,
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
            blocks,
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
            blocks,
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
            blocks,
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
}
