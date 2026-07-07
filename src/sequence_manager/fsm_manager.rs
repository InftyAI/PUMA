use super::events::*;
use super::fsm_events::*;
use super::states::*;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use std::collections::{HashMap, VecDeque};
use tracing::{debug, info, warn};

/// Sequence Manager with FSM-based state transitions
pub struct SequenceManager {
    block_manager: BlockManager,

    // Each sequence is a state machine
    sequences: HashMap<SequenceId, SequenceState>,

    tokens_per_block: usize,

    // Event-driven
    event_rx: SequenceEventReceiver,

    // Scheduling queues
    waiting_queue: VecDeque<SequenceId>,
}

impl SequenceManager {
    pub fn new(
        block_manager: BlockManager,
        tokens_per_block: usize,
    ) -> (Self, SequenceEventSender) {
        assert!(tokens_per_block > 0, "tokens_per_block must be > 0");

        let (event_tx, event_rx) = create_event_channel();
        let manager = Self {
            block_manager,
            sequences: HashMap::new(),
            tokens_per_block,
            event_rx,
            waiting_queue: VecDeque::new(),
        };

        (manager, event_tx)
    }

    /// Main event loop
    pub async fn run(mut self) {
        info!("FSM SequenceManager event loop started");

        while let Some(event) = self.event_rx.recv().await {
            match event {
                SequenceEvent::AddRequest {
                    seq_id,
                    prompt_tokens,
                    max_tokens,
                } => {
                    self.handle_add_request(seq_id, prompt_tokens, max_tokens);
                }

                SequenceEvent::AppendTokens { seq_id, num_tokens } => {
                    self.handle_append_tokens(seq_id, num_tokens);
                }

                SequenceEvent::ForkSequence {
                    parent_id,
                    child_id,
                } => {
                    self.handle_fork_sequence(parent_id, child_id);
                }

                SequenceEvent::CompleteSequence { seq_id } => {
                    self.handle_complete_sequence(seq_id);
                }

                SequenceEvent::PreemptSequence { seq_id } => {
                    self.handle_preempt_sequence(seq_id);
                }

                SequenceEvent::ResumeSequence { seq_id } => {
                    self.handle_resume_sequence(seq_id);
                }

                SequenceEvent::GetBlocks { seq_id, response } => {
                    let result = self.get_blocks(seq_id);
                    let _ = response.send(result);
                }

                SequenceEvent::GetStats { response } => {
                    let stats = self.get_stats();
                    let _ = response.send(stats);
                }
            }
        }

        info!("FSM SequenceManager event loop stopped");
    }

    fn handle_add_request(&mut self, seq_id: SequenceId, prompt_tokens: usize, max_tokens: usize) {
        debug!(
            "Adding request: seq_id={:?}, prompt_tokens={}, max_tokens={}",
            seq_id, prompt_tokens, max_tokens
        );

        // Check for duplicate seq_id
        if self.sequences.contains_key(&seq_id) {
            warn!(
                "Duplicate AddRequest for {:?} - sequence already exists, ignoring",
                seq_id
            );
            return;
        }

        let state = SequenceState::Waiting(WaitingState {
            seq_id,
            prompt_tokens,
            max_tokens,
        });

        self.sequences.insert(seq_id, state);
        self.waiting_queue.push_back(seq_id);

        // Try to schedule immediately
        self.try_schedule();
    }

    fn handle_append_tokens(&mut self, seq_id: SequenceId, num_tokens: usize) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = AppendTokensEvent {
            num_tokens,
            block_manager: &mut self.block_manager,
            tokens_per_block: self.tokens_per_block,
        };

        match event.apply(state) {
            Ok(new_state) => {
                self.sequences.insert(seq_id, new_state);
            }
            Err(Error::OutOfMemory) => {
                warn!("OOM while appending tokens to {:?}", seq_id);
                self.sequences.insert(seq_id, backup);
                self.handle_oom();
            }
            Err(e) => {
                warn!("Failed to append tokens to {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    fn handle_fork_sequence(&mut self, parent_id: SequenceId, child_id: SequenceId) {
        // Check if child_id already exists
        if self.sequences.contains_key(&child_id) {
            warn!(
                "Cannot fork {:?} → {:?}: child ID already exists",
                parent_id, child_id
            );
            return;
        }

        let parent_state = match self.sequences.remove(&parent_id) {
            Some(s) => s,
            None => {
                warn!("Parent sequence {:?} not found", parent_id);
                return;
            }
        };

        let backup = parent_state.clone();
        let event = ForkEvent {
            child_id,
            block_manager: &mut self.block_manager,
        };

        match event.apply(parent_state) {
            Ok((parent_state, child_state)) => {
                self.sequences.insert(parent_id, parent_state);
                self.sequences.insert(child_id, child_state);
                debug!("Forked sequence {:?} → {:?}", parent_id, child_id);
            }
            Err(e) => {
                warn!("Failed to fork {:?}: {:?}", parent_id, e);
                self.sequences.insert(parent_id, backup);
            }
        }
    }

    fn handle_complete_sequence(&mut self, seq_id: SequenceId) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = CompleteEvent {
            reason: FinishReason::Stop,
            block_manager: &mut self.block_manager,
        };

        match event.apply(state) {
            Ok(new_state) => {
                info!("Completed sequence {:?}", seq_id);
                // Keep finished state for stats/debugging
                self.sequences.insert(seq_id, new_state);

                // Try to schedule waiting sequences
                self.try_schedule();
            }
            Err(e) => {
                warn!("Failed to complete {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    fn handle_preempt_sequence(&mut self, seq_id: SequenceId) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = PreemptEvent {
            block_manager: &mut self.block_manager,
        };

        match event.apply(state) {
            Ok(new_state) => {
                info!("Preempted sequence {:?}", seq_id);
                self.sequences.insert(seq_id, new_state);
            }
            Err(e) => {
                warn!("Failed to preempt {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    fn handle_resume_sequence(&mut self, seq_id: SequenceId) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = ResumeEvent;

        match event.apply(state) {
            Ok(new_state) => {
                debug!("Resumed sequence {:?}", seq_id);
                self.sequences.insert(seq_id, new_state);
                self.waiting_queue.push_back(seq_id);
                self.try_schedule();
            }
            Err(e) => {
                warn!("Failed to resume {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    fn try_schedule(&mut self) {
        while let Some(&seq_id) = self.waiting_queue.front() {
            let state = match self.sequences.remove(&seq_id) {
                Some(s) => s,
                None => {
                    self.waiting_queue.pop_front();
                    continue;
                }
            };

            // Only schedule if in Waiting state
            if !matches!(state, SequenceState::Waiting(_)) {
                self.sequences.insert(seq_id, state);
                self.waiting_queue.pop_front();
                continue;
            }

            let backup = state.clone();
            let event = ScheduleEvent {
                block_manager: &mut self.block_manager,
                tokens_per_block: self.tokens_per_block,
            };

            match event.apply(state) {
                Ok(new_state) => {
                    self.sequences.insert(seq_id, new_state);
                    self.waiting_queue.pop_front();
                    info!("Scheduled sequence {:?}", seq_id);
                }
                Err(Error::OutOfMemory) => {
                    // Can't schedule - restore state and leave in waiting queue
                    self.sequences.insert(seq_id, backup);
                    break;
                }
                Err(e) => {
                    warn!("Failed to schedule {:?}: {:?}", seq_id, e);
                    self.sequences.insert(seq_id, backup);
                    self.waiting_queue.pop_front();
                }
            }
        }
    }

    fn handle_oom(&mut self) {
        warn!("Handling OOM - preemption policy not yet implemented");
        // TODO: Implement preemption policy
        // - Find lowest priority sequence
        // - Preempt it
        // - Retry scheduling
    }

    fn get_blocks(&self, seq_id: SequenceId) -> Result<Vec<BlockId>> {
        match self.sequences.get(&seq_id) {
            None => Err(Error::UnknownSequence(seq_id)),
            Some(state) => Ok(state.blocks().map(|b| b.to_vec()).unwrap_or_default()),
        }
    }

    fn get_stats(&self) -> SequenceManagerStats {
        let num_running = self.sequences.values().filter(|s| s.is_running()).count();

        let num_waiting = self.sequences.values().filter(|s| s.is_waiting()).count();

        let num_preempted = self
            .sequences
            .values()
            .filter(|s| matches!(s, SequenceState::Preempted(_)))
            .count();

        SequenceManagerStats {
            num_sequences: self.sequences.len(),
            num_running,
            num_waiting,
            num_preempted,
            block_stats: self.block_manager.get_stats(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::allocator::CpuAllocator;

    fn create_test_manager() -> (SequenceManager, SequenceEventSender) {
        let allocator = Box::new(CpuAllocator::new(1024 * 1024)); // 1MB
        let block_manager = BlockManager::new(allocator, 256); // 256 byte blocks
        SequenceManager::new(block_manager, 16) // 16 tokens per block
    }

    #[test]
    fn test_duplicate_add_request_ignored() {
        let (mut manager, _tx) = create_test_manager();

        let seq_id = SequenceId(0);

        // Add first request
        manager.handle_add_request(seq_id, 32, 128);

        // Verify it was added
        assert!(manager.sequences.contains_key(&seq_id));
        let first_state = manager.sequences.get(&seq_id).cloned();

        // Try to add duplicate - should be ignored
        manager.handle_add_request(seq_id, 64, 256); // Different params

        // Verify the original state is unchanged
        let current_state = manager.sequences.get(&seq_id);
        assert_eq!(format!("{:?}", first_state), format!("{:?}", current_state));

        // Verify original params (may have transitioned to Prefilling via try_schedule)
        match manager.sequences.get(&seq_id) {
            Some(SequenceState::Waiting(s)) => {
                assert_eq!(s.prompt_tokens, 32); // Original value
                assert_eq!(s.max_tokens, 128); // Original value
            }
            Some(SequenceState::Prefilling(s)) => {
                // Scheduled automatically, check original params
                assert_eq!(s.tokens_total, 32); // Original prompt_tokens
                assert_eq!(s.max_tokens, 128); // Original value
            }
            _ => panic!(
                "Expected Waiting or Prefilling state, got: {:?}",
                manager.sequences.get(&seq_id)
            ),
        }
    }

    #[test]
    fn test_add_different_sequences() {
        let (mut manager, _tx) = create_test_manager();

        // Add multiple different sequences - all should succeed
        manager.handle_add_request(SequenceId(0), 32, 128);
        manager.handle_add_request(SequenceId(1), 64, 256);
        manager.handle_add_request(SequenceId(2), 16, 64);

        assert_eq!(manager.sequences.len(), 3);
        assert!(manager.sequences.contains_key(&SequenceId(0)));
        assert!(manager.sequences.contains_key(&SequenceId(1)));
        assert!(manager.sequences.contains_key(&SequenceId(2)));
    }
}
