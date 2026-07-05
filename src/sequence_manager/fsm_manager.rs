use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use super::events::*;
use super::states::*;
use super::fsm_events::*;
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
    event_tx: SequenceEventSender,

    // Scheduling queues
    waiting_queue: VecDeque<SequenceId>,
}

impl SequenceManager {
    pub fn new(
        block_manager: BlockManager,
        tokens_per_block: usize,
    ) -> (Self, SequenceEventSender) {
        let (event_tx, event_rx) = create_event_channel();

        let manager = Self {
            block_manager,
            sequences: HashMap::new(),
            tokens_per_block,
            event_rx,
            event_tx: event_tx.clone(),
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

    fn handle_add_request(
        &mut self,
        seq_id: SequenceId,
        prompt_tokens: usize,
        max_tokens: usize,
    ) {
        debug!(
            "Adding request: seq_id={:?}, prompt_tokens={}, max_tokens={}",
            seq_id, prompt_tokens, max_tokens
        );

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
                self.handle_oom();
            }
            Err(e) => {
                warn!("Failed to append tokens to {:?}: {:?}", seq_id, e);
            }
        }
    }

    fn handle_fork_sequence(&mut self, parent_id: SequenceId, child_id: SequenceId) {
        let parent_state = match self.sequences.remove(&parent_id) {
            Some(s) => s,
            None => {
                warn!("Parent sequence {:?} not found", parent_id);
                return;
            }
        };

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
                    // Can't schedule - leave in waiting queue
                    // Put state back
                    if let Some(old_state) = self.sequences.get(&seq_id) {
                        // State was already removed, this shouldn't happen
                        // but handle gracefully
                    }
                    break;
                }
                Err(e) => {
                    warn!("Failed to schedule {:?}: {:?}", seq_id, e);
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
        self.sequences
            .get(&seq_id)
            .and_then(|state| state.blocks())
            .map(|blocks| blocks.to_vec())
            .ok_or(Error::UnknownSequence(seq_id))
    }

    fn get_stats(&self) -> SequenceManagerStats {
        let num_running = self
            .sequences
            .values()
            .filter(|s| s.is_running())
            .count();

        let num_waiting = self
            .sequences
            .values()
            .filter(|s| s.is_waiting())
            .count();

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
