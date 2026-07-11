use super::events::*;
use super::fsm_events::*;
use super::states::*;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

/// Scheduler - central controller
///
/// Architecture:
/// - Scheduler DRIVES inference: builds batches → calls backend → updates state
/// - External events (via channel): AddRequest, CancelRequest, GetBlocks, GetStats
/// - Internal operations: token generation, state transitions, OOM handling
///
/// Responsibilities:
/// - Owns BlockManager (memory allocation)
/// - Owns sequence state storage (FSM states)
/// - Manages batches (waiting/prefill/decode)
/// - Scheduling policy (what to schedule when)
/// - Uses FSM events for type-safe state transitions
pub struct Scheduler {
    /// Block manager for memory allocation
    block_manager: BlockManager,

    /// All sequences and their states
    sequences: HashMap<SequenceId, SequenceState>,

    /// Waiting sequences (not yet scheduled)
    waiting_queue: VecDeque<SequenceId>,

    /// Currently running sequences in prefill phase
    prefill_batch: Vec<SequenceId>,

    /// Currently running sequences in decode phase
    decode_batch: Vec<SequenceId>,

    /// Event receiver
    event_rx: mpsc::UnboundedReceiver<SchedulerEvent>,

    /// Configuration
    max_batch_size: usize,
    tokens_per_block: usize,
}

impl Scheduler {
    pub fn new(
        block_manager: BlockManager,
        max_batch_size: usize,
        tokens_per_block: usize,
    ) -> (Self, mpsc::UnboundedSender<SchedulerEvent>) {
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        let scheduler = Self {
            block_manager,
            sequences: HashMap::new(),
            waiting_queue: VecDeque::new(),
            prefill_batch: Vec::new(),
            decode_batch: Vec::new(),
            event_rx,
            max_batch_size,
            tokens_per_block,
        };

        (scheduler, event_tx)
    }

    /// Main scheduler loop - drives inference and handles external events
    ///
    /// Loop structure:
    /// 1. Build batch from ready sequences
    /// 2. Run inference (TODO: integrate backend)
    /// 3. Update state based on inference results (internal)
    /// 4. Handle external events (non-blocking)
    /// 5. Try scheduling new sequences
    pub async fn run(mut self) {
        info!("Scheduler started");

        loop {
            // 1. Build batch for inference
            // TODO: Call backend.forward(batch) when backend is integrated

            // 2. Handle external events (non-blocking)
            // Use try_recv to not block - we want to keep generating tokens
            match self.event_rx.try_recv() {
                Ok(event) => {
                    match event {
                        SchedulerEvent::AddRequest {
                            seq_id,
                            prompt_tokens,
                            max_tokens,
                        } => {
                            self.add_request(seq_id, prompt_tokens, max_tokens);
                        }

                        SchedulerEvent::CancelRequest { seq_id } => {
                            self.cancel_request(seq_id);
                        }

                        SchedulerEvent::GetBlocks { seq_id, response } => {
                            let result = self.get_blocks(seq_id);
                            let _ = response.send(result);
                        }

                        SchedulerEvent::GetStats { response } => {
                            let stats = self.get_stats();
                            let _ = response.send(stats);
                        }
                    }

                    // After event, try scheduling
                    self.try_schedule();
                }
                Err(mpsc::error::TryRecvError::Empty) => {
                    // No events - continue to next iteration
                }
                Err(mpsc::error::TryRecvError::Disconnected) => {
                    info!("Event channel closed, shutting down scheduler");
                    break;
                }
            }

            // Small yield to prevent busy loop
            // TODO: Remove once backend integration drives the timing
            tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
        }

        info!("Scheduler stopped");
    }

    // ===== External Event Handlers =====

    fn add_request(&mut self, seq_id: SequenceId, prompt_tokens: usize, max_tokens: usize) {
        debug!(
            "Adding request: seq_id={:?}, prompt_tokens={}, max_tokens={}",
            seq_id, prompt_tokens, max_tokens
        );

        // Check duplicate
        if self.sequences.contains_key(&seq_id) {
            warn!("Duplicate seq_id {:?}", seq_id);
            return;
        }

        // Create waiting state
        let state = SequenceState::Waiting(WaitingState {
            seq_id,
            prompt_tokens,
            max_tokens,
        });

        self.sequences.insert(seq_id, state);
        self.waiting_queue.push_back(seq_id);
    }

    fn cancel_request(&mut self, seq_id: SequenceId) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Cannot cancel - sequence {:?} not found", seq_id);
                return;
            }
        };

        let event = AbortEvent {
            reason: "User cancelled".to_string(),
            block_manager: &mut self.block_manager,
        };

        match event.apply(state) {
            Ok(new_state) => {
                info!("Cancelled sequence {:?}", seq_id);
                self.sequences.insert(seq_id, new_state);

                // Remove from queues/batches
                self.waiting_queue.retain(|&id| id != seq_id);
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);
            }
            Err(e) => {
                warn!("Failed to cancel {:?}: {:?}", seq_id, e);
            }
        }
    }

    // ===== Internal Methods (called by scheduler loop during inference) =====
    // These are NOT exposed as events - scheduler calls them directly when:
    // - Generating tokens (append_tokens)
    // - Detecting stop tokens (complete_sequence)
    // - Handling OOM (preempt_sequence, resume_sequence)
    // - Beam search (fork_sequence)

    fn append_tokens(&mut self, seq_id: SequenceId, num_tokens: usize) {
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

    fn complete_sequence(&mut self, seq_id: SequenceId, reason: FinishReason) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = CompleteEvent {
            reason,
            block_manager: &mut self.block_manager,
        };

        match event.apply(state) {
            Ok(new_state) => {
                info!("Completed sequence {:?}", seq_id);
                self.sequences.insert(seq_id, new_state);

                // Remove from batches
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);
            }
            Err(e) => {
                warn!("Failed to complete {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    fn fork_sequence(&mut self, parent_id: SequenceId, child_id: SequenceId) {
        // Check duplicate child_id
        if self.sequences.contains_key(&child_id) {
            warn!("Child ID {:?} already exists", child_id);
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

    fn preempt_sequence(&mut self, seq_id: SequenceId) {
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

                // Remove from batches
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);
            }
            Err(e) => {
                warn!("Failed to preempt {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    fn resume_sequence(&mut self, seq_id: SequenceId) {
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
            }
            Err(e) => {
                warn!("Failed to resume {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);
            }
        }
    }

    // ===== Scheduling Logic =====

    fn try_schedule(&mut self) {
        self.schedule_prefill();
        self.schedule_decode();
    }

    fn schedule_prefill(&mut self) {
        while let Some(seq_id) = self.waiting_queue.pop_front() {
            // Check batch limit
            if self.prefill_batch.len() >= self.max_batch_size {
                self.waiting_queue.push_front(seq_id);
                debug!("Prefill batch full");
                break;
            }

            // Get sequence state
            let state = match self.sequences.get(&seq_id) {
                Some(SequenceState::Waiting(s)) => s,
                _ => {
                    warn!("Sequence {:?} not in Waiting state", seq_id);
                    continue;
                }
            };

            // Calculate blocks needed
            let blocks_needed = state.prompt_tokens.div_ceil(self.tokens_per_block);

            // Check memory
            if !self.block_manager.can_allocate(&BlockType::StandardKV) {
                self.waiting_queue.push_front(seq_id);
                debug!("OOM - cannot schedule {:?}", seq_id);
                break;
            }

            // Allocate blocks
            let mut blocks = Vec::new();
            for _ in 0..blocks_needed {
                match self.block_manager.allocate() {
                    Ok(block_id) => blocks.push(block_id),
                    Err(Error::OutOfMemory) => {
                        // Cleanup and stop
                        for b in blocks {
                            let _ = self.block_manager.free(b);
                        }
                        self.waiting_queue.push_front(seq_id);
                        warn!("OOM during allocation for {:?}", seq_id);
                        return;
                    }
                    Err(e) => {
                        for b in blocks {
                            let _ = self.block_manager.free(b);
                        }
                        warn!("Allocation error: {:?}", e);
                        return;
                    }
                }
            }

            // Transition: Waiting → Prefilling
            let state = self.sequences.remove(&seq_id).unwrap();
            if let SequenceState::Waiting(w) = state {
                let new_state = SequenceState::Prefilling(PrefillingState {
                    seq_id,
                    blocks: Arc::new(blocks),
                    tokens_filled: 0,
                    tokens_total: w.prompt_tokens,
                    max_tokens: w.max_tokens,
                });

                self.sequences.insert(seq_id, new_state);
                self.prefill_batch.push(seq_id);

                info!("Scheduled {:?} for prefill", seq_id);
            }
        }
    }

    fn schedule_decode(&mut self) {
        // TODO: Continuous batching
        // - Move completed prefill → decode batch
        // - Remove finished sequences
        // - Respect max_batch_size

        debug!("Decode batch size: {}", self.decode_batch.len());
    }

    fn handle_oom(&mut self) {
        warn!("Handling OOM");
        // TODO: Preemption policy
        // Find lowest priority sequence
        // Preempt it
        // Retry scheduling
    }

    // ===== Query Methods =====

    fn get_blocks(&self, seq_id: SequenceId) -> Result<Vec<BlockId>> {
        match self.sequences.get(&seq_id) {
            Some(state) => Ok(state.blocks().map(|b| b.to_vec()).unwrap_or_default()),
            None => Err(Error::UnknownSequence(seq_id)),
        }
    }

    fn get_stats(&self) -> SchedulerStats {
        let num_running = self.sequences.values().filter(|s| s.is_running()).count();
        let num_waiting = self.waiting_queue.len();
        let num_preempted = self
            .sequences
            .values()
            .filter(|s| matches!(s, SequenceState::Preempted(_)))
            .count();

        SchedulerStats {
            num_sequences: self.sequences.len(),
            num_running,
            num_waiting,
            num_preempted,
            block_stats: self.block_manager.get_stats(),
        }
    }
}

/// Create scheduler event channel
pub fn create_scheduler_channel() -> (
    mpsc::UnboundedSender<SchedulerEvent>,
    mpsc::UnboundedReceiver<SchedulerEvent>,
) {
    mpsc::unbounded_channel()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::allocator::CpuAllocator;

    fn create_test_scheduler() -> (Scheduler, mpsc::UnboundedSender<SchedulerEvent>) {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let block_manager = BlockManager::new(allocator, 1024);
        Scheduler::new(block_manager, 10, 16)
    }

    #[test]
    fn test_add_request() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);

        assert_eq!(scheduler.sequences.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 1);
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Waiting(_))
        ));
    }

    #[test]
    fn test_add_duplicate_request() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.add_request(SequenceId(1), 100, 200); // Duplicate

        // Should not add duplicate
        assert_eq!(scheduler.sequences.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 1);
    }

    #[test]
    fn test_cancel_request_waiting() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.cancel_request(SequenceId(1));

        // Should transition to Aborted
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Aborted(_))
        ));
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_cancel_request_not_found() {
        let (mut scheduler, _tx) = create_test_scheduler();

        // Should not panic
        scheduler.cancel_request(SequenceId(999));
        assert_eq!(scheduler.sequences.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_success() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();

        // Should move to prefilling
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Prefilling(_))
        ));
        assert_eq!(scheduler.prefill_batch.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_multiple() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.add_request(SequenceId(2), 50, 100);
        scheduler.schedule_prefill();

        // Both should be scheduled
        assert_eq!(scheduler.prefill_batch.len(), 2);
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_batch_limit() {
        let (mut scheduler, _tx) = create_test_scheduler();

        // Add more than max_batch_size (10)
        for i in 0..15 {
            scheduler.add_request(SequenceId(i), 100, 200);
        }
        scheduler.schedule_prefill();

        // Only 10 should be scheduled
        assert_eq!(scheduler.prefill_batch.len(), 10);
        assert_eq!(scheduler.waiting_queue.len(), 5);
    }

    #[test]
    fn test_schedule_prefill_oom() {
        // Small allocator - only 2 blocks
        let allocator = Box::new(CpuAllocator::new(2048));
        let block_manager = BlockManager::new(allocator, 1024);
        let (mut scheduler, _tx) = Scheduler::new(block_manager, 10, 16);

        // First request needs 7 blocks (100 tokens / 16 tokens_per_block)
        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();

        // Should fail - not enough memory
        assert_eq!(scheduler.prefill_batch.len(), 0);
        assert_eq!(scheduler.waiting_queue.len(), 1);
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Waiting(_))
        ));
    }

    #[test]
    fn test_append_tokens_prefilling() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();

        // Append all tokens
        scheduler.append_tokens(SequenceId(1), 100);

        // Should transition to Decoding
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Decoding(_))
        ));
    }

    #[test]
    fn test_append_tokens_decoding() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        // Append more tokens
        scheduler.append_tokens(SequenceId(1), 10);

        if let Some(SequenceState::Decoding(state)) = scheduler.sequences.get(&SequenceId(1)) {
            assert_eq!(state.num_tokens, 110);
        } else {
            panic!("Expected Decoding state");
        }
    }

    #[test]
    fn test_append_tokens_reaches_max() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        // Reach max_tokens
        scheduler.append_tokens(SequenceId(1), 100); // 100 + 100 = 200 = max_tokens

        // Should transition to Finished
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Finished(_))
        ));
    }

    #[test]
    fn test_complete_sequence() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.complete_sequence(SequenceId(1), FinishReason::Stop);

        // Should be Finished
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Finished(_))
        ));
        assert_eq!(scheduler.prefill_batch.len(), 0);
    }

    #[test]
    fn test_fork_sequence() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.fork_sequence(SequenceId(1), SequenceId(2));

        // Both should exist
        assert!(scheduler.sequences.contains_key(&SequenceId(1)));
        assert!(scheduler.sequences.contains_key(&SequenceId(2)));
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(2)),
            Some(SequenceState::Decoding(_))
        ));
    }

    #[test]
    fn test_fork_duplicate_child_id() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100);

        scheduler.add_request(SequenceId(2), 50, 100); // Child ID exists
        scheduler.fork_sequence(SequenceId(1), SequenceId(2)); // Should fail

        // Original seq 2 should be unchanged (Waiting)
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(2)),
            Some(SequenceState::Waiting(_))
        ));
    }

    #[test]
    fn test_preempt_sequence() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.preempt_sequence(SequenceId(1));

        // Should be Preempted
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Preempted(_))
        ));
        assert_eq!(scheduler.decode_batch.len(), 0);
    }

    #[test]
    fn test_resume_sequence() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100);
        scheduler.preempt_sequence(SequenceId(1));

        scheduler.resume_sequence(SequenceId(1));

        // Should be back to Waiting
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
            Some(SequenceState::Waiting(_))
        ));
        assert_eq!(scheduler.waiting_queue.len(), 1);
    }

    #[test]
    fn test_get_blocks() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();

        let blocks = scheduler.get_blocks(SequenceId(1)).unwrap();
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_get_blocks_not_found() {
        let (scheduler, _tx) = create_test_scheduler();

        let result = scheduler.get_blocks(SequenceId(999));
        assert!(matches!(result, Err(Error::UnknownSequence(_))));
    }

    #[test]
    fn test_get_stats() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.add_request(SequenceId(2), 50, 100);
        scheduler.schedule_prefill();

        let stats = scheduler.get_stats();
        assert_eq!(stats.num_sequences, 2);
        assert_eq!(stats.num_running, 2); // Both in prefill
        assert_eq!(stats.num_waiting, 0);
    }

    #[test]
    fn test_try_schedule() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.try_schedule();

        // Should schedule automatically
        assert_eq!(scheduler.prefill_batch.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_cancel_removes_from_batches() {
        let (mut scheduler, _tx) = create_test_scheduler();

        scheduler.add_request(SequenceId(1), 100, 200);
        scheduler.schedule_prefill();
        assert_eq!(scheduler.prefill_batch.len(), 1);

        scheduler.cancel_request(SequenceId(1));

        // Should remove from batch
        assert_eq!(scheduler.prefill_batch.len(), 0);
    }
}
