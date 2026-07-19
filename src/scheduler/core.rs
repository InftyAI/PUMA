use super::events::{ResponseSender, SchedulerEvent, SchedulerEventReceiver, SchedulerStats};
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use crate::fsm::{Event, FinishReason, SequenceState};
use crate::sequence_manager::SequenceManager;
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Scheduler - batch and scheduling policy (synchronous API)
///
/// Architecture
/// - Pure synchronous methods (no async, no loop)
/// - LLMEngine calls: add_request() → schedule() → process_outputs()
/// - Drives a `SequenceManager` (which owns memory and executes FSM
///   transitions) via `create`/`advance`/`fork`
///
/// Responsibilities:
/// - This is the *policy* layer; the `SequenceManager` is the *mechanism*
///   layer that owns `BlockManager` and the FSM transition logic
/// - Manages batches (waiting/prefill/decode)
/// - Scheduling policy (what to schedule when)
/// - Owns client response channels and delivers results/errors
pub struct Scheduler {
    /// Owns sequence state + block memory and executes FSM transitions.
    sequences: SequenceManager,

    /// Response channels to send results back to clients (supports streaming)
    response_channels: HashMap<SequenceId, ResponseSender>,

    /// Waiting sequences (not yet scheduled)
    waiting_queue: VecDeque<SequenceId>,

    /// Currently running sequences in prefill phase
    prefill_batch: Vec<SequenceId>,

    /// Currently running sequences in decode phase
    decode_batch: Vec<SequenceId>,

    /// Receiver for external events (add/cancel/query) from clients.
    ///
    /// Public so the engine loop can drain it directly and feed each event to
    /// [`handle_event`](Self::handle_event).
    pub event_rx: SchedulerEventReceiver,

    /// Configuration
    max_batch_size: usize,
    tokens_per_block: usize,
}

impl Scheduler {
    pub fn new(
        block_manager: BlockManager,
        event_rx: SchedulerEventReceiver,
        max_batch_size: usize,
        tokens_per_block: usize,
    ) -> Self {
        Self {
            sequences: SequenceManager::new(block_manager, tokens_per_block),
            response_channels: HashMap::new(),
            waiting_queue: VecDeque::new(),
            prefill_batch: Vec::new(),
            decode_batch: Vec::new(),
            event_rx,
            max_batch_size,
            tokens_per_block,
        }
    }

    /// Dispatch a single external event to its handler.
    ///
    /// One entry point for every `SchedulerEvent`: mutating events build the
    /// corresponding FSM event and apply it; query events reply on their own
    /// response channel.
    pub fn handle_event(&mut self, event: SchedulerEvent) {
        match event {
            SchedulerEvent::AddRequest {
                seq_id,
                token_ids,
                max_tokens,
                response_tx,
            } => self.add_request(seq_id, token_ids, max_tokens, response_tx),

            SchedulerEvent::CancelRequest { seq_id } => {
                self.abort_request(seq_id, "User cancelled".to_string())
            }

            SchedulerEvent::GetBlocks { seq_id, response } => {
                let _ = response.send(self.get_blocks(seq_id));
            }

            SchedulerEvent::GetStats { response } => {
                let _ = response.send(self.get_stats());
            }
        }
    }

    // ===== Public API (called by LLMEngine) =====

    pub fn add_request(
        &mut self,
        seq_id: SequenceId,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        response_tx: ResponseSender,
    ) {
        debug!(
            "Adding request: seq_id={:?}, num_tokens={}, max_tokens={}",
            seq_id,
            token_ids.len(),
            max_tokens
        );

        // Check duplicate
        if self.sequences.contains(seq_id) {
            warn!("Duplicate seq_id {:?}", seq_id);
            self.send_error(
                response_tx,
                Error::InvalidTransition("Duplicate sequence ID"),
            );
            return;
        }

        // Store response channel
        self.response_channels.insert(seq_id, response_tx);

        // Birth the sequence (Empty → Waiting) inside the sequence manager.
        match self.sequences.create(seq_id, token_ids, max_tokens) {
            Ok(()) => {
                self.waiting_queue.push_back(seq_id);
            }
            Err(e) => {
                warn!("Failed to create sequence {:?}: {:?}", seq_id, e);
                self.response_channels.remove(&seq_id);
            }
        }
    }

    /// Abort a sequence with a caller-supplied reason.
    ///
    /// The reason is passed through to the FSM `Abort` event, so callers (user
    /// cancel, backend failure, …) describe *why* the sequence ended.
    pub fn abort_request(&mut self, seq_id: SequenceId, reason: String) {
        match self.sequences.advance(
            seq_id,
            Event::Abort {
                reason: reason.clone(),
            },
        ) {
            Ok(_) => {
                info!("Aborted sequence {:?}", seq_id);

                // Remove from queues/batches
                self.waiting_queue.retain(|&id| id != seq_id);
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);

                // Resolve the client channel so waiters unblock and the entry
                // is not leaked. Streaming callers see the error and close;
                // single callers get an `Err` instead of hanging forever.
                if let Some(response_tx) = self.response_channels.remove(&seq_id) {
                    self.send_error(response_tx, Error::Aborted(reason));
                }
            }
            Err(e) => {
                warn!("Failed to abort {:?}: {:?}", seq_id, e);
            }
        }
    }

    /// Schedule sequences - returns whether work was scheduled
    ///
    /// Called by LLMEngine every iteration to:
    /// 1. Move waiting → prefilling
    /// 2. Allocate blocks
    /// 3. Build batches
    pub fn schedule(&mut self) -> bool {
        let before = self.prefill_batch.len();
        self.schedule_prefill();
        self.schedule_decode();
        self.prefill_batch.len() > before || !self.decode_batch.is_empty()
    }

    /// Drain the current prefill batch as inference work for the engine loop.
    ///
    /// Clears the prefill batch and returns `(seq_id, token_ids, max_tokens,
    /// streaming)` per scheduled sequence. The engine loop runs the backend on
    /// these and reports results back via [`send_chunk`](Self::send_chunk) and
    /// [`complete_sequence`](Self::complete_sequence). The `streaming` flag tells
    /// the loop whether to call the backend's streaming or single-shot path.
    pub fn take_prefill_batch(&mut self) -> Vec<(SequenceId, Arc<Vec<TokenId>>, usize, bool)> {
        let batch = std::mem::take(&mut self.prefill_batch);
        batch
            .into_iter()
            .filter_map(|seq_id| match self.sequences.get(seq_id) {
                Some(SequenceState::Prefilling(s)) => {
                    let streaming = self
                        .response_channels
                        .get(&seq_id)
                        .is_some_and(|tx| tx.is_streaming());
                    Some((seq_id, Arc::clone(&s.token_ids), s.max_tokens, streaming))
                }
                _ => None,
            })
            .collect()
    }

    // ===== Internal State Management (called by LLMEngine after inference) =====

    pub fn append_tokens(&mut self, seq_id: SequenceId, num_tokens: usize) {
        let event = Event::AppendTokens {
            num_tokens,
            tokens_per_block: self.tokens_per_block,
        };

        // On error the sequence manager restores the prior state; the scheduler
        // only applies OOM policy.
        match self.sequences.advance(seq_id, event) {
            Ok(_) => {}
            Err(Error::OutOfMemory) => {
                warn!("OOM while appending tokens to {:?}", seq_id);
                self.handle_oom();
            }
            Err(e) => {
                warn!("Failed to append tokens to {:?}: {:?}", seq_id, e);
            }
        }
    }

    pub fn complete_sequence(&mut self, seq_id: SequenceId, reason: FinishReason, result: String) {
        match self.sequences.advance(seq_id, Event::Complete { reason }) {
            Ok(_) => {
                info!("Completed sequence {:?}", seq_id);

                // Remove from batches
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);

                // Send result back to client
                self.send_result(seq_id, result);
            }
            Err(e) => {
                warn!("Failed to complete {:?}: {:?}", seq_id, e);

                // Send error back to client
                if let Some(response_tx) = self.response_channels.remove(&seq_id) {
                    self.send_error(response_tx, e);
                }
            }
        }
    }

    pub fn fork_sequence(&mut self, parent_id: SequenceId, child_id: SequenceId) {
        let _ = self.sequences.fork(parent_id, child_id);
    }

    fn preempt_sequence(&mut self, seq_id: SequenceId) {
        match self.sequences.advance(seq_id, Event::Preempt) {
            Ok(_) => {
                info!("Preempted sequence {:?}", seq_id);

                // Remove from batches
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);
            }
            Err(e) => {
                warn!("Failed to preempt {:?}: {:?}", seq_id, e);
            }
        }
    }

    fn resume_sequence(&mut self, seq_id: SequenceId) {
        match self.sequences.advance(seq_id, Event::Resume) {
            Ok(_) => {
                debug!("Resumed sequence {:?}", seq_id);
                self.waiting_queue.push_back(seq_id);
            }
            Err(e) => {
                warn!("Failed to resume {:?}: {:?}", seq_id, e);
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

            let event = Event::Schedule {
                tokens_per_block: self.tokens_per_block,
            };

            // The sequence manager restores prior state on any error, so the
            // scheduler only applies queueing policy.
            match self.sequences.advance(seq_id, event) {
                Ok(_) => {
                    self.prefill_batch.push(seq_id);
                    info!("Scheduled {:?} for prefill", seq_id);
                }
                Err(Error::OutOfMemory) => {
                    // Put back at the front and stop scheduling this round.
                    self.waiting_queue.push_front(seq_id);
                    debug!("OOM - cannot schedule {:?}", seq_id);
                    break;
                }
                Err(Error::InvalidTransition(_)) => {
                    warn!("Cannot schedule {:?} - not in Waiting state", seq_id);
                }
                Err(e) => {
                    warn!("Failed to schedule {:?}: {:?}", seq_id, e);
                }
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

    pub fn get_blocks(&self, seq_id: SequenceId) -> Result<Vec<BlockId>> {
        self.sequences.blocks(seq_id)
    }

    pub fn get_stats(&self) -> SchedulerStats {
        SchedulerStats {
            num_sequences: self.sequences.len(),
            num_running: self.sequences.num_running(),
            num_waiting: self.waiting_queue.len(),
            num_preempted: self.sequences.num_preempted(),
            block_stats: self.sequences.block_stats(),
        }
    }

    // ===== Response Channel Helpers =====

    /// Send a decoded text chunk to a streaming client.
    ///
    /// The chunk is an incremental piece of detokenized output (a delta), not a
    /// single token — see the engine's incremental decode. No-op for
    /// non-streaming clients, which receive the full text via
    /// [`complete_sequence`](Self::complete_sequence).
    pub fn send_chunk(&mut self, seq_id: SequenceId, chunk: String) {
        if let Some(response_tx) = self.response_channels.get(&seq_id) {
            match response_tx {
                ResponseSender::Stream(tx) => {
                    let _ = tx.send(Ok(chunk));
                }
                ResponseSender::Single(_) => {
                    // Single response - streamed chunks are dropped;
                    // the full result is delivered at completion.
                }
            }
        }
    }

    /// Send error to client
    fn send_error(&self, response_tx: ResponseSender, error: Error) {
        match response_tx {
            ResponseSender::Single(tx) => {
                let _ = tx.send(Err(error));
            }
            ResponseSender::Stream(tx) => {
                let _ = tx.send(Err(error));
            }
        }
    }

    /// Send final result to client
    fn send_result(&mut self, seq_id: SequenceId, result: String) {
        if let Some(response_tx) = self.response_channels.remove(&seq_id) {
            match response_tx {
                ResponseSender::Single(tx) => {
                    let _ = tx.send(Ok(result));
                }
                ResponseSender::Stream(_tx) => {
                    // Already streamed tokens, just close the channel
                    // (drop _tx automatically closes it)
                }
            }
        }
    }
}

/// Create scheduler event channel
#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::allocator::CpuAllocator;
    use crate::scheduler::events::ResponseSender;

    fn create_test_scheduler() -> Scheduler {
        let allocator = Box::new(CpuAllocator::new(100_000));
        let block_manager = BlockManager::new(allocator, 1024);
        let (_event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        Scheduler::new(block_manager, event_rx, 10, 16)
    }

    fn create_dummy_response() -> ResponseSender {
        let (tx, _rx) = tokio::sync::oneshot::channel();
        ResponseSender::Single(tx)
    }

    #[test]
    fn test_add_request() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());

        assert_eq!(scheduler.sequences.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 1);
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Waiting(_))
        ));
    }

    #[test]
    fn test_add_duplicate_request() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response()); // Duplicate

        // Should not add duplicate
        assert_eq!(scheduler.sequences.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 1);
    }

    #[test]
    fn test_cancel_request_waiting() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.abort_request(SequenceId(1), "User cancelled".to_string());

        // Should transition to Aborted
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Aborted(_))
        ));
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_abort_resolves_single_channel_and_removes_entry() {
        let mut scheduler = create_test_scheduler();

        // Keep the receiver so we can observe what the client sees.
        let (tx, rx) = tokio::sync::oneshot::channel();
        scheduler.add_request(SequenceId(1), vec![0; 100], 200, ResponseSender::Single(tx));
        assert_eq!(scheduler.response_channels.len(), 1);

        scheduler.abort_request(SequenceId(1), "User cancelled".to_string());

        // The waiter is unblocked with an Aborted error (not left hanging)...
        match rx.blocking_recv() {
            Ok(Err(Error::Aborted(reason))) => assert_eq!(reason, "User cancelled"),
            other => panic!("expected Ok(Err(Aborted)), got {:?}", other),
        }
        // ...and the channel entry is not leaked.
        assert!(scheduler.response_channels.is_empty());
    }

    #[test]
    fn test_abort_closes_streaming_channel() {
        let mut scheduler = create_test_scheduler();

        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        scheduler.add_request(SequenceId(1), vec![0; 100], 200, ResponseSender::Stream(tx));

        scheduler.abort_request(SequenceId(1), "boom".to_string());

        // Streaming client receives the error, then the channel closes.
        match rx.blocking_recv() {
            Some(Err(Error::Aborted(reason))) => assert_eq!(reason, "boom"),
            other => panic!("expected Some(Err(Aborted)), got {:?}", other),
        }
        assert!(rx.blocking_recv().is_none(), "channel should be closed");
        assert!(scheduler.response_channels.is_empty());
    }

    #[test]
    fn test_cancel_request_not_found() {
        let mut scheduler = create_test_scheduler();

        // Should not panic
        scheduler.abort_request(SequenceId(999), "User cancelled".to_string());
        assert_eq!(scheduler.sequences.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_success() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();

        // Should move to prefilling
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Prefilling(_))
        ));
        assert_eq!(scheduler.prefill_batch.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_multiple() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.add_request(SequenceId(2), vec![0; 50], 100, create_dummy_response());
        scheduler.schedule_prefill();

        // Both should be scheduled
        assert_eq!(scheduler.prefill_batch.len(), 2);
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_batch_limit() {
        let mut scheduler = create_test_scheduler();

        // Add more than max_batch_size (10)
        for i in 0..15 {
            scheduler.add_request(SequenceId(i), vec![0; 100], 200, create_dummy_response());
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
        let (_event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
        let mut scheduler = Scheduler::new(block_manager, event_rx, 10, 16);

        // First request needs 7 blocks (100 tokens / 16 tokens_per_block)
        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();

        // Should fail - not enough memory
        assert_eq!(scheduler.prefill_batch.len(), 0);
        assert_eq!(scheduler.waiting_queue.len(), 1);
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Waiting(_))
        ));
    }

    #[test]
    fn test_append_tokens_prefilling() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();

        // Append all tokens
        scheduler.append_tokens(SequenceId(1), 100);

        // Should transition to Decoding
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Decoding(_))
        ));
    }

    #[test]
    fn test_append_tokens_decoding() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        // Append more tokens
        scheduler.append_tokens(SequenceId(1), 10);

        if let Some(SequenceState::Decoding(state)) = scheduler.sequences.get(SequenceId(1)) {
            assert_eq!(state.num_tokens, 110);
        } else {
            panic!("Expected Decoding state");
        }
    }

    #[test]
    fn test_append_tokens_reaches_max() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        // Reach max_tokens
        scheduler.append_tokens(SequenceId(1), 100); // 100 + 100 = 200 = max_tokens

        // Should transition to Finished
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Finished(_))
        ));
    }

    #[test]
    fn test_complete_sequence() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.complete_sequence(SequenceId(1), FinishReason::Stop, "test result".to_string());

        // Should be Finished
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Finished(_))
        ));
        assert_eq!(scheduler.prefill_batch.len(), 0);
    }

    #[test]
    fn test_fork_sequence() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.fork_sequence(SequenceId(1), SequenceId(2));

        // Both should exist
        assert!(scheduler.sequences.contains(SequenceId(1)));
        assert!(scheduler.sequences.contains(SequenceId(2)));
        assert!(matches!(
            scheduler.sequences.get(SequenceId(2)),
            Some(SequenceState::Decoding(_))
        ));
    }

    #[test]
    fn test_fork_duplicate_child_id() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100);

        scheduler.add_request(SequenceId(2), vec![0; 50], 100, create_dummy_response()); // Child ID exists
        scheduler.fork_sequence(SequenceId(1), SequenceId(2)); // Should fail

        // Original seq 2 should be unchanged (Waiting)
        assert!(matches!(
            scheduler.sequences.get(SequenceId(2)),
            Some(SequenceState::Waiting(_))
        ));
    }

    #[test]
    fn test_preempt_sequence() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.preempt_sequence(SequenceId(1));

        // Should be Preempted
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Preempted(_))
        ));
        assert_eq!(scheduler.decode_batch.len(), 0);
    }

    #[test]
    fn test_resume_sequence() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100);
        scheduler.preempt_sequence(SequenceId(1));

        scheduler.resume_sequence(SequenceId(1));

        // Should be back to Waiting
        assert!(matches!(
            scheduler.sequences.get(SequenceId(1)),
            Some(SequenceState::Waiting(_))
        ));
        assert_eq!(scheduler.waiting_queue.len(), 1);
    }

    #[test]
    fn test_get_blocks() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();

        let blocks = scheduler.get_blocks(SequenceId(1)).unwrap();
        assert!(!blocks.is_empty());
    }

    #[test]
    fn test_get_blocks_not_found() {
        let scheduler = create_test_scheduler();

        let result = scheduler.get_blocks(SequenceId(999));
        assert!(matches!(result, Err(Error::UnknownSequence(_))));
    }

    #[test]
    fn test_get_stats() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.add_request(SequenceId(2), vec![0; 50], 100, create_dummy_response());
        scheduler.schedule_prefill();

        let stats = scheduler.get_stats();
        assert_eq!(stats.num_sequences, 2);
        assert_eq!(stats.num_running, 2); // Both in prefill
        assert_eq!(stats.num_waiting, 0);
    }

    #[test]
    fn test_try_schedule() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.try_schedule();

        // Should schedule automatically
        assert_eq!(scheduler.prefill_batch.len(), 1);
        assert_eq!(scheduler.waiting_queue.len(), 0);
    }

    #[test]
    fn test_cancel_removes_from_batches() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        assert_eq!(scheduler.prefill_batch.len(), 1);

        scheduler.abort_request(SequenceId(1), "User cancelled".to_string());

        // Should remove from batch
        assert_eq!(scheduler.prefill_batch.len(), 0);
    }
}
