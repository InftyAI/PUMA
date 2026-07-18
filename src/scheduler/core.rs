use super::events::{ResponseSender, SchedulerStats};
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use crate::fsm::{
    AbortedState, DecodingState, Event, FinishReason, FinishedState, PreemptedState,
    PrefillingState, SequenceState, WaitingState,
};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tracing::{debug, info, warn};

/// Scheduler - memory and batch management (synchronous API)
///
/// Architecture
/// - Pure synchronous methods (no async, no loop)
/// - LLMEngine calls: add_request() → schedule() → process_outputs()
/// - Owns BlockManager and performs FSM transitions directly
///
/// Responsibilities:
/// - Owns BlockManager (memory allocation)
/// - Owns sequence state storage (FSM states)
/// - Manages batches (waiting/prefill/decode)
/// - Scheduling policy (what to schedule when)
/// - FSM transitions (internal methods with direct BlockManager access)
pub struct Scheduler {
    /// Block manager for memory allocation
    block_manager: BlockManager,

    /// All sequences and their states
    sequences: HashMap<SequenceId, SequenceState>,

    /// Response channels to send results back to clients (supports streaming)
    response_channels: HashMap<SequenceId, ResponseSender>,

    /// Waiting sequences (not yet scheduled)
    waiting_queue: VecDeque<SequenceId>,

    /// Currently running sequences in prefill phase
    prefill_batch: Vec<SequenceId>,

    /// Currently running sequences in decode phase
    decode_batch: Vec<SequenceId>,

    /// Configuration
    max_batch_size: usize,
    tokens_per_block: usize,
}

impl Scheduler {
    pub fn new(
        block_manager: BlockManager,
        max_batch_size: usize,
        tokens_per_block: usize,
    ) -> Self {
        Self {
            block_manager,
            sequences: HashMap::new(),
            response_channels: HashMap::new(),
            waiting_queue: VecDeque::new(),
            prefill_batch: Vec::new(),
            decode_batch: Vec::new(),
            max_batch_size,
            tokens_per_block,
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
        if self.sequences.contains_key(&seq_id) {
            warn!("Duplicate seq_id {:?}", seq_id);
            self.send_error(response_tx, Error::InvalidTransition("Duplicate sequence ID"));
            return;
        }

        // Store response channel
        self.response_channels.insert(seq_id, response_tx);

        // Create waiting state - tokens already provided by LLMEngine
        let state = SequenceState::Waiting(WaitingState {
            seq_id,
            token_ids: Arc::new(token_ids),
            max_tokens,
        });

        self.sequences.insert(seq_id, state);
        self.waiting_queue.push_back(seq_id);
    }

    pub fn cancel_request(&mut self, seq_id: SequenceId) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Cannot cancel - sequence {:?} not found", seq_id);
                return;
            }
        };

        let event = Event::Abort {
            reason: "User cancelled".to_string(),
        };

        match self.transition(state, event) {
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
            .filter_map(|seq_id| match self.sequences.get(&seq_id) {
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
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = Event::AppendTokens {
            num_tokens,
            tokens_per_block: self.tokens_per_block,
        };

        match self.transition(state, event) {
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

    pub fn complete_sequence(&mut self, seq_id: SequenceId, reason: FinishReason, result: String) {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => {
                warn!("Sequence {:?} not found", seq_id);
                return;
            }
        };

        let backup = state.clone();
        let event = Event::Complete { reason };

        match self.transition(state, event) {
            Ok(new_state) => {
                info!("Completed sequence {:?}", seq_id);
                self.sequences.insert(seq_id, new_state);

                // Remove from batches
                self.prefill_batch.retain(|&id| id != seq_id);
                self.decode_batch.retain(|&id| id != seq_id);

                // Send result back to client
                self.send_result(seq_id, result);
            }
            Err(e) => {
                warn!("Failed to complete {:?}: {:?}", seq_id, e);
                self.sequences.insert(seq_id, backup);

                // Send error back to client
                if let Some(response_tx) = self.response_channels.remove(&seq_id) {
                    self.send_error(response_tx, e);
                }
            }
        }
    }

    pub fn fork_sequence(&mut self, parent_id: SequenceId, child_id: SequenceId) {
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

        // TODO: Fork needs special handling - returns (parent, child)
        // For now, use manual implementation
        let backup = parent_state.clone();

        if let SequenceState::Decoding(s) = parent_state {
            // Copy-on-write: increment ref counts
            for &block_id in s.blocks.iter() {
                if let Err(e) = self.block_manager.add_ref(block_id) {
                    warn!("Failed to fork {:?}: {:?}", parent_id, e);
                    self.sequences.insert(parent_id, backup);
                    return;
                }
            }

            let child = DecodingState {
                seq_id: child_id,
                token_ids: Arc::clone(&s.token_ids),
                blocks: Arc::clone(&s.blocks),
                num_tokens: s.num_tokens,
                max_tokens: s.max_tokens,
            };

            self.sequences.insert(parent_id, SequenceState::Decoding(s));
            self.sequences
                .insert(child_id, SequenceState::Decoding(child));
            debug!("Forked sequence {:?} → {:?}", parent_id, child_id);
        } else {
            warn!("Cannot fork {:?} - not in Decoding state", parent_id);
            self.sequences.insert(parent_id, backup);
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
        let event = Event::Preempt;

        match self.transition(state, event) {
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
        let event = Event::Resume;

        match self.transition(state, event) {
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

            // Get and remove state
            let state = match self.sequences.remove(&seq_id) {
                Some(s) => s,
                None => {
                    warn!("Sequence {:?} not found", seq_id);
                    continue;
                }
            };

            let event = Event::Schedule {
                tokens_per_block: self.tokens_per_block,
            };

            match self.transition(state.clone(), event) {
                Ok(new_state) => {
                    self.sequences.insert(seq_id, new_state);
                    self.prefill_batch.push(seq_id);
                    info!("Scheduled {:?} for prefill", seq_id);
                }
                Err(Error::OutOfMemory) => {
                    // Transition already cleaned up allocated blocks
                    // Restore state, put back in queue and stop
                    self.sequences.insert(seq_id, state);
                    self.waiting_queue.push_front(seq_id);
                    debug!("OOM - cannot schedule {:?}", seq_id);
                    break;
                }
                Err(Error::InvalidTransition(_)) => {
                    // Not in Waiting state - restore and skip
                    self.sequences.insert(seq_id, state);
                    warn!("Cannot schedule {:?} - not in Waiting state", seq_id);
                }
                Err(e) => {
                    // Other error - restore state
                    self.sequences.insert(seq_id, state);
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
        match self.sequences.get(&seq_id) {
            Some(state) => Ok(state.blocks().map(|b| b.to_vec()).unwrap_or_default()),
            None => Err(Error::UnknownSequence(seq_id)),
        }
    }

    pub fn get_stats(&self) -> SchedulerStats {
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

    // ===== FSM Transitions =====

    /// Transition sequence state using FSM event - single entry point for all transitions
    ///
    /// This method provides an event-based abstraction over the internal transition methods.
    ///
    /// # Design
    ///
    /// **Public API**: Event-based for maintainability
    /// - Single entry point makes it easy to add cross-cutting concerns (logging, metrics)
    /// - Event enum can be serialized for debugging/replay
    /// - Clean interface for callers
    ///
    /// **Internal Implementation**: Type-safe methods
    /// - Private `transition_*()` methods enforce correct state types at compile time
    /// - Called by this method after event dispatching
    ///
    /// # Arguments
    ///
    /// * `state` - Current sequence state (will be consumed)
    /// * `event` - FSM event to apply
    ///
    /// # Returns
    ///
    /// New state on success, or `Error::InvalidTransition` for invalid state/event combinations
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let event = Event::Schedule { tokens_per_block: 16 };
    /// let new_state = self.transition(state, event)?;
    /// self.sequences.insert(seq_id, new_state);
    /// ```
    pub fn transition(&mut self, state: SequenceState, event: Event) -> Result<SequenceState> {
        use crate::fsm::Event;

        match (state, event) {
            // Waiting → Prefilling
            (SequenceState::Waiting(s), Event::Schedule { .. }) => self.transition_schedule(s),

            // Prefilling → Prefilling/Decoding
            (SequenceState::Prefilling(s), Event::AppendTokens { num_tokens, .. }) => {
                self.transition_append_tokens_prefilling(s, num_tokens)
            }

            // Decoding → Decoding/Finished
            (SequenceState::Decoding(s), Event::AppendTokens { num_tokens, .. }) => {
                self.transition_append_tokens_decoding(s, num_tokens)
            }

            // Any running state → Finished
            (state, Event::Complete { reason }) => self.transition_complete(state, reason),

            // Running → Preempted
            (
                state @ (SequenceState::Prefilling(_) | SequenceState::Decoding(_)),
                Event::Preempt,
            ) => self.transition_preempt(state),

            // Preempted → Waiting
            (SequenceState::Preempted(s), Event::Resume) => self.transition_resume(s),

            // Any → Aborted
            (state, Event::Abort { reason }) => self.transition_abort(state, reason),

            // Invalid transitions
            _ => Err(Error::InvalidTransition("Invalid state transition")),
        }
    }

    // ===== Internal Transition Methods =====
    //
    // These private methods implement the actual transition logic.
    // Called by apply() after event dispatching.

    /// Transition: Waiting → Prefilling (allocate blocks for tokenized prompt)
    fn transition_schedule(&mut self, state: WaitingState) -> Result<SequenceState> {
        let prompt_tokens = state.token_ids.len();
        let blocks_needed = prompt_tokens.div_ceil(self.tokens_per_block);

        let mut blocks = Vec::new();
        for _ in 0..blocks_needed {
            match self.block_manager.allocate() {
                Ok(block_id) => blocks.push(block_id),
                Err(Error::OutOfMemory) => {
                    // Cleanup on OOM
                    for block in blocks {
                        let _ = self.block_manager.free(block);
                    }
                    return Err(Error::OutOfMemory);
                }
                Err(e) => {
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
            prompt_tokens
        );

        Ok(SequenceState::Prefilling(PrefillingState {
            seq_id: state.seq_id,
            token_ids: state.token_ids,
            blocks: Arc::new(blocks),
            tokens_filled: 0,
            tokens_total: prompt_tokens,
            max_tokens: state.max_tokens,
        }))
    }

    /// Transition: Prefilling → Decoding or Prefilling (append tokens)
    fn transition_append_tokens_prefilling(
        &mut self,
        mut state: PrefillingState,
        num_tokens: usize,
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
                token_ids: state.token_ids,
                blocks: state.blocks,
                num_tokens: state.tokens_filled,
                max_tokens: state.max_tokens,
            }))
        } else {
            Ok(SequenceState::Prefilling(state))
        }
    }

    /// Transition: Decoding → Decoding or Finished (append tokens, maybe allocate more blocks)
    fn transition_append_tokens_decoding(
        &mut self,
        mut state: DecodingState,
        num_tokens: usize,
    ) -> Result<SequenceState> {
        state.num_tokens += num_tokens;

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

    /// Transition: Prefilling/Decoding → Finished
    fn transition_complete(
        &mut self,
        state: SequenceState,
        reason: FinishReason,
    ) -> Result<SequenceState> {
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

        debug!("Completed seq {:?}: {:?}", seq_id, reason);

        Ok(SequenceState::Finished(FinishedState {
            seq_id,
            finish_reason: reason,
        }))
    }

    /// Transition: Prefilling/Decoding → Preempted (free blocks)
    fn transition_preempt(&mut self, state: SequenceState) -> Result<SequenceState> {
        let seq_id = state.seq_id();
        let (token_ids, num_tokens, max_tokens) = match &state {
            SequenceState::Prefilling(s) => {
                (Arc::clone(&s.token_ids), s.tokens_filled, s.max_tokens)
            }
            SequenceState::Decoding(s) => (Arc::clone(&s.token_ids), s.num_tokens, s.max_tokens),
            _ => {
                return Err(Error::InvalidTransition(
                    "Can only preempt Prefilling/Decoding",
                ))
            }
        };

        // Free blocks
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

        debug!("Preempted seq {:?}", seq_id);

        Ok(SequenceState::Preempted(PreemptedState {
            seq_id,
            token_ids,
            num_tokens,
            max_tokens,
        }))
    }

    /// Transition: Preempted → Waiting
    fn transition_resume(&mut self, state: PreemptedState) -> Result<SequenceState> {
        debug!("Resuming seq {:?}", state.seq_id);

        Ok(SequenceState::Waiting(WaitingState {
            seq_id: state.seq_id,
            token_ids: state.token_ids,
            max_tokens: state.max_tokens,
        }))
    }

    /// Transition: Any → Aborted (free blocks, cleanup)
    fn transition_abort(&mut self, state: SequenceState, reason: String) -> Result<SequenceState> {
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

        warn!("Aborted seq {:?}: {}", seq_id, reason);

        Ok(SequenceState::Aborted(AbortedState { seq_id, reason }))
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
        Scheduler::new(block_manager, 10, 16)
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
            scheduler.sequences.get(&SequenceId(1)),
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
        let mut scheduler = create_test_scheduler();

        // Should not panic
        scheduler.cancel_request(SequenceId(999));
        assert_eq!(scheduler.sequences.len(), 0);
    }

    #[test]
    fn test_schedule_prefill_success() {
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
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
        let mut scheduler = Scheduler::new(block_manager, 10, 16);

        // First request needs 7 blocks (100 tokens / 16 tokens_per_block)
        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
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
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
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
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
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
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
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
        let mut scheduler = create_test_scheduler();

        scheduler.add_request(SequenceId(1), vec![0; 100], 200, create_dummy_response());
        scheduler.schedule_prefill();
        scheduler.append_tokens(SequenceId(1), 100); // → Decoding

        scheduler.complete_sequence(SequenceId(1), FinishReason::Stop, "test result".to_string());

        // Should be Finished
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(1)),
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
        assert!(scheduler.sequences.contains_key(&SequenceId(1)));
        assert!(scheduler.sequences.contains_key(&SequenceId(2)));
        assert!(matches!(
            scheduler.sequences.get(&SequenceId(2)),
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
            scheduler.sequences.get(&SequenceId(2)),
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
            scheduler.sequences.get(&SequenceId(1)),
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
            scheduler.sequences.get(&SequenceId(1)),
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

        scheduler.cancel_request(SequenceId(1));

        // Should remove from batch
        assert_eq!(scheduler.prefill_batch.len(), 0);
    }
}
