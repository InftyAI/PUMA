//! SequenceManager - owns sequence state and the memory those sequences use.
//!
//! This is the *mechanism* layer: it holds the `sequences` map and the
//! `BlockManager`, and executes FSM transitions (which allocate/free blocks).
//! The `Scheduler` is the *policy* layer — it decides which events fire and
//! owns the queues/batches — and drives this manager without ever touching a
//! `BlockManager` or a `BlockId` directly.
//!
//! Keeping memory here (rather than in the scheduler) localizes future changes
//! like multi-GPU block pools or CPU/GPU swap: the scheduler keeps calling
//! `advance`/`create`/`can_allocate`, and the device details stay inside.

use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::*;
use crate::fsm::{
    AbortedState, DecodingState, Event, FinishReason, FinishedState, PreemptedState,
    PrefillingState, SequenceState, WaitingState,
};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{debug, warn};

/// Owns sequence state + block memory and executes FSM transitions.
pub struct SequenceManager {
    /// Block manager for memory allocation
    block_manager: BlockManager,

    /// All sequences and their states
    sequences: HashMap<SequenceId, SequenceState>,

    /// Tokens per block (used when computing block allocations)
    tokens_per_block: usize,
}

impl SequenceManager {
    pub fn new(block_manager: BlockManager, tokens_per_block: usize) -> Self {
        Self {
            block_manager,
            sequences: HashMap::new(),
            tokens_per_block,
        }
    }

    // ===== Sequence map access (used by the scheduler for policy) =====

    /// Whether a sequence with this id exists.
    pub fn contains(&self, seq_id: SequenceId) -> bool {
        self.sequences.contains_key(&seq_id)
    }

    /// Look up a sequence's current state.
    pub fn get(&self, seq_id: SequenceId) -> Option<&SequenceState> {
        self.sequences.get(&seq_id)
    }

    /// Total number of tracked sequences.
    pub fn len(&self) -> usize {
        self.sequences.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sequences.is_empty()
    }

    /// Count sequences currently running (prefill or decode).
    pub fn num_running(&self) -> usize {
        self.sequences.values().filter(|s| s.is_running()).count()
    }

    /// Count sequences currently preempted.
    pub fn num_preempted(&self) -> usize {
        self.sequences
            .values()
            .filter(|s| matches!(s, SequenceState::Preempted(_)))
            .count()
    }

    /// Block ids held by a sequence (for KV-cache addressing).
    pub fn blocks(&self, seq_id: SequenceId) -> Result<Vec<BlockId>> {
        match self.sequences.get(&seq_id) {
            Some(state) => Ok(state.blocks().map(|b| b.to_vec()).unwrap_or_default()),
            None => Err(Error::UnknownSequence(seq_id)),
        }
    }

    /// Per-block-type memory statistics.
    pub fn block_stats(&self) -> HashMap<BlockType, BlockStats> {
        self.block_manager.get_stats()
    }

    // ===== Transition entry points (used by the scheduler) =====

    /// Create a new sequence in the Waiting state and store it.
    ///
    /// Returns `Err` if the id already exists.
    pub fn create(
        &mut self,
        seq_id: SequenceId,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
    ) -> Result<()> {
        if self.sequences.contains_key(&seq_id) {
            return Err(Error::InvalidTransition("Duplicate sequence ID"));
        }
        let event = Event::Create {
            token_ids,
            max_tokens,
        };
        let state = self.transition(SequenceState::Empty(seq_id), event)?;
        self.sequences.insert(seq_id, state);
        Ok(())
    }

    /// Apply an FSM event to an existing sequence, updating the map in place.
    ///
    /// On success the sequence's state is replaced with the new state. On
    /// failure the original state is restored and the error is returned, so the
    /// caller can apply rollback/OOM policy.
    pub fn advance(&mut self, seq_id: SequenceId, event: Event) -> Result<&SequenceState> {
        let state = match self.sequences.remove(&seq_id) {
            Some(s) => s,
            None => return Err(Error::UnknownSequence(seq_id)),
        };
        let backup = state.clone();

        match self.transition(state, event) {
            Ok(new_state) => {
                self.sequences.insert(seq_id, new_state);
                Ok(&self.sequences[&seq_id])
            }
            Err(e) => {
                // Restore prior state so the map is never left missing a seq.
                self.sequences.insert(seq_id, backup);
                Err(e)
            }
        }
    }

    /// Fork a decoding sequence into a child via copy-on-write block sharing.
    pub fn fork(&mut self, parent_id: SequenceId, child_id: SequenceId) -> Result<()> {
        if self.sequences.contains_key(&child_id) {
            warn!("Child ID {:?} already exists", child_id);
            return Err(Error::InvalidTransition("Duplicate child ID"));
        }

        let parent_state = match self.sequences.remove(&parent_id) {
            Some(s) => s,
            None => {
                warn!("Parent sequence {:?} not found", parent_id);
                return Err(Error::UnknownSequence(parent_id));
            }
        };

        let backup = parent_state.clone();

        if let SequenceState::Decoding(s) = parent_state {
            // Copy-on-write: increment ref counts on shared blocks.
            for &block_id in s.blocks.iter() {
                if let Err(e) = self.block_manager.add_ref(block_id) {
                    warn!("Failed to fork {:?}: {:?}", parent_id, e);
                    self.sequences.insert(parent_id, backup);
                    return Err(e);
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
            Ok(())
        } else {
            warn!("Cannot fork {:?} - not in Decoding state", parent_id);
            self.sequences.insert(parent_id, backup);
            Err(Error::InvalidTransition("Can only fork Decoding sequences"))
        }
    }

    // ===== FSM Transitions =====

    /// Compute the next state for `(state, event)`, allocating/freeing blocks as
    /// needed. Pure with respect to the sequence map — does not read or write
    /// `self.sequences`; the caller stores the result.
    fn transition(&mut self, state: SequenceState, event: Event) -> Result<SequenceState> {
        let seq_id = state.seq_id();
        match (state, event) {
            // Empty → Waiting (new sequence)
            (
                SequenceState::Empty(_),
                Event::Create {
                    token_ids,
                    max_tokens,
                },
            ) => self.transition_create(seq_id, token_ids, max_tokens),

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

    /// Transition: Empty → Waiting (construct a new sequence's initial state)
    fn transition_create(
        &mut self,
        seq_id: SequenceId,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
    ) -> Result<SequenceState> {
        Ok(SequenceState::Waiting(WaitingState {
            seq_id,
            token_ids: Arc::new(token_ids),
            max_tokens,
        }))
    }

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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::allocator::CpuAllocator;
    use crate::fsm::FinishReason;

    /// Build a manager with `max_blocks` worth of memory and the given
    /// `tokens_per_block`. Block byte-size is fixed at 1024; capping allocator
    /// memory is how we force OOM in tests.
    fn manager(max_blocks: usize, tokens_per_block: usize) -> SequenceManager {
        let block_bytes = 1024;
        let allocator = Box::new(CpuAllocator::new(max_blocks * block_bytes));
        let block_manager = BlockManager::new(allocator, block_bytes);
        SequenceManager::new(block_manager, tokens_per_block)
    }

    fn schedule_event() -> Event {
        Event::Schedule {
            tokens_per_block: 4,
        }
    }

    fn append(num_tokens: usize) -> Event {
        Event::AppendTokens {
            num_tokens,
            tokens_per_block: 4,
        }
    }

    #[test]
    fn test_create_inserts_waiting_sequence() {
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);

        mgr.create(id, vec![1, 2, 3], 20).unwrap();

        assert!(mgr.contains(id));
        assert_eq!(mgr.len(), 1);
        assert!(!mgr.is_empty());
        assert_eq!(mgr.get(id).unwrap().variant_name(), "Waiting");
    }

    #[test]
    fn test_create_duplicate_id_errors() {
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);
        mgr.create(id, vec![1, 2, 3], 20).unwrap();

        let err = mgr.create(id, vec![4, 5], 20).unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
        // Original sequence is untouched.
        assert_eq!(mgr.len(), 1);
    }

    #[test]
    fn test_advance_unknown_sequence_errors() {
        let mut mgr = manager(10, 4);
        let err = mgr.advance(SequenceId(99), schedule_event()).unwrap_err();
        assert!(matches!(err, Error::UnknownSequence(SequenceId(99))));
    }

    #[test]
    fn test_schedule_allocates_blocks() {
        // 6 tokens, 4 tokens/block => 2 blocks needed.
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);
        mgr.create(id, vec![1, 2, 3, 4, 5, 6], 20).unwrap();

        let state = mgr.advance(id, schedule_event()).unwrap();
        assert_eq!(state.variant_name(), "Prefilling");
        assert_eq!(mgr.blocks(id).unwrap().len(), 2);
    }

    #[test]
    fn test_schedule_oom_rolls_back_to_waiting() {
        // Needs 2 blocks but only 1 is available -> OOM, no blocks leaked.
        let mut mgr = manager(1, 4);
        let id = SequenceId(1);
        mgr.create(id, vec![1, 2, 3, 4, 5, 6], 20).unwrap();

        let err = mgr.advance(id, schedule_event()).unwrap_err();
        assert!(matches!(err, Error::OutOfMemory));

        // State restored to Waiting and no blocks are held.
        assert_eq!(mgr.get(id).unwrap().variant_name(), "Waiting");
        assert!(mgr.blocks(id).unwrap().is_empty());

        // The partially-allocated block was freed: a 1-block prompt now fits.
        let id2 = SequenceId(2);
        mgr.create(id2, vec![1, 2], 20).unwrap();
        assert!(mgr.advance(id2, schedule_event()).is_ok());
    }

    #[test]
    fn test_prefill_to_decoding_transition() {
        // 4 tokens, 4/block => 1 block, prefill completes in one append.
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);
        mgr.create(id, vec![1, 2, 3, 4], 20).unwrap();
        mgr.advance(id, schedule_event()).unwrap();

        // Partial prefill stays in Prefilling.
        let state = mgr.advance(id, append(2)).unwrap();
        assert_eq!(state.variant_name(), "Prefilling");

        // Reaching tokens_total flips to Decoding.
        let state = mgr.advance(id, append(2)).unwrap();
        assert_eq!(state.variant_name(), "Decoding");
        assert_eq!(mgr.num_running(), 1);
    }

    #[test]
    fn test_decoding_reaches_max_tokens_finishes_and_frees() {
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);
        // prompt of 4 tokens, max_tokens 5.
        mgr.create(id, vec![1, 2, 3, 4], 5).unwrap();
        mgr.advance(id, schedule_event()).unwrap();
        mgr.advance(id, append(4)).unwrap(); // -> Decoding, num_tokens=4

        let state = mgr.advance(id, append(1)).unwrap(); // hits max_tokens=5
        assert_eq!(state.variant_name(), "Finished");
        // Blocks freed on finish.
        assert!(mgr.blocks(id).unwrap().is_empty());
        assert_eq!(mgr.num_running(), 0);
    }

    #[test]
    fn test_preempt_frees_blocks_then_resume_returns_to_waiting() {
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);
        mgr.create(id, vec![1, 2, 3, 4], 20).unwrap();
        mgr.advance(id, schedule_event()).unwrap();
        mgr.advance(id, append(4)).unwrap(); // Decoding

        let state = mgr.advance(id, Event::Preempt).unwrap();
        assert_eq!(state.variant_name(), "Preempted");
        assert_eq!(mgr.num_preempted(), 1);
        assert!(mgr.blocks(id).unwrap().is_empty());

        let state = mgr.advance(id, Event::Resume).unwrap();
        assert_eq!(state.variant_name(), "Waiting");
        assert_eq!(mgr.num_preempted(), 0);
    }

    #[test]
    fn test_complete_and_abort_finish_the_sequence() {
        let mut mgr = manager(10, 4);

        let a = SequenceId(1);
        mgr.create(a, vec![1, 2, 3, 4], 20).unwrap();
        mgr.advance(a, schedule_event()).unwrap();
        let state = mgr
            .advance(
                a,
                Event::Complete {
                    reason: FinishReason::Stop,
                },
            )
            .unwrap();
        assert_eq!(state.variant_name(), "Finished");
        assert!(mgr.blocks(a).unwrap().is_empty());

        let b = SequenceId(2);
        mgr.create(b, vec![1, 2, 3, 4], 20).unwrap();
        mgr.advance(b, schedule_event()).unwrap();
        let state = mgr
            .advance(
                b,
                Event::Abort {
                    reason: "cancelled".to_string(),
                },
            )
            .unwrap();
        assert_eq!(state.variant_name(), "Aborted");
        assert!(mgr.blocks(b).unwrap().is_empty());
    }

    #[test]
    fn test_invalid_transition_preserves_state() {
        // Resume on a Waiting sequence is invalid.
        let mut mgr = manager(10, 4);
        let id = SequenceId(1);
        mgr.create(id, vec![1, 2, 3], 20).unwrap();

        let err = mgr.advance(id, Event::Resume).unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
        // Sequence remains present and in its original state.
        assert!(mgr.contains(id));
        assert_eq!(mgr.get(id).unwrap().variant_name(), "Waiting");
    }

    #[test]
    fn test_fork_shares_blocks_of_decoding_parent() {
        let mut mgr = manager(10, 4);
        let parent = SequenceId(1);
        let child = SequenceId(2);
        mgr.create(parent, vec![1, 2, 3, 4], 20).unwrap();
        mgr.advance(parent, schedule_event()).unwrap();
        mgr.advance(parent, append(4)).unwrap(); // Decoding

        let parent_blocks = mgr.blocks(parent).unwrap();
        mgr.fork(parent, child).unwrap();

        // Child exists, is Decoding, and shares the parent's blocks.
        assert_eq!(mgr.get(child).unwrap().variant_name(), "Decoding");
        assert_eq!(mgr.blocks(child).unwrap(), parent_blocks);
        assert_eq!(mgr.num_running(), 2);
    }

    #[test]
    fn test_fork_non_decoding_parent_errors() {
        let mut mgr = manager(10, 4);
        let parent = SequenceId(1);
        mgr.create(parent, vec![1, 2, 3], 20).unwrap(); // Waiting

        let err = mgr.fork(parent, SequenceId(2)).unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
        // Parent is restored and child was never inserted.
        assert!(mgr.contains(parent));
        assert!(!mgr.contains(SequenceId(2)));
    }

    #[test]
    fn test_fork_duplicate_child_errors() {
        let mut mgr = manager(10, 4);
        let parent = SequenceId(1);
        let child = SequenceId(2);
        mgr.create(parent, vec![1, 2, 3, 4], 20).unwrap();
        mgr.advance(parent, schedule_event()).unwrap();
        mgr.advance(parent, append(4)).unwrap();

        mgr.create(child, vec![9], 20).unwrap(); // child id already taken
        let err = mgr.fork(parent, child).unwrap_err();
        assert!(matches!(err, Error::InvalidTransition(_)));
    }
}
