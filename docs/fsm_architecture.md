# FSM Architecture for PUMA

Finite State Machine pattern for sequence lifecycle management. The FSM states
and events are defined in `src/fsm/`; the transition logic lives in the
`SequenceManager` (`src/sequence_manager/mod.rs`), which the `Scheduler` drives.

## Core Concepts

### State Machine per Sequence

Each sequence is an independent state machine. Multiple sequences can be in different states simultaneously.

**Implementation:** `src/fsm/states.rs`

**States:**
- `Empty` - Pre-birth placeholder; the `from` state of `Create`, never persisted
- `Waiting` - In queue, no resources allocated
- `Scheduling` - Being scheduled (transitional)
- `Prefilling` - Processing prompt
- `Decoding` - Generating tokens
- `Preempted` - Temporarily suspended (blocks freed)
- `Finished` - Completed successfully
- `Aborted` - Error or cancelled

### States Own Resources

Each state owns the resources it needs:
- `WaitingState` has no blocks (not allocated yet)
- `DecodingState` owns blocks via `Arc<Vec<BlockId>>`
- `PreemptedState` has no blocks (freed)

**Benefit:** Type system enforces resource invariants - can't access blocks that don't exist.

### Events as State Transformations

Events transform states: `Event + OldState → NewState`

**Implementation:** `src/fsm/events.rs`

**Events:**
- `Create` - Empty → Waiting (birth a new sequence)
- `Schedule` - Waiting → Prefilling (allocate blocks)
- `AppendTokens` - Prefilling → Decoding or Decoding → Decoding/Finished
- `Preempt` - Decoding → Preempted (free blocks on OOM)
- `Resume` - Preempted → Waiting (re-queue)
- `Complete` - * → Finished (free blocks)
- `Abort` - * → Aborted (cleanup)

Invalid transitions return `Error::InvalidTransition`.

Forking (Decoding → (Decoding, Decoding) for beam search) is **not** an event:
it is a dedicated `SequenceManager::fork` method, since it produces a second
sequence rather than transforming one state into another.

### FSM Integration with SequenceManager

The FSM logic lives in the **`SequenceManager`** (the *mechanism* layer), which
owns `BlockManager`. The `Scheduler` (the *policy* layer) drives it with events
and never touches `BlockManager` or transition logic directly.

**Public API:** `SequenceManager::advance(seq_id, event)` - Event-based abstraction
- Single entry point for all state transitions
- Easy to add logging, metrics, debugging
- Backs up state and restores it on error (no block leaks)

**Internal Implementation:** Private `transition()` / `transition_*()` methods
- Type-safe helpers that enforce correct state types
- Direct access to `self.block_manager` - no parameter passing
- Called by `advance()` after looking up the current state

```rust
impl SequenceManager {
    pub fn advance(&mut self, seq_id: SequenceId, event: Event) -> Result<&SequenceState> {
        // Remove current state, run transition(), reinsert on success
        // or restore the backup on error.
    }

    fn transition(&mut self, state: SequenceState, event: Event) -> Result<SequenceState> {
        let seq_id = state.seq_id();
        match (state, event) {
            (SequenceState::Waiting(s), Event::Schedule{..}) =>
                self.transition_schedule(s),
            // ... dispatches to internal methods
        }
    }

    fn transition_schedule(&mut self, state: WaitingState) -> Result<SequenceState> {
        // Direct access to self.block_manager, self.tokens_per_block
    }
}
```

## State Diagram

```
                    AddRequest
                        ↓
    ┌─────────────→ Waiting
    │                   ↓ ScheduleEvent
    │              Prefilling
    │                   ↓ AppendTokens
    │              Decoding
    │                   ├─→ AppendTokens (continue)
    │                   │   ├─ Decoding (more tokens)
    │                   │   └─ Finished (max_tokens)
    │                   │
    │                   ├─→ fork() method (beam search, not an event)
    │                   │   ├─ Parent: Decoding
    │                   │   └─ Child: Decoding
    │                   │
    │                   ├─→ PreemptEvent (OOM)
    │                   │   ↓ Preempted
    │                   │   ↓ ResumeEvent
    │                   └───┘
    │
    └─ CompleteEvent → Finished
```

## Scheduler / SequenceManager Integration

The `SequenceManager` owns the FSM and executes all state transitions; the
`Scheduler` owns the queues/batches and decides *which* events to fire.

**Implementation:** `src/sequence_manager/mod.rs` (mechanism),
`src/scheduler/core.rs` (policy)

**Key features:**
- Synchronous API (LLMEngine handles async coordination)
- `SequenceManager` maintains `HashMap<SequenceId, SequenceState>`
- `SequenceManager` owns `BlockManager` directly - no parameter passing needed
- Clones state before transitions (prevents loss on failure)
- Restores backup on error (no block leaks)
- Event-based `advance()` method for maintainability

**Tests:** `cargo test --lib`

## Arc Optimization

States use `Arc<Vec<BlockId>>` for efficient cloning:

- **State cloning** (for backup/fork): **O(1)** - just increment ref count
- **Modifying blocks**: Copy-on-write via `Arc::make_mut()` - only copies if shared
- **Fork sequences**: Share blocks via Arc, diverge only when modified

**Benefit:** Hot path performance - state cloning costs ~50 bytes instead of O(n) vector copy.

## Error Handling

All state transitions are wrapped with backup/restore:

1. Remove state from HashMap
2. Clone state as backup
3. Apply event transformation
4. On success: insert new state
5. On failure: restore backup, log warning

**Prevents:**
- Sequence loss on transition failure
- Block leaks (blocks remain owned by restored state)
- Duplicate sequences (check before insert)

## Type Safety

**Runtime checks:**
- Invalid transitions caught by pattern matching
- Return `Error::InvalidTransition` with clear message

**Resource ownership:**
- States own their blocks via Arc
- Type system prevents accessing freed blocks
- Drop trait ensures cleanup

## Benefits

1. **Type Safety** - States enforce resource invariants, invalid transitions caught explicitly
2. **Clear Ownership** - Resources belong to states, automatic cleanup on drop
3. **Testability** - Events are pure functions, test state transitions in isolation
4. **Explicitness** - Every transition is explicit, state diagram maps to code
5. **Performance** - Arc enables O(1) cloning, copy-on-write for efficiency
6. **Reliability** - Backup/restore pattern prevents data loss on errors

## Two-Level Event Architecture

PUMA uses a two-level event system (inspired by TokenSpeed):

**Level 1: FSM Events (Internal)** - `src/fsm/events.rs`
- Created by the Scheduler internally
- Include scheduler context (tokens_per_block, etc.)
- Applied via `SequenceManager::advance(seq_id, event)`
- Examples: `Event::Schedule`, `Event::AppendTokens`

**Level 2: Scheduler Events (External)** - `src/scheduler/events.rs`
- Created by external components (clients, GPU workers)
- Simple data, no resource pointers
- LLMEngine translates these to FSM events
- Examples: `SchedulerEvent::AddRequest`, `SchedulerEvent::CancelRequest`

**Why two levels?** External components cannot create FSM events because:
- They don't own `BlockManager`
- They don't know scheduler policies (tokens_per_block)
- FSM events require scheduler context

## Key Files

- `src/fsm/states.rs` - State definitions and helper methods
- `src/fsm/events.rs` - FSM event enum (internal transitions)
- `src/sequence_manager/mod.rs` - Owns memory + FSM transition logic (`advance()`)
- `src/scheduler/core.rs` - Scheduling policy that drives the SequenceManager
- `src/scheduler/events.rs` - External scheduler events
- `src/engine/mod.rs` - Event coordinator

## Usage Example

```rust
// External component sends a high-level event
scheduler_tx.send(SchedulerEvent::AddRequest {
    seq_id: SequenceId(1),
    token_ids: vec![/* tokenized prompt */],
    max_tokens: 100,
    response_tx, // channel the result is delivered on
})?;

// The Scheduler translates it to an FSM event and drives the SequenceManager,
// which owns BlockManager and performs the transition.
impl Scheduler {
    fn schedule_prefill(&mut self) {
        while let Some(seq_id) = self.waiting_queue.pop_front() {
            // Build the FSM event with scheduler context
            let event = Event::Schedule {
                tokens_per_block: self.tokens_per_block,
            };

            match self.sequences.advance(seq_id, event) {
                Ok(_) => self.prefill_batch.push(seq_id),
                Err(e) => { /* apply OOM / queueing policy */ }
            }
        }
    }
}
```
