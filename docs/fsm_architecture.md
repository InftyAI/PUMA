# FSM Architecture for PUMA

Finite State Machine pattern for sequence lifecycle management, integrated into the Scheduler.

## Core Concepts

### State Machine per Sequence

Each sequence is an independent state machine. Multiple sequences can be in different states simultaneously.

**Implementation:** `src/fsm/states.rs`

**States:**
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
- `Schedule` - Waiting → Prefilling (allocate blocks)
- `AppendTokens` - Prefilling → Decoding or Decoding → Decoding/Finished
- `Fork` - Decoding → (Decoding, Decoding) for beam search
- `Preempt` - Decoding → Preempted (free blocks on OOM)
- `Resume` - Preempted → Waiting (re-queue)
- `Complete` - * → Finished (free blocks)
- `Abort` - * → Aborted (cleanup)

Invalid transitions return `Error::InvalidTransition`.

### FSM Integration with Scheduler

The FSM logic is **integrated into the Scheduler** to solve the ownership boundary problem:

**Public API:** `Scheduler::transition(state, event)` - Event-based abstraction
- Single entry point for all state transitions
- Easy to add logging, metrics, debugging
- Event replay capability for testing

**Internal Implementation:** Private `transition_*()` methods
- Type-safe helpers that enforce correct state types
- Direct access to `self.block_manager` - no parameter passing
- Called by `transition()` after event dispatching

```rust
impl Scheduler {
    pub fn transition(&mut self, state: SequenceState, event: Event) -> Result<SequenceState> {
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
    │                   ├─→ ForkEvent (beam search)
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

## Scheduler Integration

The Scheduler owns the FSM and manages all state transitions.

**Implementation:** `src/scheduler/core.rs`

**Key features:**
- Synchronous API (LLMEngine handles async coordination)
- Maintains `HashMap<SequenceId, SequenceState>`
- Owns `BlockManager` directly - no parameter passing needed
- Clones state before transitions (prevents loss on failure)
- Restores backup on error (no block leaks)
- Event-based `apply()` method for maintainability

**Tests:** `cargo test --lib` (120 tests passing)

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
- Created by Scheduler internally
- Include scheduler context (tokens_per_block, etc.)
- Used with `Scheduler::apply(state, event)`
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

See `ARCHITECTURE.md` for detailed explanation and flow diagrams.

## Key Files

- `src/fsm/states.rs` - State definitions and helper methods
- `src/fsm/events.rs` - FSM event enum (internal transitions)
- `src/scheduler/core.rs` - Scheduler with integrated FSM (`apply()` method)
- `src/scheduler/events.rs` - External scheduler events
- `src/backend/llm_engine.rs` - Event coordinator
- `ARCHITECTURE.md` - Comprehensive architecture documentation

## Usage Example

```rust
// External component sends high-level event
scheduler_tx.send(SchedulerEvent::AddRequest {
    seq_id: 1,
    prompt_tokens: 10,
    max_tokens: 100,
})?;

// Scheduler translates to FSM events internally
impl Scheduler {
    pub fn schedule(&mut self) -> bool {
        // Pop from waiting queue
        let state = self.sequences.remove(&seq_id).unwrap();
        
        // Create FSM event with scheduler context
        let event = Event::Schedule {
            tokens_per_block: self.tokens_per_block,
        };
        
        // Perform transition (direct access to self.block_manager)
        match self.transition(state, event) {
            Ok(new_state) => {
                self.sequences.insert(seq_id, new_state);
                self.prefill_batch.push(seq_id);
            }
            Err(e) => { /* handle error */ }
        }
    }
}
```
