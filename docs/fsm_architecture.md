# FSM Architecture for PUMA

Finite State Machine pattern for sequence lifecycle management.

## Core Concepts

### State Machine per Sequence

Each sequence is an independent state machine. Multiple sequences can be in different states simultaneously.

**Implementation:** `src/sequence_manager/states.rs`

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

**Implementation:** `src/sequence_manager/fsm_events.rs`

**Events:**
- `ScheduleEvent` - Waiting → Prefilling (allocate blocks)
- `AppendTokensEvent` - Prefilling → Decoding or Decoding → Decoding/Finished
- `ForkEvent` - Decoding → (Decoding, Decoding) for beam search
- `PreemptEvent` - Decoding → Preempted (free blocks on OOM)
- `ResumeEvent` - Preempted → Waiting (re-queue)
- `CompleteEvent` - * → Finished (free blocks)
- `AbortEvent` - * → Aborted (cleanup)

Invalid transitions return `Error::InvalidTransition`.

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

## Sequence Manager

Event-driven architecture with async event loop.

**Implementation:** `src/sequence_manager/fsm_manager.rs`

**Key features:**
- Receives events via async channel
- Maintains `HashMap<SequenceId, SequenceState>`
- Clones state before transitions (prevents loss on failure)
- Restores backup on error (no block leaks)
- Duplicate `seq_id` check prevents overwrites
- Duplicate `child_id` check prevents fork overwrites

**Tests:** Inline in `fsm_manager.rs` and `fsm_events.rs`

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

## Key Files

- `src/sequence_manager/states.rs` - State definitions
- `src/sequence_manager/fsm_events.rs` - Event implementations + tests
- `src/sequence_manager/fsm_manager.rs` - SequenceManager + tests
- `src/sequence_manager/events.rs` - Event channel types
- `src/sequence_manager/id_generator.rs` - Sequential ID generator

## Usage Example

See `src/sequence_manager/fsm_manager.rs` tests for complete examples.

Basic flow:
1. Create BlockManager and SequenceManager
2. Get event sender channel
3. Spawn event loop in background
4. Send `AddRequest` event
5. Send `AppendTokens` events as tokens generate
6. Sequence automatically transitions through states
7. Finished/Aborted when done
