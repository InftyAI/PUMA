# FSM Architecture for PUMA

Finite State Machine pattern for sequence lifecycle management.

## Core Concepts

### State Machine per Sequence

Each sequence is an independent state machine:

```rust
pub enum SequenceState {
    Waiting(WaitingState),
    Scheduling(SchedulingState),
    Prefilling(PrefillingState),
    Decoding(DecodingState),
    Preempted(PreemptedState),
    Finished(FinishedState),
    Aborted(AbortedState),
}

// Multiple sequences, different states
sequences: HashMap<SequenceId, SequenceState>
```

### States Own Resources

```rust
pub struct WaitingState {
    seq_id: SequenceId,
    prompt_tokens: usize,
    max_tokens: usize,
    // No blocks - not allocated yet
}

pub struct DecodingState {
    seq_id: SequenceId,
    blocks: Vec<BlockId>,  // ← State owns blocks
    num_tokens: usize,
    max_tokens: usize,
}

pub struct PreemptedState {
    seq_id: SequenceId,
    num_tokens: usize,
    max_tokens: usize,
    // No blocks - they were freed
}
```

**Benefit:** Type system enforces "Waiting has no blocks, Decoding must have blocks"

### Events as State Transformations

```rust
pub struct ScheduleEvent<'a> {
    block_manager: &'a mut BlockManager,
    tokens_per_block: usize,
}

impl ScheduleEvent {
    pub fn apply(self, state: SequenceState) -> Result<SequenceState> {
        match state {
            SequenceState::Waiting(s) => {
                // Allocate blocks
                let blocks = allocate_blocks(s.prompt_tokens)?;

                // Transform to new state
                Ok(SequenceState::Prefilling(PrefillingState {
                    seq_id: s.seq_id,
                    blocks,  // Ownership transferred
                    tokens_filled: 0,
                    tokens_total: s.prompt_tokens,
                    max_tokens: s.max_tokens,
                }))
            }
            _ => Err(Error::invalid_transition("ScheduleEvent requires Waiting")),
        }
    }
}
```

**Pattern:** `Event + OldState → NewState`

## State Diagram

```
                    AddRequest
                        ↓
    ┌─────────────→ Waiting
    │                   ↓ ScheduleEvent
    │              Prefilling
    │                   ↓ AppendTokens
    │                   ↓ (tokens_filled >= tokens_total)
    │              Decoding
    │                   ├─→ AppendTokens (continue)
    │                   │   ↓
    │                   │   ├─ Decoding (more tokens)
    │                   │   └─ Finished (max_tokens reached)
    │                   │
    │                   ├─→ ForkEvent
    │                   │   ├─ Parent: Decoding
    │                   │   └─ Child: Decoding
    │                   │
    │                   ├─→ PreemptEvent (OOM)
    │                   │   ↓
    │                   │   Preempted
    │                   │   ↓ ResumeEvent
    │                   └───┘
    │
    └─ CompleteEvent → Finished
```

## State Definitions

```rust
/// Waiting in queue for scheduling
pub struct WaitingState {
    pub seq_id: SequenceId,
    pub prompt_tokens: usize,
    pub max_tokens: usize,
}

/// Running prefill phase
pub struct PrefillingState {
    pub seq_id: SequenceId,
    pub blocks: Vec<BlockId>,      // Owns blocks
    pub tokens_filled: usize,
    pub tokens_total: usize,
    pub max_tokens: usize,
}

/// Running decode phase
pub struct DecodingState {
    pub seq_id: SequenceId,
    pub blocks: Vec<BlockId>,      // Owns blocks
    pub num_tokens: usize,
    pub max_tokens: usize,
}

/// Preempted due to OOM
pub struct PreemptedState {
    pub seq_id: SequenceId,
    pub num_tokens: usize,
    pub max_tokens: usize,
    // Blocks freed
}

/// Finished generation
pub struct FinishedState {
    pub seq_id: SequenceId,
    pub finish_reason: FinishReason,
    // All resources freed
}
```

## Event Definitions

### ScheduleEvent: Waiting → Prefilling

```rust
let event = ScheduleEvent {
    block_manager: &mut block_manager,
    tokens_per_block: 16,
};

match event.apply(state) {
    Ok(new_state) => {
        // Waiting → Prefilling (with blocks allocated)
    }
    Err(Error::OutOfMemory) => {
        // Can't allocate, stay in queue
    }
}
```

### AppendTokensEvent: Prefilling → Decoding or Decoding → Decoding

```rust
let event = AppendTokensEvent {
    num_tokens: 100,
    block_manager: &mut block_manager,
    tokens_per_block: 16,
};

// From Prefilling
Prefilling(tokens_filled: 0) + AppendTokens(100)
  → Decoding(num_tokens: 100)  // Transition

// From Decoding
Decoding(num_tokens: 100) + AppendTokens(1)
  → Decoding(num_tokens: 101)  // Stay in state
```

### ForkEvent: Decoding → (Decoding, Decoding)

```rust
let event = ForkEvent {
    child_id: SequenceId(2),
    block_manager: &mut block_manager,
};

// Copy-on-write: both sequences share blocks
match event.apply(parent_state) {
    Ok((parent, child)) => {
        // Both in Decoding state
        // Blocks ref-counted (shared)
    }
}
```

### PreemptEvent: Decoding → Preempted

```rust
let event = PreemptEvent {
    block_manager: &mut block_manager,
};

// Frees blocks
Decoding(blocks: [1,2,3]) → Preempted(blocks: [])
```

### ResumeEvent: Preempted → Waiting

```rust
let event = ResumeEvent;

// Back to waiting for rescheduling
Preempted → Waiting → (Schedule) → Decoding
```

## Sequence Manager

```rust
pub struct SequenceManager {
    block_manager: BlockManager,
    sequences: HashMap<SequenceId, SequenceState>,
    tokens_per_block: usize,
    event_rx: SequenceEventReceiver,
}

impl SequenceManager {
    pub async fn run(mut self) {
        while let Some(event) = self.event_rx.recv().await {
            match event {
                SequenceEvent::AppendTokens { seq_id, num_tokens } => {
                    // Get current state
                    let state = self.sequences.remove(&seq_id).unwrap();

                    // Clone state before transition (prevents loss on failure)
                    let backup = state.clone();

                    // Apply event transformation
                    let event = AppendTokensEvent {
                        num_tokens,
                        block_manager: &mut self.block_manager,
                        tokens_per_block: self.tokens_per_block,
                    };

                    // Transform state
                    match event.apply(state) {
                        Ok(new_state) => {
                            self.sequences.insert(seq_id, new_state);
                        }
                        Err(e) => {
                            error!("Transition failed: {:?}", e);
                            // Restore original state to prevent sequence loss
                            self.sequences.insert(seq_id, backup);
                        }
                    }
                }
                // ... other events
            }
        }
    }
}
```

## Type Safety

**Compile-time checks:**
```rust
// ✅ Valid
let waiting = SequenceState::Waiting(...);
let event = ScheduleEvent { ... };
event.apply(waiting)?;  // OK

// ❌ Invalid (caught at runtime, enforced by match)
let decoding = SequenceState::Decoding(...);
let event = ScheduleEvent { ... };
event.apply(decoding)?;  // Error: invalid_transition
```

**Resource ownership:**
```rust
fn transition(state: DecodingState) -> FinishedState {
    // state.blocks moved/consumed
    for block in state.blocks {
        block_manager.free(block);
    }

    FinishedState {
        seq_id: state.seq_id,
        finish_reason: FinishReason::Stop,
    }
    // Old state dropped, can't access blocks anymore
}
```

## Benefits

1. **Type Safety**
   - States enforce resource invariants
   - Invalid transitions caught explicitly

2. **Clear Ownership**
   - Resources belong to states
   - Automatic cleanup on state drop

3. **Testability**
   - Events are pure functions
   - Test state transitions in isolation

4. **Explicitness**
   - Every transition is a function call
   - State diagram maps directly to code

## Example Flow

```rust
// 1. Add request → Waiting
let state = SequenceState::Waiting(WaitingState {
    seq_id: SequenceId(1),
    prompt_tokens: 100,
    max_tokens: 150,
});

// 2. Schedule → Prefilling (allocate 7 blocks)
let event = ScheduleEvent { ... };
let state = event.apply(state)?;
// state = Prefilling(blocks: [1,2,3,4,5,6,7], tokens_filled: 0)

// 3. Append tokens → Decoding (transition)
let event = AppendTokensEvent { num_tokens: 100 };
let state = event.apply(state)?;
// state = Decoding(blocks: [1,2,3,4,5,6,7], num_tokens: 100)

// 4. Fork → 2 sequences
let event = ForkEvent { child_id: SequenceId(2) };
let (parent, child) = event.apply(state)?;
// parent = Decoding(blocks: [1,2,3,4,5,6,7], ref_count: 2)
// child = Decoding(blocks: [1,2,3,4,5,6,7], ref_count: 2)

// 5. Continue generating
let event = AppendTokensEvent { num_tokens: 10 };
let parent = event.apply(parent)?;
// parent = Decoding(blocks: [1,2,3,4,5,6,7,8], num_tokens: 110)

// 6. Complete → Finished (free blocks)
let event = CompleteEvent { ... };
let state = event.apply(parent)?;
// state = Finished, blocks freed
```

## Comparison with Simple Design

| Simple | FSM |
|--------|-----|
| `seq.state = Running` | `state = ScheduleEvent.apply(state)` |
| Resources in struct | Resources in state variant |
| Manual cleanup | Automatic cleanup |
| Runtime checks | Type + runtime checks |

## Error Handling and State Recovery

On transition failure, the state is cloned before applying the event:

```rust
fn handle_append_tokens(&mut self, seq_id: SequenceId, num_tokens: usize) {
    let state = self.sequences.remove(&seq_id).unwrap();
    let backup = state.clone();  // Clone before transition

    let event = AppendTokensEvent { ... };

    match event.apply(state) {
        Ok(new_state) => {
            self.sequences.insert(seq_id, new_state);
        }
        Err(Error::OutOfMemory) => {
            // Restore state and trigger preemption
            self.sequences.insert(seq_id, backup);
            self.handle_oom();
        }
        Err(e) => {
            // Restore state on any error (prevents block leaks)
            self.sequences.insert(seq_id, backup);
            warn!("Transition failed: {:?}", e);
        }
    }
}
```

**Benefits:**
- No sequence loss on transition failure
- No block leaks (blocks remain owned by restored state)
- Can retry or handle errors without losing context
- Clone cost is negligible (~50-100 bytes per state)

## Usage

```rust
use puma::block_manager::{BlockManager, CpuAllocator};
use puma::sequence_manager::{SequenceManager, SequenceEvent, SequenceIdGenerator};

// Setup
let allocator = Box::new(CpuAllocator::new(100_000_000));
let block_manager = BlockManager::new(allocator, 4096);
let (seq_manager, event_tx) = SequenceManager::new(block_manager, 16);

// ID generator for sequential IDs starting from 0
let id_gen = SequenceIdGenerator::new();
let seq_id = id_gen.next();  // SequenceId(0)

// Run event loop
tokio::spawn(async move { seq_manager.run().await });

// Send events
event_tx.send(SequenceEvent::AddRequest {
    seq_id,
    prompt_tokens: 100,
    max_tokens: 150,
}).unwrap();

event_tx.send(SequenceEvent::AppendTokens {
    seq_id,
    num_tokens: 100,
}).unwrap();
```
