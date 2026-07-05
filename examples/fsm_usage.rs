use puma::block_manager::{BlockManager, CpuAllocator};
use puma::sequence_manager::{SequenceEvent, SequenceIdGenerator, SequenceManager};

#[tokio::main]
async fn main() {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("info,puma=debug")
        .init();

    println!("=== FSM-Based Sequence Manager Demo ===\n");

    // Setup block manager
    let allocator = Box::new(CpuAllocator::new(100_000_000)); // 100MB
    let block_size = 4096; // 4KB per block
    let block_manager = BlockManager::new(allocator, block_size);

    // Setup sequence manager
    let tokens_per_block = 16;
    let (seq_manager, event_tx) = SequenceManager::new(block_manager, tokens_per_block);

    // Spawn event loop
    let manager_handle = tokio::spawn(async move {
        seq_manager.run().await;
    });

    // ID generator (starts from 0)
    let id_gen = SequenceIdGenerator::new();

    println!("1. Adding sequence (Waiting state)");
    let seq_id = id_gen.next();
    event_tx
        .send(SequenceEvent::AddRequest {
            seq_id,
            prompt_tokens: 100,
            max_tokens: 150,
        })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    println!("\n2. Appending tokens during prefill (Prefilling → Decoding)");
    event_tx
        .send(SequenceEvent::AppendTokens {
            seq_id,
            num_tokens: 100, // Complete prefill
        })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    println!("\n3. Generating tokens (Decoding state)");
    for i in 0..10 {
        event_tx
            .send(SequenceEvent::AppendTokens {
                seq_id,
                num_tokens: 1,
            })
            .unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;
    }

    println!("\n4. Forking sequence for beam search");
    let child_id = id_gen.next();
    event_tx
        .send(SequenceEvent::ForkSequence {
            parent_id: seq_id,
            child_id,
        })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    println!("\n5. Continuing both sequences");
    event_tx
        .send(SequenceEvent::AppendTokens {
            seq_id,
            num_tokens: 5,
        })
        .unwrap();
    event_tx
        .send(SequenceEvent::AppendTokens {
            seq_id: child_id,
            num_tokens: 5,
        })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Query stats
    let (tx, rx) = tokio::sync::oneshot::channel();
    event_tx
        .send(SequenceEvent::GetStats { response: tx })
        .unwrap();

    let stats = rx.await.unwrap();
    println!("\n=== Stats ===");
    println!("Total sequences: {}", stats.num_sequences);
    println!("Running: {}", stats.num_running);
    println!("Waiting: {}", stats.num_waiting);
    println!("Preempted: {}", stats.num_preempted);

    for (block_type, block_stats) in &stats.block_stats {
        println!("\nBlock type: {}", block_type);
        println!("  Total blocks: {}", block_stats.total_blocks);
        println!("  Allocated: {}", block_stats.allocated_blocks);
        println!("  Free: {}", block_stats.free_blocks);
        println!("  Total memory: {} bytes", block_stats.total_memory);
    }

    println!("\n6. Completing sequences (Finished state)");
    event_tx
        .send(SequenceEvent::CompleteSequence { seq_id })
        .unwrap();
    event_tx
        .send(SequenceEvent::CompleteSequence { seq_id: child_id })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Final stats
    let (tx, rx) = tokio::sync::oneshot::channel();
    event_tx
        .send(SequenceEvent::GetStats { response: tx })
        .unwrap();

    let stats = rx.await.unwrap();
    println!("\n=== Final Stats ===");
    println!("Total sequences: {}", stats.num_sequences);
    println!("Running: {}", stats.num_running);

    println!("\n7. Testing preemption");
    let seq3 = id_gen.next();
    event_tx
        .send(SequenceEvent::AddRequest {
            seq_id: seq3,
            prompt_tokens: 50,
            max_tokens: 100,
        })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    event_tx
        .send(SequenceEvent::AppendTokens {
            seq_id: seq3,
            num_tokens: 50,
        })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    println!("\n   Preempting sequence (Decoding → Preempted)");
    event_tx
        .send(SequenceEvent::PreemptSequence { seq_id: seq3 })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    println!("   Resuming sequence (Preempted → Waiting → Decoding)");
    event_tx
        .send(SequenceEvent::ResumeSequence { seq_id: seq3 })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    event_tx
        .send(SequenceEvent::CompleteSequence { seq_id: seq3 })
        .unwrap();

    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    println!("\n=== Sequence Manager Demo Complete! ===");
    println!("\nState transitions demonstrated:");
    println!("  Waiting → Prefilling → Decoding → Finished");
    println!("  Decoding → Fork → Two Decoding sequences");
    println!("  Decoding → Preempted → Waiting → Decoding");

    // Cleanup
    drop(event_tx);
    manager_handle.await.unwrap();
}
