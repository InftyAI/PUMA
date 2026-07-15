use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use super::engine::{GenerateResponse, InferenceEngine};
use crate::block_manager::allocator::CpuAllocator;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::SequenceId;
use crate::scheduler::core::Scheduler;
use crate::scheduler::events::SchedulerEvent;

/// LLM Engine - integrates scheduler with inference backend
///
/// Architecture
/// - LLMEngine drives the main loop
/// - Owns tokenizer (model-specific)
/// - Scheduler is synchronous (just methods, no async)
/// - Direct ownership (no Arc/Mutex overhead)
/// - Loop: handle events → schedule() → forward() → process_outputs()
pub struct LLMEngine<B: InferenceEngine> {
    backend: B,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    event_rx: mpsc::UnboundedReceiver<SchedulerEvent>,
    event_tx: mpsc::UnboundedSender<SchedulerEvent>,
    seq_id_counter: AtomicU64,
    model: String,
}

impl<B: InferenceEngine + Clone + 'static> LLMEngine<B> {
    pub fn new(backend: B, tokenizer: Tokenizer, model: String) -> Self {
        // Create block manager (100MB memory pool, 512 bytes per block)
        let allocator = Box::new(CpuAllocator::new(1024 * 1024 * 100));
        let block_manager = BlockManager::new(allocator, 512);

        // Create scheduler (max 32 batch size, 16 tokens per block)
        let scheduler = Scheduler::new(block_manager, 32, 16);

        // Event channel
        let (event_tx, event_rx) = mpsc::unbounded_channel();

        Self {
            backend,
            scheduler,
            tokenizer,
            event_rx,
            event_tx,
            seq_id_counter: AtomicU64::new(0),
            model,
        }
    }

    fn next_seq_id(&self) -> SequenceId {
        SequenceId(self.seq_id_counter.fetch_add(1, Ordering::Relaxed))
    }

    /// Main loop - drives scheduling and inference
    ///
    /// Pattern
    /// 1. Handle events (add/cancel requests)
    /// 2. Schedule (decide what to run)
    /// 3. Forward (run inference if work exists)
    /// 4. Process outputs (update scheduler state)
    pub async fn run(mut self) {
        tracing::info!("LLMEngine started");

        loop {
            // 1. Handle events (non-blocking, drain all)
            while let Ok(event) = self.event_rx.try_recv() {
                match event {
                    SchedulerEvent::AddRequest {
                        seq_id,
                        token_ids,
                        max_tokens,
                    } => {
                        self.scheduler.add_request(seq_id, token_ids, max_tokens);
                    }

                    SchedulerEvent::CancelRequest { seq_id } => {
                        self.scheduler.cancel_request(seq_id);
                    }

                    SchedulerEvent::GetBlocks { seq_id, response } => {
                        let result = self.scheduler.get_blocks(seq_id);
                        let _ = response.send(result);
                    }

                    SchedulerEvent::GetStats { response } => {
                        let stats = self.scheduler.get_stats();
                        let _ = response.send(stats);
                    }
                }
            }

            // 2. Schedule (always, not event-driven)
            let has_work = self.scheduler.schedule();

            // 3. Forward (if work exists)
            if has_work {
                // TODO: Build batch and call backend.forward()
                // For now, just yield
            }

            // 4. Small yield to prevent busy loop
            tokio::task::yield_now().await;
        }
    }
}

// Implement InferenceEngine directly for LLMEngine
impl<B: InferenceEngine + Clone + 'static> InferenceEngine for LLMEngine<B> {
    async fn generate(
        &self,
        _model: &str,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<GenerateResponse, io::Error> {
        let seq_id = self.next_seq_id();

        // Tokenize prompt using HuggingFace tokenizer
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| io::Error::other(format!("Tokenization failed: {}", e)))?;
        let token_ids = encoding.get_ids().to_vec();
        let num_tokens = token_ids.len();

        // Send request to scheduler
        self.event_tx
            .send(SchedulerEvent::AddRequest {
                seq_id,
                token_ids,
                max_tokens,
            })
            .map_err(|e| io::Error::other(format!("Engine send failed: {}", e)))?;

        // TODO: Wait for actual result from scheduler/GPU
        // For now, return dummy response
        Ok(GenerateResponse {
            text: format!("Generated response for: {}", prompt),
            prompt_tokens: num_tokens,
            completion_tokens: 0,
        })
    }

    async fn generate_stream(
        &self,
        _model: &str,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<std::pin::Pin<Box<dyn tokio_stream::Stream<Item = String> + Send>>, io::Error> {
        let seq_id = self.next_seq_id();

        // Tokenize prompt using HuggingFace tokenizer
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| io::Error::other(format!("Tokenization failed: {}", e)))?;
        let token_ids = encoding.get_ids().to_vec();

        // Send request to scheduler
        self.event_tx
            .send(SchedulerEvent::AddRequest {
                seq_id,
                token_ids,
                max_tokens,
            })
            .map_err(|e| io::Error::other(format!("Engine send failed: {}", e)))?;

        // TODO: Stream results from scheduler/GPU
        // For now, return dummy stream
        let stream = tokio_stream::iter(vec![
            format!("Token 1 for: {}\n", prompt),
            "Token 2\n".to_string(),
            "Token 3\n".to_string(),
        ]);
        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockEngine;

    fn create_test_tokenizer() -> Tokenizer {
        // Create a simple BPE tokenizer for testing
        use tokenizers::models::bpe::BPE;
        use tokenizers::Tokenizer as TokenizerBuilder;

        let bpe = BPE::default();
        TokenizerBuilder::new(bpe)
    }

    #[tokio::test]
    async fn test_llm_engine() {
        let backend = MockEngine::new();
        let tokenizer = create_test_tokenizer();
        let engine = LLMEngine::new(backend, tokenizer, "test-model".to_string());

        // TODO: Spawn engine loop when real backend integration is done
        // tokio::spawn(async move {
        //     engine_copy.run().await;
        // });

        let result = engine.generate("test-model", "Hello world", 100, 0.7).await;
        assert!(result.is_ok());
    }
}
