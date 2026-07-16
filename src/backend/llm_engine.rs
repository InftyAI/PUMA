use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use tokenizers::Tokenizer;
use tokio::sync::mpsc;

use super::engine::Backend;

/// User-facing response for generate()
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}
use crate::block_manager::allocator::CpuAllocator;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::SequenceId;
use crate::scheduler::core::Scheduler;
use crate::scheduler::events::{ResponseSender, SchedulerEvent};

/// LLM Engine - integrates scheduler with inference backend
///
/// Architecture
/// - LLMEngine drives the main loop
/// - Owns tokenizer (model-specific)
/// - Scheduler is synchronous (just methods, no async)
/// - Direct ownership (no Arc/Mutex overhead)
/// - Loop: handle events → schedule() → forward() → process_outputs()
pub struct LLMEngine<B: Backend> {
    backend: B,
    scheduler: Scheduler,
    tokenizer: Tokenizer,
    event_rx: mpsc::UnboundedReceiver<SchedulerEvent>,
    event_tx: mpsc::UnboundedSender<SchedulerEvent>,
    seq_id_counter: AtomicU64,
    model: String,
}

impl<B: Backend + Clone + 'static> LLMEngine<B> {
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

    /// Tokenize prompt
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, io::Error> {
        self.tokenizer
            .encode(prompt, false)
            .map(|encoding| encoding.get_ids().to_vec())
            .map_err(|e| io::Error::other(format!("Tokenization failed: {}", e)))
    }

    /// Decode token IDs to text
    fn decode_tokens(&self, token_ids: &[u32]) -> Result<String, io::Error> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|e| io::Error::other(format!("Detokenization failed: {}", e)))
    }

    /// Send request helper
    fn send_request(
        &self,
        token_ids: Vec<u32>,
        max_tokens: usize,
        response_tx: ResponseSender,
    ) -> Result<(), io::Error> {
        let seq_id = self.next_seq_id();
        self.event_tx
            .send(SchedulerEvent::AddRequest {
                seq_id,
                token_ids,
                max_tokens,
                response_tx,
            })
            .map_err(|e| io::Error::other(format!("Engine send failed: {}", e)))
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
                        response_tx,
                    } => {
                        self.scheduler
                            .add_request(seq_id, token_ids, max_tokens, response_tx);
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

// User-facing API methods
impl<B: Backend + Clone + 'static> LLMEngine<B> {
    /// Generate text completion (single response)
    pub async fn generate(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<GenerateResponse, io::Error> {
        // Tokenize
        let token_ids = self.tokenize(prompt)?;
        let prompt_tokens = token_ids.len();

        // Create response channel
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        // Send request
        self.send_request(token_ids, max_tokens, ResponseSender::Single(response_tx))?;

        // Wait for result from scheduler
        let text = response_rx
            .await
            .map_err(|e| io::Error::other(format!("Response channel closed: {}", e)))?
            .map_err(|e| io::Error::other(format!("Scheduler error: {:?}", e)))?;

        Ok(GenerateResponse {
            text,
            prompt_tokens,
            completion_tokens: 0,
        })
    }

    /// Generate with streaming
    pub async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: usize,
    ) -> Result<std::pin::Pin<Box<dyn tokio_stream::Stream<Item = String> + Send>>, io::Error> {
        // Tokenize
        let token_ids = self.tokenize(prompt)?;

        // Create channel
        let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();

        // Send request
        self.send_request(token_ids, max_tokens, ResponseSender::Stream(response_tx))?;

        // Create stream that receives tokens from scheduler
        let stream = async_stream::stream! {
            while let Some(result) = response_rx.recv().await {
                match result {
                    Ok(token) => yield token,
                    Err(e) => {
                        tracing::error!("Stream error: {:?}", e);
                        break;
                    }
                }
            }
        };

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
