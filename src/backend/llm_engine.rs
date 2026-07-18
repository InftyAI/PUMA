use std::io;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;

use super::engine::Backend;
use crate::block_manager::allocator::CpuAllocator;
use crate::block_manager::manager::BlockManager;
use crate::block_manager::types::{SequenceId, TokenId};
use crate::fsm::FinishReason;
use crate::scheduler::core::Scheduler;
use crate::scheduler::events::{ResponseSender, SchedulerEvent};

/// User-facing response for generate()
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}

/// Send-side handle to the engine (cheap to clone).
///
/// Holds only what's needed to translate a user request into a
/// `SchedulerEvent` and push it onto the event channel:
/// - tokenizer (text → tokens)
/// - event sender (fire the event)
/// - sequence id counter (assign ids)
///
/// The actual work happens in [`EngineRunner::serve`], which owns the scheduler
/// and backend and consumes these events. This handle is `Clone` so the API
/// layer and CLI can share it freely across tasks.
#[derive(Clone)]
pub struct EngineHandle {
    tokenizer: Arc<Tokenizer>,
    event_tx: mpsc::UnboundedSender<SchedulerEvent>,
    seq_id_counter: Arc<AtomicU64>,
    model: String,
}

impl EngineHandle {
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

    /// Send an AddRequest event to the engine loop
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

    /// Generate text completion (single response)
    pub async fn generate(
        &self,
        _model: &str,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<GenerateResponse, io::Error> {
        let token_ids = self.tokenize(prompt)?;
        let prompt_tokens = token_ids.len();

        let (response_tx, response_rx) = tokio::sync::oneshot::channel();
        self.send_request(token_ids, max_tokens, ResponseSender::Single(response_tx))?;

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
        _model: &str,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<std::pin::Pin<Box<dyn tokio_stream::Stream<Item = String> + Send>>, io::Error> {
        let token_ids = self.tokenize(prompt)?;

        let (response_tx, mut response_rx) = tokio::sync::mpsc::unbounded_channel();
        self.send_request(token_ids, max_tokens, ResponseSender::Stream(response_tx))?;

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

    /// Model name this handle serves
    pub fn model(&self) -> &str {
        &self.model
    }
}

/// The engine itself: owns the scheduler and backend, drives the event loop.
///
/// Drains the event channel and runs inference. Spawn its
/// [`serve`](Self::serve) on a task; hold the paired [`EngineHandle`] everywhere
/// else to submit work.
pub struct EngineRunner<B: Backend> {
    backend: B,
    scheduler: Scheduler,
    tokenizer: Arc<Tokenizer>,
}

impl<B: Backend + Clone + 'static> EngineRunner<B> {
    /// Decode token IDs to text
    fn decode_tokens(&self, token_ids: &[u32]) -> Result<String, io::Error> {
        self.tokenizer
            .decode(token_ids, false)
            .map_err(|e| io::Error::other(format!("Detokenization failed: {}", e)))
    }

    /// Run one-shot inference for a non-streaming request.
    ///
    /// Calls [`Backend::generate`], decodes the completion, advances the FSM,
    /// and delivers the full text to the client's single-response channel.
    async fn forward_single(
        &mut self,
        seq_id: SequenceId,
        token_ids: Arc<Vec<TokenId>>,
        max_tokens: usize,
    ) {
        let prompt_len = token_ids.len();
        let completion_tokens = match self
            .backend
            .generate((*token_ids).clone(), max_tokens, 0.0)
            .await
        {
            Ok(tokens) => tokens,
            Err(e) => {
                tracing::error!("Backend inference failed for {:?}: {}", seq_id, e);
                self.scheduler
                    .abort_request(seq_id, format!("Backend inference failed: {}", e));
                return;
            }
        };

        // The backend returns only the newly generated completion tokens.
        let num_completion = completion_tokens.len();
        let text = self.decode_tokens(&completion_tokens).unwrap_or_default();

        // Advance FSM state (prefill → decode → finished) so blocks are freed,
        // then deliver the result to the client channel.
        self.scheduler.append_tokens(seq_id, prompt_len);
        self.scheduler
            .complete_sequence(seq_id, FinishReason::Stop, text);
        tracing::debug!(
            "Completed {:?}: {} completion tokens",
            seq_id,
            num_completion
        );
    }

    /// Run streaming inference for a streaming request.
    ///
    /// Calls [`Backend::generate_stream`] and forwards each decoded text chunk to
    /// the client via [`send_chunk`](Scheduler::send_chunk) as it arrives, then
    /// advances the FSM and closes the stream once generation finishes.
    async fn forward_stream(
        &mut self,
        seq_id: SequenceId,
        token_ids: Arc<Vec<TokenId>>,
        max_tokens: usize,
    ) {
        let prompt_len = token_ids.len();
        let mut stream = match self
            .backend
            .generate_stream((*token_ids).clone(), max_tokens, 0.0)
            .await
        {
            Ok(stream) => stream,
            Err(e) => {
                tracing::error!("Backend stream failed for {:?}: {}", seq_id, e);
                self.scheduler
                    .abort_request(seq_id, format!("Backend stream failed: {}", e));
                return;
            }
        };

        // The backend streams only the newly generated completion tokens.
        //
        // Decode incrementally rather than one token at a time: subword/byte-BPE
        // tokenizers split multi-byte characters (emoji, CJK) across several
        // tokens, so a lone token often decodes to `�` or "". We keep the running
        // list of completion tokens and decode the whole prefix each step, which
        // is always well-formed. When the decode ends in the Unicode replacement
        // character the trailing multi-byte char is still incomplete, so we hold
        // the output back until the next token completes it; otherwise we emit the
        // newly-decoded suffix. This is the streaming detokenization scheme vLLM
        // and TGI use.
        let mut completion_ids: Vec<TokenId> = Vec::new();
        let mut sent_len = 0usize; // bytes of the decoded completion already sent
        while let Some(token_id) = stream.next().await {
            completion_ids.push(token_id);

            let decoded = self.decode_tokens(&completion_ids).unwrap_or_default();
            // A trailing replacement char means an incomplete multi-byte
            // character; wait for the next token before emitting more.
            if decoded.ends_with('\u{FFFD}') {
                continue;
            }
            if decoded.len() > sent_len {
                let delta = decoded[sent_len..].to_string();
                sent_len = decoded.len();
                self.scheduler.send_chunk(seq_id, delta);
            }
        }
        let num_completion = completion_ids.len();

        // Advance FSM state (prefill → decode → finished) so blocks are freed,
        // then close the stream channel.
        self.scheduler.append_tokens(seq_id, prompt_len);
        self.scheduler
            .complete_sequence(seq_id, FinishReason::Stop, String::new());
        tracing::debug!(
            "Streamed {:?}: {} completion tokens",
            seq_id,
            num_completion
        );
    }

    /// Main loop - drives scheduling and inference
    ///
    /// Pattern
    /// 1. Handle events (add/cancel requests)
    /// 2. Schedule (decide what to run)
    /// 3. Forward (run inference if work exists)
    /// 4. Process outputs (update scheduler state)
    pub async fn serve(mut self) {
        tracing::info!("EngineRunner started");

        loop {
            // 1. Handle events (non-blocking, drain all). The scheduler owns the
            //    event channel; the loop drains it and dispatches each event.
            while let Ok(event) = self.scheduler.event_rx.try_recv() {
                self.scheduler.handle_event(event);
            }

            // 2. Schedule (always, not event-driven)
            let has_work = self.scheduler.schedule();

            // 3. Forward: run inference on the scheduled batch and report
            //    results back to the scheduler, which fulfills client channels.
            //    Streaming requests use the backend's streaming path (token by
            //    token); single requests use the one-shot path.
            //
            // TODO: this processes one sequence at a time, awaiting each backend
            // call in turn. For real batch inference, partition the batch by mode
            // (see the `PrefillBatch` design) and hand the non-streaming bucket to
            // a single `Backend::generate_batch` call instead of looping.
            if has_work {
                for work in self.scheduler.take_prefill_batch() {
                    let (seq_id, token_ids, max_tokens, streaming) = work;
                    if streaming {
                        self.forward_stream(seq_id, token_ids, max_tokens).await;
                    } else {
                        self.forward_single(seq_id, token_ids, max_tokens).await;
                    }
                }
            }

            // 4. Small yield to prevent busy loop
            tokio::task::yield_now().await;
        }
    }
}

/// Construct a paired [`EngineHandle`] and [`EngineRunner`].
///
/// Spawn `runner.serve()` on a task and share the returned handle with the API /
/// CLI. All request submission goes through events, so the handle never
/// touches the scheduler directly.
pub fn engine<B: Backend + Clone + 'static>(
    backend: B,
    tokenizer: Tokenizer,
    model: String,
) -> (EngineHandle, EngineRunner<B>) {
    // Create block manager (100MB memory pool, 512 bytes per block)
    let allocator = Box::new(CpuAllocator::new(1024 * 1024 * 100));
    let block_manager = BlockManager::new(allocator, 512);

    // Event channel: the handle produces events, the scheduler consumes them.
    let (event_tx, event_rx) = mpsc::unbounded_channel();

    // Create scheduler (max 32 batch size, 16 tokens per block); it owns the
    // event receiver.
    let scheduler = Scheduler::new(block_manager, event_rx, 32, 16);

    let tokenizer = Arc::new(tokenizer);

    let handle = EngineHandle {
        tokenizer: tokenizer.clone(),
        event_tx,
        seq_id_counter: Arc::new(AtomicU64::new(1)),
        model,
    };

    let runner = EngineRunner {
        backend,
        scheduler,
        tokenizer,
    };

    (handle, runner)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockEngine;

    fn create_test_tokenizer() -> Tokenizer {
        use tokenizers::models::bpe::BPE;
        use tokenizers::Tokenizer as TokenizerBuilder;

        let bpe = BPE::default();
        TokenizerBuilder::new(bpe)
    }

    #[tokio::test]
    async fn test_llm_engine() {
        let backend = MockEngine::new();
        let tokenizer = create_test_tokenizer();
        let (handle, runner) = engine(backend, tokenizer, "test-model".to_string());

        tokio::spawn(runner.serve());

        let result = handle.generate("test-model", "Hello world", 100, 0.7).await;
        assert!(result.is_ok());
    }
}
