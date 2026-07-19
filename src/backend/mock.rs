use futures::stream::{self, StreamExt};
use std::io;
use std::pin::Pin;
use tokio_stream::Stream;

use super::engine::Backend;
use crate::block_manager::types::TokenId;

/// Default vocab size the mock samples completion tokens from.
///
/// Small enough that ids land in the low end of any real tokenizer's vocab
/// (so they decode to real text), large enough for varied output.
const DEFAULT_VOCAB_SIZE: u32 = 1000;

/// Mock inference engine that behaves like a real autoregressive model.
///
/// Unlike a naive echo, this consumes the prompt as *context* and emits only
/// **new** completion tokens — never the prompt back. Tokens are produced by a
/// deterministic pseudo-random walk seeded from the context, so:
/// - the same prompt always yields the same completion (reproducible tests),
/// - different prompts yield different completions (input-dependent),
/// - ids stay within the vocab range, so they decode to real text.
///
/// Generation stops at `max_tokens` (the mock has no EOS concept yet).
#[derive(Clone)]
pub struct MockEngine {
    vocab_size: u32,
}

impl MockEngine {
    pub fn new() -> Self {
        Self {
            vocab_size: DEFAULT_VOCAB_SIZE,
        }
    }

    /// Construct with an explicit vocab size (useful for tests pinning output).
    pub fn with_vocab_size(vocab_size: u32) -> Self {
        Self {
            vocab_size: vocab_size.max(1),
        }
    }

    /// Deterministically generate `max_tokens` completion token ids from the
    /// prompt context, mimicking an autoregressive decode.
    ///
    /// Each step hashes the running context (prompt + tokens produced so far)
    /// into the vocab range and appends the result — exactly the shape of real
    /// inference (next token conditioned on all prior tokens), just with a hash
    /// standing in for a learned distribution.
    fn generate_tokens(&self, prompt: &[TokenId], max_tokens: usize) -> Vec<TokenId> {
        let mut context: Vec<TokenId> = prompt.to_vec();
        let mut completion = Vec::with_capacity(max_tokens);
        for _ in 0..max_tokens {
            let next = (hash_tokens(&context) % self.vocab_size as u64) as TokenId;
            completion.push(next);
            context.push(next);
        }
        completion
    }
}

/// FNV-1a hash over a token sequence — order-sensitive and fast, so each
/// distinct context maps to a distinct next token deterministically.
fn hash_tokens(tokens: &[TokenId]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &tok in tokens {
        for byte in tok.to_le_bytes() {
            hash ^= byte as u64;
            hash = hash.wrapping_mul(0x100000001b3);
        }
    }
    hash
}

impl Backend for MockEngine {
    async fn generate(
        &self,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Vec<TokenId>, io::Error> {
        // Return ONLY the new completion tokens, as a real model does.
        Ok(self.generate_tokens(&token_ids, max_tokens))
    }

    async fn generate_stream(
        &self,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = TokenId> + Send>>, io::Error> {
        // Same completion, but yielded one token at a time with a small delay to
        // mimic a real GPU's inter-token latency.
        let completion = self.generate_tokens(&token_ids, max_tokens);
        let stream = stream::iter(completion).then(|token_id| async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(20)).await;
            token_id
        });
        Ok(Box::pin(stream))
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn returns_only_completion_tokens() {
        let engine = MockEngine::new();
        let prompt = vec![5, 9, 2];
        let out = engine.generate(prompt.clone(), 4, 0.0).await.unwrap();
        // Completion-only: exactly max_tokens, and it does not start with the
        // prompt (a real model returns a continuation, not an echo).
        assert_eq!(out.len(), 4);
        assert_ne!(&out[..prompt.len().min(out.len())], &prompt[..]);
    }

    #[tokio::test]
    async fn is_deterministic() {
        let engine = MockEngine::new();
        let a = engine.generate(vec![1, 2, 3], 8, 0.0).await.unwrap();
        let b = engine.generate(vec![1, 2, 3], 8, 0.0).await.unwrap();
        assert_eq!(a, b, "same prompt must yield same completion");
    }

    #[tokio::test]
    async fn is_input_dependent() {
        let engine = MockEngine::new();
        let a = engine.generate(vec![1, 2, 3], 8, 0.0).await.unwrap();
        let b = engine.generate(vec![3, 2, 1], 8, 0.0).await.unwrap();
        assert_ne!(a, b, "different prompts should yield different completions");
    }

    #[tokio::test]
    async fn tokens_stay_within_vocab() {
        let engine = MockEngine::with_vocab_size(50);
        let out = engine.generate(vec![7, 7, 7], 32, 0.0).await.unwrap();
        assert!(
            out.iter().all(|&t| t < 50),
            "ids must be within vocab range"
        );
    }

    #[tokio::test]
    async fn stream_matches_generate() {
        let engine = MockEngine::new();
        let prompt = vec![10, 20, 30];
        let batched = engine.generate(prompt.clone(), 6, 0.0).await.unwrap();
        let mut streamed = Vec::new();
        let mut s = engine.generate_stream(prompt, 6, 0.0).await.unwrap();
        while let Some(tok) = s.next().await {
            streamed.push(tok);
        }
        assert_eq!(batched, streamed, "streaming and batched output must agree");
    }
}
