use futures::stream::{self, StreamExt};
use std::io;
use std::pin::Pin;
use tokio_stream::Stream;

use super::engine::Backend;
use crate::block_manager::types::TokenId;

/// Mock engine for testing (replace with MLX later)
#[derive(Clone)]
pub struct MockEngine;

impl MockEngine {
    pub fn new() -> Self {
        Self
    }
}

impl Backend for MockEngine {
    async fn generate(
        &self,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Vec<TokenId>, io::Error> {
        // Mock: echo input + add generated tokens
        // This makes different inputs produce different outputs
        let mut result = token_ids;
        for i in 0..max_tokens.min(10) {
            result.push(1000 + i as u32); // Append tokens 1000, 1001, 1002...
        }
        Ok(result)
    }

    async fn generate_stream(
        &self,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = TokenId> + Send>>, io::Error> {
        // Mock: echo input tokens first, then generate new ones
        // This makes different inputs produce different outputs

        // First yield all input tokens (the prompt)
        let mut all_tokens = token_ids;

        // Then add generated tokens
        for i in 0..max_tokens.min(10) {
            all_tokens.push(1000 + i as u32); // Generate tokens 1000, 1001, 1002...
        }

        // Simulate delay between tokens (like real GPU)
        let stream = stream::iter(all_tokens).then(|token_id| async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
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
