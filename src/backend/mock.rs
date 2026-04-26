use futures::stream::{self, StreamExt};
use std::io;
use std::pin::Pin;
use tokio_stream::Stream;

use super::engine::{GenerateResponse, InferenceEngine};

/// Mock engine for testing (replace with MLX later)
#[derive(Clone)]
pub struct MockEngine;

impl MockEngine {
    pub fn new() -> Self {
        Self
    }
}

impl InferenceEngine for MockEngine {
    async fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<GenerateResponse, io::Error> {
        // Mock response for testing
        let response_text = format!(
            "This is a mock response from model '{}' for prompt: '{}' (max_tokens: {})",
            model,
            prompt.chars().take(50).collect::<String>(),
            max_tokens
        );

        Ok(GenerateResponse {
            text: response_text,
            prompt_tokens: prompt.split_whitespace().count(),
            completion_tokens: 20,
        })
    }

    async fn generate_stream(
        &self,
        model: &str,
        _prompt: &str,
        max_tokens: usize,
        _temperature: f32,
    ) -> Result<Pin<Box<dyn Stream<Item = String> + Send>>, io::Error> {
        // Mock streaming response
        let tokens = vec![
            "This ".to_string(),
            "is ".to_string(),
            "a ".to_string(),
            "mock ".to_string(),
            "streaming ".to_string(),
            "response ".to_string(),
            format!("from model '{}' ", model),
            format!("(max_tokens: {}).", max_tokens),
        ];

        // Simulate delay between tokens
        let stream = stream::iter(tokens).then(|token| async move {
            tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            token
        });

        Ok(Box::pin(stream))
    }
}

impl Default for MockEngine {
    fn default() -> Self {
        Self::new()
    }
}
