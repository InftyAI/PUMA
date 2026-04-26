use std::io;
use std::pin::Pin;
use tokio_stream::Stream;

/// Inference engine trait
pub trait InferenceEngine: Send + Sync {
    /// Generate text completion
    fn generate(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> impl std::future::Future<Output = Result<GenerateResponse, io::Error>> + Send;

    /// Generate text with streaming
    fn generate_stream(
        &self,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
    ) -> impl std::future::Future<
        Output = Result<Pin<Box<dyn Stream<Item = String> + Send>>, io::Error>,
    > + Send;
}

/// Generation response
#[derive(Debug, Clone)]
pub struct GenerateResponse {
    pub text: String,
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
}
