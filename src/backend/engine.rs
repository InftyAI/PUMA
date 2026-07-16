use std::io;
use std::pin::Pin;
use tokio_stream::Stream;
use crate::block_manager::types::TokenId;

/// Backend trait - low-level inference that works with token IDs
///
/// LLMEngine handles tokenization (text → tokens)
/// Backend handles inference (tokens → tokens)
pub trait Backend: Send + Sync {
    /// Generate tokens from input token_ids
    /// Returns generated token IDs
    fn generate(
        &self,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        temperature: f32,
    ) -> impl std::future::Future<Output = Result<Vec<TokenId>, io::Error>> + Send;

    /// Generate tokens with streaming
    /// Returns stream of token IDs as they're generated
    fn generate_stream(
        &self,
        token_ids: Vec<TokenId>,
        max_tokens: usize,
        temperature: f32,
    ) -> impl std::future::Future<
        Output = Result<Pin<Box<dyn Stream<Item = TokenId> + Send>>, io::Error>,
    > + Send;
}
