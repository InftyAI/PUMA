use serde::{Deserialize, Serialize};

/// Chat completion request (OpenAI compatible)
#[derive(Debug, Clone, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default = "default_top_p")]
    pub top_p: Option<f32>,
    #[serde(default = "default_n")]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

/// Chat message
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String, // "system", "user", "assistant"
    pub content: String,
}

/// Legacy text completion request
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(alias = "prompt")]
    pub prompt: StringOrArray,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default = "default_top_p")]
    pub top_p: Option<f32>,
    #[serde(default = "default_n")]
    pub n: Option<usize>,
    #[serde(default)]
    pub stream: bool,
    pub stop: Option<Vec<String>>,
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
}

/// Prompt can be string or array of strings
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum StringOrArray {
    String(String),
    Array(Vec<String>),
}

impl StringOrArray {
    pub fn to_string(&self) -> String {
        match self {
            StringOrArray::String(s) => s.clone(),
            StringOrArray::Array(arr) => arr.join("\n"),
        }
    }
}

// Default values
fn default_max_tokens() -> Option<usize> {
    Some(100)
}

fn default_temperature() -> Option<f32> {
    Some(0.7)
}

fn default_top_p() -> Option<f32> {
    Some(1.0)
}

fn default_n() -> Option<usize> {
    Some(1)
}
