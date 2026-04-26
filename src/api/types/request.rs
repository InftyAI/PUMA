use serde::{Deserialize, Serialize};

/// Chat completion request (OpenAI compatible)
#[derive(Debug, Clone, Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
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
#[allow(dead_code)]
pub struct CompletionRequest {
    pub model: String,
    #[serde(alias = "prompt")]
    pub prompt: StringOrArray,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: Option<usize>,
    #[serde(default = "default_temperature")]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
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

impl std::fmt::Display for StringOrArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StringOrArray::String(s) => write!(f, "{}", s),
            StringOrArray::Array(arr) => write!(f, "{}", arr.join("\n")),
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
