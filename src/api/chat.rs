use axum::{
    extract::State,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use futures::stream::StreamExt;
use std::sync::Arc;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::api::routes::AppState;
use crate::api::types::{
    ChatChoice, ChatChoiceDelta, ChatCompletionChunk, ChatCompletionRequest,
    ChatCompletionResponse, ChatMessage, ChatMessageDelta, ErrorResponse, Usage,
};
use crate::backend::InferenceEngine;

/// Main handler for chat completions
pub async fn chat_completions<E: InferenceEngine + 'static>(
    State(state): State<AppState<E>>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let engine = state.engine;
    let registry = state.registry;

    // Validate request
    if req.messages.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "messages cannot be empty".to_string(),
                "invalid_request_error".to_string(),
            )),
        )
            .into_response();
    }

    // Validate model exists
    match registry.get_model(&req.model) {
        Ok(None) => {
            return (
                axum::http::StatusCode::NOT_FOUND,
                Json(ErrorResponse::new(
                    format!("Model '{}' not found", req.model),
                    "model_not_found".to_string(),
                )),
            )
                .into_response();
        }
        Err(e) => {
            return (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    format!("Failed to check model: {}", e),
                    "internal_error".to_string(),
                )),
            )
                .into_response();
        }
        Ok(Some(_)) => {
            // Model exists, continue
        }
    }

    if req.stream {
        chat_completions_stream(engine, req).await.into_response()
    } else {
        match chat_completions_non_stream(engine, req).await {
            Ok(response) => Json(response).into_response(),
            Err(err) => (
                axum::http::StatusCode::INTERNAL_SERVER_ERROR,
                Json(ErrorResponse::new(
                    err.to_string(),
                    "internal_error".to_string(),
                )),
            )
                .into_response(),
        }
    }
}

/// Non-streaming chat completion
async fn chat_completions_non_stream<E: InferenceEngine>(
    engine: Arc<E>,
    req: ChatCompletionRequest,
) -> Result<ChatCompletionResponse, Box<dyn std::error::Error>> {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    // Convert messages to prompt
    let prompt = format_chat_messages(&req.messages);

    // Generate
    let response = engine
        .generate(
            &req.model,
            &prompt,
            req.max_tokens.unwrap_or(100),
            req.temperature.unwrap_or(0.7),
        )
        .await?;

    Ok(ChatCompletionResponse {
        id,
        object: "chat.completion".to_string(),
        created,
        model: req.model,
        choices: vec![ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: response.text,
            },
            finish_reason: "stop".to_string(),
        }],
        usage: Usage {
            prompt_tokens: response.prompt_tokens,
            completion_tokens: response.completion_tokens,
            total_tokens: response.prompt_tokens + response.completion_tokens,
        },
    })
}

/// Streaming chat completion
async fn chat_completions_stream<E: InferenceEngine + 'static>(
    engine: Arc<E>,
    req: ChatCompletionRequest,
) -> Sse<impl futures::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let id = format!("chatcmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();
    let model = req.model.clone();

    let (tx, rx) = tokio::sync::mpsc::channel(100);

    // Spawn task to generate tokens
    tokio::spawn(async move {
        let prompt = format_chat_messages(&req.messages);

        // Send initial chunk with role
        let initial_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatChoiceDelta {
                index: 0,
                delta: ChatMessageDelta {
                    role: Some("assistant".to_string()),
                    content: None,
                },
                finish_reason: None,
            }],
        };

        if tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&initial_chunk).unwrap())
            ))
            .await
            .is_err()
        {
            return;
        }

        // Stream tokens
        match engine
            .generate_stream(
                &model,
                &prompt,
                req.max_tokens.unwrap_or(100),
                req.temperature.unwrap_or(0.7),
            )
            .await
        {
            Ok(mut stream) => {
                while let Some(token) = stream.next().await {
                    let chunk = ChatCompletionChunk {
                        id: id.clone(),
                        object: "chat.completion.chunk".to_string(),
                        created,
                        model: model.clone(),
                        choices: vec![ChatChoiceDelta {
                            index: 0,
                            delta: ChatMessageDelta {
                                role: None,
                                content: Some(token),
                            },
                            finish_reason: None,
                        }],
                    };

                    if tx
                        .send(Ok(
                            Event::default().data(serde_json::to_string(&chunk).unwrap())
                        ))
                        .await
                        .is_err()
                    {
                        break;
                    }
                }
            }
            Err(e) => {
                log::error!("Error generating stream: {}", e);
                return;
            }
        }

        // Send final chunk
        let final_chunk = ChatCompletionChunk {
            id: id.clone(),
            object: "chat.completion.chunk".to_string(),
            created,
            model: model.clone(),
            choices: vec![ChatChoiceDelta {
                index: 0,
                delta: ChatMessageDelta {
                    role: None,
                    content: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
        };

        let _ = tx
            .send(Ok(
                Event::default().data(serde_json::to_string(&final_chunk).unwrap())
            ))
            .await;
        let _ = tx.send(Ok(Event::default().data("[DONE]"))).await;
    });

    Sse::new(ReceiverStream::new(rx)).keep_alive(KeepAlive::default())
}

/// Format chat messages into a prompt
fn format_chat_messages(messages: &[ChatMessage]) -> String {
    messages
        .iter()
        .map(|m| {
            if m.role == "system" {
                format!("System: {}", m.content)
            } else if m.role == "user" {
                format!("User: {}", m.content)
            } else {
                format!("Assistant: {}", m.content)
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}
