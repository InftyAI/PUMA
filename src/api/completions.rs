use axum::{extract::State, response::IntoResponse, Json};
use uuid::Uuid;

use crate::api::routes::AppState;
use crate::api::types::{
    CompletionChoice, CompletionRequest, CompletionResponse, ErrorResponse, Usage,
};
use crate::backend::InferenceEngine;

/// Handler for legacy text completions
pub async fn completions<E: InferenceEngine + 'static>(
    State(state): State<AppState<E>>,
    Json(req): Json<CompletionRequest>,
) -> impl IntoResponse {
    let engine = state.engine;
    let registry = state.registry;

    // Validate request
    let prompt = req.prompt.to_string();
    if prompt.is_empty() {
        return (
            axum::http::StatusCode::BAD_REQUEST,
            Json(ErrorResponse::new(
                "prompt cannot be empty".to_string(),
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

    let id = format!("cmpl-{}", Uuid::new_v4());
    let created = chrono::Utc::now().timestamp();

    // Generate
    match engine
        .generate(
            &req.model,
            &prompt,
            req.max_tokens.unwrap_or(100),
            req.temperature.unwrap_or(0.7),
        )
        .await
    {
        Ok(response) => {
            let completion = CompletionResponse {
                id,
                object: "text_completion".to_string(),
                created,
                model: req.model,
                choices: vec![CompletionChoice {
                    text: response.text,
                    index: 0,
                    logprobs: None,
                    finish_reason: "stop".to_string(),
                }],
                usage: Usage {
                    prompt_tokens: response.prompt_tokens,
                    completion_tokens: response.completion_tokens,
                    total_tokens: response.prompt_tokens + response.completion_tokens,
                },
            };

            Json(completion).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                e.to_string(),
                "internal_error".to_string(),
            )),
        )
            .into_response(),
    }
}
