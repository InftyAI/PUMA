use axum::{
    extract::{Path, State},
    response::IntoResponse,
    Json,
};

use crate::api::routes::AppState;
use crate::api::types::{ErrorResponse, Model, ModelList};
use crate::backend::InferenceEngine;

/// List all available models
pub async fn list_models<E: InferenceEngine + 'static>(
    State(state): State<AppState<E>>,
) -> impl IntoResponse {
    let registry = state.registry;
    match registry.load_models(None) {
        Ok(models) => {
            let model_list = ModelList {
                object: "list".to_string(),
                data: models
                    .into_iter()
                    .map(|m| Model {
                        id: m.name.clone(),
                        object: "model".to_string(),
                        created: chrono::DateTime::parse_from_rfc3339(&m.created_at)
                            .map(|dt| dt.timestamp())
                            .unwrap_or(0),
                        owned_by: m.author.unwrap_or_else(|| "puma".to_string()),
                    })
                    .collect(),
            };
            Json(model_list).into_response()
        }
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                format!("Failed to load models: {}", e),
                "internal_error".to_string(),
            )),
        )
            .into_response(),
    }
}

/// Get a specific model by ID
pub async fn get_model<E: InferenceEngine + 'static>(
    State(state): State<AppState<E>>,
    Path(model_id): Path<String>,
) -> impl IntoResponse {
    let registry = state.registry;
    match registry.get_model(&model_id) {
        Ok(Some(model)) => {
            let model_info = Model {
                id: model.name.clone(),
                object: "model".to_string(),
                created: chrono::DateTime::parse_from_rfc3339(&model.created_at)
                    .map(|dt| dt.timestamp())
                    .unwrap_or(0),
                owned_by: model.author.unwrap_or_else(|| "puma".to_string()),
            };
            Json(model_info).into_response()
        }
        Ok(None) => (
            axum::http::StatusCode::NOT_FOUND,
            Json(ErrorResponse::new(
                format!("Model '{}' not found", model_id),
                "model_not_found".to_string(),
            )),
        )
            .into_response(),
        Err(e) => (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse::new(
                format!("Failed to get model: {}", e),
                "internal_error".to_string(),
            )),
        )
            .into_response(),
    }
}
