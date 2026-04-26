use axum::{
    routing::{get, post},
    Json, Router,
};
use serde::Serialize;
use std::sync::Arc;
use tower_http::{
    cors::CorsLayer,
    trace::{DefaultMakeSpan, DefaultOnRequest, DefaultOnResponse, TraceLayer},
    LatencyUnit,
};

use crate::backend::InferenceEngine;
use crate::registry::model_registry::ModelRegistry;

use super::{chat, completions, models};

/// Shared application state
#[derive(Clone)]
pub struct AppState<E: InferenceEngine> {
    pub engine: Arc<E>,
    pub registry: Arc<ModelRegistry>,
}

/// Create the API router with all endpoints
pub fn create_router<E: InferenceEngine + Clone + 'static>(
    engine: Arc<E>,
    registry: Arc<ModelRegistry>,
) -> Router {
    let state = AppState { engine, registry };

    Router::new()
        // Chat completions (most important)
        .route("/v1/chat/completions", post(chat::chat_completions::<E>))
        // Legacy completions
        .route("/v1/completions", post(completions::completions::<E>))
        // Models
        .route("/v1/models", get(models::list_models::<E>))
        .route("/v1/models/:model", get(models::get_model::<E>))
        // Health check
        .route("/health", get(health_check))
        // Pass state
        .with_state(state)
        // Enable request/response logging at INFO level
        .layer(
            TraceLayer::new_for_http()
                .make_span_with(DefaultMakeSpan::new().level(tracing::Level::INFO))
                .on_request(DefaultOnRequest::new().level(tracing::Level::INFO))
                .on_response(
                    DefaultOnResponse::new()
                        .level(tracing::Level::INFO)
                        .latency_unit(LatencyUnit::Millis),
                ),
        )
        // Enable CORS for browser clients
        .layer(CorsLayer::permissive())
}

/// Health check response
#[derive(Serialize)]
struct HealthResponse {
    status: String,
    version: String,
}

/// Health check endpoint
async fn health_check() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}
