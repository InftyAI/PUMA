//! API Integration Tests
//!
//! Tests the PUMA API endpoints using the Axum test utilities.
//! These tests verify the entire request/response cycle through the router.

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use serde_json::{json, Value};
use std::sync::Arc;
use tempfile::TempDir;
use tower::ServiceExt; // for `oneshot` and `ready`

use super::routes::create_router;
use crate::backend::mock::MockEngine;
use crate::registry::model_registry::{ArtifactInfo, ModelInfo, ModelMetadata, ModelRegistry};

/// Helper to create test app with a pre-registered test model
/// Returns the router and the temp directory (which must be kept alive)
fn create_test_app() -> (axum::Router, TempDir) {
    let engine = Arc::new(MockEngine::new());
    let temp_dir = TempDir::new().unwrap();
    let registry = Arc::new(ModelRegistry::new(Some(temp_dir.path().to_path_buf())));

    // Register a test model
    let test_model = ModelInfo {
        uuid: "test-uuid".to_string(),
        name: "test-model".to_string(),
        author: Some("test-author".to_string()),
        task: Some("text-generation".to_string()),
        model_series: Some("test-series".to_string()),
        provider: "test".to_string(),
        license: Some("MIT".to_string()),
        created_at: chrono::Utc::now().to_rfc3339(),
        updated_at: chrono::Utc::now().to_rfc3339(),
        metadata: ModelMetadata {
            artifact: ArtifactInfo {
                revision: "test-rev".to_string(),
                size: 1000,
                path: "/tmp/test-model".to_string(),
            },
            context_window: Some(2048),
            safetensors: None,
        },
    };

    registry
        .register_model(test_model)
        .expect("failed to register test model");

    (create_router(engine, registry), temp_dir)
}

/// Helper to make a JSON request
async fn make_json_request(
    app: axum::Router,
    method: &str,
    uri: &str,
    body: Option<Value>,
) -> (StatusCode, Value) {
    let mut request = Request::builder().uri(uri).method(method);

    if body.is_some() {
        request = request.header("content-type", "application/json");
    }

    let request = if let Some(body) = body {
        request.body(Body::from(serde_json::to_vec(&body).unwrap()))
    } else {
        request.body(Body::empty())
    }
    .unwrap();

    let response = app.oneshot(request).await.unwrap();

    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let json: Value = serde_json::from_slice(&body).unwrap_or(json!({}));

    // Debug output for failed requests
    if !status.is_success() {
        eprintln!("Request failed: {} {}", method, uri);
        eprintln!("Status: {}", status);
        eprintln!(
            "Response: {}",
            serde_json::to_string_pretty(&json).unwrap_or_default()
        );
    }

    (status, json)
}

#[tokio::test]
async fn test_health_check() {
    let (app, _temp_dir) = create_test_app();
    let (status, json) = make_json_request(app, "GET", "/health", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["status"], "ok");
    assert!(json["version"].is_string());
}

#[tokio::test]
async fn test_list_models() {
    let (app, _temp_dir) = create_test_app();
    let (status, json) = make_json_request(app, "GET", "/v1/models", None).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "list");
    assert!(json["data"].is_array());
}

#[tokio::test]
async fn test_chat_completion_non_streaming() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 50,
        "stream": false
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/chat/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
    assert!(json["id"].is_string());
    assert_eq!(json["model"], "test-model");
    assert!(json["choices"].is_array());
    assert_eq!(json["choices"][0]["index"], 0);
    assert_eq!(json["choices"][0]["message"]["role"], "assistant");
    assert!(json["choices"][0]["message"]["content"].is_string());
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(json["usage"]["prompt_tokens"].is_number());
    assert!(json["usage"]["completion_tokens"].is_number());
    assert!(json["usage"]["total_tokens"].is_number());
}

#[tokio::test]
async fn test_chat_completion_empty_messages() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "messages": [],
        "stream": false
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/chat/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("messages cannot be empty"));
}

#[tokio::test]
async fn test_text_completion() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "prompt": "Once upon a time",
        "max_tokens": 50
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "text_completion");
    assert!(json["id"].is_string());
    assert_eq!(json["model"], "test-model");
    assert!(json["choices"].is_array());
    assert_eq!(json["choices"][0]["index"], 0);
    assert!(json["choices"][0]["text"].is_string());
    assert_eq!(json["choices"][0]["finish_reason"], "stop");
    assert!(json["usage"]["prompt_tokens"].is_number());
}

#[tokio::test]
async fn test_text_completion_empty_prompt() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "prompt": ""
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(json["error"]["type"], "invalid_request_error");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("prompt cannot be empty"));
}

#[tokio::test]
async fn test_chat_completion_with_system_message() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 50
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/chat/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
    assert!(json["choices"][0]["message"]["content"].is_string());
}

#[tokio::test]
async fn test_chat_completion_with_temperature() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "temperature": 0.5,
        "max_tokens": 50
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/chat/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
}

#[tokio::test]
async fn test_chat_completion_default_values() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "test-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ]
        // No max_tokens, temperature, etc. - should use defaults
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/chat/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::OK);
    assert_eq!(json["object"], "chat.completion");
}

#[tokio::test]
async fn test_cors_headers() {
    let (app, _temp_dir) = create_test_app();
    let request = Request::builder()
        .uri("/health")
        .method("GET")
        .header("Origin", "https://example.com")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();

    // CORS should be permissive
    assert!(response
        .headers()
        .contains_key("access-control-allow-origin"));
}

#[tokio::test]
async fn test_invalid_route() {
    let (app, _temp_dir) = create_test_app();
    let request = Request::builder()
        .uri("/invalid/route")
        .method("GET")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn test_method_not_allowed() {
    let (app, _temp_dir) = create_test_app();
    // Try POST on GET-only endpoint
    let request = Request::builder()
        .uri("/health")
        .method("POST")
        .body(Body::empty())
        .unwrap();

    let response = app.oneshot(request).await.unwrap();
    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
}

#[tokio::test]
async fn test_chat_completion_nonexistent_model() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "nonexistent-model",
        "messages": [
            {"role": "user", "content": "Hello"}
        ],
        "max_tokens": 50
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/chat/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["type"], "model_not_found");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("nonexistent-model"));
}

#[tokio::test]
async fn test_text_completion_nonexistent_model() {
    let (app, _temp_dir) = create_test_app();
    let request_body = json!({
        "model": "nonexistent-model",
        "prompt": "Hello world"
    });

    let (status, json) =
        make_json_request(app, "POST", "/v1/completions", Some(request_body)).await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(json["error"]["type"], "model_not_found");
    assert!(json["error"]["message"]
        .as_str()
        .unwrap()
        .contains("nonexistent-model"));
}
