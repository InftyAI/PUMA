use colored::Colorize;
use std::sync::Arc;
use tokenizers::models::bpe::BPE;
use tokenizers::Tokenizer;
use tracing::{debug, info};

use crate::api::routes::create_router;
use crate::backend::engine;
use crate::backend::mock::MockEngine;
use crate::registry::model_registry::ModelRegistry;

/// Execute the serve command
pub async fn execute(
    host: &str,
    port: u16,
    model_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{}",
        "
 ███████████  █████  █████ ██████   ██████   █████████
░░███░░░░░███░░███  ░░███ ░░██████ ██████   ███░░░░░███
 ░███    ░███ ░███   ░███  ░███░█████░███  ░███    ░███
 ░██████████  ░███   ░███  ░███░░███ ░███  ░███████████
 ░███░░░░░░   ░███   ░███  ░███ ░░░  ░███  ░███░░░░░███
 ░███         ░███   ░███  ░███      ░███  ░███    ░███
 █████        ░░████████   █████     █████ █████   █████
░░░░░          ░░░░░░░░   ░░░░░     ░░░░░ ░░░░░   ░░░░░
                                                        "
        .bright_blue()
        .bold()
    );
    info!("Starting PUMA to serve model: {}", model_name);

    // Initialize backend (MockEngine for now, replace with MLX later)
    let backend = MockEngine::new();
    debug!("Using MockEngine backend");

    // TODO: Load the model's real tokenizer; placeholder BPE for now
    let tokenizer = Tokenizer::new(BPE::default());

    // Create engine: cheap send-side handle + runner that owns the scheduler
    let (handle, runner) = engine(backend, tokenizer, model_name.to_string());

    // Spawn the runner's event loop; the handle submits work via events
    tokio::spawn(runner.serve());
    info!("Inference engine initialized");

    // Initialize model registry
    let registry = Arc::new(ModelRegistry::new(None));
    info!("Model registry loaded");

    // Create router
    let app = create_router(handle, registry);

    // Bind address
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    info!("Server listening on http://{}", addr);
    info!("Available endpoints:");
    info!("  POST /v1/chat/completions");
    info!("  POST /v1/completions");
    info!("  GET  /v1/models");
    info!("  GET  /v1/models/:model");
    info!("  GET  /health");

    // Start server
    debug!("Starting server");
    axum::serve(listener, app).await?;

    info!("Server shutdown");
    Ok(())
}
