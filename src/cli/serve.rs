use colored::Colorize;
use log::{debug, info};
use std::sync::Arc;

use crate::api::create_router;
use crate::backend::mock::MockEngine;
use crate::registry::model_registry::ModelRegistry;

/// Execute the serve command
pub async fn execute(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
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
    info!("Starting PUMA inference server");

    // Initialize backend (MockEngine for now, replace with MLX later)
    let engine = Arc::new(MockEngine::new());
    info!("Inference engine initialized");
    debug!("Using MockEngine backend");

    // Initialize model registry
    let registry = Arc::new(ModelRegistry::new(None));
    info!("Model registry loaded");

    // Create router
    let app = create_router(engine, registry);

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
    debug!("Starting axum server");
    axum::serve(listener, app).await?;

    info!("Server shutdown");
    Ok(())
}
