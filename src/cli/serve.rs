use colored::Colorize;
use std::sync::Arc;

use crate::api::create_router;
use crate::backend::mock::MockEngine;
use crate::registry::model_registry::ModelRegistry;

/// Execute the serve command
pub async fn execute(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    println!("{} Starting PUMA inference server...", "🚀".bright_green());

    // Initialize backend (MockEngine for now, replace with MLX later)
    let engine = Arc::new(MockEngine::new());
    println!("{} Inference engine initialized", "✓".green());

    // Initialize model registry
    let registry = Arc::new(ModelRegistry::new(None));
    println!("{} Model registry loaded", "✓".green());

    // Create router
    let app = create_router(engine, registry);

    // Bind address
    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    println!();
    println!("{}", "PUMA Inference Server".bright_cyan().bold());
    println!("{}", "━━━━━━━━━━━━━━━━━━━━━━".bright_black());
    println!(
        "  Listening on: {}",
        format!("http://{}", addr).bright_white().underline()
    );
    println!();
    println!("  Endpoints:");
    println!("    {} /v1/chat/completions", "POST".bright_yellow());
    println!("    {} /v1/completions", "POST".bright_yellow());
    println!("    {} /v1/models", "GET ".bright_green());
    println!("    {} /health", "GET ".bright_green());
    println!();
    println!("  {}", "Press Ctrl+C to stop".bright_black());
    println!();

    // Start server
    axum::serve(listener, app).await?;

    Ok(())
}
