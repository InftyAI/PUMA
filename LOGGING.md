# Logging Guidelines for PUMA

## Overview

PUMA uses a hybrid approach for output:
- **User-facing output**: `println!` for CLI commands
- **Internal logging**: `log::*` macros for debugging and monitoring

---

## When to Use Each

### Use `println!` / `eprintln!`

For **direct user communication** in CLI commands:

✅ **Good Examples:**
```rust
// Success messages
println!("✓ Model downloaded successfully");

// Progress updates
println!("Downloading... 50%");

// Command output (ls, inspect, etc.)
println!("MODEL          PROVIDER    SIZE");

// User-facing startup banners
println!("🚀 Starting PUMA inference server...");
```

❌ **Bad Examples:**
```rust
// Internal state (use log::debug instead)
println!("Initializing database connection");

// Error details (use log::error instead)  
println!("Failed to parse config: {}", err);
```

---

### Use `log::*` Macros

For **internal operations** and **debugging**:

#### `log::error!` - Errors
```rust
log::error!("Failed to connect to database: {}", err);
log::error!("Model validation failed for: {}", model_name);
```

#### `log::warn!` - Warnings
```rust
log::warn!("Cache directory not found, creating new one");
log::warn!("Model file corrupt, re-downloading");
```

#### `log::info!` - Important events
```rust
log::info!("Server listening on http://{}", addr);
log::info!("Model registry loaded with {} models", count);
log::info!("Starting inference for model: {}", model);
```

#### `log::debug!` - Debug details
```rust
log::debug!("Using MockEngine backend");
log::debug!("Cache hit for model: {}", model);
log::debug!("Request took {}ms", elapsed);
```

---

## Examples from Codebase

### Good: `serve.rs`

Hybrid approach - both user-facing and logging:

```rust
pub async fn execute(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    // Logging for monitoring
    info!("Starting PUMA inference server");
    
    // User-facing output
    println!("🚀 Starting PUMA inference server...");

    let engine = Arc::new(MockEngine::new());
    
    // Internal logging
    info!("Inference engine initialized");
    debug!("Using MockEngine backend");
    
    // User-facing confirmation
    println!("✓ Inference engine initialized");
    
    // ... server starts
    
    // Logging for monitoring
    info!("Server listening on http://{}", addr);
    
    // User-facing banner
    println!("  Listening on: http://{}", addr);
}
```

### Good: `downloader.rs`

Internal operations use logging only:

```rust
pub async fn download_model(&self, name: &str) -> Result<(), Error> {
    debug!("Downloading model {} from Hugging Face...", name);
    
    // ... download logic
    
    debug!("Model info for {}: {:?}", name, model_info);
}
```

---

## Log Levels in Practice

### Development
```bash
# See all logs
RUST_LOG=debug cargo run -- serve

# See only warnings and errors
RUST_LOG=warn cargo run -- serve
```

### Production
```bash
# Default: info level
RUST_LOG=info ./puma serve

# Quiet mode (errors only)
RUST_LOG=error ./puma serve
```

---

## Best Practices

### ✅ DO

1. **Use both** when appropriate:
   ```rust
   info!("Server starting on {}", addr);  // For logs
   println!("🚀 Starting server...");      // For users
   ```

2. **Log structured data**:
   ```rust
   info!("Request completed: method={} path={} status={} duration={}ms", 
         method, path, status, duration);
   ```

3. **Use appropriate levels**:
   ```rust
   error!("Critical failure");  // Something broke
   warn!("Potential issue");    // Might be a problem
   info!("Important event");    // Normal but notable
   debug!("Internal detail");   // For debugging only
   ```

### ❌ DON'T

1. **Don't log sensitive data**:
   ```rust
   // Bad
   debug!("API key: {}", api_key);
   
   // Good  
   debug!("API key provided: {}", !api_key.is_empty());
   ```

2. **Don't use println for internal operations**:
   ```rust
   // Bad
   println!("Database connection established");
   
   // Good
   info!("Database connection established");
   ```

3. **Don't be too verbose**:
   ```rust
   // Bad - logs every iteration
   for item in items {
       debug!("Processing item: {}", item);
   }
   
   // Good - log summary
   debug!("Processing {} items", items.len());
   ```

---

## Testing Logs

Use `env_logger` for tests:

```rust
#[test]
fn test_something() {
    let _ = env_logger::builder().is_test(true).try_init();
    
    info!("Running test");
    // ... test code
}
```

Run with logs:
```bash
RUST_LOG=debug cargo test -- --nocapture
```

---

## Summary

| Output Type | Use | Example |
|-------------|-----|---------|
| **User-facing** | `println!` | "✓ Model downloaded" |
| **Errors** | `log::error!` | "Failed to connect: {}" |
| **Warnings** | `log::warn!` | "Cache miss for model" |
| **Events** | `log::info!` | "Server started on :8000" |
| **Debug** | `log::debug!` | "Using MockEngine" |

**Key principle**: If a human is directly waiting for the output → `println!`. If it's for monitoring/debugging → `log::*`.
