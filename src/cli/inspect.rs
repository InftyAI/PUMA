use crate::registry::model_registry::{ModelInfo, ModelRegistry};
use crate::utils::format::{format_parameters, format_size_decimal, format_time_ago};

/// Execute the INSPECT command logic
pub fn execute(registry: &ModelRegistry, model_name: &str) -> Result<ModelInfo, String> {
    match registry.get_model(model_name) {
        Ok(Some(model)) => Ok(model),
        Ok(None) => Err(format!("Model not found: {}", model_name)),
        Err(e) => Err(format!("Failed to load registry: {}", e)),
    }
}

/// Display the model information
pub fn display(model: &ModelInfo) {
    println!("name: {}", model.name);
    println!("kind: Model");
    println!("spec:");
    println!(
        "  author:         {}",
        model.author.as_deref().unwrap_or("N/A")
    );
    println!("  provider:       {}", model.provider);
    println!(
        "  model_series:   {}",
        model.model_series.as_deref().unwrap_or("N/A")
    );
    println!(
        "  task:           {}",
        model.task.as_deref().unwrap_or("N/A")
    );
    println!(
        "  license:        {}",
        model
            .license
            .as_ref()
            .map(|s| s.to_uppercase())
            .unwrap_or_else(|| "N/A".to_string())
    );
    println!(
        "  context_window: {}",
        model
            .metadata
            .context_window
            .map(|w| format_parameters(w as u64))
            .unwrap_or_else(|| "N/A".to_string())
    );

    if let Some(st) = &model.metadata.safetensors {
        println!("  safetensors:");
        if let Some(total) = st.get("total").and_then(|v| v.as_u64()) {
            println!("    total:        {}", format_parameters(total));
        }
        if let Some(params) = st.get("parameters").and_then(|v| v.as_object()) {
            println!("    parameters:");
            for (dtype, count) in params {
                if let Some(num) = count.as_u64() {
                    println!(
                        "      {:<12} {}",
                        format!("{}:", dtype.to_lowercase()),
                        format_parameters(num)
                    );
                }
            }
        }
    } else {
        println!("  safetensors:    N/A");
    }

    // Cache section
    println!("  cache:");
    println!("    revision:       {}", model.metadata.cache.revision);
    println!(
        "    size:           {}",
        format_size_decimal(model.metadata.cache.size)
    );
    println!("    path:           {}", model.metadata.cache.path);
    println!("status:");
    println!("  created:        {}", format_time_ago(&model.created_at));
    println!("  updated:        {}", format_time_ago(&model.updated_at));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::model_registry::{CacheInfo, ModelInfo, ModelMetadata};
    use tempfile::TempDir;

    fn create_test_model(name: &str, uuid: &str) -> ModelInfo {
        let safetensors = serde_json::json!({
            "parameters": {"F32": 7000000000u64},
            "total": 7000000000u64
        });

        ModelInfo {
            uuid: uuid.to_string(),
            name: name.to_string(),
            provider: "huggingface".to_string(),
            author: Some("test-author".to_string()),
            task: Some("text-generation".to_string()),
            model_series: Some("gpt2".to_string()),
            license: Some("mit".to_string()),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            metadata: ModelMetadata {
                cache: CacheInfo {
                    revision: uuid.to_string(),
                    size: 1000,
                    path: "/tmp/test".to_string(),
                },
                context_window: Some(2048),
                safetensors: Some(safetensors),
            },
        }
    }

    #[test]
    fn test_execute_inspect() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = create_test_model("inftyai/test-model", "abc123");
        registry.register_model(model).unwrap();

        let result = execute(&registry, "inftyai/test-model");
        assert!(result.is_ok());

        let model_info = result.unwrap();
        assert_eq!(model_info.name, "inftyai/test-model");
        assert_eq!(model_info.provider, "huggingface");
    }

    #[test]
    fn test_execute_inspect_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let result = execute(&registry, "nonexistent/model");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Model not found"));
    }

    #[test]
    fn test_execute_inspect_case_insensitive() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = create_test_model("InftyAI/TestModel", "abc123");
        registry.register_model(model).unwrap();

        // Can query with different cases
        let result = execute(&registry, "InftyAI/TestModel");
        assert!(result.is_ok());

        let result = execute(&registry, "inftyai/testmodel");
        assert!(result.is_ok());

        let result = execute(&registry, "INFTYAI/TESTMODEL");
        assert!(result.is_ok());
    }
}
