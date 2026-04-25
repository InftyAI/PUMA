use crate::registry::model_registry::ModelRegistry;

/// Execute the RM command logic
pub fn execute(registry: &ModelRegistry, model_name: &str) -> Result<(), String> {
    match registry.get_model(model_name) {
        Ok(Some(_)) => registry
            .remove_model(model_name)
            .map_err(|e| format!("Failed to remove model: {}", e)),
        Ok(None) => Err(format!("Model not found: {}", model_name)),
        Err(e) => Err(format!("Failed to load registry: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::model_registry::{ArtifactInfo, ModelInfo, ModelMetadata};
    use tempfile::TempDir;

    fn create_test_model(name: &str, uuid: &str) -> ModelInfo {
        let safetensors = serde_json::json!({
            "parameters": {"F32": 7000000000u64},
            "total": 7000000000u64
        });

        ModelInfo {
            uuid: uuid.to_string(),
            name: name.to_string(),
            author: Some("test-author".to_string()),
            r#type: Some("text-generation".to_string()),
            model_series: Some("gpt2".to_string()),
            provider: "huggingface".to_string(),
            license: Some("mit".to_string()),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            metadata: ModelMetadata {
                artifact: ArtifactInfo {
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
    fn test_execute_rm() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let cache_dir = temp_dir.path().join("cache");
        std::fs::create_dir_all(&cache_dir).unwrap();
        std::fs::write(cache_dir.join("model.safetensors"), "fake data").unwrap();

        let mut model = create_test_model("test/remove-model", "abc123");
        model.metadata.artifact.path = cache_dir.to_string_lossy().to_string();

        registry.register_model(model).unwrap();
        assert!(registry.get_model("test/remove-model").unwrap().is_some());
        assert!(cache_dir.exists());

        let result = execute(&registry, "test/remove-model");
        assert!(result.is_ok());

        assert!(registry.get_model("test/remove-model").unwrap().is_none());
        assert!(!cache_dir.exists());
    }

    #[test]
    fn test_execute_rm_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let result = execute(&registry, "nonexistent/model");
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Model not found"));
    }
}
