use crate::registry::model_registry::{ModelInfo, ModelRegistry};
use std::collections::HashMap;

/// Execute the LS command logic
pub fn execute(
    registry: &ModelRegistry,
    pattern: Option<&str>,
    query: Option<&str>,
) -> Result<Vec<ModelInfo>, String> {
    // Parse query filters if provided
    let mut query_filters = HashMap::new();
    if let Some(query_str) = query {
        for pair in query_str.split(',') {
            if let Some((key, value)) = pair.split_once('=') {
                query_filters.insert(key.trim().to_string(), value.trim().to_string());
            } else {
                return Err(format!(
                    "Invalid query format: {}. Expected key=value pairs separated by commas.",
                    pair
                ));
            }
        }
    }

    // Load models with optional SQL filters
    let filter_ref = if query_filters.is_empty() {
        None
    } else {
        Some(&query_filters)
    };

    let mut models = registry
        .load_models(filter_ref)
        .map_err(|e| format!("Failed to query models: {}", e))?;

    // Filter models by name pattern if provided (supports regex)
    if let Some(pattern_str) = pattern {
        let pattern_lower = pattern_str.to_lowercase();
        match regex::Regex::new(&pattern_lower) {
            Ok(re) => {
                models.retain(|model| re.is_match(&model.name));
            }
            Err(e) => {
                return Err(format!("Invalid regex pattern '{}': {}", pattern_str, e));
            }
        }
    }

    Ok(models)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::model_registry::{ArtifactInfo, ModelInfo, ModelMetadata};
    use tempfile::TempDir;

    fn create_test_model(name: &str, uuid: &str, author: &str) -> ModelInfo {
        let safetensors = serde_json::json!({
            "parameters": {"F32": 7000000000u64},
            "total": 7000000000u64
        });

        ModelInfo {
            uuid: uuid.to_string(),
            name: name.to_string(),
            author: Some(author.to_string()),
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
    fn test_execute_ls_substring() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        registry
            .register_model(create_test_model("inftyai/model1", "uuid1", "inftyai"))
            .unwrap();
        registry
            .register_model(create_test_model("openai/gpt2", "uuid2", "openai"))
            .unwrap();
        registry
            .register_model(create_test_model("inftyai/model2", "uuid3", "inftyai"))
            .unwrap();

        let models = execute(&registry, Some("inftyai"), None).unwrap();
        assert_eq!(models.len(), 2);
        assert!(models.iter().all(|m| m.name.contains("inftyai")));
    }

    #[test]
    fn test_execute_ls_prefix() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        registry
            .register_model(create_test_model("inftyai/model1", "uuid1", "inftyai"))
            .unwrap();
        registry
            .register_model(create_test_model("openai/gpt2", "uuid2", "openai"))
            .unwrap();

        let models = execute(&registry, Some("^inftyai/"), None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "inftyai/model1");
    }

    #[test]
    fn test_execute_ls_case_insensitive() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        registry
            .register_model(create_test_model("InftyAI/Model1", "uuid1", "InftyAI"))
            .unwrap();

        let models = execute(&registry, Some("InftyAI"), None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "inftyai/model1");
    }

    #[test]
    fn test_execute_ls_sql_filter() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        registry
            .register_model(create_test_model("inftyai/model1", "uuid1", "inftyai"))
            .unwrap();
        registry
            .register_model(create_test_model("openai/gpt2", "uuid2", "openai"))
            .unwrap();

        let models = execute(&registry, None, Some("author=inftyai")).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "inftyai/model1");
    }

    #[test]
    fn test_execute_ls_pattern_and_filter() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        registry
            .register_model(create_test_model("inftyai/llama-2", "uuid1", "inftyai"))
            .unwrap();
        registry
            .register_model(create_test_model("inftyai/gpt2", "uuid2", "inftyai"))
            .unwrap();
        registry
            .register_model(create_test_model("openai/llama-2", "uuid3", "openai"))
            .unwrap();

        let models = execute(&registry, Some("llama"), Some("author=inftyai")).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "inftyai/llama-2");
    }
}
