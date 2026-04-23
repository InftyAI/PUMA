use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::utils::file;
use crate::utils::format::format_parameters;

#[derive(Debug, Serialize, Deserialize, Clone, PartialEq)]
pub struct ModelArchitecture {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub classes: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<String>,
}

impl ModelArchitecture {
    /// Extract model architecture from config.json
    pub fn from_config(config: &serde_json::Value) -> Option<Self> {
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let classes = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect::<Vec<String>>()
            })
            .filter(|v| !v.is_empty());

        let context_window = config
            .get("n_positions")
            .or_else(|| config.get("max_position_embeddings"))
            .or_else(|| config.get("n_ctx"))
            .and_then(|v| v.as_u64())
            .map(|v| v as u32);

        let parameters = Self::estimate_parameters(config);

        if model_type.is_some()
            || classes.is_some()
            || context_window.is_some()
            || parameters.is_some()
        {
            Some(ModelArchitecture {
                model_type,
                classes,
                context_window,
                parameters,
            })
        } else {
            None
        }
    }

    /// Estimate model parameters from config
    fn estimate_parameters(config: &serde_json::Value) -> Option<String> {
        let n_layer = config
            .get("n_layer")
            .or_else(|| config.get("num_hidden_layers"))
            .and_then(|v| v.as_u64())?;

        let n_embd = config
            .get("n_embd")
            .or_else(|| config.get("hidden_size"))
            .and_then(|v| v.as_u64())?;

        let vocab_size = config.get("vocab_size").and_then(|v| v.as_u64())?;

        let n_positions = config
            .get("n_positions")
            .or_else(|| config.get("max_position_embeddings"))
            .and_then(|v| v.as_u64())
            .unwrap_or(2048);

        // Rough parameter estimation for transformer models
        // Each layer: ~12 * n_embd^2 (attention + FFN)
        // Embeddings: vocab_size * n_embd + n_positions * n_embd
        let layer_params = 12 * n_layer * n_embd * n_embd;
        let embedding_params = vocab_size * n_embd + n_positions * n_embd;
        let total_params = layer_params + embedding_params;

        Some(format_parameters(total_params))
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub revision: String,
    pub size: u64,
    pub modified_at: String,
    pub cache_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub arch: Option<ModelArchitecture>,
}

pub struct ModelRegistry {
    home_dir: PathBuf,
}

impl ModelRegistry {
    pub fn new(home_dir: Option<PathBuf>) -> Self {
        Self {
            home_dir: home_dir.unwrap_or_else(file::root_home),
        }
    }

    fn registry_file(&self) -> PathBuf {
        self.home_dir.join("models.json")
    }

    pub fn load_models(&self) -> Result<Vec<ModelInfo>, std::io::Error> {
        let registry_file = self.registry_file();

        if !registry_file.exists() {
            return Ok(Vec::new());
        }

        let contents = fs::read_to_string(registry_file)?;
        let models: Vec<ModelInfo> = serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(models)
    }

    pub fn save_models(&self, models: &[ModelInfo]) -> Result<(), std::io::Error> {
        // Ensure home directory exists
        fs::create_dir_all(&self.home_dir)?;

        let registry_file = self.registry_file();
        let json = serde_json::to_string_pretty(models)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        fs::write(registry_file, json)?;
        Ok(())
    }

    pub fn register_model(&self, model: ModelInfo) -> Result<(), std::io::Error> {
        let mut models = self.load_models()?;

        // Remove existing model with same name if exists
        models.retain(|m| m.name != model.name);

        models.push(model);
        self.save_models(&models)?;

        Ok(())
    }

    pub fn unregister_model(&self, name: &str) -> Result<(), std::io::Error> {
        let mut models = self.load_models()?;
        models.retain(|m| m.name != name);
        self.save_models(&models)?;

        Ok(())
    }

    pub fn get_model(&self, name: &str) -> Result<Option<ModelInfo>, std::io::Error> {
        let models = self.load_models()?;
        Ok(models.into_iter().find(|m| m.name == name))
    }

    pub fn remove_model(&self, name: &str) -> Result<(), std::io::Error> {
        // Get model info first
        let model_info = self.get_model(name)?;

        if let Some(info) = model_info {
            // Delete cache directory if it exists
            let cache_path = std::path::Path::new(&info.cache_path);
            if cache_path.exists() {
                fs::remove_dir_all(cache_path)?;
            }

            // Remove from registry
            self.unregister_model(name)?;

            println!(
                "{} {} {}",
                "✓".green().bold(),
                "Successfully removed model".bright_white(),
                name.cyan().bold()
            );
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn test_add_and_load_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123".to_string(),
            size: 1000,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(model.clone()).unwrap();

        let models = registry.load_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model");
    }

    #[test]
    fn test_unregister_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123".to_string(),
            size: 1000,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();
        assert_eq!(registry.load_models().unwrap().len(), 1);

        registry.unregister_model("test/model").unwrap();
        assert_eq!(registry.load_models().unwrap().len(), 0);
    }

    #[test]
    fn test_get_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123".to_string(),
            size: 1000,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();

        let result = registry.get_model("test/model").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "test/model");

        let not_found = registry.get_model("nonexistent").unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_remove_nonexistent_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        // Should not error when removing non-existent model
        let result = registry.unregister_model("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_update_existing_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model1 = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123".to_string(),
            size: 1000,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(model1).unwrap();

        let model2 = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "def456".to_string(),
            size: 2000,
            modified_at: "2025-01-02T00:00:00Z".to_string(),
            cache_path: "/tmp/test2".to_string(),
            arch: None,
        };

        registry.register_model(model2).unwrap();

        let models = registry.load_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].revision, "def456");
        assert_eq!(models[0].size, 2000);
    }

    #[test]
    fn test_remove_model_with_cache() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        // Create a fake cache directory
        let cache_dir = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::write(cache_dir.join("test.txt"), "test data").unwrap();

        let model = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123".to_string(),
            size: 1000,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: cache_dir.to_string_lossy().to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();
        assert_eq!(registry.load_models().unwrap().len(), 1);
        assert!(cache_dir.exists());

        // Delete model
        registry.remove_model("test/model").unwrap();

        // Verify model removed from registry
        assert_eq!(registry.load_models().unwrap().len(), 0);

        // Verify cache directory deleted
        assert!(!cache_dir.exists());
    }

    #[test]
    fn test_delete_nonexistent_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        // Should not error when deleting non-existent model
        let result = registry.remove_model("nonexistent");
        assert!(result.is_ok());
    }

    #[test]
    fn test_inspect_model_with_full_spec() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/gpt-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123def456".to_string(),
            size: 7_000_000_000,
            modified_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test/gpt".to_string(),
            arch: Some(ModelArchitecture {
                model_type: Some("gpt2".to_string()),
                classes: Some(vec!["GPT2LMHeadModel".to_string()]),
                context_window: Some(2048),
                parameters: Some("7.00B".to_string()),
            }),
        };

        registry.register_model(model).unwrap();

        let retrieved = registry.get_model("test/gpt-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/gpt-model");
        assert_eq!(model_info.provider, "huggingface");
        assert_eq!(model_info.revision, "abc123def456");
        assert_eq!(model_info.size, 7_000_000_000);

        let arch = model_info.arch.unwrap();
        assert_eq!(arch.model_type, Some("gpt2".to_string()));
        assert_eq!(arch.classes, Some(vec!["GPT2LMHeadModel".to_string()]));
        assert_eq!(arch.context_window, Some(2048));
        assert_eq!(arch.parameters, Some("7.00B".to_string()));
    }

    #[test]
    fn test_inspect_model_without_spec() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/legacy-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "legacy123".to_string(),
            size: 1_000_000,
            modified_at: "2024-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test/legacy".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();

        let retrieved = registry.get_model("test/legacy-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/legacy-model");
        assert!(model_info.arch.is_none());
    }

    #[test]
    fn test_model_architecture_from_config_gpt2() {
        use serde_json::json;

        let config = json!({
            "model_type": "gpt2",
            "architectures": ["GPT2LMHeadModel"],
            "n_layer": 5,
            "n_embd": 32,
            "vocab_size": 1000,
            "n_positions": 512
        });

        let arch = ModelArchitecture::from_config(&config);
        assert!(arch.is_some());

        let arch = arch.unwrap();
        assert_eq!(arch.model_type, Some("gpt2".to_string()));
        assert_eq!(arch.classes, Some(vec!["GPT2LMHeadModel".to_string()]));
        assert_eq!(arch.context_window, Some(512));
        assert_eq!(arch.parameters, Some("109.82K".to_string()));
    }

    #[test]
    fn test_model_architecture_from_config_bert_style() {
        use serde_json::json;

        let config = json!({
            "model_type": "bert",
            "num_hidden_layers": 12,
            "hidden_size": 768,
            "vocab_size": 30000,
            "max_position_embeddings": 512
        });

        let arch = ModelArchitecture::from_config(&config);
        assert!(arch.is_some());

        let arch = arch.unwrap();
        assert_eq!(arch.model_type, Some("bert".to_string()));
        assert_eq!(arch.context_window, Some(512));
        assert!(arch.parameters.unwrap().contains("M"));
    }

    #[test]
    fn test_model_architecture_from_config_partial() {
        use serde_json::json;

        let config = json!({
            "model_type": "llama",
            "n_ctx": 4096
        });

        let arch = ModelArchitecture::from_config(&config);
        assert!(arch.is_some());

        let arch = arch.unwrap();
        assert_eq!(arch.model_type, Some("llama".to_string()));
        assert_eq!(arch.context_window, Some(4096));
        assert_eq!(arch.parameters, None);
    }

    #[test]
    fn test_model_architecture_from_config_empty() {
        use serde_json::json;

        let config = json!({});
        let arch = ModelArchitecture::from_config(&config);
        assert_eq!(arch, None);
    }
}
