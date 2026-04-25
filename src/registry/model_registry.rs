use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::storage::{ModelStorage, SqliteStorage};
use crate::utils::file;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ArtifactInfo {
    pub revision: String,
    pub size: u64,
    pub path: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelMetadata {
    pub artifact: ArtifactInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context_window: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub safetensors: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub uuid: String,
    pub name: String,
    pub author: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub task: Option<String>, // Task type (image-text-to-text, text-generation)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_series: Option<String>, // Architecture series (qwen3_5, gpt2, llama3)
    pub provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
    pub metadata: ModelMetadata,
    pub created_at: String,
    pub updated_at: String,
}

pub struct ModelRegistry {
    storage: Box<dyn ModelStorage>,
}

impl ModelRegistry {
    pub fn new(home_dir: Option<PathBuf>) -> Self {
        let home_dir = home_dir.unwrap_or_else(file::root_home);
        fs::create_dir_all(&home_dir).ok();

        let db_path = home_dir.join("models.db");
        let storage = SqliteStorage::new(db_path).expect("Failed to initialize storage");

        Self {
            storage: Box::new(storage),
        }
    }

    pub fn load_models(&self, filters: Option<&HashMap<String, String>>) -> Result<Vec<ModelInfo>, std::io::Error> {
        self.storage.load_models(filters)
    }

    pub fn register_model(&self, model: ModelInfo) -> Result<(), std::io::Error> {
        self.storage.register_model(model)
    }

    pub fn unregister_model(&self, name: &str) -> Result<(), std::io::Error> {
        self.storage.unregister_model(name)
    }

    pub fn get_model(&self, name: &str) -> Result<Option<ModelInfo>, std::io::Error> {
        self.storage.get_model(name)
    }

    pub fn remove_model(&self, name: &str) -> Result<(), std::io::Error> {
        // Get model info first
        let model_info = self.get_model(name)?;

        if let Some(info) = model_info {
            // Delete artifact directory if it exists
            let artifact_path = std::path::Path::new(&info.metadata.artifact.path);
            if artifact_path.exists() {
                fs::remove_dir_all(artifact_path)?;
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

    // Helper to create a test model
    fn create_test_model(name: &str, revision: &str) -> ModelInfo {
        let safetensors = serde_json::json!({
            "parameters": {
                "F32": 7000000000u64
            },
            "total": 7000000000u64
        });

        ModelInfo {
            uuid: revision.to_string(),
            name: name.to_string(),
            author: Some("test-author".to_string()),
            task: Some("text-generation".to_string()),
            model_series: Some("gpt2".to_string()),
            provider: "huggingface".to_string(),
            license: Some("mit".to_string()),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            metadata: ModelMetadata {
                artifact: ArtifactInfo {
                    revision: revision.to_string(),
                    size: 1000,
                    path: "/tmp/test".to_string(),
                },
                context_window: Some(2048),
                safetensors: Some(safetensors),
            },
        }
    }

    #[test]
    fn test_add_and_load_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = create_test_model("test/model", "abc123");

        registry.register_model(model.clone()).unwrap();

        let models = registry.load_models(None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model");
    }

    #[test]
    fn test_unregister_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = create_test_model("test/model", "abc123");

        registry.register_model(model).unwrap();
        assert_eq!(registry.load_models(None).unwrap().len(), 1);

        registry.unregister_model("test/model").unwrap();
        assert_eq!(registry.load_models(None).unwrap().len(), 0);
    }

    #[test]
    fn test_get_model() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = create_test_model("test/model", "abc123");

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

        let model1 = create_test_model("test/model", "abc123");
        registry.register_model(model1).unwrap();

        let mut model2 = create_test_model("test/model", "def456");
        model2.metadata.artifact.size = 2000;
        model2.metadata.artifact.path = "/tmp/test2".to_string();
        model2.created_at = "2025-01-02T00:00:00Z".to_string();
        model2.updated_at = "2025-01-02T00:00:00Z".to_string();

        registry.register_model(model2).unwrap();

        let models = registry.load_models(None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].metadata.artifact.revision, "def456");
        assert_eq!(models[0].metadata.artifact.size, 2000);
        // created_at should be preserved from model1
        assert_eq!(models[0].created_at, "2025-01-01T00:00:00Z");
        // updated_at should be from model2
        assert_eq!(models[0].updated_at, "2025-01-02T00:00:00Z");
    }

    #[test]
    fn test_remove_model_with_cache() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        // Create a fake cache directory
        let cache_dir = temp_dir.path().join("cache");
        fs::create_dir_all(&cache_dir).unwrap();
        fs::write(cache_dir.join("test.txt"), "test data").unwrap();

        let mut model = create_test_model("test/model", "abc123");
        model.metadata.artifact.path = cache_dir.to_string_lossy().to_string();

        registry.register_model(model).unwrap();
        assert_eq!(registry.load_models(None).unwrap().len(), 1);
        assert!(cache_dir.exists());

        // Delete model
        registry.remove_model("test/model").unwrap();

        // Verify model removed from registry
        assert_eq!(registry.load_models(None).unwrap().len(), 0);

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

        let model = create_test_model("test/gpt-model", "abc123def456");

        registry.register_model(model).unwrap();

        let retrieved = registry.get_model("test/gpt-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/gpt-model");
        assert_eq!(model_info.provider, "huggingface");
        assert_eq!(model_info.metadata.artifact.revision, "abc123def456");
        assert_eq!(model_info.model_series, Some("gpt2".to_string()));
        assert_eq!(model_info.metadata.context_window, Some(2048));
        assert_eq!(
            model_info
                .metadata
                .safetensors
                .as_ref()
                .unwrap()
                .get("total")
                .unwrap()
                .as_u64()
                .unwrap(),
            7_000_000_000
        );
    }

    #[test]
    fn test_inspect_model_without_spec() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let mut model = create_test_model("test/legacy-model", "legacy123");
        model.metadata.safetensors = None;
        model.metadata.context_window = None;

        registry.register_model(model).unwrap();

        let retrieved = registry.get_model("test/legacy-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/legacy-model");
        assert!(model_info.metadata.safetensors.is_none());
        assert!(model_info.metadata.context_window.is_none());
    }
}
