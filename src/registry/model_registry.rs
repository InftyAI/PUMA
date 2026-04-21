use colored::Colorize;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::utils::file;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub revision: String,
    pub size: u64,
    pub modified_at: String,
    pub cache_path: String,
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
        };

        registry.register_model(model1).unwrap();

        let model2 = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "def456".to_string(),
            size: 2000,
            modified_at: "2025-01-02T00:00:00Z".to_string(),
            cache_path: "/tmp/test2".to_string(),
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
}
