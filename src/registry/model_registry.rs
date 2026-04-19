use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::util::file;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ModelInfo {
    pub name: String,
    pub provider: String,
    pub revision: String,
    pub size: u64,
    pub created_at: String,
    pub cache_path: String,
}

pub struct ModelRegistry;

impl ModelRegistry {
    fn registry_file() -> PathBuf {
        file::root_home().join("models.json")
    }

    pub fn load_models() -> Result<Vec<ModelInfo>, std::io::Error> {
        let registry_file = Self::registry_file();

        if !registry_file.exists() {
            return Ok(Vec::new());
        }

        let contents = fs::read_to_string(registry_file)?;
        let models: Vec<ModelInfo> = serde_json::from_str(&contents)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        Ok(models)
    }

    pub fn save_models(models: &[ModelInfo]) -> Result<(), std::io::Error> {
        let registry_file = Self::registry_file();
        let json = serde_json::to_string_pretty(models)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        fs::write(registry_file, json)?;
        Ok(())
    }

    pub fn add_model(model: ModelInfo) -> Result<(), std::io::Error> {
        let mut models = Self::load_models()?;

        // Remove existing model with same name if exists
        models.retain(|m| m.name != model.name);

        models.push(model);
        Self::save_models(&models)?;

        Ok(())
    }

    pub fn remove_model(name: &str) -> Result<(), std::io::Error> {
        let mut models = Self::load_models()?;
        models.retain(|m| m.name != name);
        Self::save_models(&models)?;

        Ok(())
    }

    pub fn get_model(name: &str) -> Result<Option<ModelInfo>, std::io::Error> {
        let models = Self::load_models()?;
        Ok(models.into_iter().find(|m| m.name == name))
    }
}
