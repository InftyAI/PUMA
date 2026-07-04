use crate::registry::model_registry::ModelInfo;
use std::io;

use std::collections::HashMap;

/// Trait for model storage backends
pub trait ModelStorage: Send + Sync {
    /// Load models from storage with optional filtering by column values (e.g., author=InftyAI, license=mit)
    fn load_models(
        &self,
        filters: Option<&HashMap<String, String>>,
    ) -> Result<Vec<ModelInfo>, io::Error>;

    /// Register (insert or update) a single model
    fn register_model(&self, model: ModelInfo) -> Result<(), io::Error>;

    /// Unregister (delete) a model by name
    fn unregister_model(&self, name: &str) -> Result<(), io::Error>;

    /// Get a single model by name
    fn get_model(&self, name: &str) -> Result<Option<ModelInfo>, io::Error>;
}
