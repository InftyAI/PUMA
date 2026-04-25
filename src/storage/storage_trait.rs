use crate::registry::model_registry::ModelInfo;
use std::io;

/// Trait for model storage backends
pub trait ModelStorage {
    /// Load all models from storage
    fn load_models(&self) -> Result<Vec<ModelInfo>, io::Error>;

    /// Register (insert or update) a single model
    fn register_model(&self, model: ModelInfo) -> Result<(), io::Error>;

    /// Unregister (delete) a model by name
    fn unregister_model(&self, name: &str) -> Result<(), io::Error>;

    /// Get a single model by name
    fn get_model(&self, name: &str) -> Result<Option<ModelInfo>, io::Error>;
}
