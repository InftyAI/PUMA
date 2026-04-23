use colored::Colorize;
use log::debug;

use hf_hub::api::tokio::{ApiBuilder, Progress};
use indicatif::{ProgressBar, ProgressStyle};

use crate::downloader::downloader::{DownloadError, Downloader};
use crate::downloader::progress::{DownloadProgressManager, FileProgress};
use crate::registry::model_registry::{ModelInfo, ModelRegistry, ModelSpec};
use crate::utils::file::{self, format_model_name};
use crate::utils::format::format_parameters;

/// Adapter to bridge HuggingFace's Progress trait with our FileProgress
#[derive(Clone)]
struct HfProgressAdapter {
    progress: FileProgress,
}

impl Progress for HfProgressAdapter {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.progress.init(size as u64);
    }

    async fn update(&mut self, size: usize) {
        self.progress.update(size as u64);
    }

    async fn finish(&mut self) {
        self.progress.finish();
    }
}

pub struct HuggingFaceDownloader;

impl HuggingFaceDownloader {
    pub fn new() -> Self {
        Self
    }

    fn estimate_parameters(config: &serde_json::Value) -> Option<String> {
        // Try to extract architecture dimensions for parameter estimation
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

impl Default for HuggingFaceDownloader {
    fn default() -> Self {
        Self::new()
    }
}

impl Downloader for HuggingFaceDownloader {
    async fn download_model(&self, name: &str) -> Result<(), DownloadError> {
        let start_time = std::time::Instant::now();

        debug!("Downloading model {} from Hugging Face...", name);

        // Use unified PUMA cache directory
        let cache_dir = file::huggingface_cache_dir();
        file::create_folder_if_not_exists(&cache_dir).map_err(|e| {
            DownloadError::IoError(format!("Failed to create cache directory: {}", e))
        })?;

        // Build API with PUMA cache directory
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir.clone())
            .build()
            .map_err(|e| {
                DownloadError::ApiError(format!("Failed to initialize Hugging Face API: {}", e))
            })?;

        // Create a simple spinner for manifest pulling
        let manifest_spinner = ProgressBar::new_spinner();
        manifest_spinner.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("pulling manifest {spinner:.white}")
                .unwrap(),
        );
        manifest_spinner.enable_steady_tick(std::time::Duration::from_millis(80));

        // Download the entire model repository using snapshot download
        let repo = api.model(name.to_string());

        // Get model info to list all files
        let model_info = repo.info().await.map_err(|e| {
            let err_str = e.to_string();
            if err_str.contains("404") || err_str.contains("not found") {
                DownloadError::ModelNotFound(format!("Model '{}' not found", name))
            } else if err_str.contains("401") || err_str.contains("403") {
                DownloadError::AuthError(format!("Authentication failed: {}", e))
            } else if err_str.contains("network") || err_str.contains("connection") {
                DownloadError::NetworkError(format!("Network error: {}", e))
            } else {
                DownloadError::ApiError(format!("Failed to fetch model info: {}", e))
            }
        })?;

        // Stop manifest spinner and print clean message
        manifest_spinner.finish_and_clear();
        println!("🐆 pulling manifest");

        debug!("Model info for {}: {:?}", name, model_info);

        // Calculate the longest filename for proper alignment
        let max_filename_len = model_info
            .siblings
            .iter()
            .map(|s| s.rfilename.len())
            .max()
            .unwrap_or(30);

        // Create progress manager
        let progress_manager = DownloadProgressManager::new(max_filename_len);

        // Calculate cache paths
        let model_cache_path = cache_dir.join(format_model_name(name));
        let sha = model_info.sha.clone();
        let snapshot_path = model_cache_path.join("snapshots").join(&sha);

        // Check if all files are already cached
        let model_totally_cached = model_info
            .siblings
            .iter()
            .all(|sibling| snapshot_path.join(&sibling.rfilename).exists());

        // Process all files in manifest order (cached files show as instantly complete)
        let mut tasks = Vec::new();

        for sibling in model_info.siblings {
            let api_clone = api.clone();
            let model_name = name.to_string();
            let filename = sibling.rfilename.clone();
            let progress_manager_clone = progress_manager.clone();
            let snapshot_path_clone = snapshot_path.clone();

            let task = tokio::spawn(async move {
                let repo = api_clone.model(model_name);

                // Check if file exists in cache
                let cached_file_path = snapshot_path_clone.join(&filename);
                if cached_file_path.exists() {
                    debug!("File {} found in cache, showing as complete", filename);

                    // Create progress bar for cached file (no speed display)
                    let mut file_progress =
                        progress_manager_clone.create_cached_file_progress(&filename);
                    let file_size = cached_file_path.metadata().map(|m| m.len()).unwrap_or(0);
                    file_progress.init(file_size);
                    file_progress.update(file_size);
                    file_progress.finish();

                    return Ok(());
                }

                // File not in cache, download with progress
                debug!("Downloading: {}", filename);
                let file_progress = progress_manager_clone.create_file_progress(&filename);
                let progress = HfProgressAdapter {
                    progress: file_progress,
                };

                repo.download_with_progress(&filename, progress)
                    .await
                    .map_err(|e| {
                        DownloadError::NetworkError(format!(
                            "Failed to download {}: {}",
                            filename, e
                        ))
                    })?;

                Ok(())
            });

            tasks.push(task);
        }

        // Give tasks a moment to start and create their progress bars
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

        // Show spinner at the bottom after all progress bars are created (only if not fully cached)
        let spinner = if !model_totally_cached {
            Some(progress_manager.create_spinner())
        } else {
            None
        };

        // Wait for all downloads to complete
        for task in tasks {
            task.await
                .map_err(|e| DownloadError::ApiError(format!("Task join error: {}", e)))??;
        }

        // Finish spinner after downloads complete
        if let Some(spinner) = &spinner {
            spinner.finish_and_clear();
        }

        let elapsed_time = start_time.elapsed();

        // Get accumulated size from downloads
        let downloaded_size = progress_manager.total_downloaded_bytes();
        let model_cache_path = cache_dir.join(format_model_name(name));

        // Register the model only if not totally cached
        if !model_totally_cached {
            // Extract architecture info from config.json
            let config_path = snapshot_path.join("config.json");
            let spec = if config_path.exists() {
                match std::fs::read_to_string(&config_path) {
                    Ok(config_content) => {
                        match serde_json::from_str::<serde_json::Value>(&config_content) {
                            Ok(config) => {
                                let model_type = config
                                    .get("model_type")
                                    .and_then(|v| v.as_str())
                                    .map(|s| s.to_string());

                                let architectures = config
                                    .get("architectures")
                                    .and_then(|v| v.as_array())
                                    .map(|arr| {
                                        arr.iter()
                                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                                            .collect::<Vec<String>>()
                                    })
                                    .filter(|v| !v.is_empty());

                                let context_window = config
                                    .get("max_position_embeddings")
                                    .or_else(|| config.get("n_ctx"))
                                    .or_else(|| config.get("n_positions"))
                                    .and_then(|v| v.as_u64())
                                    .map(|v| v as u32);

                                let parameters = Self::estimate_parameters(&config);

                                if model_type.is_some()
                                    || architectures.is_some()
                                    || context_window.is_some()
                                    || parameters.is_some()
                                {
                                    Some(ModelSpec {
                                        model_type,
                                        architectures,
                                        context_window,
                                        parameters,
                                    })
                                } else {
                                    None
                                }
                            }
                            Err(_) => None,
                        }
                    }
                    Err(_) => None,
                }
            } else {
                None
            };

            let model_info_record = ModelInfo {
                name: name.to_string(),
                provider: "huggingface".to_string(),
                revision: sha,
                size: downloaded_size,
                modified_at: chrono::Local::now().to_rfc3339(),
                cache_path: model_cache_path.to_string_lossy().to_string(),
                spec,
            };

            let registry = ModelRegistry::new(None);
            registry
                .register_model(model_info_record)
                .map_err(|e| DownloadError::ApiError(format!("Failed to register model: {}", e)))?;
        }

        // Print success message
        println!(
            "{} {} {} {} {:.2?}",
            "✓".green().bold(),
            "Successfully downloaded model".bright_white(),
            name.cyan().bold(),
            "in".bright_white(),
            elapsed_time
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_download_model_invalid() {
        let downloader = HuggingFaceDownloader::new();
        let result = downloader
            .download_model("invalid-model-that-does-not-exist-12345")
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_download_real_tiny_model() {
        let downloader = HuggingFaceDownloader::new();
        // Use HF's official tiny test model (only a few KB)
        let result = downloader.download_model("InftyAI/tiny-random-gpt2").await;
        assert!(
            result.is_ok(),
            "Failed to download tiny model: {:?}",
            result
        );

        // Cleanup: remove the downloaded files from PUMA cache
        let cache_dir = file::huggingface_cache_dir().join("models--InftyAI--tiny-random-gpt2");

        if cache_dir.exists() {
            let _ = std::fs::remove_dir_all(&cache_dir);
        }
    }
}
