use colored::Colorize;
use log::{debug, info};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use hf_hub::api::tokio::{ApiBuilder, Progress};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::downloader::downloader::{DownloadError, Downloader};
use crate::registry::model_registry::{ModelInfo, ModelRegistry};
use crate::util::file;

#[derive(Clone)]
struct FileProgressBar {
    pb: ProgressBar,
    total_size: Arc<AtomicU64>,
}

impl Progress for FileProgressBar {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.pb.set_length(size as u64);
        self.pb.reset();
        self.pb.tick(); // Force render with correct size
        self.total_size.fetch_add(size as u64, Ordering::Relaxed);
    }

    async fn update(&mut self, size: usize) {
        self.pb.inc(size as u64);
    }

    async fn finish(&mut self) {}
}

pub struct HuggingFaceDownloader;

impl HuggingFaceDownloader {
    pub fn new() -> Self {
        Self
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

        info!("Downloading model {} from Hugging Face...", name);

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

        debug!("Model info for {}: {:?}", name, model_info);

        // Create multi-progress for parallel downloads
        let multi_progress = Arc::new(MultiProgress::new());

        // Calculate the longest filename for proper alignment
        let max_filename_len = model_info
            .siblings
            .iter()
            .map(|s| s.rfilename.len())
            .max()
            .unwrap_or(30);

        // Progress bar style with block characters (chart-like, not #)
        let template = format!(
            "{{msg:<{width}}} [{{elapsed_precise}}] {{bar:60.white}} {{bytes}}/{{total_bytes}}",
            width = max_filename_len
        );
        let style = ProgressStyle::default_bar()
            .template(&template)
            .unwrap()
            .progress_chars("▇▆▅▄▃▂▁ ");

        // Download all files in parallel
        let mut tasks = Vec::new();
        let sha = model_info.sha.clone();
        let total_size = Arc::new(AtomicU64::new(0));

        for sibling in model_info.siblings {
            let api_clone = api.clone();
            let model_name = name.to_string();
            let filename = sibling.rfilename.clone();
            let total_size_clone = Arc::clone(&total_size);

            let pb = multi_progress.add(ProgressBar::hidden());
            pb.set_style(style.clone());
            pb.set_message(filename.clone());

            let task = tokio::spawn(async move {
                debug!("Downloading: {}", filename);

                let repo = api_clone.model(model_name);
                let progress = FileProgressBar {
                    pb: pb.clone(),
                    total_size: total_size_clone,
                };

                let result = repo.download_with_progress(&filename, progress).await;

                match &result {
                    Ok(_) => {
                        pb.finish();
                    }
                    Err(_) => {
                        pb.abandon();
                    }
                }

                result.map_err(|e| {
                    DownloadError::NetworkError(format!("Failed to download {}: {}", filename, e))
                })
            });

            tasks.push(task);
        }

        // Wait for all downloads to complete
        for task in tasks {
            task.await
                .map_err(|e| DownloadError::ApiError(format!("Task join error: {}", e)))??;
        }

        let elapsed_time = start_time.elapsed();

        // Get accumulated size from downloads
        let downloaded_size = total_size.load(Ordering::Relaxed);
        let model_cache_path = cache_dir.join(format!("models--{}", name.replace("/", "--")));

        // Register the model
        let model_info_record = ModelInfo {
            name: name.to_string(),
            provider: "huggingface".to_string(),
            revision: sha,
            size: downloaded_size,
            created_at: chrono::Local::now().to_rfc3339(),
            cache_path: model_cache_path.to_string_lossy().to_string(),
        };

        let registry = ModelRegistry::new(None);
        registry
            .register_model(model_info_record)
            .map_err(|e| DownloadError::ApiError(format!("Failed to register model: {}", e)))?;

        println!(
            "\n{} {} {} {} {:.2?}",
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
