use log::{debug, info};
use std::path::PathBuf;

use hf_hub::api::tokio::ApiBuilder;

use crate::downloader::downloader::{DownloadError, Downloader};

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
    async fn download_model(&self, name: &str, cache_dir: &PathBuf) -> Result<(), DownloadError> {
        info!("Downloading model {} from Hugging Face...", name);

        let start_time = std::time::Instant::now();

        // Build API - disable default progress bars (we have our own implementation)
        let api = if cache_dir.as_os_str().is_empty() {
            // Use default HF cache with progress disabled
            ApiBuilder::new()
                .with_progress(false)
                .build()
                .map_err(|e| {
                    DownloadError::ApiError(format!("Failed to initialize Hugging Face API: {}", e))
                })?
        } else {
            // Use custom cache directory with progress disabled
            ApiBuilder::new()
                .with_cache_dir(cache_dir.clone())
                .with_progress(false)
                .build()
                .map_err(|e| {
                    DownloadError::ApiError(format!(
                        "Failed to initialize Hugging Face API with custom cache: {}",
                        e
                    ))
                })?
        };

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

        // Download all files in the repository
        for sibling in &model_info.siblings {
            debug!("Downloading: {}", sibling.rfilename);
            repo.get(&sibling.rfilename).await.map_err(|e| {
                DownloadError::NetworkError(format!(
                    "Failed to download {}: {}",
                    sibling.rfilename, e
                ))
            })?;
        }

        let elapsed_time = start_time.elapsed();
        info!(
            "Download model {} successfully with {:.2?}",
            name, elapsed_time
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
            .download_model("invalid-model-that-does-not-exist-12345", &PathBuf::new())
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_download_real_tiny_model() {
        let downloader = HuggingFaceDownloader::new();
        // Use HF's official tiny test model (only a few KB)
        let result = downloader
            .download_model("InftyAI/tiny-random-gpt2", &PathBuf::new())
            .await;
        assert!(
            result.is_ok(),
            "Failed to download tiny model: {:?}",
            result
        );

        // Cleanup: remove the downloaded files from the default HF cache (~/.cache/huggingface/hub)
        if let Some(home_dir) = dirs::home_dir() {
            let cache_dir = home_dir
                .join(".cache")
                .join("huggingface")
                .join("hub")
                .join("models--InftyAI--tiny-random-gpt2");

            if cache_dir.exists() {
                let _ = std::fs::remove_dir_all(&cache_dir);
            }
        }
    }

    #[tokio::test]
    async fn test_download_with_custom_cache() {
        use std::env;
        use std::fs;

        let downloader = HuggingFaceDownloader::new();
        let temp_dir = env::temp_dir().join("puma_test_cache");

        print!("Using temporary cache directory: {:?}\n", temp_dir);

        // Create the directory first
        fs::create_dir_all(&temp_dir).unwrap();

        let result = downloader
            .download_model("InftyAI/tiny-random-gpt2", &temp_dir)
            .await;

        assert!(
            result.is_ok(),
            "Failed to download with custom cache: {:?}",
            result
        );

        // Cleanup
        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
