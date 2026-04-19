use colored::Colorize;
use log::{debug, info};
use std::path::PathBuf;
use std::sync::Arc;

use hf_hub::api::tokio::{ApiBuilder, Progress};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

use crate::downloader::downloader::{DownloadError, Downloader};

#[derive(Clone)]
struct FileProgressBar {
    pb: ProgressBar,
}

impl Progress for FileProgressBar {
    async fn init(&mut self, size: usize, _filename: &str) {
        self.pb.set_length(size as u64);
        self.pb.reset();
        self.pb.tick(); // Force render with correct size
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
    async fn download_model(&self, name: &str, cache_dir: &PathBuf) -> Result<(), DownloadError> {
        let start_time = std::time::Instant::now();

        info!("Downloading model {} from Hugging Face...", name);

        // Build API without default progress bars (we have our own implementation)
        let api = if cache_dir.as_os_str().is_empty() {
            // Use default HF cache
            ApiBuilder::new().build().map_err(|e| {
                DownloadError::ApiError(format!("Failed to initialize Hugging Face API: {}", e))
            })?
        } else {
            // Use custom cache directory
            ApiBuilder::new()
                .with_cache_dir(cache_dir.clone())
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

        // Create multi-progress for parallel downloads
        let multi_progress = Arc::new(MultiProgress::new());

        // Progress bar style with block characters (chart-like, not #)
        let style = ProgressStyle::default_bar()
            .template("{msg:<30} [{elapsed_precise}] {bar:40.white} {bytes}/{total_bytes}\n")
            .unwrap()
            .progress_chars("█▉▊▋▌▍▎▏░");

        // Download all files in parallel
        let mut tasks = Vec::new();

        for sibling in model_info.siblings {
            let api_clone = api.clone();
            let model_name = name.to_string();
            let filename = sibling.rfilename.clone();

            let pb = multi_progress.add(ProgressBar::hidden());
            pb.set_style(style.clone());
            pb.set_message(filename.clone());

            let task = tokio::spawn(async move {
                debug!("Downloading: {}", filename);

                let repo = api_clone.model(model_name);
                let progress = FileProgressBar { pb: pb.clone() };

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
