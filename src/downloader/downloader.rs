use core::fmt;
use std::path::PathBuf;

#[derive(Debug)]
pub enum DownloadError {
    NetworkError(String),
    AuthError(String),
    ModelNotFound(String),
    IoError(String),
    ApiError(String),
}

impl std::error::Error for DownloadError {}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DownloadError::NetworkError(e) => write!(f, "Network error: {}", e),
            DownloadError::AuthError(e) => write!(f, "Authentication error: {}", e),
            DownloadError::ModelNotFound(e) => write!(f, "Model not found: {}", e),
            DownloadError::IoError(e) => write!(f, "IO error: {}", e),
            DownloadError::ApiError(e) => write!(f, "API error: {}", e),
        }
    }
}

pub trait Downloader {
    async fn download_model(&self, name: &str, cache_dir: &PathBuf) -> Result<(), DownloadError>;
}
