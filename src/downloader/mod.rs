use core::fmt;

pub mod huggingface;
pub mod progress;

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
    fn download_model(
        &self,
        name: &str,
    ) -> impl std::future::Future<Output = Result<(), DownloadError>> + Send;
}

/// Provider for downloading models
#[derive(Debug, Clone, Copy, Default, clap::ValueEnum)]
pub enum Provider {
    #[default]
    #[value(alias = "hf")]
    Huggingface,
    #[value(alias = "ms")]
    Modelscope,
}

/// Download a model from the specified provider
pub async fn download_model(model_name: &str, provider: Provider) -> Result<(), DownloadError> {
    match provider {
        Provider::Huggingface => {
            let downloader = huggingface::HuggingFaceDownloader::new();
            downloader.download_model(&model_name.to_lowercase()).await
        }
        Provider::Modelscope => Err(DownloadError::ApiError(
            "Modelscope provider not yet implemented".to_string(),
        )),
    }
}
