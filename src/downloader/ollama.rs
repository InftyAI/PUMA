use std::sync::Arc;

use log::{error, info};
use reqwest::Client;
use serde::Deserialize;
use tokio;
use tokio::sync::Semaphore;

use crate::downloader::downloader::DownloadError;
use crate::util::request;

const MAX_FILE_CONCURRENCY: usize = 5;

pub struct OllamaDownloader {
    model_name: String,
}

#[derive(Deserialize, Debug)]
struct OllamaResponse {
    #[serde(rename = "schemaVersion")]
    schema_version: u32,
    layers: Vec<Layer>,
}

#[derive(Deserialize, Debug)]
struct Layer {
    #[serde(rename = "mediaType")]
    media_type: String,
    size: u64,
    digest: String,
}

impl Layer {
    fn path(&self) -> &str {
        return self.media_type.split(".").last().unwrap();
    }
}

impl OllamaDownloader {
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
        }
    }

    pub async fn download_model(&self, path: &str) -> Result<(), DownloadError> {
        info!("Downloading model from ollama provider...");

        let start_time = std::time::Instant::now();

        let splits: Vec<&str> = self.model_name.split(":").collect();
        let [model_name, tag] = [splits[0], splits[1]];
        let client = Arc::new(Client::new());

        let manifest_url = format!(
            "https://registry.ollama.ai/v2/library/{}/manifests/{}",
            model_name, tag
        );
        let resp = query_manifest(&client, &manifest_url).await?;

        let mut tasks = Vec::new();
        let semaphore = Arc::new(Semaphore::new(MAX_FILE_CONCURRENCY));

        for layer in resp.layers {
            let layer_url = format!(
                "https://registry.ollama.ai/v2/library/{}/blobs/{}",
                model_name, layer.digest
            );

            let client: Arc<Client> = Arc::clone(&client);
            let semaphore: Arc<Semaphore> = Arc::clone(&semaphore);

            let full_path = format!("{}/{}", path, layer.path());
            let size = layer.size;

            let task = tokio::spawn(async move {
                let _permit = semaphore.acquire().await.unwrap();
                let _ = request::download_file(client, layer_url.clone(), size, full_path)
                    .await
                    .map_err(|e| {
                        error!(
                            "Failed to download file from {}: {}",
                            layer_url,
                            e.to_string()
                        );
                    });
            });
            tasks.push(task);
        }

        for task in tasks {
            let _ = task.await;
        }

        let elapsed_time = start_time.elapsed();
        info!(
            "Download model {} takes {:.2?}.",
            self.model_name, elapsed_time,
        );
        Ok(())
    }
}

async fn query_manifest(client: &Client, url: &str) -> Result<OllamaResponse, DownloadError> {
    let response = client.get(url).send().await.map_err(|e| {
        DownloadError::RequestError(format!(
            "failed to query the manifest, error: {}",
            e.to_string()
        ))
    })?;

    if response.status() == 200 {
        let resp: OllamaResponse = response.json().await.map_err(|e| {
            DownloadError::ParseError(format!(
                "failed to parse the manifest, error: {}",
                e.to_string()
            ))
        })?;

        return Ok(resp);
    }

    Err(DownloadError::RequestError(format!(
        "HTTP Code not right: {}",
        response.status()
    )))
}
