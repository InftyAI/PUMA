use std::error::Error;
use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::sync::Arc;

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use log::{debug, error, info};
use reqwest::Client;
use tokio::sync::Semaphore;

const MAX_CHUNK_CONCURRENCY: usize = 100;
const CHUNK_SIZE: usize = 1000 * 1000 * 10; // 10MB
const MAX_RETRIES: usize = 5;

pub async fn download_file(
    client: Arc<Client>,
    url: String,
    content_length: u64,
    filename: String,
    output_path: String,
    m: Arc<MultiProgress>,
    sty: ProgressStyle,
) -> Result<(), Box<dyn Error>> {
    debug!("Start to download file {} to {}", filename, output_path);

    let mut tasks = Vec::new();
    let mut start = 0;
    let mut end = CHUNK_SIZE as u64 - 1;
    end = end.min(content_length - 1);

    let semaphore = Arc::new(Semaphore::new(MAX_CHUNK_CONCURRENCY));
    // TODO: verify the file not downloaded yet.
    let file = Arc::new(File::create(&output_path)?);
    let arc_url = Arc::new(url);

    let pb = m.add(ProgressBar::new(content_length).with_style(sty));
    pb.set_message(filename.clone());
    let arc_pb = Arc::new(pb);

    while start < content_length {
        let client = Arc::clone(&client);
        let semaphore = Arc::clone(&semaphore);
        let file = Arc::clone(&file);
        let url = Arc::clone(&arc_url);
        let pb = Arc::clone(&arc_pb);

        let fname = filename.clone();

        let task = tokio::spawn(async move {
            let _permit = semaphore.acquire().await.unwrap();
            let _ = download_chunk_with_retries(
                client,
                file,
                fname,
                url,
                start.clone(),
                end.clone(),
                MAX_RETRIES,
            )
            .await;

            pb.inc(end - start + 1);
        });
        tasks.push(task);

        start = end + 1;
        end = (end + CHUNK_SIZE as u64).min(content_length - 1);
    }

    for task in tasks {
        let _ = task.await;
        // TODO: write to a file about the chunk info.
    }

    arc_pb.finish();
    Ok(())
}

async fn download_chunk_with_retries(
    client: Arc<Client>,
    file: Arc<File>,
    filename: String,
    url: Arc<String>,
    start: u64,
    end: u64,
    retries: usize,
) -> Result<(), Box<dyn Error>> {
    debug!(
        "Start to download file chunk {} from {} to {}",
        filename, start, end,
    );

    let mut retries = retries;
    loop {
        match download_chunk(&client, &file, &url, start, end).await {
            Ok(_) => {
                debug!(
                    "Download chunk {} from {} to {} successfully.",
                    url, start, end
                );
                break;
            }
            // TODO: retry only when http error.
            Err(e) => {
                if retries == 0 {
                    error!("Reach the maximum retries {}, return.", MAX_RETRIES);
                    return Err(e);
                }
                retries -= 1;
                // TODO: sleep for a while
                info!(
                    "Failed to download chunk {}, retrying {}...",
                    e.to_string(),
                    retries
                );
            }
        }
    }
    Ok(())
}

async fn download_chunk(
    client: &Client,
    file: &File,
    url: &str,
    start: u64,
    end: u64,
) -> Result<(), Box<dyn Error>> {
    let response = client
        .get(url)
        .header("Range", format!("bytes={}-{}", start, end))
        .send()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    let chunk = response
        .bytes()
        .await
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;

    file.write_all_at(&chunk, start)?;
    Ok(())
}
