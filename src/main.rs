mod api;
mod backend;
mod cli;
mod downloader;
mod registry;
mod storage;
mod system;
mod utils;

use clap::Parser;
use tokio::runtime::Builder;

use crate::cli::commands::{run, Cli};
use crate::utils::file;

fn main() {
    // Setup tracing subscriber for tower-http TraceLayer
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,hf_hub=warn,tower_http=info,rusqlite_migration=warn".into()),
        )
        .init();

    // Create the root folder if it doesn't exist.
    file::create_folder_if_not_exists(&file::root_home()).unwrap();

    let cli = Cli::parse();

    let runtime = Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(run(cli));
}
