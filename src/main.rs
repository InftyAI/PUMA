#![allow(dead_code)]

mod api;
mod backend;
mod block_manager;
mod cli;
mod downloader;
mod fsm;
mod registry;
mod scheduler;
mod storage;
mod system;
mod utils;

use clap::Parser;
use tokio::runtime::Builder;

use crate::cli::commands::{run, Cli};
use crate::utils::file;

fn main() {
    let cli = Cli::parse();

    // Setup tracing subscriber for tower-http TraceLayer.
    //
    // An explicit RUST_LOG always wins. Otherwise only the long-running server
    // logs by default; one-shot/interactive CLI commands stay quiet so their
    // output isn't buried under per-request INFO lines.
    let default_filter = if cli.wants_default_logging() {
        "info,hf_hub=warn,tower_http=info,rusqlite_migration=warn"
    } else {
        "off"
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| default_filter.into()),
        )
        .init();

    // Create the root folder if it doesn't exist.
    file::create_folder_if_not_exists(&file::root_home()).unwrap();

    let runtime = Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(run(cli));
}
