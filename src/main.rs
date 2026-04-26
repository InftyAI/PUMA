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
    // set hf_hub to warn to disable the info logs.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info,hf_hub=warn"))
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
