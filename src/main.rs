mod cli;
mod downloader;
mod registry;
mod system;
mod util;

use clap::Parser;
use tokio::runtime::Builder;

use crate::cli::commands::{run, Cli};
use crate::util::file;

fn main() {
    // Initialize logger.
    env_logger::Builder::from_env(env_logger::Env::default()).init();

    // Create the root folder if it doesn't exist.
    file::create_folder_if_not_exists(&file::root_home()).unwrap();

    let runtime = Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let cli = Cli::parse();
    runtime.block_on(run(cli));
}
