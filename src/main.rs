mod cli;
mod downloader;
mod util;

use clap::Parser;
use env_logger;
use tokio::runtime::Builder;

use crate::cli::cmds::{run, Cli};
use crate::util::file;

fn main() {
    env_logger::init();

    // create the root folder.
    file::create_folder_if_not_exists(&file::root_home()).unwrap();

    let runtime = Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let cli = Cli::parse();
    runtime.block_on(run(cli));
}
