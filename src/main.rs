#[macro_use]
extern crate prettytable;
use clap::Parser;

mod cli;
use crate::cli::cmds::{run, Cli};

fn main() {
    let cli = Cli::parse();
    run(cli);
}
