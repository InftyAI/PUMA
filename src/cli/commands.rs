use clap::{Parser, Subcommand};
use prettytable::{format, row, Table};
use std::path::PathBuf;

use crate::downloader::downloader::Downloader;
use crate::downloader::huggingface::HuggingFaceDownloader;
use crate::registry::model_registry::ModelRegistry;

#[derive(Parser)]
#[command(name = "PUMA")]
#[command(about = "PUMA CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List running models
    PS,
    /// List local models
    LS,
    /// Download a model from a model provider
    PULL(PullArgs),
    /// Create and run a new model
    RUN,
    /// Stop one running model
    STOP,
    /// Remove one model
    RM,
    /// Display system-wide information
    INFO,
    /// Return detailed information about a model
    INSPECT,
    /// Returns the version of PUMA.
    VERSION,
}

#[derive(Parser)]
struct PullArgs {
    #[arg(short = 'm', long, value_name = "model name")]
    model: String,
    #[arg(
        short = 'p',
        long,
        value_name = "model provider",
        value_enum,
        default_value = "huggingface"
    )]
    provider: Provider,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Provider {
    Huggingface,
    Modelscope,
}

impl Default for Provider {
    fn default() -> Self {
        Provider::Huggingface
    }
}

// Support commands like: pull, ls, run, ps, stop, rm, info, inspect, show.
pub async fn run(cli: Cli) {
    match cli.command {
        Commands::PS => {
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_CLEAN);
            table.add_row(row!["NAME", "PROVIDER", "MODEL", "STATUS", "AGE"]);
            table.add_row(row![
                "deepseek-r1",
                "huggingface",
                "deepseek-ai/DeepSeek-R1",
                "Running",
                "8m",
            ]);

            table.printstd();
        }

        Commands::LS => {
            let models = ModelRegistry::load_models().unwrap_or_default();

            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_CLEAN);
            table.add_row(row!["MODEL", "PROVIDER", "REVISION", "SIZE", "CREATED"]);

            for model in models {
                let size_gb = if model.size > 1_000_000_000 {
                    format!("{:.2} GB", model.size as f64 / 1_000_000_000.0)
                } else if model.size > 1_000_000 {
                    format!("{:.2} MB", model.size as f64 / 1_000_000.0)
                } else if model.size > 1_000 {
                    format!("{:.2} KB", model.size as f64 / 1_000.0)
                } else {
                    format!("{} B", model.size)
                };

                let revision_short = if model.revision.len() > 8 {
                    &model.revision[..8]
                } else {
                    &model.revision
                };

                table.add_row(row![
                    model.name,
                    model.provider,
                    revision_short,
                    size_gb,
                    model.created_at
                ]);
            }

            table.printstd();
        }

        Commands::PULL(args) => match args.provider {
            Provider::Huggingface => {
                let downloader = HuggingFaceDownloader::new();
                match downloader.download_model(&args.model).await {
                    Ok(_) => {}
                    Err(e) => {
                        eprintln!("Error downloading model: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Provider::Modelscope => {
                println!("Downloading model from Modelscope...");
            }
        },

        Commands::RUN => {
            println!("Creating and running a new model...");
        }

        Commands::STOP => {
            println!("Stopping one running model...");
        }

        Commands::RM => {
            println!("Removing one model...");
        }

        Commands::INFO => {
            println!("Displaying system-wide information...");
        }

        Commands::INSPECT => {
            println!("Returning detailed information about model...");
        }

        Commands::VERSION => {
            println!("PUMA {}", env!("CARGO_PKG_VERSION"));
        }
    }
}
