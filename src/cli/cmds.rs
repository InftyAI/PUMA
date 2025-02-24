use clap::{Parser, Subcommand};
use prettytable::{format, row, Table};

use crate::downloader::ollama::OllamaDownloader;

#[derive(Parser)]
#[command(name = "PUMA")]
#[command(about = "PUMA CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// List running inference services
    PS,
    /// List local models
    LS,
    /// Download a model from a model provider
    PULL(PullArgs),
    /// Create and run a new inference service from a model
    RUN,
    /// Stop one running inference service
    STOP,
    /// Remove one model
    RM,
    /// Display system-wide information
    INFO,
    /// Return detailed information about inference service
    INSPECT,
    /// Return detailed information about model
    SHOW,
    /// Returns the version of PUMA.
    VERSION,
}

#[derive(Parser)]
struct PullArgs {
    #[arg(long, value_name = "model name")]
    model: String,
    #[arg(long, value_name = "model provider", value_enum)]
    provider: Provider,
}

#[derive(Debug, Clone, clap::ValueEnum)]
pub enum Provider {
    Huggingface,
    Ollama,
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
            table.add_row(row!["llama3-8b", "ollama", "llama3.1:8b", "Running", "2d",]);
            table.add_row(row![
                "llama3-70b",
                "ollama",
                "llama3.1:70b",
                "Downloading",
                "10s",
            ]);

            table.printstd();
        }

        Commands::LS => {
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_CLEAN);
            table.add_row(row!["NAME", "PROVIDER", "REVISION", "SIZE", "CREATED"]);
            table.add_row(row![
                "deepseek-ai/DeepSeek-R1",
                "huggingface",
                "main",
                "2 weeks ago",
                "800GB"
            ]);
            table.add_row(row!["llama3.1", "ollama", "8b", "2 weeks ago", "4.9GB"]);
            table.add_row(row!["llama3.1", "ollama", "70b", "2 weeks ago", "43GB"]);
            table.add_row(row!["llama3.1", "ollama", "405b", "2 weeks ago", "243GB"]);
            table.printstd();
        }

        Commands::PULL(args) => match args.provider {
            Provider::Huggingface => {
                println!("Downloading model from Huggingface...");
            }
            Provider::Ollama => {
                let d = OllamaDownloader::new(&args.model);
                d.download_model("/Users/kerthcet/Workspaces/InftyAI/puma/tmp")
                    .await
                    .unwrap();
            }
            Provider::Modelscope => {
                println!("Downloading model from Modelscope...");
            }
        },

        Commands::RUN => {
            println!("Creating and running a new inference service from a model...");
        }

        Commands::STOP => {
            println!("Stopping one running inference service...");
        }

        Commands::RM => {
            println!("Removing one model...");
        }

        Commands::INFO => {
            println!("Displaying system-wide information...");
        }

        Commands::INSPECT => {
            println!("Returning detailed information about inference service...");
        }

        Commands::SHOW => {
            println!("Returning detailed information about model...");
        }

        Commands::VERSION => {
            println!("PUMA {}", env!("CARGO_PKG_VERSION"));
        }
    }
}
