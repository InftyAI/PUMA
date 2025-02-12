use clap::{Parser, Subcommand};
use prettytable::{format, Table};

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
    PULL,
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

// Support commands like: pull, ls, run, ps, stop, rm, info, inspect, show.
pub fn run(cli: Cli) {
    match cli.command {
        Commands::PS => {
            let mut table = Table::new();
            table.set_format(*format::consts::FORMAT_CLEAN);
            table.add_row(row!["NAME", "PROVIDER", "MODEL", "AGE"]);
            table.add_row(row![
                "deepseek-r1",
                "huggingface",
                "deepseek-ai/DeepSeek-R1",
                "8m",
            ]);
            table.add_row(row!["llama3-8b", "ollama", "llama3.1:8b", "2d",]);

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

        Commands::PULL => {
            println!("Downloading model from model provider...");
        }

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
