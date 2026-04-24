use clap::{Parser, Subcommand};
use prettytable::{format, row, Table};

use crate::downloader::downloader::Downloader;
use crate::downloader::huggingface::HuggingFaceDownloader;
use crate::registry::model_registry::ModelRegistry;
use crate::system::system_info::SystemInfo;
use crate::utils::format::{format_size_decimal, format_time_ago};

#[derive(Parser)]
#[command(name = "PUMA")]
#[command(about = "PUMA CLI")]
pub struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
#[allow(clippy::upper_case_acronyms)]
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
    RM(RmArgs),
    /// Display system-wide information
    INFO,
    /// Return detailed information about a model
    INSPECT(InspectArgs),
    /// Returns the version of PUMA.
    VERSION,
}

#[derive(Parser)]
struct PullArgs {
    /// Model name to download (e.g., InftyAI/tiny-random-gpt2)
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

#[derive(Parser)]
struct RmArgs {
    /// Model name to remove (e.g., InftyAI/tiny-random-gpt2)
    model: String,
}

#[derive(Parser)]
struct InspectArgs {
    /// Model name to inspect (e.g., InftyAI/tiny-random-gpt2)
    model: String,
}

#[derive(Debug, Clone, Default, clap::ValueEnum)]
pub enum Provider {
    #[default]
    Huggingface,
    Modelscope,
}

// Support commands like: pull, ls, run, ps, stop, rm, info, inspect, show.
pub async fn run(cli: Cli) {
    match cli.command {
        Commands::PS => {
            let mut table = Table::new();
            table.set_format(
                format::FormatBuilder::new()
                    .column_separator(' ')
                    .padding(0, 1)
                    .build(),
            );
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
            let registry = ModelRegistry::new(None);
            let models = registry.load_models().unwrap_or_default();

            let mut table = Table::new();
            table.set_format(
                format::FormatBuilder::new()
                    .column_separator(' ')
                    .padding(0, 1)
                    .build(),
            );
            table.add_row(row!["MODEL", "PROVIDER", "REVISION", "SIZE", "AGE"]);
            for model in models {
                let size_str = format_size_decimal(model.size);

                let revision_short = if model.revision.len() > 8 {
                    &model.revision[..8]
                } else {
                    &model.revision
                };

                let created_str = format_time_ago(&model.created_at);

                table.add_row(row![
                    model.name,
                    model.provider,
                    revision_short,
                    size_str,
                    created_str
                ]);
            }

            table.printstd();
        }

        Commands::PULL(args) => match args.provider {
            Provider::Huggingface => {
                let downloader = HuggingFaceDownloader::new();
                if let Err(e) = downloader.download_model(&args.model).await {
                    eprintln!("❌ Error downloading model: {}", e);
                    std::process::exit(1);
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

        Commands::RM(args) => {
            let registry = ModelRegistry::new(None);

            // Check if model exists first
            match registry.get_model(&args.model) {
                Ok(Some(_)) => {
                    // Delete model (cache + registry)
                    if let Err(e) = registry.remove_model(&args.model) {
                        eprintln!("Failed to remove model: {}", e);
                        std::process::exit(1);
                    }
                }
                Ok(None) => {
                    eprintln!("Model not found: {}", args.model);
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Failed to load registry: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::INFO => {
            let info = SystemInfo::collect();
            info.display();
        }

        Commands::INSPECT(args) => {
            let registry = ModelRegistry::new(None);

            match registry.get_model(&args.model) {
                Ok(Some(model)) => {
                    println!("Name: {}", model.name);
                    println!("Kind: Model");
                    println!("Metadata:");
                    println!("  Created:        {}", format_time_ago(&model.created_at));
                    println!("  Updated:        {}", format_time_ago(&model.updated_at));

                    println!("Spec:");
                    // Architecture section (only if info is available)
                    if let Some(arch) = &model.arch {
                        println!("  Architecture:");
                        if let Some(model_type) = &arch.model_type {
                            println!("    Type:           {}", model_type);
                        }
                        if let Some(classes) = &arch.classes {
                            println!("    Classes:        {}", classes.join(", "));
                        }
                        if let Some(parameters) = &arch.parameters {
                            println!("    Parameters:     {}", parameters);
                        }
                        if let Some(context_window) = arch.context_window {
                            println!("    Context Window: {}", context_window);
                        }
                    }
                    // Registry section
                    println!("  Registry:");
                    println!("    Provider:       {}", model.provider);
                    println!("    Revision:       {}", model.revision);
                    println!("    Size:           {}", format_size_decimal(model.size));
                    println!("    Cache Path:     {}", model.cache_path);
                }
                Ok(None) => {
                    eprintln!("Model not found: {}", args.model);
                    std::process::exit(1);
                }
                Err(e) => {
                    eprintln!("Failed to load registry: {}", e);
                    std::process::exit(1);
                }
            }
        }

        Commands::VERSION => {
            println!("PUMA {}", env!("CARGO_PKG_VERSION"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::model_registry::{ModelArchitecture, ModelInfo};
    use tempfile::TempDir;

    #[test]
    fn test_ls_command_empty() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let models = registry.load_models().unwrap_or_default();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_ls_command_with_models() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123def456".to_string(),
            size: 1_000_000,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();

        let models = registry.load_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model");
        assert_eq!(models[0].provider, "huggingface");
    }

    #[test]
    fn test_inspect_command_with_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/gpt-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123def456".to_string(),
            size: 7_000_000_000,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-02T00:00:00Z".to_string(),
            cache_path: "/tmp/test/gpt".to_string(),
            arch: Some(ModelArchitecture {
                model_type: Some("gpt2".to_string()),
                classes: Some(vec!["GPT2LMHeadModel".to_string()]),
                context_window: Some(2048),
                parameters: Some("7.00B".to_string()),
            }),
        };

        registry.register_model(model.clone()).unwrap();

        let retrieved = registry.get_model("test/gpt-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/gpt-model");
        assert_eq!(model_info.created_at, "2025-01-01T00:00:00Z");
        assert_eq!(model_info.updated_at, "2025-01-02T00:00:00Z");

        let arch = model_info.arch.unwrap();
        assert_eq!(arch.model_type, Some("gpt2".to_string()));
        assert_eq!(arch.classes, Some(vec!["GPT2LMHeadModel".to_string()]));
        assert_eq!(arch.context_window, Some(2048));
        assert_eq!(arch.parameters, Some("7.00B".to_string()));
    }

    #[test]
    fn test_inspect_command_without_architecture() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/simple-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "xyz789".to_string(),
            size: 500_000,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test/simple".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();

        let retrieved = registry.get_model("test/simple-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/simple-model");
        assert!(model_info.arch.is_none());
    }

    #[test]
    fn test_rm_command() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/remove-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "abc123".to_string(),
            size: 1000,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test/remove".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();
        assert!(registry.get_model("test/remove-model").unwrap().is_some());

        // Simulate RM command
        let result = registry.get_model("test/remove-model");
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_rm_command_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let result = registry.get_model("nonexistent/model");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_revision_truncation() {
        let long_revision = "abc123def456ghi789jkl012";
        let short = if long_revision.len() > 8 {
            &long_revision[..8]
        } else {
            long_revision
        };
        assert_eq!(short, "abc123de");

        let short_revision = "abc123";
        let short = if short_revision.len() > 8 {
            &short_revision[..8]
        } else {
            short_revision
        };
        assert_eq!(short, "abc123");
    }

    #[test]
    fn test_metadata_timestamps_differ() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = ModelInfo {
            name: "test/updated-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "v1".to_string(),
            size: 1000,
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(model).unwrap();

        // Update the model
        let updated_model = ModelInfo {
            name: "test/updated-model".to_string(),
            provider: "huggingface".to_string(),
            revision: "v2".to_string(),
            size: 2000,
            created_at: "2025-01-05T00:00:00Z".to_string(),
            updated_at: "2025-01-05T00:00:00Z".to_string(),
            cache_path: "/tmp/test".to_string(),
            arch: None,
        };

        registry.register_model(updated_model).unwrap();

        let result = registry.get_model("test/updated-model").unwrap().unwrap();
        // created_at should remain the same
        assert_eq!(result.created_at, "2025-01-01T00:00:00Z");
        // updated_at should be new
        assert_eq!(result.updated_at, "2025-01-05T00:00:00Z");
        // Other fields should be updated
        assert_eq!(result.revision, "v2");
        assert_eq!(result.size, 2000);
    }
}
