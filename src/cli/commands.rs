use clap::{Parser, Subcommand};
use prettytable::{format, row, Table};

use crate::cli::{inspect, ls, rm};
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
    LS(LsArgs),
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
struct LsArgs {
    /// Optional model name pattern to filter (e.g., qwen, openai/*)
    pattern: Option<String>,

    /// Advanced filter using SQL WHERE conditions (e.g., author=inftyai,license=mit)
    #[arg(short = 'l', long, value_name = "KEY=VALUE,...")]
    query: Option<String>,
}

#[derive(Parser)]
struct PullArgs {
    /// Model name to download (e.g., inftyai/tiny-random-gpt2)
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
    /// Model name to remove (e.g., inftyai/tiny-random-gpt2)
    model: String,
}

#[derive(Parser)]
struct InspectArgs {
    /// Model name to inspect (e.g., inftyai/tiny-random-gpt2)
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

        Commands::LS(args) => {
            let registry = ModelRegistry::new(None);

            let models =
                match ls::execute(&registry, args.pattern.as_deref(), args.query.as_deref()) {
                    Ok(models) => models,
                    Err(e) => {
                        eprintln!("{}", e);
                        std::process::exit(1);
                    }
                };

            let mut table = Table::new();
            table.set_format(
                format::FormatBuilder::new()
                    .column_separator(' ')
                    .padding(0, 1)
                    .build(),
            );
            table.add_row(row![
                "MODEL", "TASK", "PROVIDER", "REVISION", "SIZE", "CREATED"
            ]);
            for model in models {
                let size_str = format_size_decimal(model.metadata.artifact.size);

                let revision_short = if model.metadata.artifact.revision.len() > 8 {
                    &model.metadata.artifact.revision[..8]
                } else {
                    &model.metadata.artifact.revision
                };

                let created_str = format_time_ago(&model.created_at);

                let model_task = model.task.as_deref().unwrap_or("N/A");

                table.add_row(row![
                    model.name,
                    model_task,
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

            if let Err(e) = rm::execute(&registry, &args.model) {
                eprintln!("{}", e);
                std::process::exit(1);
            }
        }

        Commands::INFO => {
            let info = SystemInfo::collect();
            info.display();
        }

        Commands::INSPECT(args) => {
            let registry = ModelRegistry::new(None);

            match inspect::execute(&registry, &args.model) {
                Ok(model) => inspect::display(&model),
                Err(e) => {
                    eprintln!("{}", e);
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
    use crate::registry::model_registry::{ArtifactInfo, ModelInfo, ModelMetadata};
    use tempfile::TempDir;

    // Helper to create a test model
    fn create_test_model(name: &str, revision: &str) -> ModelInfo {
        let safetensors = serde_json::json!({
            "parameters": {
                "F32": 7000000000u64
            },
            "total": 7000000000u64
        });

        ModelInfo {
            uuid: revision.to_string(),
            name: name.to_string(),
            author: Some("test-author".to_string()),
            task: Some("text-generation".to_string()),
            model_series: Some("gpt2".to_string()),
            provider: "huggingface".to_string(),
            license: Some("mit".to_string()),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            metadata: ModelMetadata {
                artifact: ArtifactInfo {
                    revision: revision.to_string(),
                    size: 1000,
                    path: "/tmp/test".to_string(),
                },
                context_window: Some(2048),
                safetensors: Some(safetensors),
            },
        }
    }

    #[test]
    fn test_ls_command_empty() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let models = registry.load_models(None).unwrap_or_default();
        assert_eq!(models.len(), 0);
    }

    #[test]
    fn test_ls_command_with_models() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let model = create_test_model("test/model", "abc123def456");

        registry.register_model(model).unwrap();

        let models = registry.load_models(None).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model");
        assert_eq!(models[0].provider, "huggingface");
    }

    #[test]
    fn test_inspect_command_with_metadata() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let mut model = create_test_model("test/gpt-model", "abc123def456");
        model.author = Some("test-org".to_string());
        model.task = Some("text-generation".to_string());
        model.license = Some("mit".to_string());
        model.updated_at = "2025-01-02T00:00:00Z".to_string();

        registry.register_model(model.clone()).unwrap();

        let retrieved = registry.get_model("test/gpt-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/gpt-model");
        assert_eq!(model_info.created_at, "2025-01-01T00:00:00Z");
        assert_eq!(model_info.updated_at, "2025-01-02T00:00:00Z");
        assert_eq!(model_info.author, Some("test-org".to_string()));
        assert_eq!(model_info.task, Some("text-generation".to_string()));
        assert_eq!(model_info.license, Some("mit".to_string()));
        assert_eq!(model_info.model_series, Some("gpt2".to_string()));
        assert_eq!(model_info.metadata.context_window, Some(2048));
        assert_eq!(
            model_info
                .metadata
                .safetensors
                .as_ref()
                .unwrap()
                .get("total")
                .unwrap()
                .as_u64()
                .unwrap(),
            7_000_000_000
        );
    }

    #[test]
    fn test_inspect_command_without_architecture() {
        let temp_dir = TempDir::new().unwrap();
        let registry = ModelRegistry::new(Some(temp_dir.path().to_path_buf()));

        let mut model = create_test_model("test/simple-model", "xyz789");
        model.metadata.safetensors = None;
        model.metadata.context_window = None;

        registry.register_model(model).unwrap();

        let retrieved = registry.get_model("test/simple-model").unwrap();
        assert!(retrieved.is_some());

        let model_info = retrieved.unwrap();
        assert_eq!(model_info.name, "test/simple-model");
        assert!(model_info.metadata.safetensors.is_none());
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

        let model = create_test_model("test/updated-model", "v1");
        registry.register_model(model).unwrap();

        // Update the model
        let mut updated_model = create_test_model("test/updated-model", "v2");
        updated_model.metadata.artifact.size = 2000;
        updated_model.created_at = "2025-01-05T00:00:00Z".to_string();
        updated_model.updated_at = "2025-01-05T00:00:00Z".to_string();

        registry.register_model(updated_model).unwrap();

        let result = registry.get_model("test/updated-model").unwrap().unwrap();
        // created_at should remain the same
        assert_eq!(result.created_at, "2025-01-01T00:00:00Z");
        // updated_at should be new
        assert_eq!(result.updated_at, "2025-01-05T00:00:00Z");
        // Other fields should be updated
        assert_eq!(result.metadata.artifact.revision, "v2");
        assert_eq!(result.metadata.artifact.size, 2000);
    }
}
