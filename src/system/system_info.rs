use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use sysinfo::System;

use crate::registry::model_registry::ModelRegistry;
use crate::utils::file;
use crate::utils::format::format_size;

#[derive(Debug, Serialize, Deserialize)]
pub struct SystemInfo {
    pub version: String,
    pub os: String,
    pub architecture: String,
    pub cpu_cores: usize,
    pub total_memory: String,
    pub available_memory: String,
    pub cache_dir: String,
    pub cache_size: String,
    pub models_count: usize,
    pub running_models: usize,
}

impl SystemInfo {
    pub fn collect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_all();

        let cache_dir = file::cache_dir();
        let cache_size = Self::calculate_cache_size(&cache_dir);

        let registry = ModelRegistry::new(None);
        let models_count = registry.load_models().unwrap_or_default().len();

        SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            os: System::name().unwrap_or_else(|| "Unknown".to_string()),
            architecture: System::cpu_arch().unwrap_or_else(|| "Unknown".to_string()),
            cpu_cores: sys.cpus().len(),
            total_memory: format_size(sys.total_memory()),
            available_memory: format_size(sys.available_memory()),
            cache_dir: cache_dir.to_string_lossy().to_string(),
            cache_size: format_size(cache_size),
            models_count,
            running_models: 0, // TODO: implement running models tracking
        }
    }

    fn calculate_cache_size(cache_dir: &PathBuf) -> u64 {
        if !cache_dir.exists() {
            return 0;
        }

        let mut total_size = 0u64;

        if let Ok(entries) = fs::read_dir(cache_dir) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        // Use blocks * 512 to get actual disk usage (handles sparse files)
                        total_size += metadata.blocks() * 512;
                    } else if metadata.is_dir() {
                        total_size += Self::dir_size(&entry.path());
                    }
                }
            }
        }

        total_size
    }

    fn dir_size(path: &PathBuf) -> u64 {
        let mut total_size = 0u64;

        if let Ok(entries) = fs::read_dir(path) {
            for entry in entries.flatten() {
                if let Ok(metadata) = entry.metadata() {
                    if metadata.is_file() {
                        // Use blocks * 512 to get actual disk usage (handles sparse files)
                        total_size += metadata.blocks() * 512;
                    } else if metadata.is_dir() {
                        total_size += Self::dir_size(&entry.path());
                    }
                }
            }
        }

        total_size
    }

    pub fn display(&self) {
        println!("System Information:");
        println!("  Operating System:   {}", self.os);
        println!("  Architecture:       {}", self.architecture);
        println!("  CPU Cores:          {}", self.cpu_cores);
        println!("  Total Memory:       {}", self.total_memory);
        println!("  Available Memory:   {}", self.available_memory);
        println!();
        println!("PUMA Information:");
        println!("  PUMA Version:       {}", self.version);
        println!("  Cache Directory:    {}", self.cache_dir);
        println!("  Cache Size:         {}", self.cache_size);
        println!("  Models:             {}", self.models_count);
        println!("  Running Models:     {}", self.running_models);
    }
}
