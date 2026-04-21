use serde::{Deserialize, Serialize};
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::PathBuf;
use std::process::Command;
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
    pub gpu_info: Vec<GpuInfo>,
    pub cache_dir: String,
    pub cache_size: String,
    pub models_count: usize,
    pub running_models: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String, // "CUDA", "Metal", "ROCm", or "Unknown"
    pub memory: Option<String>,
}

impl SystemInfo {
    pub fn collect() -> Self {
        let mut sys = System::new_all();
        sys.refresh_memory();

        let cache_dir = file::cache_dir();
        let cache_size = Self::calculate_cache_size(&cache_dir);

        let registry = ModelRegistry::new(None);
        let models_count = registry.load_models().unwrap_or_default().len();

        let gpu_info = Self::detect_gpus();

        SystemInfo {
            version: env!("CARGO_PKG_VERSION").to_string(),
            os: System::name().unwrap_or_else(|| "Unknown".to_string()),
            architecture: System::cpu_arch().unwrap_or_else(|| "Unknown".to_string()),
            cpu_cores: sys.cpus().len(),
            total_memory: format_size(sys.total_memory()),
            gpu_info,
            cache_dir: cache_dir.to_string_lossy().to_string(),
            cache_size: format_size(cache_size),
            models_count,
            running_models: 0, // TODO: implement running models tracking
        }
    }

    fn detect_gpus() -> Vec<GpuInfo> {
        let mut gpus = Vec::new();

        // Try NVIDIA GPUs first (Linux/Windows)
        if let Some(nvidia_gpus) = Self::detect_nvidia_gpus() {
            gpus.extend(nvidia_gpus);
        }

        // Try Metal (macOS)
        if let Some(metal_gpu) = Self::detect_metal_gpu() {
            gpus.push(metal_gpu);
        }

        // Try AMD ROCm (Linux)
        if let Some(amd_gpus) = Self::detect_amd_gpus() {
            gpus.extend(amd_gpus);
        }

        gpus
    }

    fn detect_nvidia_gpus() -> Option<Vec<GpuInfo>> {
        let output = Command::new("nvidia-smi")
            .args(&[
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let output_str = String::from_utf8(output.stdout).ok()?;
        let mut gpus = Vec::new();

        for line in output_str.lines() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 2 {
                gpus.push(GpuInfo {
                    name: parts[0].to_string(),
                    backend: "CUDA".to_string(),
                    memory: Some(format!("{} MB", parts[1])),
                });
            }
        }

        if gpus.is_empty() {
            None
        } else {
            Some(gpus)
        }
    }

    fn detect_metal_gpu() -> Option<GpuInfo> {
        // Check if running on macOS
        if !cfg!(target_os = "macos") {
            return None;
        }

        let output = Command::new("system_profiler")
            .arg("SPDisplaysDataType")
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let output_str = String::from_utf8(output.stdout).ok()?;
        let lines: Vec<&str> = output_str.lines().collect();

        // Find GPU name and cores
        let mut gpu_name = None;
        let mut core_count = None;

        for (i, line) in lines.iter().enumerate() {
            if line.contains("Chipset Model:") {
                let parts: Vec<&str> = line.split("Chipset Model:").collect();
                if parts.len() >= 2 {
                    let name = parts[1].trim();
                    if !name.is_empty() {
                        gpu_name = Some(name.to_string());

                        // Look for core count in the next few lines
                        for j in (i + 1)..std::cmp::min(i + 10, lines.len()) {
                            if lines[j].contains("Total Number of Cores:") {
                                let core_parts: Vec<&str> =
                                    lines[j].split("Total Number of Cores:").collect();
                                if core_parts.len() >= 2 {
                                    core_count = Some(core_parts[1].trim().to_string());
                                }
                                break;
                            }
                        }
                        break;
                    }
                }
            }
        }

        if let Some(name) = gpu_name {
            let memory_str = core_count.map(|cores| format!("{} GPU cores", cores));

            return Some(GpuInfo {
                name,
                backend: "Metal".to_string(),
                memory: memory_str,
            });
        }

        None
    }

    fn detect_amd_gpus() -> Option<Vec<GpuInfo>> {
        let output = Command::new("rocm-smi")
            .arg("--showproductname")
            .output()
            .ok()?;

        if !output.status.success() {
            return None;
        }

        let output_str = String::from_utf8(output.stdout).ok()?;
        let mut gpus = Vec::new();

        for line in output_str.lines() {
            if line.contains("Card series:") || line.contains("Card model:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() >= 2 {
                    let name = parts[1].trim().to_string();
                    if !name.is_empty() {
                        gpus.push(GpuInfo {
                            name,
                            backend: "ROCm".to_string(),
                            memory: None,
                        });
                    }
                }
            }
        }

        if gpus.is_empty() {
            None
        } else {
            Some(gpus)
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

        if !self.gpu_info.is_empty() {
            for (i, gpu) in self.gpu_info.iter().enumerate() {
                if i == 0 {
                    print!("  GPU:                ");
                } else {
                    print!("                      ");
                }
                print!("{} ({})", gpu.name, gpu.backend);
                if let Some(ref memory) = gpu.memory {
                    print!(" - {}", memory);
                }
                println!();
            }
        }

        println!();
        println!("PUMA Information:");
        println!("  PUMA Version:       {}", self.version);
        println!("  Cache Directory:    {}", self.cache_dir);
        println!("  Cache Size:         {}", self.cache_size);
        println!("  Models:             {}", self.models_count);
        println!("  Running Models:     {}", self.running_models);
    }
}
