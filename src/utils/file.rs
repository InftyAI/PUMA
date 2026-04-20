use std::fs;
use std::path::PathBuf;

use dirs::home_dir;

pub fn create_folder_if_not_exists(folder_path: &PathBuf) -> std::io::Result<()> {
    fs::create_dir_all(folder_path)?;
    Ok(())
}

pub fn root_home() -> PathBuf {
    // Allow tests to override PUMA home directory
    if let Ok(test_home) = std::env::var("PUMA_HOME") {
        PathBuf::from(test_home)
    } else {
        let home = home_dir().expect("Failed to get home directory");
        home.join(".puma")
    }
}

pub fn cache_dir() -> PathBuf {
    root_home().join("cache")
}

pub fn huggingface_cache_dir() -> PathBuf {
    cache_dir().join("huggingface")
}

#[allow(dead_code)]
pub fn modelscope_cache_dir() -> PathBuf {
    cache_dir().join("modelscope")
}

/// Format model name for HuggingFace cache directory
/// Converts "owner/model" to "models--owner--model"
pub fn format_model_name(name: &str) -> String {
    format!("models--{}", name.replace("/", "--"))
}

/// List all files recursively in a directory
#[allow(dead_code)]
pub fn list_files_recursive(dir: &std::path::Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if dir.is_dir() {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                files.extend(list_files_recursive(&path)?);
            } else {
                files.push(path);
            }
        }
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_model_name_basic() {
        assert_eq!(format_model_name("owner/model"), "models--owner--model");
    }

    #[test]
    fn test_format_model_name_complex() {
        assert_eq!(
            format_model_name("Qwen/Qwen3.5-2B"),
            "models--Qwen--Qwen3.5-2B"
        );
    }

    #[test]
    fn test_format_model_name_multiple_slashes() {
        assert_eq!(
            format_model_name("org/team/model"),
            "models--org--team--model"
        );
    }

    #[test]
    fn test_format_model_name_no_slash() {
        assert_eq!(format_model_name("model"), "models--model");
    }

    #[test]
    fn test_format_model_name_special_chars() {
        assert_eq!(
            format_model_name("InftyAI/tiny-random-gpt2"),
            "models--InftyAI--tiny-random-gpt2"
        );
    }
}
