use std::fs;
use std::path::PathBuf;

use dirs::home_dir;

pub fn create_folder_if_not_exists(folder_path: &PathBuf) -> std::io::Result<()> {
    fs::create_dir_all(folder_path)?;
    Ok(())
}

pub fn root_home() -> PathBuf {
    let home = home_dir().expect("Failed to get home directory");
    home.join(".puma")
}

pub fn cache_dir() -> PathBuf {
    root_home().join("cache")
}

pub fn huggingface_cache_dir() -> PathBuf {
    cache_dir().join("huggingface")
}

pub fn modelscope_cache_dir() -> PathBuf {
    cache_dir().join("modelscope")
}
