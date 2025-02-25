use std::fs;
use std::path::PathBuf;

use dirs::home_dir;

pub fn create_folder_if_not_exists(folder_path: &PathBuf) -> std::io::Result<()> {
    fs::create_dir_all(folder_path)?;
    Ok(())
}

pub fn model_folder_name(model_name: &str) -> String {
    model_name.replace(":", "--").replace("/", "--")
}

pub fn root_home() -> PathBuf {
    let home = home_dir().expect("Failed to get home directory");
    home.join(".puma")
}
