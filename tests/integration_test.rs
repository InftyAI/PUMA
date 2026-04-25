use std::process::Command;
use tempfile::TempDir;

/// Helper to run puma command with custom PUMA_HOME
fn run_puma(home_dir: &str, args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_puma"))
        .env("PUMA_HOME", home_dir)
        .args(args)
        .output()
        .expect("Failed to execute puma command")
}

/// Helper to check if output contains a string
fn output_contains(output: &std::process::Output, text: &str) -> bool {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    stdout.contains(text) || stderr.contains(text)
}

#[test]
fn test_pull_command_with_provider() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // Pull a real model
    let output = run_puma(
        home,
        &["pull", "inftyai/tiny-random-gpt2", "-p", "huggingface"],
    );
    assert!(output.status.success());

    // Verify model appears in ls
    let output = run_puma(home, &["ls"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("inftyai/tiny-random-gpt2"));

    // Verify model can be inspected
    let output = run_puma(home, &["inspect", "inftyai/tiny-random-gpt2"]);
    assert!(output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("name: inftyai/tiny-random-gpt2"));
    assert!(stdout.contains("provider"));
    assert!(stdout.contains("huggingface"));

    // Verify model can be removed
    let output = run_puma(home, &["rm", "inftyai/tiny-random-gpt2"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Successfully removed model"));

    // Verify model is gone
    let output = run_puma(home, &["ls"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(!stdout.contains("inftyai/tiny-random-gpt2"));
}

#[test]
fn test_rm_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["rm", "nonexistent/model"]);
    assert!(!output.status.success());
    assert!(output_contains(&output, "Model not found"));
}

#[test]
fn test_inspect_nonexistent() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["inspect", "nonexistent/model"]);
    assert!(!output.status.success());
    assert!(output_contains(&output, "Model not found"));
}

#[test]
fn test_version() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["version"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "PUMA"));
}

#[test]
fn test_info() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["info"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Version"));
    assert!(stdout.contains("Models"));
}

#[test]
fn test_ls_with_invalid_regex() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["ls", "[invalid"]);
    assert!(!output.status.success());
    assert!(output_contains(&output, "Invalid regex pattern"));
}

#[test]
fn test_ls_with_invalid_query() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["ls", "-l", "invalid_format"]);
    assert!(!output.status.success());
    assert!(output_contains(&output, "Invalid query format"));
}

#[test]
fn test_ls_with_invalid_filter_column() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["ls", "-l", "invalid_column=value"]);
    assert!(!output.status.success());
    assert!(output_contains(&output, "Invalid filter column"));
}

#[test]
fn test_ps_command() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["ps"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("NAME"));
    assert!(stdout.contains("PROVIDER"));
    assert!(stdout.contains("MODEL"));
}

#[test]
fn test_pull_command_invalid_model() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // Pull with invalid model name should fail
    let output = run_puma(home, &["pull", "invalid/nonexistent-model-12345"]);
    assert!(!output.status.success());
}

#[test]
fn test_pull_command_modelscope_provider() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // Test modelscope provider (currently just prints message)
    let output = run_puma(home, &["pull", "test/model", "-p", "modelscope"]);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Downloading model from Modelscope") || !output.status.success());
}

#[test]
fn test_run_command() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["run"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Creating and running a new model"));
}

#[test]
fn test_stop_command() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["stop"]);
    assert!(output.status.success());
    assert!(output_contains(&output, "Stopping one running model"));
}

#[test]
fn test_ls_with_pattern_no_models() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // Pattern matching on empty registry should succeed
    let output = run_puma(home, &["ls", "test"]);
    assert!(output.status.success());
}

#[test]
fn test_ls_with_sql_filter_no_models() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // SQL filter on empty registry should succeed
    let output = run_puma(home, &["ls", "-l", "author=test"]);
    assert!(output.status.success());
}

#[test]
fn test_ls_with_multiple_filters() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // Multiple filters separated by comma
    let output = run_puma(home, &["ls", "-l", "author=test,license=mit"]);
    assert!(output.status.success());
}

#[test]
fn test_ls_with_pattern_and_filter_combined() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    // Both pattern and filter should work together
    let output = run_puma(home, &["ls", "test", "-l", "author=test"]);
    assert!(output.status.success());
}

#[test]
fn test_invalid_command() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["invalid-command"]);
    assert!(!output.status.success());
}

#[test]
fn test_help_command() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("PUMA CLI"));
    assert!(stdout.contains("Commands:"));
}

#[test]
fn test_ls_help() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["ls", "--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("List local models"));
}

#[test]
fn test_rm_help() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["rm", "--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Remove one model"));
}

#[test]
fn test_inspect_help() {
    let temp_dir = TempDir::new().unwrap();
    let home = temp_dir.path().to_str().unwrap();

    let output = run_puma(home, &["inspect", "--help"]);
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("Return detailed information about a model"));
}
