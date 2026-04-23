use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Manages multi-file download progress tracking
///
/// # Example
/// ```rust
/// use puma::downloader::progress::DownloadProgressManager;
///
/// let progress_manager = DownloadProgressManager::new(30);
/// let mut file_progress = progress_manager.create_file_progress("model.bin");
///
/// file_progress.init(1024 * 1024); // 1 MB
/// file_progress.update(512 * 1024); // Downloaded 512 KB
/// file_progress.finish();
///
/// let total = progress_manager.total_downloaded_bytes();
/// ```
#[derive(Clone)]
pub struct DownloadProgressManager {
    multi_progress: Arc<MultiProgress>,
    total_size: Arc<AtomicU64>,
    style: ProgressStyle,
    cached_style: ProgressStyle,
}

impl DownloadProgressManager {
    /// Create a new progress manager with aligned file names
    pub fn new(max_filename_len: usize) -> Self {
        let multi_progress = Arc::new(MultiProgress::new());

        let template = format!(
            "{{msg:<{width}}} [{{elapsed_precise}}] {{bar:60.white}} {{bytes}}/{{total_bytes}} {{bytes_per_sec}}",
            width = max_filename_len
        );
        let style = ProgressStyle::default_bar()
            .template(&template)
            .unwrap()
            .progress_chars("▇▆▅▄▃▂▁ ");

        // Cached file style without speed
        let cached_template = format!(
            "{{msg:<{width}}} [{{elapsed_precise}}] {{bar:60.white}} {{bytes}}/{{total_bytes}}",
            width = max_filename_len
        );
        let cached_style = ProgressStyle::default_bar()
            .template(&cached_template)
            .unwrap()
            .progress_chars("▇▆▅▄▃▂▁ ");

        Self {
            multi_progress,
            total_size: Arc::new(AtomicU64::new(0)),
            style,
            cached_style,
        }
    }

    /// Create a new progress bar for a file download
    pub fn create_file_progress(&self, filename: &str) -> FileProgress {
        let pb = self.multi_progress.add(ProgressBar::hidden());
        pb.set_style(self.style.clone());
        pb.set_message(filename.to_string());

        FileProgress {
            pb,
            total_size: Arc::clone(&self.total_size),
        }
    }

    /// Create a new progress bar for a cached file (no speed display)
    pub fn create_cached_file_progress(&self, filename: &str) -> FileProgress {
        let pb = self.multi_progress.add(ProgressBar::hidden());
        pb.set_style(self.cached_style.clone());
        pb.set_message(filename.to_string());

        FileProgress {
            pb,
            total_size: Arc::clone(&self.total_size),
        }
    }

    /// Get the total accumulated download size
    pub fn total_downloaded_bytes(&self) -> u64 {
        self.total_size.load(Ordering::Relaxed)
    }

    /// Create a spinner progress bar (for post-download operations)
    pub fn create_spinner(&self) -> ProgressBar {
        let pb = self.multi_progress.add(ProgressBar::new_spinner());
        pb.set_style(
            ProgressStyle::default_spinner()
                .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
                .template("{spinner} ")
                .unwrap(),
        );
        pb.enable_steady_tick(std::time::Duration::from_millis(80));
        pb
    }
}

/// Tracks progress for a single file download
#[derive(Clone)]
pub struct FileProgress {
    pb: ProgressBar,
    total_size: Arc<AtomicU64>,
}

impl FileProgress {
    /// Initialize progress bar with file size
    pub fn init(&mut self, size: u64) {
        self.pb.set_length(size);
        self.pb.reset();
        self.pb.tick();
        self.total_size.fetch_add(size, Ordering::Relaxed);
    }

    /// Update progress with downloaded bytes
    pub fn update(&mut self, bytes: u64) {
        self.pb.inc(bytes);
    }

    /// Mark download as complete
    pub fn finish(&mut self) {
        self.pb.finish();
    }

    /// Mark download as failed
    #[allow(dead_code)]
    pub fn abandon(&mut self) {
        self.pb.abandon();
    }

    /// Get the inner progress bar (for provider-specific adapters)
    #[allow(dead_code)]
    pub fn progress_bar(&self) -> &ProgressBar {
        &self.pb
    }

    /// Get the total size tracker (for provider-specific adapters)
    #[allow(dead_code)]
    pub fn total_size_tracker(&self) -> Arc<AtomicU64> {
        Arc::clone(&self.total_size)
    }
}
