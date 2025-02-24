use core::fmt;

#[derive(Debug)]
pub enum DownloadError {
    RequestError(String),
    ParseError(String),
}

impl std::error::Error for DownloadError {}

impl fmt::Display for DownloadError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            DownloadError::RequestError(e) => write!(f, "RequestError: {}", e),
            DownloadError::ParseError(e) => write!(f, "ParseError: {}", e),
        }
    }
}
