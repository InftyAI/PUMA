use chrono::{DateTime, Local, Utc};

/// Format byte size to human-readable format (B, KB, MB, GB)
pub fn format_size(bytes: u64) -> String {
    if bytes > 1_000_000_000 {
        format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
    } else if bytes > 1_000_000 {
        format!("{:.2} MB", bytes as f64 / 1_000_000.0)
    } else if bytes > 1_000 {
        format!("{:.2} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
    }
}

/// Format RFC3339 timestamp to human-readable relative time (e.g., "2 hours ago")
pub fn format_time_ago(timestamp: &str) -> String {
    // Try to parse as RFC3339
    let created_time = match DateTime::parse_from_rfc3339(timestamp) {
        Ok(dt) => dt.with_timezone(&Utc),
        Err(_) => return timestamp.to_string(), // Return original if parse fails
    };

    let now = Utc::now();
    let duration = now.signed_duration_since(created_time);

    let seconds = duration.num_seconds();

    if seconds < 0 {
        return "just now".to_string();
    } else if seconds < 60 {
        return format!("{} seconds ago", seconds);
    } else if seconds < 3600 {
        let minutes = seconds / 60;
        return format!("{} {} ago", minutes, if minutes == 1 { "minute" } else { "minutes" });
    } else if seconds < 86400 {
        let hours = seconds / 3600;
        return format!("{} {} ago", hours, if hours == 1 { "hour" } else { "hours" });
    } else if seconds < 2592000 {
        let days = seconds / 86400;
        return format!("{} {} ago", days, if days == 1 { "day" } else { "days" });
    } else if seconds < 31536000 {
        let months = seconds / 2592000;
        return format!("{} {} ago", months, if months == 1 { "month" } else { "months" });
    } else {
        let years = seconds / 31536000;
        return format!("{} {} ago", years, if years == 1 { "year" } else { "years" });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_size_bytes() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(1), "1 B");
        assert_eq!(format_size(999), "999 B");
        assert_eq!(format_size(1000), "1000 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1_001), "1.00 KB");
        assert_eq!(format_size(1_500), "1.50 KB");
        assert_eq!(format_size(10_000), "10.00 KB");
        assert_eq!(format_size(999_999), "1000.00 KB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_000_001), "1.00 MB");
        assert_eq!(format_size(1_500_000), "1.50 MB");
        assert_eq!(format_size(10_000_000), "10.00 MB");
        assert_eq!(format_size(500_000_000), "500.00 MB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1_000_000_001), "1.00 GB");
        assert_eq!(format_size(1_500_000_000), "1.50 GB");
        assert_eq!(format_size(10_000_000_000), "10.00 GB");
        assert_eq!(format_size(100_000_000_000), "100.00 GB");
    }

    #[test]
    fn test_format_size_edge_cases() {
        // Boundary between KB and MB
        assert_eq!(format_size(1_000_000), "1000.00 KB");
        assert_eq!(format_size(1_000_001), "1.00 MB");

        // Boundary between MB and GB
        assert_eq!(format_size(1_000_000_000), "1000.00 MB");
        assert_eq!(format_size(1_000_000_001), "1.00 GB");
    }

    #[test]
    fn test_format_size_realistic_model_sizes() {
        // Small model (100 MB)
        assert_eq!(format_size(104_857_600), "104.86 MB");

        // Medium model (7 GB)
        assert_eq!(format_size(7_516_192_768), "7.52 GB");

        // Large model (65 GB)
        assert_eq!(format_size(69_793_218_560), "69.79 GB");
    }

    #[test]
    fn test_format_time_ago_seconds() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now - Duration::seconds(30)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "30 seconds ago");

        let timestamp = (now - Duration::seconds(1)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "1 seconds ago");
    }

    #[test]
    fn test_format_time_ago_minutes() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now - Duration::minutes(5)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "5 minutes ago");

        let timestamp = (now - Duration::minutes(1)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "1 minute ago");
    }

    #[test]
    fn test_format_time_ago_hours() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now - Duration::hours(3)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "3 hours ago");

        let timestamp = (now - Duration::hours(1)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "1 hour ago");
    }

    #[test]
    fn test_format_time_ago_days() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now - Duration::days(7)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "7 days ago");

        let timestamp = (now - Duration::days(1)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "1 day ago");
    }

    #[test]
    fn test_format_time_ago_months() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now - Duration::days(60)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "2 months ago");

        let timestamp = (now - Duration::days(30)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "1 month ago");
    }

    #[test]
    fn test_format_time_ago_years() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now - Duration::days(730)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "2 years ago");

        let timestamp = (now - Duration::days(365)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "1 year ago");
    }

    #[test]
    fn test_format_time_ago_future() {
        use chrono::Duration;

        let now = Utc::now();
        let timestamp = (now + Duration::hours(5)).to_rfc3339();
        assert_eq!(format_time_ago(&timestamp), "just now");
    }

    #[test]
    fn test_format_time_ago_invalid() {
        let invalid = "not-a-timestamp";
        assert_eq!(format_time_ago(invalid), "not-a-timestamp");
    }
}
