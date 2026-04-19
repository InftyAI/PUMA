use chrono::{DateTime, Utc};

/// Format byte size to human-readable format (B, KiB, MiB, GiB)
pub fn format_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    if bytes as f64 >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB)
    } else if bytes as f64 >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB)
    } else if bytes as f64 >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB)
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
        "just now".to_string()
    } else if seconds < 60 {
        format!("{} seconds ago", seconds)
    } else if seconds < 3600 {
        let minutes = seconds / 60;
        format!(
            "{} {} ago",
            minutes,
            if minutes == 1 { "minute" } else { "minutes" }
        )
    } else if seconds < 86400 {
        let hours = seconds / 3600;
        format!(
            "{} {} ago",
            hours,
            if hours == 1 { "hour" } else { "hours" }
        )
    } else if seconds < 2592000 {
        let days = seconds / 86400;
        format!("{} {} ago", days, if days == 1 { "day" } else { "days" })
    } else if seconds < 31536000 {
        let months = seconds / 2592000;
        format!(
            "{} {} ago",
            months,
            if months == 1 { "month" } else { "months" }
        )
    } else {
        let years = seconds / 31536000;
        format!(
            "{} {} ago",
            years,
            if years == 1 { "year" } else { "years" }
        )
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
        assert_eq!(format_size(1023), "1023 B");
    }

    #[test]
    fn test_format_size_kilobytes() {
        assert_eq!(format_size(1024), "1.00 KiB");
        assert_eq!(format_size(1536), "1.50 KiB");
        assert_eq!(format_size(10240), "10.00 KiB");
        assert_eq!(format_size(1_048_575), "1024.00 KiB");
    }

    #[test]
    fn test_format_size_megabytes() {
        assert_eq!(format_size(1_048_576), "1.00 MiB");
        assert_eq!(format_size(1_572_864), "1.50 MiB");
        assert_eq!(format_size(10_485_760), "10.00 MiB");
        assert_eq!(format_size(524_288_000), "500.00 MiB");
    }

    #[test]
    fn test_format_size_gigabytes() {
        assert_eq!(format_size(1_073_741_824), "1.00 GiB");
        assert_eq!(format_size(1_610_612_736), "1.50 GiB");
        assert_eq!(format_size(10_737_418_240), "10.00 GiB");
        assert_eq!(format_size(107_374_182_400), "100.00 GiB");
    }

    #[test]
    fn test_format_size_edge_cases() {
        // Boundary between KiB and MiB
        assert_eq!(format_size(1_048_575), "1024.00 KiB");
        assert_eq!(format_size(1_048_576), "1.00 MiB");

        // Boundary between MiB and GiB
        assert_eq!(format_size(1_073_741_823), "1024.00 MiB");
        assert_eq!(format_size(1_073_741_824), "1.00 GiB");
    }

    #[test]
    fn test_format_size_realistic_model_sizes() {
        // Small model (100 MiB)
        assert_eq!(format_size(104_857_600), "100.00 MiB");

        // Medium model (7 GiB)
        assert_eq!(format_size(7_516_192_768), "7.00 GiB");

        // Large model (65 GiB)
        assert_eq!(format_size(69_793_218_560), "65.00 GiB");
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
