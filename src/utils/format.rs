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

/// Format byte size to human-readable format using decimal units (B, KB, MB, GB)
pub fn format_size_decimal(bytes: u64) -> String {
    const KB: f64 = 1000.0;
    const MB: f64 = 1000.0 * 1000.0;
    const GB: f64 = 1000.0 * 1000.0 * 1000.0;

    if bytes as f64 >= GB {
        format!("{:.2} GB", bytes as f64 / GB)
    } else if bytes as f64 >= MB {
        format!("{:.2} MB", bytes as f64 / MB)
    } else if bytes as f64 >= KB {
        format!("{:.2} KB", bytes as f64 / KB)
    } else {
        format!("{} B", bytes)
    }
}

/// Format parameter count to human-readable format (K, M, B)
pub fn format_parameters(count: u64) -> String {
    const K: f64 = 1_000.0;
    const M: f64 = 1_000_000.0;
    const B: f64 = 1_000_000_000.0;

    if count as f64 >= B {
        format!("{:.2}B", count as f64 / B)
    } else if count as f64 >= M {
        format!("{:.2}M", count as f64 / M)
    } else if count as f64 >= K {
        format!("{:.2}K", count as f64 / K)
    } else {
        count.to_string()
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

    #[test]
    fn test_format_size_decimal_bytes() {
        assert_eq!(format_size_decimal(0), "0 B");
        assert_eq!(format_size_decimal(1), "1 B");
        assert_eq!(format_size_decimal(999), "999 B");
    }

    #[test]
    fn test_format_size_decimal_kilobytes() {
        assert_eq!(format_size_decimal(1000), "1.00 KB");
        assert_eq!(format_size_decimal(1500), "1.50 KB");
        assert_eq!(format_size_decimal(10000), "10.00 KB");
        assert_eq!(format_size_decimal(999_999), "1000.00 KB");
    }

    #[test]
    fn test_format_size_decimal_megabytes() {
        assert_eq!(format_size_decimal(1_000_000), "1.00 MB");
        assert_eq!(format_size_decimal(1_500_000), "1.50 MB");
        assert_eq!(format_size_decimal(10_000_000), "10.00 MB");
        assert_eq!(format_size_decimal(500_000_000), "500.00 MB");
    }

    #[test]
    fn test_format_size_decimal_gigabytes() {
        assert_eq!(format_size_decimal(1_000_000_000), "1.00 GB");
        assert_eq!(format_size_decimal(1_500_000_000), "1.50 GB");
        assert_eq!(format_size_decimal(10_000_000_000), "10.00 GB");
        assert_eq!(format_size_decimal(100_000_000_000), "100.00 GB");
    }

    #[test]
    fn test_format_size_decimal_realistic_model_sizes() {
        // Small model (100 MB)
        assert_eq!(format_size_decimal(100_000_000), "100.00 MB");

        // Medium model (7 GB)
        assert_eq!(format_size_decimal(7_000_000_000), "7.00 GB");

        // Large model (65 GB)
        assert_eq!(format_size_decimal(65_000_000_000), "65.00 GB");
    }

    #[test]
    fn test_format_parameters_raw() {
        assert_eq!(format_parameters(0), "0");
        assert_eq!(format_parameters(1), "1");
        assert_eq!(format_parameters(999), "999");
    }

    #[test]
    fn test_format_parameters_thousands() {
        assert_eq!(format_parameters(1_000), "1.00K");
        assert_eq!(format_parameters(1_500), "1.50K");
        assert_eq!(format_parameters(10_000), "10.00K");
        assert_eq!(format_parameters(999_999), "1000.00K");
    }

    #[test]
    fn test_format_parameters_millions() {
        assert_eq!(format_parameters(1_000_000), "1.00M");
        assert_eq!(format_parameters(1_500_000), "1.50M");
        assert_eq!(format_parameters(7_000_000), "7.00M");
        assert_eq!(format_parameters(350_000_000), "350.00M");
    }

    #[test]
    fn test_format_parameters_billions() {
        assert_eq!(format_parameters(1_000_000_000), "1.00B");
        assert_eq!(format_parameters(1_500_000_000), "1.50B");
        assert_eq!(format_parameters(7_000_000_000), "7.00B");
        assert_eq!(format_parameters(175_000_000_000), "175.00B");
    }

    #[test]
    fn test_format_parameters_realistic_models() {
        // Tiny model (109K parameters)
        assert_eq!(format_parameters(109_824), "109.82K");

        // Small model (125M parameters)
        assert_eq!(format_parameters(125_000_000), "125.00M");

        // Medium model (7B parameters)
        assert_eq!(format_parameters(7_000_000_000), "7.00B");

        // Large model (70B parameters)
        assert_eq!(format_parameters(70_000_000_000), "70.00B");

        // Very large model (405B parameters)
        assert_eq!(format_parameters(405_000_000_000), "405.00B");
    }

    #[test]
    fn test_format_parameters_edge_cases() {
        // Boundary between K and M
        assert_eq!(format_parameters(999_999), "1000.00K");
        assert_eq!(format_parameters(1_000_000), "1.00M");

        // Boundary between M and B
        assert_eq!(format_parameters(999_999_999), "1000.00M");
        assert_eq!(format_parameters(1_000_000_000), "1.00B");
    }
}
