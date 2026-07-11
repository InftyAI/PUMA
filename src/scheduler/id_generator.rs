use crate::block_manager::types::SequenceId;
use std::sync::atomic::{AtomicU64, Ordering};

/// Sequential ID generator for sequences
/// Thread-safe, starts from 0
pub struct SequenceIdGenerator {
    next_id: AtomicU64,
}

impl SequenceIdGenerator {
    /// Create a new ID generator starting from 0
    pub fn new() -> Self {
        Self {
            next_id: AtomicU64::new(0),
        }
    }

    /// Create an ID generator starting from a specific value
    pub fn with_start(start: u64) -> Self {
        Self {
            next_id: AtomicU64::new(start),
        }
    }

    /// Generate the next sequential ID
    /// Thread-safe, can be called concurrently
    pub fn next(&self) -> SequenceId {
        SequenceId(self.next_id.fetch_add(1, Ordering::Relaxed))
    }

    /// Peek at the next ID without consuming it
    pub fn peek(&self) -> SequenceId {
        SequenceId(self.next_id.load(Ordering::Relaxed))
    }
}

impl Default for SequenceIdGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_generation() {
        let gen = SequenceIdGenerator::new();
        assert_eq!(gen.next().0, 0);
        assert_eq!(gen.next().0, 1);
        assert_eq!(gen.next().0, 2);
    }

    #[test]
    fn test_with_start() {
        let gen = SequenceIdGenerator::with_start(100);
        assert_eq!(gen.next().0, 100);
        assert_eq!(gen.next().0, 101);
    }

    #[test]
    fn test_peek() {
        let gen = SequenceIdGenerator::new();
        assert_eq!(gen.peek().0, 0);
        assert_eq!(gen.peek().0, 0); // Doesn't consume
        assert_eq!(gen.next().0, 0); // Now consume
        assert_eq!(gen.peek().0, 1);
    }
}
