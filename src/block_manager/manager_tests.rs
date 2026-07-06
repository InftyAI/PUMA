#[cfg(test)]
mod tests {
    use super::super::allocator::CpuAllocator;
    use super::super::manager::BlockManager;
    use super::super::types::*;

    #[test]
    fn test_allocate_and_free() {
        let allocator = Box::new(CpuAllocator::new(10_000));
        let mut manager = BlockManager::new(allocator, 1024);

        // Allocate a block
        let block_id = manager.allocate().unwrap();
        assert_eq!(block_id, BlockId(0));

        // Free it
        manager.free(block_id).unwrap();

        // Should be back in free pool
        let stats = manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.free_blocks, 1);
        assert_eq!(block_stats.allocated_blocks, 0);
    }

    #[test]
    fn test_ref_counting() {
        let allocator = Box::new(CpuAllocator::new(10_000));
        let mut manager = BlockManager::new(allocator, 1024);

        let block_id = manager.allocate().unwrap();

        // Add ref
        manager.add_ref(block_id).unwrap();

        // Free once - should still be allocated (ref_count = 1)
        manager.free(block_id).unwrap();
        let stats = manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.allocated_blocks, 1);
        assert_eq!(block_stats.free_blocks, 0);

        // Free again - now should be in free pool (ref_count = 0)
        manager.free(block_id).unwrap();
        let stats = manager.get_stats();
        let block_stats = stats.get(&BlockType::StandardKV).unwrap();
        assert_eq!(block_stats.free_blocks, 1);
        assert_eq!(block_stats.allocated_blocks, 0);
    }

    #[test]
    fn test_double_free_error() {
        let allocator = Box::new(CpuAllocator::new(10_000));
        let mut manager = BlockManager::new(allocator, 1024);

        let block_id = manager.allocate().unwrap();
        manager.free(block_id).unwrap();

        // Should fail on double free
        let result = manager.free(block_id);
        assert!(matches!(result, Err(Error::DoubleFree(_))));
    }

    #[test]
    fn test_oom() {
        // Small allocator - only enough for 2 blocks
        let allocator = Box::new(CpuAllocator::new(2048));
        let mut manager = BlockManager::new(allocator, 1024);

        // First block succeeds
        let block1 = manager.allocate().unwrap();
        assert_eq!(block1, BlockId(0));

        // Second block succeeds
        let block2 = manager.allocate().unwrap();
        assert_eq!(block2, BlockId(1));

        // Third block should fail (OOM)
        let result = manager.allocate();
        assert!(matches!(result, Err(Error::OutOfMemory)));
    }

    #[test]
    fn test_free_pool_reuse() {
        let allocator = Box::new(CpuAllocator::new(10_000));
        let mut manager = BlockManager::new(allocator, 1024);

        // Allocate and free
        let block1 = manager.allocate().unwrap();
        manager.free(block1).unwrap();

        // Next allocation should reuse from free pool (same ID)
        let block2 = manager.allocate().unwrap();
        assert_eq!(block1, block2);
    }

    #[test]
    fn test_get_memory_address() {
        let allocator = Box::new(CpuAllocator::new(10_000));
        let mut manager = BlockManager::new(allocator, 1024);

        let block_id = manager.allocate().unwrap();
        let addr = manager.get_memory_address(block_id);
        assert!(addr.is_ok());
    }

    #[test]
    fn test_invalid_block_id() {
        let allocator = Box::new(CpuAllocator::new(10_000));
        let manager = BlockManager::new(allocator, 1024);

        let invalid_id = BlockId(999);
        let result = manager.get_memory_address(invalid_id);
        assert!(matches!(result, Err(Error::InvalidBlockId(_))));
    }

    #[test]
    fn test_can_allocate() {
        let allocator = Box::new(CpuAllocator::new(2048));
        let mut manager = BlockManager::new(allocator, 1024);

        // Should be able to allocate
        assert!(manager.can_allocate(&BlockType::StandardKV));

        // Allocate both blocks
        manager.allocate().unwrap();
        manager.allocate().unwrap();

        // Should not be able to allocate more
        assert!(!manager.can_allocate(&BlockType::StandardKV));
    }
}
