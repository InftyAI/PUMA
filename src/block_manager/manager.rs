use super::allocator::*;
use super::types::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Manages memory blocks with reference counting and pooling.
///
/// Features:
/// - Reference counting: blocks can be shared, freed only when ref_count = 0
/// - Free pools: reuse freed blocks instead of allocating new memory
/// - Type-specific blocks: different block types with different sizes
/// - Thread-safe ID generation: atomic counter for BlockId
pub struct BlockManager {
    /// Underlying memory allocator (CPU or GPU)
    allocator: Box<dyn MemoryAllocator>,

    /// All blocks (allocated + free) - tracks metadata and physical memory
    block_table: HashMap<BlockId, Block>,

    /// Free block pools per type - blocks ready for reuse (ref_count = 0)
    /// Keyed by BlockType, values are BlockIds ready to be allocated
    free_pools: HashMap<BlockType, Vec<BlockId>>,

    /// Configuration for each block type (size, etc.)
    block_configs: HashMap<BlockType, BlockConfig>,

    /// Default block type for allocate() calls
    default_block_type: BlockType,

    /// Atomic counter for generating unique BlockIds
    next_block_id: AtomicU64,
}

impl BlockManager {
    pub fn new(allocator: Box<dyn MemoryAllocator>, default_block_size: usize) -> Self {
        let mut manager = Self {
            allocator,
            block_table: HashMap::new(),
            free_pools: HashMap::new(),
            block_configs: HashMap::new(),
            default_block_type: BlockType::StandardKV,
            next_block_id: AtomicU64::new(0),
        };

        // Register default type
        manager.register_block_type(BlockConfig {
            block_type: BlockType::StandardKV,
            size_bytes: default_block_size,
        });

        manager
    }

    /// Register a new block type with specific configuration.
    pub fn register_block_type(&mut self, config: BlockConfig) {
        self.free_pools.insert(config.block_type, Vec::new());
        self.block_configs.insert(config.block_type, config);
    }

    /// Allocate a block of specific type.
    ///
    /// Flow:
    /// 1. Check free pool first (reuse if available)
    /// 2. If pool empty, allocate new physical memory
    /// 3. Return BlockId with ref_count = 1
    pub fn allocate_typed(&mut self, block_type: &BlockType) -> Result<BlockId> {
        // Try free pool first - reuse freed blocks
        if let Some(block_id) = self
            .free_pools
            .get_mut(block_type)
            .and_then(|pool| pool.pop())
        {
            let block = self.block_table.get_mut(&block_id).unwrap();
            block.ref_count = 1;
            return Ok(block_id);
        }

        // No free blocks - allocate new physical memory
        let config = self
            .block_configs
            .get(block_type)
            .ok_or_else(|| Error::UnknownBlockType(block_type.as_str().to_string()))?;

        let mem_addr = self.allocator.allocate(config.size_bytes)?;

        let block_id = BlockId(self.next_block_id.fetch_add(1, Ordering::Relaxed));

        self.block_table.insert(
            block_id,
            Block {
                block_id,
                block_type: *block_type,
                mem_addr,
                ref_count: 1,
            },
        );

        Ok(block_id)
    }

    /// Allocate a block using the default type.
    pub fn allocate(&mut self) -> Result<BlockId> {
        let default_type = self.default_block_type;
        self.allocate_typed(&default_type)
    }

    /// Decrement reference count. When ref_count reaches 0, block goes to free pool.
    ///
    /// Note: Physical memory is NOT freed - block is just marked as reusable.
    /// This enables fast reallocation without syscalls.
    pub fn free(&mut self, block_id: BlockId) -> Result<()> {
        let block = self
            .block_table
            .get_mut(&block_id)
            .ok_or(Error::InvalidBlockId(block_id))?;

        if block.ref_count == 0 {
            return Err(Error::DoubleFree(block_id));
        }

        block.ref_count -= 1;

        // When ref_count hits 0, move to free pool for reuse
        if block.ref_count == 0 {
            self.free_pools
                .get_mut(&block.block_type)
                .unwrap()
                .push(block_id);
        }

        Ok(())
    }

    /// Increment reference count (for sharing blocks between sequences).
    ///
    /// Used for fork/beam search where multiple sequences share the same blocks.
    pub fn add_ref(&mut self, block_id: BlockId) -> Result<()> {
        let block = self
            .block_table
            .get_mut(&block_id)
            .ok_or(Error::InvalidBlockId(block_id))?;
        block.ref_count += 1;
        Ok(())
    }

    /// Get the physical memory address for a block.
    pub fn get_memory_address(&self, block_id: BlockId) -> Result<MemoryAddress> {
        self.block_table
            .get(&block_id)
            .map(|b| b.mem_addr)
            .ok_or(Error::InvalidBlockId(block_id))
    }

    /// Get the block type for a block.
    pub fn get_block_type(&self, block_id: BlockId) -> Result<&BlockType> {
        self.block_table
            .get(&block_id)
            .map(|b| &b.block_type)
            .ok_or(Error::InvalidBlockId(block_id))
    }

    /// Get statistics for all block types (allocated, free, total memory, etc.).
    pub fn get_stats(&self) -> HashMap<BlockType, BlockStats> {
        let mut stats = HashMap::new();

        for (type_id, pool) in &self.free_pools {
            let config = &self.block_configs[type_id];
            let free_count = pool.len();

            let allocated_count = self
                .block_table
                .values()
                .filter(|b| b.block_type == *type_id && b.ref_count > 0)
                .count();

            stats.insert(
                *type_id,
                BlockStats {
                    total_blocks: free_count + allocated_count,
                    allocated_blocks: allocated_count,
                    free_blocks: free_count,
                    block_size: config.size_bytes,
                    total_memory: (free_count + allocated_count) * config.size_bytes,
                },
            );
        }

        stats
    }

    /// Check if a block can be allocated (either from free pool or new memory).
    pub fn can_allocate(&self, block_type: &BlockType) -> bool {
        // Check free pool first
        if let Some(pool) = self.free_pools.get(block_type) {
            if !pool.is_empty() {
                return true;
            }
        }

        // Check if allocator has enough memory
        if let Some(config) = self.block_configs.get(block_type) {
            return self.allocator.get_available_memory() >= config.size_bytes;
        }

        false
    }
}

impl Drop for BlockManager {
    /// Free all physical memory when BlockManager is dropped.
    ///
    /// This ensures no memory leaks even if blocks weren't explicitly freed.
    fn drop(&mut self) {
        // Free all backing memory allocations
        for block in self.block_table.values() {
            if let Err(e) = self.allocator.free(block.mem_addr) {
                eprintln!(
                    "Warning: failed to free block {:?} memory: {:?}",
                    block.block_id, e
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block_manager::allocator::CpuAllocator;

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
