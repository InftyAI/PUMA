use super::allocator::*;
use super::types::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct BlockManager {
    allocator: Box<dyn MemoryAllocator>,

    // Block registry
    block_table: HashMap<BlockId, Block>,

    // Free pools per type
    free_pools: HashMap<BlockType, Vec<BlockId>>,

    // Block configurations
    block_configs: HashMap<BlockType, BlockConfig>,

    // Default type
    default_block_type: BlockType,

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

    pub fn register_block_type(&mut self, config: BlockConfig) {
        self.free_pools.insert(config.block_type, Vec::new());
        self.block_configs.insert(config.block_type, config);
    }

    pub fn allocate_typed(&mut self, block_type: &BlockType) -> Result<BlockId> {
        // Try free pool first
        if let Some(block_id) = self
            .free_pools
            .get_mut(block_type)
            .and_then(|pool| pool.pop())
        {
            let block = self.block_table.get_mut(&block_id).unwrap();
            block.ref_count = 1;
            return Ok(block_id);
        }

        // Allocate new physical memory
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

    pub fn allocate(&mut self) -> Result<BlockId> {
        let default_type = self.default_block_type;
        self.allocate_typed(&default_type)
    }

    pub fn free(&mut self, block_id: BlockId) -> Result<()> {
        let block = self
            .block_table
            .get_mut(&block_id)
            .ok_or(Error::InvalidBlockId(block_id))?;

        if block.ref_count == 0 {
            return Err(Error::DoubleFree(block_id));
        }

        block.ref_count -= 1;

        if block.ref_count == 0 {
            self.free_pools
                .get_mut(&block.block_type)
                .unwrap()
                .push(block_id);
        }

        Ok(())
    }

    pub fn add_ref(&mut self, block_id: BlockId) -> Result<()> {
        let block = self
            .block_table
            .get_mut(&block_id)
            .ok_or(Error::InvalidBlockId(block_id))?;
        block.ref_count += 1;
        Ok(())
    }

    pub fn get_memory_address(&self, block_id: BlockId) -> Result<MemoryAddress> {
        self.block_table
            .get(&block_id)
            .map(|b| b.mem_addr)
            .ok_or(Error::InvalidBlockId(block_id))
    }

    pub fn get_block_type(&self, block_id: BlockId) -> Result<&BlockType> {
        self.block_table
            .get(&block_id)
            .map(|b| &b.block_type)
            .ok_or(Error::InvalidBlockId(block_id))
    }

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

    pub fn can_allocate(&self, block_type: &BlockType) -> bool {
        if let Some(pool) = self.free_pools.get(block_type) {
            if !pool.is_empty() {
                return true;
            }
        }

        if let Some(config) = self.block_configs.get(block_type) {
            return self.allocator.get_available_memory() >= config.size_bytes;
        }

        false
    }
}

impl Drop for BlockManager {
    fn drop(&mut self) {
        // Free all backing memory allocations to prevent leaks
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
