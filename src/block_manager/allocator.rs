use super::types::*;

/// Memory allocator trait
pub trait MemoryAllocator: Send + Sync {
    fn allocate(&mut self, size_bytes: usize) -> Result<MemoryAddress>;
    fn free(&mut self, addr: MemoryAddress) -> Result<()>;
    fn get_total_memory(&self) -> usize;
    fn get_available_memory(&self) -> usize;
}

/// Simple CPU memory allocator (for testing/CPU inference)
pub struct CpuAllocator {
    total_memory: usize,
    used_memory: usize,
}

impl CpuAllocator {
    pub fn new(total_memory: usize) -> Self {
        Self {
            total_memory,
            used_memory: 0,
        }
    }
}

impl MemoryAllocator for CpuAllocator {
    fn allocate(&mut self, size_bytes: usize) -> Result<MemoryAddress> {
        if self.used_memory + size_bytes > self.total_memory {
            return Err(Error::OutOfMemory);
        }

        // Allocate aligned memory
        let layout = std::alloc::Layout::from_size_align(size_bytes, 64)
            .map_err(|e| Error::AllocationError(e.to_string()))?;

        let ptr = unsafe { std::alloc::alloc(layout) };

        if ptr.is_null() {
            return Err(Error::OutOfMemory);
        }

        self.used_memory += size_bytes;

        Ok(MemoryAddress {
            ptr,
            size: size_bytes,
        })
    }

    fn free(&mut self, addr: MemoryAddress) -> Result<()> {
        let layout = std::alloc::Layout::from_size_align(addr.size, 64)
            .map_err(|e| Error::AllocationError(e.to_string()))?;

        unsafe {
            std::alloc::dealloc(addr.ptr, layout);
        }

        self.used_memory -= addr.size;

        Ok(())
    }

    fn get_total_memory(&self) -> usize {
        self.total_memory
    }

    fn get_available_memory(&self) -> usize {
        self.total_memory - self.used_memory
    }
}

// TODO: Implement CudaAllocator later
// pub struct CudaAllocator { ... }
