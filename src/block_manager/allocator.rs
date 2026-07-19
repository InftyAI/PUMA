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
            .map_err(|e| Error::AllocationFailed(e.to_string()))?;

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
        if addr.size > self.used_memory {
            return Err(Error::FreeFailed(
                "free() called with size larger than used_memory".to_string(),
            ));
        }

        let layout = std::alloc::Layout::from_size_align(addr.size, 64)
            .map_err(|e| Error::FreeFailed(e.to_string()))?;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_tracks_used_memory() {
        let mut alloc = CpuAllocator::new(10_000);
        assert_eq!(alloc.get_total_memory(), 10_000);
        assert_eq!(alloc.get_available_memory(), 10_000);

        let addr = alloc.allocate(1024).unwrap();
        assert_eq!(addr.size, 1024);
        assert!(!addr.ptr.is_null());
        assert_eq!(alloc.get_available_memory(), 10_000 - 1024);

        alloc.free(addr).unwrap();
        assert_eq!(alloc.get_available_memory(), 10_000);
    }

    #[test]
    fn test_allocate_out_of_memory() {
        let mut alloc = CpuAllocator::new(1024);

        // First allocation fits exactly.
        let addr = alloc.allocate(1024).unwrap();
        assert_eq!(alloc.get_available_memory(), 0);

        // Any further allocation exceeds capacity.
        assert!(matches!(alloc.allocate(1), Err(Error::OutOfMemory)));

        alloc.free(addr).unwrap();
    }

    #[test]
    fn test_allocate_exact_boundary() {
        let mut alloc = CpuAllocator::new(2048);
        let a = alloc.allocate(1024).unwrap();
        let b = alloc.allocate(1024).unwrap();
        assert_eq!(alloc.get_available_memory(), 0);
        alloc.free(a).unwrap();
        alloc.free(b).unwrap();
        assert_eq!(alloc.get_available_memory(), 2048);
    }

    #[test]
    fn test_free_more_than_used_errors() {
        let mut alloc = CpuAllocator::new(10_000);
        let addr = alloc.allocate(512).unwrap();

        // Fabricate an address claiming a larger size than is actually used.
        let bogus = MemoryAddress {
            ptr: addr.ptr,
            size: 4096,
        };
        assert!(matches!(alloc.free(bogus), Err(Error::FreeFailed(_))));

        // The real allocation is still accounted for and can be freed.
        assert_eq!(alloc.get_available_memory(), 10_000 - 512);
        alloc.free(addr).unwrap();
        assert_eq!(alloc.get_available_memory(), 10_000);
    }

    #[test]
    fn test_reuse_after_free() {
        let mut alloc = CpuAllocator::new(1024);
        for _ in 0..3 {
            let addr = alloc.allocate(1024).unwrap();
            assert_eq!(alloc.get_available_memory(), 0);
            alloc.free(addr).unwrap();
            assert_eq!(alloc.get_available_memory(), 1024);
        }
    }
}
