/// Block type - defines the kind of memory block
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// Standard KV cache blocks (for normal attention)
    StandardKV,

    /// Compressed KV cache blocks (for DeepSeek-style compression)
    CompressedKV,

    /// Compressor state blocks (for DeepSeek RNN compressor)
    CompressorState,
}

impl BlockType {
    pub fn as_str(&self) -> &'static str {
        match self {
            BlockType::StandardKV => "standard_kv",
            BlockType::CompressedKV => "compressed_kv",
            BlockType::CompressorState => "compressor_state",
        }
    }
}

impl std::fmt::Display for BlockType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u64);

impl BlockId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SequenceId(pub u64);

impl SequenceId {
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct MemoryAddress {
    pub ptr: *mut u8,
    pub size: usize,
}

unsafe impl Send for MemoryAddress {}
unsafe impl Sync for MemoryAddress {}

pub struct BlockConfig {
    pub block_type: BlockType,
    pub size_bytes: usize,
}

pub struct Block {
    pub block_id: BlockId,
    pub block_type: BlockType,
    pub mem_addr: MemoryAddress,
    pub ref_count: usize,
}

#[derive(Debug, Clone)]
pub struct BlockStats {
    pub total_blocks: usize,
    pub allocated_blocks: usize,
    pub free_blocks: usize,
    pub block_size: usize,
    pub total_memory: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unknown block type: {0}")]
    UnknownBlockType(String),

    #[error("Invalid block ID: {0:?}")]
    InvalidBlockId(BlockId),

    #[error("Double free detected for block: {0:?}")]
    DoubleFree(BlockId),

    #[error("Unknown sequence: {0:?}")]
    UnknownSequence(SequenceId),

    #[error("Out of memory")]
    OutOfMemory,

    #[error("Allocation error: {0}")]
    AllocationError(String),
}

pub type Result<T> = std::result::Result<T, Error>;
