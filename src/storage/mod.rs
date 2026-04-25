pub mod sqlite;
pub mod storage_trait;

pub use storage_trait::ModelStorage;
pub use sqlite::SqliteStorage;
