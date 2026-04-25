use crate::registry::model_registry::{ModelInfo, ModelMetadata};
use crate::storage::ModelStorage;
use rusqlite::{params, Connection, Result as SqlResult};
use rusqlite_migration::{Migrations, M};
use std::io;
use std::path::PathBuf;

pub struct SqliteStorage {
    db_path: PathBuf,
}

impl SqliteStorage {
    pub fn new(db_path: PathBuf) -> Result<Self, io::Error> {
        let storage = Self { db_path };
        storage.run_migrations()?;
        Ok(storage)
    }

    fn run_migrations(&self) -> Result<(), io::Error> {
        let mut conn = self.get_connection()?;

        let migrations = Migrations::new(vec![
            M::up(
                "CREATE TABLE models (
                    uuid TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    author TEXT,
                    type TEXT,
                    model_series TEXT,
                    provider TEXT NOT NULL,
                    license TEXT,
                    metadata JSON NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    CHECK(json_valid(metadata))
                );
                CREATE INDEX idx_name ON models(name);
                CREATE INDEX idx_provider ON models(provider);
                CREATE INDEX idx_created_at ON models(created_at);",
            ),
            // Future migrations go here
        ]);

        migrations.to_latest(&mut conn).map_err(io::Error::other)?;

        Ok(())
    }

    fn get_connection(&self) -> Result<Connection, io::Error> {
        Connection::open(&self.db_path).map_err(io::Error::other)
    }
}

impl ModelStorage for SqliteStorage {
    fn load_models(&self) -> Result<Vec<ModelInfo>, io::Error> {
        let conn = self.get_connection()?;

        let mut stmt = conn
            .prepare(
                "SELECT uuid, name, author, type, model_series, provider, license,
                        metadata, created_at, updated_at
                 FROM models",
            )
            .map_err(io::Error::other)?;

        let models = stmt
            .query_map([], |row| {
                let metadata_json: String = row.get(7)?;
                let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                Ok(ModelInfo {
                    uuid: row.get(0)?,
                    name: row.get(1)?,
                    author: row.get(2)?,
                    r#type: row.get(3)?,
                    model_series: row.get(4)?,
                    provider: row.get(5)?,
                    license: row.get(6)?,
                    metadata,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                })
            })
            .map_err(io::Error::other)?
            .collect::<SqlResult<Vec<_>>>()
            .map_err(io::Error::other)?;

        Ok(models)
    }

    fn register_model(&self, model: ModelInfo) -> Result<(), io::Error> {
        let conn = self.get_connection()?;

        // Check if model exists to preserve created_at
        let existing_created_at: Option<String> = conn
            .query_row(
                "SELECT created_at FROM models WHERE name = ?1",
                params![&model.name],
                |row| row.get(0),
            )
            .ok();

        let created_at = existing_created_at.unwrap_or_else(|| model.created_at.clone());

        let metadata_json = serde_json::to_string(&model.metadata)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        conn.execute(
            "INSERT OR REPLACE INTO models
             (uuid, name, author, type, model_series, provider, license,
              metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                &model.uuid,
                &model.name,
                model.author.as_deref(),
                model.r#type.as_deref(),
                model.model_series.as_deref(),
                &model.provider,
                model.license.as_deref(),
                &metadata_json,
                &created_at,
                &model.updated_at,
            ],
        )
        .map_err(io::Error::other)?;

        Ok(())
    }

    fn unregister_model(&self, name: &str) -> Result<(), io::Error> {
        let conn = self.get_connection()?;

        conn.execute("DELETE FROM models WHERE name = ?1", params![name])
            .map_err(io::Error::other)?;

        Ok(())
    }

    fn get_model(&self, name: &str) -> Result<Option<ModelInfo>, io::Error> {
        let conn = self.get_connection()?;

        let result = conn.query_row(
            "SELECT uuid, name, author, type, model_series, provider, license,
                    metadata, created_at, updated_at
             FROM models WHERE name = ?1",
            params![name],
            |row| {
                let metadata_json: String = row.get(7)?;
                let metadata: ModelMetadata = serde_json::from_str(&metadata_json)
                    .map_err(|e| rusqlite::Error::ToSqlConversionFailure(Box::new(e)))?;

                Ok(ModelInfo {
                    uuid: row.get(0)?,
                    name: row.get(1)?,
                    author: row.get(2)?,
                    r#type: row.get(3)?,
                    model_series: row.get(4)?,
                    provider: row.get(5)?,
                    license: row.get(6)?,
                    created_at: row.get(8)?,
                    updated_at: row.get(9)?,
                    metadata,
                })
            },
        );

        match result {
            Ok(model) => Ok(Some(model)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(io::Error::other(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::model_registry::{ArtifactInfo, ModelMetadata};
    use tempfile::TempDir;

    fn create_test_model(name: &str, uuid: &str) -> ModelInfo {
        let safetensors = serde_json::json!({
            "parameters": {"F32": 1000u64},
            "total": 1000u64
        });

        ModelInfo {
            uuid: uuid.to_string(),
            name: name.to_string(),
            author: Some("test-author".to_string()),
            r#type: Some("text-generation".to_string()),
            model_series: Some("gpt2".to_string()),
            provider: "huggingface".to_string(),
            license: Some("mit".to_string()),
            created_at: "2025-01-01T00:00:00Z".to_string(),
            updated_at: "2025-01-01T00:00:00Z".to_string(),
            metadata: ModelMetadata {
                artifact: ArtifactInfo {
                    revision: "abc123".to_string(),
                    size: 1000,
                    path: "/tmp/test".to_string(),
                },
                context_window: Some(2048),
                safetensors: Some(safetensors),
            },
        }
    }

    #[test]
    fn test_sqlite_register_and_load() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let model = create_test_model("test/model", "uuid123");
        storage.register_model(model.clone()).unwrap();

        let models = storage.load_models().unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model");
        assert_eq!(models[0].uuid, "uuid123");
    }

    #[test]
    fn test_sqlite_get_model() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let model = create_test_model("test/model", "uuid123");
        storage.register_model(model).unwrap();

        let result = storage.get_model("test/model").unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().name, "test/model");

        let not_found = storage.get_model("nonexistent").unwrap();
        assert!(not_found.is_none());
    }

    #[test]
    fn test_sqlite_unregister() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let model = create_test_model("test/model", "uuid123");
        storage.register_model(model).unwrap();
        assert_eq!(storage.load_models().unwrap().len(), 1);

        storage.unregister_model("test/model").unwrap();
        assert_eq!(storage.load_models().unwrap().len(), 0);
    }

    #[test]
    fn test_sqlite_update_preserves_created_at() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let model1 = create_test_model("test/model", "uuid1");
        storage.register_model(model1).unwrap();

        let mut model2 = create_test_model("test/model", "uuid2");
        model2.created_at = "2025-01-02T00:00:00Z".to_string();
        model2.updated_at = "2025-01-02T00:00:00Z".to_string();
        storage.register_model(model2).unwrap();

        let models = storage.load_models().unwrap();
        assert_eq!(models.len(), 1);
        // created_at should be preserved
        assert_eq!(models[0].created_at, "2025-01-01T00:00:00Z");
        // updated_at should be new
        assert_eq!(models[0].updated_at, "2025-01-02T00:00:00Z");
        // uuid should be updated
        assert_eq!(models[0].uuid, "uuid2");
    }

    #[test]
    fn test_sqlite_metadata_json() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let model = create_test_model("test/model", "uuid123");
        storage.register_model(model).unwrap();

        let retrieved = storage.get_model("test/model").unwrap().unwrap();
        assert_eq!(retrieved.metadata.artifact.revision, "abc123");
        assert_eq!(retrieved.metadata.artifact.size, 1000);
        assert_eq!(retrieved.metadata.context_window, Some(2048));
        assert!(retrieved.metadata.safetensors.is_some());

        let st = retrieved.metadata.safetensors.unwrap();
        assert_eq!(st.get("total").unwrap().as_u64().unwrap(), 1000);
    }
}
