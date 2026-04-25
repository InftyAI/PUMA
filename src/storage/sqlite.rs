use crate::registry::model_registry::{ModelInfo, ModelMetadata};
use crate::storage::ModelStorage;
use rusqlite::{params, Connection, Result as SqlResult};
use rusqlite_migration::{Migrations, M};
use std::collections::HashMap;
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
                CREATE INDEX idx_author ON models(author);
                CREATE INDEX idx_type ON models(type);
                CREATE INDEX idx_model_series ON models(model_series);
                CREATE INDEX idx_provider ON models(provider);
                CREATE INDEX idx_license ON models(license);
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
    fn load_models(
        &self,
        filters: Option<&HashMap<String, String>>,
    ) -> Result<Vec<ModelInfo>, io::Error> {
        let conn = self.get_connection()?;

        // Build WHERE clause from filters
        let mut where_clauses = Vec::new();
        let mut params: Vec<String> = Vec::new();

        if let Some(filter_map) = filters {
            // Allowed columns for filtering (prevent SQL injection)
            let allowed_columns = ["author", "type", "model_series", "provider", "license"];

            for (key, value) in filter_map {
                if allowed_columns.contains(&key.as_str()) {
                    where_clauses.push(format!("{} = ?", key));
                    params.push(value.clone());
                } else {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("Invalid filter column: {}", key),
                    ));
                }
            }
        }

        let query = if where_clauses.is_empty() {
            "SELECT uuid, name, author, type, model_series, provider, license,
                    metadata, created_at, updated_at
             FROM models"
                .to_string()
        } else {
            format!(
                "SELECT uuid, name, author, type, model_series, provider, license,
                        metadata, created_at, updated_at
                 FROM models
                 WHERE {}",
                where_clauses.join(" AND ")
            )
        };

        let mut stmt = conn.prepare(&query).map_err(io::Error::other)?;

        let param_refs: Vec<&dyn rusqlite::ToSql> =
            params.iter().map(|p| p as &dyn rusqlite::ToSql).collect();

        let models = stmt
            .query_map(param_refs.as_slice(), |row| {
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

        let metadata_json = serde_json::to_string(&model.metadata)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Normalize name and author to lowercase
        let name_lower = model.name.to_lowercase();
        let author_lower = model.author.as_ref().map(|a| a.to_lowercase());

        conn.execute(
            "INSERT INTO models
             (uuid, name, author, type, model_series, provider, license,
              metadata, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(name) DO UPDATE SET
                uuid = excluded.uuid,
                author = excluded.author,
                type = excluded.type,
                model_series = excluded.model_series,
                provider = excluded.provider,
                license = excluded.license,
                metadata = excluded.metadata,
                updated_at = excluded.updated_at",
            params![
                &model.uuid,
                &name_lower,
                author_lower.as_deref(),
                model.r#type.as_deref(),
                model.model_series.as_deref(),
                &model.provider,
                model.license.as_deref(),
                &metadata_json,
                &model.created_at,
                &model.updated_at,
            ],
        )
        .map_err(io::Error::other)?;

        Ok(())
    }

    fn unregister_model(&self, name: &str) -> Result<(), io::Error> {
        let conn = self.get_connection()?;

        // Normalize name to lowercase for case-insensitive lookup
        let name_lower = name.to_lowercase();

        conn.execute("DELETE FROM models WHERE name = ?1", params![name_lower])
            .map_err(io::Error::other)?;

        Ok(())
    }

    fn get_model(&self, name: &str) -> Result<Option<ModelInfo>, io::Error> {
        let conn = self.get_connection()?;

        // Normalize name to lowercase for case-insensitive lookup
        let name_lower = name.to_lowercase();

        let result = conn.query_row(
            "SELECT uuid, name, author, type, model_series, provider, license,
                    metadata, created_at, updated_at
             FROM models WHERE name = ?1",
            params![name_lower],
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

        let models = storage.load_models(None).unwrap();
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
        assert_eq!(storage.load_models(None).unwrap().len(), 1);

        storage.unregister_model("test/model").unwrap();
        assert_eq!(storage.load_models(None).unwrap().len(), 0);
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

        let models = storage.load_models(None).unwrap();
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

    #[test]
    fn test_load_models_with_single_filter() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let mut model1 = create_test_model("test/model1", "uuid1");
        model1.author = Some("author1".to_string());
        storage.register_model(model1).unwrap();

        let mut model2 = create_test_model("test/model2", "uuid2");
        model2.author = Some("author2".to_string());
        storage.register_model(model2).unwrap();

        let mut filters = HashMap::new();
        filters.insert("author".to_string(), "author1".to_string());

        let models = storage.load_models(Some(&filters)).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model1");
    }

    #[test]
    fn test_load_models_with_multiple_filters() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let mut model1 = create_test_model("test/model1", "uuid1");
        model1.author = Some("InftyAI".to_string());
        model1.license = Some("mit".to_string());
        storage.register_model(model1).unwrap();

        let mut model2 = create_test_model("test/model2", "uuid2");
        model2.author = Some("InftyAI".to_string());
        model2.license = Some("apache-2.0".to_string());
        storage.register_model(model2).unwrap();

        let mut model3 = create_test_model("test/model3", "uuid3");
        model3.author = Some("other-author".to_string());
        model3.license = Some("mit".to_string());
        storage.register_model(model3).unwrap();

        let mut filters = HashMap::new();
        filters.insert("author".to_string(), "inftyai".to_string());
        filters.insert("license".to_string(), "mit".to_string());

        let models = storage.load_models(Some(&filters)).unwrap();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "test/model1");
    }

    #[test]
    fn test_load_models_with_invalid_filter_column() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let mut filters = HashMap::new();
        filters.insert("invalid_column".to_string(), "value".to_string());

        let result = storage.load_models(Some(&filters));
        assert!(result.is_err());
        assert_eq!(result.unwrap_err().kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn test_name_and_author_stored_lowercase() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let mut model = create_test_model("InftyAI/TestModel", "uuid123");
        model.author = Some("InftyAI".to_string());
        storage.register_model(model).unwrap();

        // Query with original case should work
        let retrieved = storage.get_model("InftyAI/TestModel").unwrap();
        assert!(retrieved.is_some());
        let model_info = retrieved.unwrap();
        // Verify stored as lowercase
        assert_eq!(model_info.name, "inftyai/testmodel");
        assert_eq!(model_info.author, Some("inftyai".to_string()));

        // Query with different case should also work
        let retrieved2 = storage.get_model("inftyai/testmodel").unwrap();
        assert!(retrieved2.is_some());

        let retrieved3 = storage.get_model("INFTYAI/TESTMODEL").unwrap();
        assert!(retrieved3.is_some());
    }

    #[test]
    fn test_author_filter_case_sensitive() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("test.db");
        let storage = SqliteStorage::new(db_path).unwrap();

        let mut model = create_test_model("test/model", "uuid123");
        model.author = Some("InftyAI".to_string());
        storage.register_model(model).unwrap();

        // Filter must use lowercase since data is stored in lowercase
        let mut filters = HashMap::new();
        filters.insert("author".to_string(), "inftyai".to_string());
        assert_eq!(storage.load_models(Some(&filters)).unwrap().len(), 1);

        // Non-lowercase filter won't match
        filters.clear();
        filters.insert("author".to_string(), "InftyAI".to_string());
        assert_eq!(storage.load_models(Some(&filters)).unwrap().len(), 0);
    }
}
