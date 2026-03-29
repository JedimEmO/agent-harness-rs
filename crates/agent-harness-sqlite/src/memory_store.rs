use async_trait::async_trait;
use diesel::prelude::*;
use agent_harness_core::{AgentError, Memory, MemoryCategory, MemoryStore};

use crate::pool::{DbPool, with_conn};
use crate::schema::agent_memories;

#[derive(Queryable, Selectable, Clone)]
#[diesel(table_name = agent_memories)]
struct MemoryRow {
    id: String,
    scope_id: String,
    key: String,
    content: String,
    category: String,
    created_at: String,
    updated_at: String,
}

pub struct SqliteMemoryStore {
    pool: DbPool,
}

impl SqliteMemoryStore {
    pub fn new(pool: DbPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl MemoryStore for SqliteMemoryStore {
    async fn save(
        &self,
        scope_id: &str,
        key: &str,
        content: &str,
        category: MemoryCategory,
    ) -> Result<Memory, AgentError> {
        let scope_id = scope_id.to_string();
        let key = key.to_string();
        let content = content.to_string();
        let cat_str = category.to_string();

        let row = with_conn(&self.pool, move |conn| {
            let now = chrono::Utc::now().to_rfc3339();

            let existing = agent_memories::table
                .filter(agent_memories::scope_id.eq(&scope_id))
                .filter(agent_memories::key.eq(&key))
                .select(MemoryRow::as_select())
                .first(conn)
                .optional()?;

            if let Some(existing) = existing {
                diesel::update(agent_memories::table.filter(agent_memories::id.eq(&existing.id)))
                    .set((
                        agent_memories::content.eq(&content),
                        agent_memories::category.eq(&cat_str),
                        agent_memories::updated_at.eq(&now),
                    ))
                    .execute(conn)?;

                Ok(MemoryRow {
                    id: existing.id,
                    scope_id,
                    key,
                    content,
                    category: cat_str,
                    created_at: existing.created_at,
                    updated_at: now,
                })
            } else {
                let id = uuid::Uuid::new_v4().to_string();
                diesel::insert_into(agent_memories::table)
                    .values((
                        agent_memories::id.eq(&id),
                        agent_memories::scope_id.eq(&scope_id),
                        agent_memories::key.eq(&key),
                        agent_memories::content.eq(&content),
                        agent_memories::category.eq(&cat_str),
                        agent_memories::created_at.eq(&now),
                        agent_memories::updated_at.eq(&now),
                    ))
                    .execute(conn)?;

                Ok(MemoryRow {
                    id,
                    scope_id,
                    key,
                    content,
                    category: cat_str,
                    created_at: now.clone(),
                    updated_at: now,
                })
            }
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(row_to_memory(row))
    }

    async fn search(
        &self,
        scope_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, AgentError> {
        let scope_id = scope_id.to_string();
        let pattern = format!("%{}%", query);
        let limit = limit as i64;

        let rows = with_conn(&self.pool, move |conn| {
            agent_memories::table
                .filter(agent_memories::scope_id.eq(&scope_id))
                .filter(
                    agent_memories::key.like(&pattern)
                        .or(agent_memories::content.like(&pattern))
                )
                .order(agent_memories::updated_at.desc())
                .limit(limit)
                .select(MemoryRow::as_select())
                .load(conn)
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(rows.into_iter().map(row_to_memory).collect())
    }

    async fn list(
        &self,
        scope_id: &str,
        category: Option<MemoryCategory>,
    ) -> Result<Vec<Memory>, AgentError> {
        let scope_id = scope_id.to_string();
        let cat_str = category.map(|c| c.to_string());

        let rows = with_conn(&self.pool, move |conn| {
            let mut query = agent_memories::table
                .filter(agent_memories::scope_id.eq(&scope_id))
                .order(agent_memories::updated_at.desc())
                .into_boxed();

            if let Some(cat) = &cat_str {
                query = query.filter(agent_memories::category.eq(cat));
            }

            query.select(MemoryRow::as_select()).load(conn)
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(rows.into_iter().map(row_to_memory).collect())
    }

    async fn delete(&self, id: &str) -> Result<bool, AgentError> {
        let id = id.to_string();
        let count = with_conn(&self.pool, move |conn| {
            diesel::delete(agent_memories::table.filter(agent_memories::id.eq(&id)))
                .execute(conn)
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(count > 0)
    }
}

fn row_to_memory(row: MemoryRow) -> Memory {
    Memory {
        id: row.id,
        scope_id: row.scope_id,
        key: row.key,
        content: row.content,
        category: row.category.parse().unwrap_or(MemoryCategory::Fact),
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::{MemoryCategory, MemoryStore};
    use crate::init_db;

    fn test_pool() -> DbPool {
        init_db(":memory:").expect("failed to init in-memory db")
    }

    #[tokio::test]
    async fn save_and_search() {
        let store = SqliteMemoryStore::new(test_pool());
        store.save("scope1", "color", "user likes blue", MemoryCategory::Preference).await.unwrap();

        let results = store.search("scope1", "blue", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].key, "color");
        assert_eq!(results[0].content, "user likes blue");
        assert_eq!(results[0].category, MemoryCategory::Preference);
    }

    #[tokio::test]
    async fn save_upserts_existing_key() {
        let store = SqliteMemoryStore::new(test_pool());
        store.save("scope1", "color", "blue", MemoryCategory::Preference).await.unwrap();
        store.save("scope1", "color", "red", MemoryCategory::Preference).await.unwrap();

        let results = store.search("scope1", "color", 10).await.unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "red");
    }

    #[tokio::test]
    async fn search_respects_scope() {
        let store = SqliteMemoryStore::new(test_pool());
        store.save("scope-a", "key", "value", MemoryCategory::Fact).await.unwrap();

        let results = store.search("scope-b", "value", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn search_by_key() {
        let store = SqliteMemoryStore::new(test_pool());
        store.save("s", "project_goals", "build an AI", MemoryCategory::Fact).await.unwrap();

        let results = store.search("s", "project", 10).await.unwrap();
        assert_eq!(results.len(), 1);
    }

    #[tokio::test]
    async fn search_respects_limit() {
        let store = SqliteMemoryStore::new(test_pool());
        for i in 0..5 {
            store.save("s", &format!("key{}", i), "matching content", MemoryCategory::Fact).await.unwrap();
        }

        let results = store.search("s", "matching", 3).await.unwrap();
        assert_eq!(results.len(), 3);
    }

    #[tokio::test]
    async fn list_by_category() {
        let store = SqliteMemoryStore::new(test_pool());
        store.save("s", "fact1", "the sky is blue", MemoryCategory::Fact).await.unwrap();
        store.save("s", "pref1", "likes dark mode", MemoryCategory::Preference).await.unwrap();

        let facts = store.list("s", Some(MemoryCategory::Fact)).await.unwrap();
        assert_eq!(facts.len(), 1);
        assert_eq!(facts[0].key, "fact1");

        let prefs = store.list("s", Some(MemoryCategory::Preference)).await.unwrap();
        assert_eq!(prefs.len(), 1);
        assert_eq!(prefs[0].key, "pref1");

        let all = store.list("s", None).await.unwrap();
        assert_eq!(all.len(), 2);
    }

    #[tokio::test]
    async fn delete_memory() {
        let store = SqliteMemoryStore::new(test_pool());
        let mem = store.save("s", "key", "value", MemoryCategory::Fact).await.unwrap();

        let deleted = store.delete(&mem.id).await.unwrap();
        assert!(deleted);

        let results = store.search("s", "value", 10).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn delete_nonexistent() {
        let store = SqliteMemoryStore::new(test_pool());
        let deleted = store.delete("nonexistent-id").await.unwrap();
        assert!(!deleted);
    }

    #[test]
    fn category_roundtrip() {
        for cat in [MemoryCategory::Fact, MemoryCategory::Preference, MemoryCategory::Instruction, MemoryCategory::Context] {
            let s = cat.to_string();
            let back: MemoryCategory = s.parse().unwrap();
            assert_eq!(cat, back);
        }
    }

    #[test]
    fn unknown_category_defaults_to_fact() {
        assert_eq!("bogus".parse::<MemoryCategory>().unwrap_or(MemoryCategory::Fact), MemoryCategory::Fact);
    }
}
