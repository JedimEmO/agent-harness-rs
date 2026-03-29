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
        let cat_str = category_to_str(category).to_string();

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
        let cat_str = category.map(|c| category_to_str(c).to_string());

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

fn category_to_str(cat: MemoryCategory) -> &'static str {
    match cat {
        MemoryCategory::Fact => "fact",
        MemoryCategory::Preference => "preference",
        MemoryCategory::Instruction => "instruction",
        MemoryCategory::Context => "context",
    }
}

fn str_to_category(s: &str) -> MemoryCategory {
    match s {
        "fact" => MemoryCategory::Fact,
        "preference" => MemoryCategory::Preference,
        "instruction" => MemoryCategory::Instruction,
        "context" => MemoryCategory::Context,
        _ => MemoryCategory::Fact,
    }
}

fn row_to_memory(row: MemoryRow) -> Memory {
    Memory {
        id: row.id,
        scope_id: row.scope_id,
        key: row.key,
        content: row.content,
        category: str_to_category(&row.category),
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}
