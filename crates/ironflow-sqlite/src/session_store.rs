use async_trait::async_trait;
use diesel::prelude::*;
use ironflow_core::{
    AgentError, MessageContent, MessageRole, Session, SessionMessage, SessionStore,
};

use crate::pool::{DbPool, with_conn};
use crate::schema::{agent_sessions, agent_messages};

#[derive(Queryable, Selectable, Clone)]
#[diesel(table_name = agent_sessions)]
struct SessionRow {
    id: String,
    scope_id: String,
    title: Option<String>,
    created_at: String,
    updated_at: String,
}

#[derive(Queryable, Selectable, Clone)]
#[diesel(table_name = agent_messages)]
struct MessageRow {
    id: String,
    session_id: String,
    role: String,
    content_type: String,
    content: String,
    created_at: String,
}

pub struct SqliteSessionStore {
    pool: DbPool,
}

impl SqliteSessionStore {
    pub fn new(pool: DbPool) -> Self {
        Self { pool }
    }
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn create_session(&self, scope_id: &str) -> Result<Session, AgentError> {
        let scope_id = scope_id.to_string();
        let row = with_conn(&self.pool, move |conn| {
            let id = uuid::Uuid::new_v4().to_string();
            let now = chrono::Utc::now().to_rfc3339();

            diesel::insert_into(agent_sessions::table)
                .values((
                    agent_sessions::id.eq(&id),
                    agent_sessions::scope_id.eq(&scope_id),
                    agent_sessions::created_at.eq(&now),
                    agent_sessions::updated_at.eq(&now),
                ))
                .execute(conn)?;

            Ok(SessionRow {
                id,
                scope_id,
                title: None,
                created_at: now.clone(),
                updated_at: now,
            })
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(row_to_session(row))
    }

    async fn get_session(&self, id: &str) -> Result<Option<Session>, AgentError> {
        let id = id.to_string();
        let row = with_conn(&self.pool, move |conn| {
            agent_sessions::table
                .filter(agent_sessions::id.eq(&id))
                .select(SessionRow::as_select())
                .first(conn)
                .optional()
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(row.map(row_to_session))
    }

    async fn list_sessions(&self, scope_id: &str) -> Result<Vec<Session>, AgentError> {
        let scope_id = scope_id.to_string();
        let rows = with_conn(&self.pool, move |conn| {
            agent_sessions::table
                .filter(agent_sessions::scope_id.eq(&scope_id))
                .order(agent_sessions::updated_at.desc())
                .select(SessionRow::as_select())
                .load(conn)
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        Ok(rows.into_iter().map(row_to_session).collect())
    }

    async fn update_title(&self, id: &str, title: &str) -> Result<(), AgentError> {
        let id = id.to_string();
        let title = title.to_string();
        with_conn(&self.pool, move |conn| {
            let now = chrono::Utc::now().to_rfc3339();
            diesel::update(agent_sessions::table.filter(agent_sessions::id.eq(&id)))
                .set((
                    agent_sessions::title.eq(&title),
                    agent_sessions::updated_at.eq(&now),
                ))
                .execute(conn)?;
            Ok(())
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))
    }

    async fn append_message(&self, msg: &SessionMessage) -> Result<(), AgentError> {
        let (content_type, content_json) = serialize_message_content(&msg.content)?;
        let role_str = match msg.role {
            MessageRole::User => "user",
            MessageRole::Assistant => "assistant",
            MessageRole::System => "system",
            MessageRole::Tool => "tool",
        };

        let id = msg.id.clone();
        let session_id = msg.session_id.clone();
        let role = role_str.to_string();
        let ct = content_type.to_string();
        let content = content_json;
        let created_at = msg.created_at.clone();

        with_conn(&self.pool, move |conn| {
            diesel::insert_into(agent_messages::table)
                .values((
                    agent_messages::id.eq(&id),
                    agent_messages::session_id.eq(&session_id),
                    agent_messages::role.eq(&role),
                    agent_messages::content_type.eq(&ct),
                    agent_messages::content.eq(&content),
                    agent_messages::created_at.eq(&created_at),
                ))
                .execute(conn)?;

            let now = chrono::Utc::now().to_rfc3339();
            diesel::update(agent_sessions::table.filter(agent_sessions::id.eq(&session_id)))
                .set(agent_sessions::updated_at.eq(&now))
                .execute(conn)?;

            Ok(())
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))
    }

    async fn get_messages(&self, session_id: &str) -> Result<Vec<SessionMessage>, AgentError> {
        let session_id = session_id.to_string();
        let rows = with_conn(&self.pool, move |conn| {
            agent_messages::table
                .filter(agent_messages::session_id.eq(&session_id))
                .order(agent_messages::created_at.asc())
                .select(MessageRow::as_select())
                .load(conn)
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))?;

        rows.into_iter()
            .map(|r| {
                let role = match r.role.as_str() {
                    "user" => MessageRole::User,
                    "assistant" => MessageRole::Assistant,
                    "system" => MessageRole::System,
                    "tool" => MessageRole::Tool,
                    _ => MessageRole::User,
                };
                let content = deserialize_message_content(&r.content_type, &r.content)?;
                Ok(SessionMessage {
                    id: r.id,
                    session_id: r.session_id,
                    role,
                    content,
                    created_at: r.created_at,
                })
            })
            .collect()
    }

    async fn delete_session(&self, id: &str) -> Result<(), AgentError> {
        let id = id.to_string();
        with_conn(&self.pool, move |conn| {
            diesel::delete(agent_sessions::table.filter(agent_sessions::id.eq(&id)))
                .execute(conn)?;
            Ok(())
        })
        .await
        .map_err(|e| AgentError::StorageError(e.to_string()))
    }
}

fn row_to_session(row: SessionRow) -> Session {
    Session {
        id: row.id,
        scope_id: row.scope_id,
        title: row.title,
        created_at: row.created_at,
        updated_at: row.updated_at,
    }
}

fn serialize_message_content(content: &MessageContent) -> Result<(&str, String), AgentError> {
    match content {
        MessageContent::Text(text) => Ok(("text", text.clone())),
        MessageContent::ToolCalls(calls) => {
            let json = serde_json::to_string(calls)
                .map_err(AgentError::from)?;
            Ok(("tool_calls", json))
        }
        MessageContent::ToolResults(results) => {
            let json = serde_json::to_string(results)
                .map_err(AgentError::from)?;
            Ok(("tool_results", json))
        }
        MessageContent::Summary(text) => Ok(("summary", text.clone())),
    }
}

fn deserialize_message_content(
    content_type: &str,
    content: &str,
) -> Result<MessageContent, AgentError> {
    match content_type {
        "text" => Ok(MessageContent::Text(content.to_string())),
        "tool_calls" => {
            let calls = serde_json::from_str(content)
                .map_err(AgentError::from)?;
            Ok(MessageContent::ToolCalls(calls))
        }
        "tool_results" => {
            let results = serde_json::from_str(content)
                .map_err(AgentError::from)?;
            Ok(MessageContent::ToolResults(results))
        }
        "summary" => Ok(MessageContent::Summary(content.to_string())),
        _ => Ok(MessageContent::Text(content.to_string())),
    }
}
