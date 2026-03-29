use async_trait::async_trait;
use diesel::prelude::*;
use agent_harness_core::{
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::{
        MessageContent, MessageRole, SessionMessage, SessionStore,
        ToolCallRecord, ToolResultRecord,
    };
    use crate::init_db;

    fn test_pool() -> DbPool {
        init_db(":memory:").expect("failed to init in-memory db")
    }

    #[tokio::test]
    async fn create_and_get_session() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test-scope").await.unwrap();
        assert!(!session.id.is_empty());
        assert_eq!(session.scope_id, "test-scope");
        assert!(session.title.is_none());

        let fetched = store.get_session(&session.id).await.unwrap();
        assert!(fetched.is_some());
        let fetched = fetched.unwrap();
        assert_eq!(fetched.id, session.id);
        assert_eq!(fetched.scope_id, "test-scope");
    }

    #[tokio::test]
    async fn get_nonexistent_session() {
        let store = SqliteSessionStore::new(test_pool());
        let result = store.get_session("nonexistent").await.unwrap();
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn list_sessions_by_scope() {
        let store = SqliteSessionStore::new(test_pool());
        store.create_session("scope-a").await.unwrap();
        store.create_session("scope-a").await.unwrap();
        store.create_session("scope-b").await.unwrap();

        let a_sessions = store.list_sessions("scope-a").await.unwrap();
        assert_eq!(a_sessions.len(), 2);

        let b_sessions = store.list_sessions("scope-b").await.unwrap();
        assert_eq!(b_sessions.len(), 1);

        let empty = store.list_sessions("scope-c").await.unwrap();
        assert!(empty.is_empty());
    }

    #[tokio::test]
    async fn update_title() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();
        assert!(session.title.is_none());

        store.update_title(&session.id, "My Chat").await.unwrap();
        let fetched = store.get_session(&session.id).await.unwrap().unwrap();
        assert_eq!(fetched.title.as_deref(), Some("My Chat"));
    }

    #[tokio::test]
    async fn append_and_get_text_messages() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();

        let msg1 = SessionMessage {
            id: "msg-1".into(),
            session_id: session.id.clone(),
            role: MessageRole::User,
            content: MessageContent::Text("Hello".into()),
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        let msg2 = SessionMessage {
            id: "msg-2".into(),
            session_id: session.id.clone(),
            role: MessageRole::Assistant,
            content: MessageContent::Text("Hi there!".into()),
            created_at: "2026-01-01T00:00:01Z".into(),
        };

        store.append_message(&msg1).await.unwrap();
        store.append_message(&msg2).await.unwrap();

        let messages = store.get_messages(&session.id).await.unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, MessageRole::User);
        assert_eq!(messages[1].role, MessageRole::Assistant);
        match &messages[0].content {
            MessageContent::Text(t) => assert_eq!(t, "Hello"),
            _ => panic!("expected Text"),
        }
        match &messages[1].content {
            MessageContent::Text(t) => assert_eq!(t, "Hi there!"),
            _ => panic!("expected Text"),
        }
    }

    #[tokio::test]
    async fn tool_calls_roundtrip() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();

        let calls = vec![ToolCallRecord {
            call_id: "c1".into(),
            tool_name: "search".into(),
            arguments: serde_json::json!({"query": "rust"}),
        }];
        let msg = SessionMessage {
            id: "msg-tc".into(),
            session_id: session.id.clone(),
            role: MessageRole::Assistant,
            content: MessageContent::ToolCalls(calls),
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        store.append_message(&msg).await.unwrap();

        let messages = store.get_messages(&session.id).await.unwrap();
        assert_eq!(messages.len(), 1);
        match &messages[0].content {
            MessageContent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].call_id, "c1");
                assert_eq!(calls[0].tool_name, "search");
                assert_eq!(calls[0].arguments, serde_json::json!({"query": "rust"}));
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[tokio::test]
    async fn tool_results_roundtrip() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();

        let results = vec![ToolResultRecord {
            call_id: "c1".into(),
            tool_name: "search".into(),
            content: serde_json::json!({"results": ["a", "b"]}),
        }];
        let msg = SessionMessage {
            id: "msg-tr".into(),
            session_id: session.id.clone(),
            role: MessageRole::Tool,
            content: MessageContent::ToolResults(results),
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        store.append_message(&msg).await.unwrap();

        let messages = store.get_messages(&session.id).await.unwrap();
        assert_eq!(messages.len(), 1);
        match &messages[0].content {
            MessageContent::ToolResults(results) => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].call_id, "c1");
                assert_eq!(results[0].content, serde_json::json!({"results": ["a", "b"]}));
            }
            _ => panic!("expected ToolResults"),
        }
    }

    #[tokio::test]
    async fn summary_roundtrip() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();

        let msg = SessionMessage {
            id: "msg-s".into(),
            session_id: session.id.clone(),
            role: MessageRole::Assistant,
            content: MessageContent::Summary("previous context".into()),
            created_at: "2026-01-01T00:00:00Z".into(),
        };
        store.append_message(&msg).await.unwrap();

        let messages = store.get_messages(&session.id).await.unwrap();
        match &messages[0].content {
            MessageContent::Summary(s) => assert_eq!(s, "previous context"),
            _ => panic!("expected Summary"),
        }
    }

    #[tokio::test]
    async fn delete_session() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();
        assert!(store.get_session(&session.id).await.unwrap().is_some());

        store.delete_session(&session.id).await.unwrap();
        assert!(store.get_session(&session.id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn messages_ordered_by_created_at() {
        let store = SqliteSessionStore::new(test_pool());
        let session = store.create_session("test").await.unwrap();

        // Insert in reverse order
        for i in (1..=3).rev() {
            let msg = SessionMessage {
                id: format!("msg-{}", i),
                session_id: session.id.clone(),
                role: MessageRole::User,
                content: MessageContent::Text(format!("message {}", i)),
                created_at: format!("2026-01-01T00:00:0{}Z", i),
            };
            store.append_message(&msg).await.unwrap();
        }

        let messages = store.get_messages(&session.id).await.unwrap();
        assert_eq!(messages.len(), 3);
        // Should be sorted by created_at ascending
        assert_eq!(messages[0].id, "msg-1");
        assert_eq!(messages[1].id, "msg-2");
        assert_eq!(messages[2].id, "msg-3");
    }

    #[test]
    fn serialize_deserialize_text() {
        let mc = MessageContent::Text("hello".into());
        let (ct, content) = serialize_message_content(&mc).unwrap();
        assert_eq!(ct, "text");
        assert_eq!(content, "hello");
        let round = deserialize_message_content(ct, &content).unwrap();
        match round {
            MessageContent::Text(t) => assert_eq!(t, "hello"),
            _ => panic!("expected Text"),
        }
    }

    #[test]
    fn serialize_deserialize_tool_calls() {
        let calls = vec![ToolCallRecord {
            call_id: "c1".into(),
            tool_name: "test".into(),
            arguments: serde_json::json!({"x": 1}),
        }];
        let mc = MessageContent::ToolCalls(calls);
        let (ct, content) = serialize_message_content(&mc).unwrap();
        assert_eq!(ct, "tool_calls");
        let round = deserialize_message_content(ct, &content).unwrap();
        match round {
            MessageContent::ToolCalls(c) => {
                assert_eq!(c.len(), 1);
                assert_eq!(c[0].call_id, "c1");
            }
            _ => panic!("expected ToolCalls"),
        }
    }

    #[test]
    fn deserialize_unknown_content_type_falls_back_to_text() {
        let result = deserialize_message_content("unknown_type", "some data").unwrap();
        match result {
            MessageContent::Text(t) => assert_eq!(t, "some data"),
            _ => panic!("expected Text fallback"),
        }
    }
}
