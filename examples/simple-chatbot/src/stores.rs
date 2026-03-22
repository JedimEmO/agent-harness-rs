//! In-memory implementations of SessionStore and MemoryStore.
//!
//! These are intentionally simple — they demonstrate how to implement
//! the traits without any external dependencies.

use std::collections::HashMap;
use std::sync::Arc;
use async_trait::async_trait;
use tokio::sync::RwLock;
use ironflow_core::*;

// --- In-Memory Session Store ---

pub struct InMemorySessionStore {
    sessions: Arc<RwLock<HashMap<String, Session>>>,
    messages: Arc<RwLock<HashMap<String, Vec<SessionMessage>>>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self {
            sessions: Arc::new(RwLock::new(HashMap::new())),
            messages: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn create_session(&self, scope_id: &str) -> Result<Session, AgentError> {
        let session = Session {
            id: uuid::Uuid::new_v4().to_string(),
            scope_id: scope_id.to_string(),
            title: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            updated_at: chrono::Utc::now().to_rfc3339(),
        };
        self.sessions.write().await.insert(session.id.clone(), session.clone());
        Ok(session)
    }

    async fn get_session(&self, id: &str) -> Result<Option<Session>, AgentError> {
        Ok(self.sessions.read().await.get(id).cloned())
    }

    async fn list_sessions(&self, scope_id: &str) -> Result<Vec<Session>, AgentError> {
        Ok(self.sessions.read().await.values()
            .filter(|s| s.scope_id == scope_id)
            .cloned()
            .collect())
    }

    async fn update_title(&self, id: &str, title: &str) -> Result<(), AgentError> {
        if let Some(s) = self.sessions.write().await.get_mut(id) {
            s.title = Some(title.to_string());
        }
        Ok(())
    }

    async fn append_message(&self, msg: &SessionMessage) -> Result<(), AgentError> {
        self.messages.write().await
            .entry(msg.session_id.clone())
            .or_default()
            .push(msg.clone());
        Ok(())
    }

    async fn get_messages(&self, session_id: &str) -> Result<Vec<SessionMessage>, AgentError> {
        Ok(self.messages.read().await
            .get(session_id)
            .cloned()
            .unwrap_or_default())
    }

    async fn delete_session(&self, id: &str) -> Result<(), AgentError> {
        self.sessions.write().await.remove(id);
        self.messages.write().await.remove(id);
        Ok(())
    }
}

// --- In-Memory Memory Store ---

pub struct InMemoryMemoryStore {
    memories: Arc<RwLock<Vec<Memory>>>,
}

impl InMemoryMemoryStore {
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

#[async_trait]
impl MemoryStore for InMemoryMemoryStore {
    async fn save(
        &self,
        scope_id: &str,
        key: &str,
        content: &str,
        category: MemoryCategory,
    ) -> Result<Memory, AgentError> {
        let mut memories = self.memories.write().await;
        let now = chrono::Utc::now().to_rfc3339();

        // Update if exists
        if let Some(existing) = memories.iter_mut().find(|m| m.scope_id == scope_id && m.key == key) {
            existing.content = content.to_string();
            existing.category = category;
            existing.updated_at = now;
            return Ok(existing.clone());
        }

        let memory = Memory {
            id: uuid::Uuid::new_v4().to_string(),
            scope_id: scope_id.to_string(),
            key: key.to_string(),
            content: content.to_string(),
            category,
            created_at: now.clone(),
            updated_at: now,
        };
        memories.push(memory.clone());
        Ok(memory)
    }

    async fn search(
        &self,
        scope_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, AgentError> {
        let memories = self.memories.read().await;
        let query_lower = query.to_lowercase();
        Ok(memories.iter()
            .filter(|m| m.scope_id == scope_id && (
                m.key.to_lowercase().contains(&query_lower) ||
                m.content.to_lowercase().contains(&query_lower)
            ))
            .take(limit)
            .cloned()
            .collect())
    }

    async fn list(
        &self,
        scope_id: &str,
        category: Option<MemoryCategory>,
    ) -> Result<Vec<Memory>, AgentError> {
        let memories = self.memories.read().await;
        Ok(memories.iter()
            .filter(|m| {
                m.scope_id == scope_id &&
                category.map_or(true, |c| m.category == c)
            })
            .cloned()
            .collect())
    }

    async fn delete(&self, id: &str) -> Result<bool, AgentError> {
        let mut memories = self.memories.write().await;
        let len_before = memories.len();
        memories.retain(|m| m.id != id);
        Ok(memories.len() < len_before)
    }
}
