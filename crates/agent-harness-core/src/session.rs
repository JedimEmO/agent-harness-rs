use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AgentError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: String,
    pub scope_id: String,
    pub title: Option<String>,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageRole {
    User,
    Assistant,
    System,
    Tool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
pub enum MessageContent {
    Text(String),
    ToolCalls(Vec<ToolCallRecord>),
    ToolResults(Vec<ToolResultRecord>),
    Summary(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRecord {
    pub call_id: String,
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultRecord {
    pub call_id: String,
    pub tool_name: String,
    /// Structured result value. Stored as JSON for rich introspection.
    pub content: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMessage {
    pub id: String,
    pub session_id: String,
    pub role: MessageRole,
    pub content: MessageContent,
    pub created_at: String,
}

// ---------------------------------------------------------------------------
// Conversions between session and provider types
// ---------------------------------------------------------------------------

impl From<&crate::provider::ToolCall> for ToolCallRecord {
    fn from(tc: &crate::provider::ToolCall) -> Self {
        Self {
            call_id: tc.id.clone(),
            tool_name: tc.name.clone(),
            arguments: tc.arguments.clone(),
        }
    }
}

impl From<&ToolCallRecord> for crate::provider::ToolCall {
    fn from(r: &ToolCallRecord) -> Self {
        Self {
            id: r.call_id.clone(),
            name: r.tool_name.clone(),
            arguments: r.arguments.clone(),
        }
    }
}

impl From<&ToolResultRecord> for crate::provider::ToolResult {
    fn from(r: &ToolResultRecord) -> Self {
        Self {
            call_id: r.call_id.clone(),
            content: r.content.clone(),
        }
    }
}

/// Trait for persisting agent sessions and messages.
///
/// Implement this to store sessions in your preferred backend
/// (SQLite, Postgres, in-memory, etc.).
#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn create_session(&self, scope_id: &str) -> Result<Session, AgentError>;
    async fn get_session(&self, id: &str) -> Result<Option<Session>, AgentError>;
    async fn list_sessions(&self, scope_id: &str) -> Result<Vec<Session>, AgentError>;
    async fn update_title(&self, id: &str, title: &str) -> Result<(), AgentError>;
    async fn append_message(&self, msg: &SessionMessage) -> Result<(), AgentError>;
    async fn get_messages(&self, session_id: &str) -> Result<Vec<SessionMessage>, AgentError>;
    async fn delete_session(&self, id: &str) -> Result<(), AgentError>;
}
