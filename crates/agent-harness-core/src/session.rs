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

impl SessionMessage {
    pub fn new(session_id: impl Into<String>, role: MessageRole, content: MessageContent) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: session_id.into(),
            role,
            content,
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

impl ToolResultRecord {
    pub fn from_call(tc: &crate::provider::ToolCall, content: serde_json::Value) -> Self {
        Self {
            call_id: tc.id.clone(),
            tool_name: tc.name.clone(),
            content,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_content_text_serde_roundtrip() {
        let content = MessageContent::Text("hello world".into());
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn message_content_tool_calls_serde_roundtrip() {
        let content = MessageContent::ToolCalls(vec![ToolCallRecord {
            call_id: "c1".into(),
            tool_name: "my_tool".into(),
            arguments: serde_json::json!({"key": "value"}),
        }]);
        let json = serde_json::to_string(&content).unwrap();
        let deserialized: MessageContent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn message_role_serde_snake_case() {
        let user_json = serde_json::to_string(&MessageRole::User).unwrap();
        assert_eq!(user_json, "\"user\"");

        let assistant_json = serde_json::to_string(&MessageRole::Assistant).unwrap();
        assert_eq!(assistant_json, "\"assistant\"");

        let system_json = serde_json::to_string(&MessageRole::System).unwrap();
        assert_eq!(system_json, "\"system\"");

        let tool_json = serde_json::to_string(&MessageRole::Tool).unwrap();
        assert_eq!(tool_json, "\"tool\"");

        // Roundtrip
        let deserialized: MessageRole = serde_json::from_str("\"user\"").unwrap();
        assert_eq!(deserialized, MessageRole::User);
        let deserialized: MessageRole = serde_json::from_str("\"assistant\"").unwrap();
        assert_eq!(deserialized, MessageRole::Assistant);
    }

    #[test]
    fn tool_call_record_from_provider_tool_call() {
        let provider_tc = crate::provider::ToolCall {
            id: "call-123".into(),
            name: "read_file".into(),
            arguments: serde_json::json!({"path": "/tmp/test.txt"}),
        };
        let record = ToolCallRecord::from(&provider_tc);
        assert_eq!(record.call_id, "call-123");
        assert_eq!(record.tool_name, "read_file");
        assert_eq!(record.arguments, serde_json::json!({"path": "/tmp/test.txt"}));
    }

    #[test]
    fn tool_result_record_to_provider_tool_result() {
        let record = ToolResultRecord {
            call_id: "call-456".into(),
            tool_name: "write_file".into(),
            content: serde_json::json!({"status": "ok"}),
        };
        let provider_tr = crate::provider::ToolResult::from(&record);
        assert_eq!(provider_tr.call_id, "call-456");
        assert_eq!(provider_tr.content, serde_json::json!({"status": "ok"}));
    }
}
