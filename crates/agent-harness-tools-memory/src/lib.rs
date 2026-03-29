//! # agent-harness-tools-memory
//!
//! Memory tools for agent-harness agents.
//!
//! Provides tools that use the [`MemoryStore`] trait
//! to save and recall memories:
//! - [`SaveMemoryTool`] — save a key-value memory with a category
//! - [`RecallMemoriesTool`] — search stored memories by query

use std::sync::Arc;
use async_trait::async_trait;
use agent_harness_core::{
    AgentError, AgentTool, MemoryCategory, MemoryStore, ToolDefinition, ToolExecResult,
    ToolPermission,
};

// --- Save Memory ---

pub struct SaveMemoryTool {
    store: Arc<dyn MemoryStore>,
}

impl SaveMemoryTool {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl AgentTool for SaveMemoryTool {
    fn name(&self) -> &str { "save_memory" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "save_memory".to_string(),
            description: "Save a piece of information to memory for future reference. Use this to remember important decisions, user preferences, and key facts.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "key": { "type": "string", "description": "A short key/label for this memory (e.g., 'tone_preference', 'project_goals')" },
                    "content": { "type": "string", "description": "The content to remember" },
                    "category": { "type": "string", "description": "Category: fact, preference, instruction, or context" }
                },
                "required": ["key", "content"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::RequiresApproval }

    async fn execute(&self, scope_id: &str, arguments: serde_json::Value) -> Result<ToolExecResult, AgentError> {
        let key = arguments.get("key").and_then(|v| v.as_str()).unwrap_or("");
        let content = arguments.get("content").and_then(|v| v.as_str()).unwrap_or("");
        let category_str = arguments.get("category").and_then(|v| v.as_str()).unwrap_or("fact");
        let category = category_str.parse().unwrap_or(MemoryCategory::Fact);

        match self.store.save(scope_id, key, content, category).await {
            Ok(_) => Ok(ToolExecResult::text(format!("Saved memory: '{}'", key))),
            Err(e) => Err(AgentError::ToolError { tool_name: "save_memory".to_string(), message: e.to_string() }),
        }
    }
}

// --- Recall Memories ---

pub struct RecallMemoriesTool {
    store: Arc<dyn MemoryStore>,
}

impl RecallMemoriesTool {
    pub fn new(store: Arc<dyn MemoryStore>) -> Self {
        Self { store }
    }
}

#[async_trait]
impl AgentTool for RecallMemoriesTool {
    fn name(&self) -> &str { "recall_memories" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "recall_memories".to_string(),
            description: "Search through saved memories for relevant context. Use this before making major decisions or when the user asks about previous conversations.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query to find relevant memories" },
                    "category": { "type": "string", "description": "Optional category filter: fact, preference, instruction, or context" }
                },
                "required": ["query"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(&self, scope_id: &str, arguments: serde_json::Value) -> Result<ToolExecResult, AgentError> {
        let query = arguments.get("query").and_then(|v| v.as_str()).unwrap_or("");
        let category_filter = arguments.get("category").and_then(|v| v.as_str()).map(|s| match s {
            "preference" => MemoryCategory::Preference,
            "instruction" => MemoryCategory::Instruction,
            "context" => MemoryCategory::Context,
            _ => MemoryCategory::Fact,
        });
        match self.store.search(scope_id, query, 10).await {
            Ok(memories) => {
                let memories = if let Some(cat) = category_filter {
                    memories.into_iter().filter(|m| m.category == cat).collect()
                } else {
                    memories
                };
                if memories.is_empty() {
                    Ok(ToolExecResult::text("No relevant memories found."))
                } else {
                    let list = memories.iter()
                        .map(|m| format!("- [{}] ({:?}) {}", m.key, m.category, m.content))
                        .collect::<Vec<_>>()
                        .join("\n");
                    Ok(ToolExecResult::text(list))
                }
            }
            Err(e) => Err(AgentError::ToolError { tool_name: "recall_memories".to_string(), message: e.to_string() }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::sync::RwLock;
    use agent_harness_core::{AgentTool, Memory, MemoryCategory, MemoryStore, AgentError};

    struct TestMemoryStore {
        memories: RwLock<Vec<Memory>>,
    }

    impl TestMemoryStore {
        fn new() -> Self {
            Self { memories: RwLock::new(Vec::new()) }
        }
    }

    #[async_trait]
    impl MemoryStore for TestMemoryStore {
        async fn save(&self, scope_id: &str, key: &str, content: &str, category: MemoryCategory) -> Result<Memory, AgentError> {
            let memory = Memory {
                id: format!("mem_{}", self.memories.read().await.len()),
                scope_id: scope_id.to_string(),
                key: key.to_string(),
                content: content.to_string(),
                category,
                created_at: "2026-01-01".into(),
                updated_at: "2026-01-01".into(),
            };
            self.memories.write().await.push(memory.clone());
            Ok(memory)
        }

        async fn search(&self, scope_id: &str, query: &str, limit: usize) -> Result<Vec<Memory>, AgentError> {
            let memories = self.memories.read().await;
            Ok(memories.iter()
                .filter(|m| m.scope_id == scope_id && (m.key.contains(query) || m.content.contains(query)))
                .take(limit)
                .cloned()
                .collect())
        }

        async fn list(&self, scope_id: &str, category: Option<MemoryCategory>) -> Result<Vec<Memory>, AgentError> {
            let memories = self.memories.read().await;
            Ok(memories.iter()
                .filter(|m| m.scope_id == scope_id && category.map_or(true, |c| m.category == c))
                .cloned()
                .collect())
        }

        async fn delete(&self, _id: &str) -> Result<bool, AgentError> {
            Ok(true)
        }
    }

    fn make_store() -> Arc<TestMemoryStore> {
        Arc::new(TestMemoryStore::new())
    }

    #[tokio::test]
    async fn save_memory_basic() {
        let store = make_store();
        let tool = SaveMemoryTool::new(store.clone());
        let result = tool.execute("scope1", serde_json::json!({
            "key": "favorite_color",
            "content": "blue"
        })).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                let text = v.as_str().unwrap();
                assert!(text.contains("Saved memory"), "got: {}", text);
                assert!(text.contains("favorite_color"), "got: {}", text);
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn save_memory_with_category() {
        let store = make_store();
        let tool = SaveMemoryTool::new(store.clone());
        tool.execute("scope1", serde_json::json!({
            "key": "tone",
            "content": "casual",
            "category": "preference"
        })).await.unwrap();
        let memories = store.memories.read().await;
        assert_eq!(memories[0].category, MemoryCategory::Preference);
    }

    #[tokio::test]
    async fn save_memory_default_category() {
        let store = make_store();
        let tool = SaveMemoryTool::new(store.clone());
        tool.execute("scope1", serde_json::json!({
            "key": "sky",
            "content": "is blue"
        })).await.unwrap();
        let memories = store.memories.read().await;
        assert_eq!(memories[0].category, MemoryCategory::Fact);
    }

    #[test]
    fn save_memory_requires_approval() {
        let store = make_store();
        let tool = SaveMemoryTool::new(store);
        assert_eq!(tool.permission(), ToolPermission::RequiresApproval);
    }

    #[tokio::test]
    async fn recall_memories_found() {
        let store = make_store();
        // Pre-populate via the store directly
        store.save("scope1", "color", "blue is best", MemoryCategory::Preference).await.unwrap();
        let tool = RecallMemoriesTool::new(store.clone());
        let result = tool.execute("scope1", serde_json::json!({"query": "color"})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                let text = v.as_str().unwrap();
                assert!(text.contains("color"), "got: {}", text);
                assert!(text.contains("blue is best"), "got: {}", text);
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn recall_memories_empty() {
        let store = make_store();
        let tool = RecallMemoriesTool::new(store);
        let result = tool.execute("scope1", serde_json::json!({"query": "anything"})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                assert_eq!(v.as_str().unwrap(), "No relevant memories found.");
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[test]
    fn recall_memories_is_auto_execute() {
        let store = make_store();
        let tool = RecallMemoriesTool::new(store);
        assert_eq!(tool.permission(), ToolPermission::AutoExecute);
    }

    #[tokio::test]
    async fn save_and_recall_roundtrip() {
        let store = make_store();
        let save_tool = SaveMemoryTool::new(store.clone());
        let recall_tool = RecallMemoriesTool::new(store.clone());

        save_tool.execute("scope1", serde_json::json!({
            "key": "project_goal",
            "content": "Build the best agent framework"
        })).await.unwrap();

        let result = recall_tool.execute("scope1", serde_json::json!({"query": "agent"})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                let text = v.as_str().unwrap();
                assert!(text.contains("project_goal"), "got: {}", text);
                assert!(text.contains("Build the best agent framework"), "got: {}", text);
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }
}
