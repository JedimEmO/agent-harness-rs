//! # ironflow-tools-memory
//!
//! Memory tools for ironflow agents.
//!
//! Provides tools that use the [`MemoryStore`] trait
//! to save and recall memories:
//! - [`SaveMemoryTool`] — save a key-value memory with a category
//! - [`RecallMemoriesTool`] — search stored memories by query

use std::sync::Arc;
use async_trait::async_trait;
use ironflow_core::{
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
        let category = match category_str {
            "preference" => MemoryCategory::Preference,
            "instruction" => MemoryCategory::Instruction,
            "context" => MemoryCategory::Context,
            _ => MemoryCategory::Fact,
        };

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
        match self.store.search(scope_id, query, 10).await {
            Ok(memories) => {
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
