use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::error::AgentError;
use crate::interaction::InteractionRequest;
use crate::provider::ToolDefinition;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ToolPermission {
    AutoExecute,
    RequiresApproval,
}

#[derive(Debug, Clone)]
pub enum ToolExecResult {
    /// Tool completed with a structured result value.
    Completed(serde_json::Value),
    NeedsInteraction(InteractionRequest),
}

impl ToolExecResult {
    /// Convenience: complete with a plain text string.
    pub fn text(s: impl Into<String>) -> Self {
        ToolExecResult::Completed(serde_json::Value::String(s.into()))
    }
}

/// Core tool trait — implement this for each tool in your agent.
///
/// The `scope_id` parameter is an opaque string that groups related data.
/// Your application decides what it represents (project_id, user_id, etc.).
#[async_trait]
pub trait AgentTool: Send + Sync {
    fn name(&self) -> &str;
    fn definition(&self) -> ToolDefinition;
    fn permission(&self) -> ToolPermission;

    /// Execute the tool with the given arguments within a scope.
    async fn execute(
        &self,
        scope_id: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecResult, AgentError>;
}

/// Validate tool arguments against the JSON Schema defined in the ToolDefinition.
///
/// Performs basic validation: required fields, type checking, and enum constraints.
pub(crate) fn validate_arguments(
    schema: &serde_json::Value,
    arguments: &serde_json::Value,
) -> Result<(), Vec<String>> {
    let mut errors = Vec::new();

    let schema_type = schema.get("type").and_then(|t| t.as_str());
    if schema_type == Some("object") {
        if !arguments.is_object() {
            errors.push("Arguments must be an object".to_string());
            return Err(errors);
        }

        let obj = arguments.as_object().unwrap();

        // Check required fields
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            for req in required {
                if let Some(field_name) = req.as_str() {
                    if !obj.contains_key(field_name) {
                        errors.push(format!("Missing required field: '{}'", field_name));
                    }
                }
            }
        }

        // Check property types
        if let Some(properties) = schema.get("properties").and_then(|p| p.as_object()) {
            for (key, prop_schema) in properties {
                if let Some(value) = obj.get(key) {
                    if let Some(expected_type) = prop_schema.get("type").and_then(|t| t.as_str()) {
                        let type_matches = match expected_type {
                            "string" => value.is_string(),
                            "number" => value.is_number(),
                            "integer" => value.is_i64() || value.is_u64(),
                            "boolean" => value.is_boolean(),
                            "array" => value.is_array(),
                            "object" => value.is_object(),
                            _ => true,
                        };
                        if !type_matches {
                            errors.push(format!(
                                "Field '{}': expected type '{}', got '{}'",
                                key,
                                expected_type,
                                value_type_name(value)
                            ));
                        }
                    }

                    // Check enum constraints
                    if let Some(enum_values) = prop_schema.get("enum").and_then(|e| e.as_array()) {
                        if !enum_values.contains(value) {
                            let allowed: Vec<String> =
                                enum_values.iter().map(|v| v.to_string()).collect();
                            errors.push(format!(
                                "Field '{}': value must be one of [{}]",
                                key,
                                allowed.join(", ")
                            ));
                        }
                    }
                }
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn value_type_name(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "boolean",
        serde_json::Value::Number(_) => "number",
        serde_json::Value::String(_) => "string",
        serde_json::Value::Array(_) => "array",
        serde_json::Value::Object(_) => "object",
    }
}

/// Dynamic tool registry supporting runtime registration and removal.
pub struct ToolRegistry {
    tools: RwLock<Vec<Arc<dyn AgentTool>>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self {
            tools: RwLock::new(Vec::new()),
        }
    }

    /// Register a new tool. Can be called at any time.
    pub async fn register(&self, tool: Arc<dyn AgentTool>) {
        self.tools.write().await.push(tool);
    }

    /// Register a tool from a boxed trait object (convenience for migration).
    pub async fn register_boxed(&self, tool: Box<dyn AgentTool>) {
        self.tools.write().await.push(Arc::from(tool));
    }

    /// Unregister a tool by name. Returns true if a tool was removed.
    pub async fn unregister(&self, name: &str) -> bool {
        let mut tools = self.tools.write().await;
        let len_before = tools.len();
        tools.retain(|t| t.name() != name);
        tools.len() < len_before
    }

    /// Get all tool definitions.
    pub async fn definitions(&self) -> Vec<ToolDefinition> {
        self.tools.read().await.iter().map(|t| t.definition()).collect()
    }

    /// Get tool definitions filtered by a predicate on the tool name.
    pub async fn definitions_filtered(
        &self,
        filter: &(dyn Fn(&str) -> bool + Send + Sync),
    ) -> Vec<ToolDefinition> {
        self.tools
            .read()
            .await
            .iter()
            .filter(|t| filter(t.name()))
            .map(|t| t.definition())
            .collect()
    }

    /// Look up a tool by name.
    pub async fn get(&self, name: &str) -> Option<Arc<dyn AgentTool>> {
        self.tools
            .read()
            .await
            .iter()
            .find(|t| t.name() == name)
            .cloned()
    }

    /// List all registered tool names.
    pub async fn tool_names(&self) -> Vec<String> {
        self.tools
            .read()
            .await
            .iter()
            .map(|t| t.name().to_string())
            .collect()
    }
}

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}
