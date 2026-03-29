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

    // If schema has no "type" field, default to "object" validation
    // so that required/type/enum checks are not silently skipped.
    let schema_type = schema.get("type").and_then(|t| t.as_str());
    if schema_type.is_none() || schema_type == Some("object") {
        if !arguments.is_object() {
            errors.push("Arguments must be an object".to_string());
            return Err(errors);
        }

        let obj = arguments.as_object().unwrap();

        // Check required fields — null values don't count as present
        if let Some(required) = schema.get("required").and_then(|r| r.as_array()) {
            for req in required {
                if let Some(field_name) = req.as_str() {
                    match obj.get(field_name) {
                        None | Some(serde_json::Value::Null) => {
                            errors.push(format!("Missing required field: '{}'", field_name));
                        }
                        _ => {}
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    struct DummyTool {
        name: String,
    }

    #[async_trait]
    impl AgentTool for DummyTool {
        fn name(&self) -> &str {
            &self.name
        }

        fn definition(&self) -> ToolDefinition {
            ToolDefinition {
                name: self.name.clone(),
                description: "test".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }

        fn permission(&self) -> ToolPermission {
            ToolPermission::AutoExecute
        }

        async fn execute(
            &self,
            _scope_id: &str,
            _arguments: serde_json::Value,
        ) -> Result<ToolExecResult, AgentError> {
            Ok(ToolExecResult::text("ok"))
        }
    }

    #[test]
    fn validate_required_field_present() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({"name": "Alice"});
        assert!(validate_arguments(&schema, &args).is_ok());
    }

    #[test]
    fn validate_required_field_missing() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({});
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("Missing required field: 'name'")));
    }

    #[test]
    fn validate_type_string_correct() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({"name": "Alice"});
        assert!(validate_arguments(&schema, &args).is_ok());
    }

    #[test]
    fn validate_type_string_wrong() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({"name": 42});
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("expected type 'string'")));
    }

    #[test]
    fn validate_type_integer_accepts_i64() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "count": { "type": "integer" } }
        });
        let args = serde_json::json!({"count": 42});
        assert!(validate_arguments(&schema, &args).is_ok());
    }

    #[test]
    fn validate_enum_constraint_passes() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "color": { "type": "string", "enum": ["red", "green", "blue"] } }
        });
        let args = serde_json::json!({"color": "red"});
        assert!(validate_arguments(&schema, &args).is_ok());
    }

    #[test]
    fn validate_enum_constraint_fails() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "color": { "type": "string", "enum": ["red", "green", "blue"] } }
        });
        let args = serde_json::json!({"color": "yellow"});
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("value must be one of")));
    }

    #[test]
    fn validate_non_object_arguments() {
        let schema = serde_json::json!({"type": "object"});
        let args = serde_json::json!("not an object");
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("Arguments must be an object")));
    }

    #[test]
    fn validate_multiple_errors() {
        let schema = serde_json::json!({
            "type": "object",
            "required": ["a", "b"],
            "properties": {
                "a": { "type": "string" },
                "b": { "type": "integer" },
                "c": { "type": "boolean" }
            }
        });
        let args = serde_json::json!({"c": "not a bool"});
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.len() >= 3, "expected at least 3 errors, got {}: {:?}", err.len(), err);
    }

    #[tokio::test]
    async fn tool_registry_register_get_unregister() {
        let registry = ToolRegistry::new();
        let tool: Arc<dyn AgentTool> = Arc::new(DummyTool { name: "mytool".into() });

        registry.register(tool).await;
        assert!(registry.get("mytool").await.is_some());

        let removed = registry.unregister("mytool").await;
        assert!(removed);
        assert!(registry.get("mytool").await.is_none());
    }

    #[tokio::test]
    async fn tool_registry_definitions_filtered() {
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool { name: "alpha".into() })).await;
        registry.register(Arc::new(DummyTool { name: "beta".into() })).await;
        registry.register(Arc::new(DummyTool { name: "gamma".into() })).await;

        let defs = registry.definitions_filtered(&|name: &str| name.starts_with('a') || name.starts_with('g')).await;
        let names: Vec<&str> = defs.iter().map(|d| d.name.as_str()).collect();
        assert_eq!(names, vec!["alpha", "gamma"]);
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[test]
    fn validate_schema_with_no_type_field() {
        // Schema without "type" should default to object validation.
        let schema = serde_json::json!({
            "required": ["name"],
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({});
        let result = validate_arguments(&schema, &args);
        assert!(result.is_err(), "missing 'name' should be caught even without 'type' in schema");
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("Missing required field: 'name'")));
    }

    #[test]
    fn validate_null_arguments() {
        // BUG PROBE: arguments is JSON null.
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::Value::Null;
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("Arguments must be an object")));
    }

    #[test]
    fn validate_required_field_contains_null_value() {
        // BUG PROBE: A required field is present but its value is null.
        // The check is `obj.contains_key(field_name)` — it only checks presence,
        // not whether the value is null.
        let schema = serde_json::json!({
            "type": "object",
            "required": ["name"],
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({"name": null});
        let result = validate_arguments(&schema, &args);
        // Null values should be treated as missing for required field checks.
        match result {
            Err(errors) => {
                assert!(
                    errors.iter().any(|e| e.contains("Missing required field")),
                    "null value should be treated as missing for required check"
                );
            }
            Ok(()) => panic!("null value for required field should fail validation"),
        }
    }

    #[test]
    fn validate_integer_rejects_float() {
        // BUG PROBE: JSON number 3.14 for "integer" type.
        // is_i64() and is_u64() both return false for 3.14.
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "count": { "type": "integer" } }
        });
        let args = serde_json::json!({"count": 3.14});
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("expected type 'integer'")));
    }

    #[test]
    fn validate_integer_accepts_negative() {
        // Sanity check: negative integers should pass
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "count": { "type": "integer" } }
        });
        let args = serde_json::json!({"count": -5});
        assert!(validate_arguments(&schema, &args).is_ok());
    }

    #[test]
    fn validate_number_accepts_float() {
        // "number" type should accept both integers and floats
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "score": { "type": "number" } }
        });
        let args = serde_json::json!({"score": 3.14});
        assert!(validate_arguments(&schema, &args).is_ok());
    }

    #[test]
    fn validate_number_rejects_string() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "score": { "type": "number" } }
        });
        let args = serde_json::json!({"score": "3.14"});
        let err = validate_arguments(&schema, &args).unwrap_err();
        assert!(err.iter().any(|e| e.contains("expected type 'number'")));
    }

    #[test]
    fn validate_extra_fields_are_silently_accepted() {
        // BUG PROBE: Extra fields not in schema are silently accepted.
        // No additionalProperties check is implemented.
        let schema = serde_json::json!({
            "type": "object",
            "properties": { "name": { "type": "string" } }
        });
        let args = serde_json::json!({"name": "Alice", "injection": "malicious_payload"});
        let result = validate_arguments(&schema, &args);
        assert!(
            result.is_ok(),
            "BUG: extra fields not in schema are silently accepted (no additionalProperties check)"
        );
    }

    #[test]
    fn validate_empty_schema_accepts_anything() {
        // BUG PROBE: Completely empty schema. Not even "type" is set.
        let schema = serde_json::json!({});
        let args = serde_json::json!("not even an object");
        let result = validate_arguments(&schema, &args);
        // Empty schema defaults to object validation, so non-object is rejected.
        assert!(result.is_err(), "empty schema should default to object validation");
    }

    #[test]
    fn validate_nested_object_not_validated() {
        // BUG PROBE: Nested objects are not recursively validated.
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "required": ["x"],
                    "properties": { "x": { "type": "string" } }
                }
            }
        });
        // inner is present and is an object, but missing required field "x"
        let args = serde_json::json!({"inner": {}});
        let result = validate_arguments(&schema, &args);
        // Only top-level type check is done (inner is object? yes.)
        // Nested required fields are NOT checked.
        assert!(
            result.is_ok(),
            "BUG: nested object schema is not recursively validated"
        );
    }

    #[tokio::test]
    async fn tool_registry_duplicate_names() {
        // BUG PROBE: Two tools with the same name. get() returns the first one.
        // unregister() removes ALL with that name.
        let registry = ToolRegistry::new();
        registry.register(Arc::new(DummyTool { name: "dup".into() })).await;
        registry.register(Arc::new(DummyTool { name: "dup".into() })).await;

        let names = registry.tool_names().await;
        assert_eq!(names.len(), 2, "duplicate names are allowed in registry");

        let defs = registry.definitions().await;
        assert_eq!(defs.len(), 2, "both duplicates appear in definitions");

        // get() returns first match
        assert!(registry.get("dup").await.is_some());

        // unregister removes ALL with that name
        let removed = registry.unregister("dup").await;
        assert!(removed);
        assert!(registry.get("dup").await.is_none());
        assert_eq!(registry.tool_names().await.len(), 0);
    }
}
