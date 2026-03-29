use std::sync::Arc;
use async_trait::async_trait;
use agent_harness_core::{AgentTool, ToolDefinition, ToolExecResult, ToolPermission, AgentError};
use tracing::debug;

use crate::client::McpClient;

/// Bridges a single MCP tool into the agent-harness [`AgentTool`] trait.
///
/// Each `McpToolBridge` represents one tool discovered from an MCP server.
/// It delegates `execute()` to `McpClient::call_tool()`.
pub struct McpToolBridge {
    client: Arc<McpClient>,
    /// Name used in the agent registry (may be namespaced).
    tool_name: String,
    /// Original name as known by the MCP server (for call_tool).
    mcp_name: String,
    definition: ToolDefinition,
    permission: ToolPermission,
}

impl McpToolBridge {
    pub fn new(
        client: Arc<McpClient>,
        tool_name: String,
        description: String,
        input_schema: serde_json::Value,
    ) -> Self {
        Self {
            client,
            definition: ToolDefinition {
                name: tool_name.clone(),
                description,
                parameters: input_schema,
            },
            mcp_name: tool_name.clone(),
            tool_name,
            permission: ToolPermission::AutoExecute,
        }
    }

    /// Set this tool to require user approval before execution.
    pub fn with_permission(mut self, permission: ToolPermission) -> Self {
        self.permission = permission;
        self
    }

    /// Override the tool name (e.g., to add a namespace prefix).
    /// The original MCP name is preserved for server calls.
    pub fn with_name(mut self, name: String) -> Self {
        self.tool_name = name.clone();
        self.definition.name = name;
        self
    }
}

#[async_trait]
impl AgentTool for McpToolBridge {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn definition(&self) -> ToolDefinition {
        self.definition.clone()
    }

    fn permission(&self) -> ToolPermission {
        self.permission.clone()
    }

    async fn execute(
        &self,
        _scope_id: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecResult, AgentError> {
        debug!(tool = %self.tool_name, "executing MCP tool");

        let result = self
            .client
            .call_tool(&self.mcp_name, arguments)
            .await
            .map_err(|e| AgentError::ToolError {
                tool_name: self.tool_name.clone(),
                message: e.to_string(),
            })?;

        Ok(ToolExecResult::Completed(result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::McpClient;
    use crate::transport::MockTransport;

    const INIT_RESPONSE: &str =
        r#"{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"test"}}}"#;

    async fn make_client() -> Arc<McpClient> {
        let transport = Arc::new(MockTransport::new(vec![INIT_RESPONSE.to_string()]));
        Arc::new(McpClient::with_transport(transport).await.unwrap())
    }

    #[tokio::test]
    async fn bridge_definition_matches() {
        let client = make_client().await;
        let bridge = McpToolBridge::new(
            client,
            "read_file".to_string(),
            "Read a file from disk".to_string(),
            serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        );
        let def = bridge.definition();
        assert_eq!(def.name, "read_file");
        assert_eq!(def.description, "Read a file from disk");
        assert_eq!(def.parameters["type"], "object");
        assert_eq!(bridge.name(), "read_file");
    }

    #[tokio::test]
    async fn bridge_with_name_overrides() {
        let client = make_client().await;
        let bridge = McpToolBridge::new(
            client,
            "status".to_string(),
            "Get status".to_string(),
            serde_json::json!({"type": "object"}),
        )
        .with_name("git_status".to_string());

        assert_eq!(bridge.name(), "git_status");
        let def = bridge.definition();
        assert_eq!(def.name, "git_status");
    }

    #[tokio::test]
    async fn bridge_default_permission_auto_execute() {
        let client = make_client().await;
        let bridge = McpToolBridge::new(
            client,
            "test".to_string(),
            "desc".to_string(),
            serde_json::json!({"type": "object"}),
        );
        assert_eq!(bridge.permission(), ToolPermission::AutoExecute);
    }
}
