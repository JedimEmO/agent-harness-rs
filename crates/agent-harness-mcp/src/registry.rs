use std::sync::Arc;
use tracing::info;

use crate::bridge::McpToolBridge;
use crate::client::McpClient;
use crate::error::McpError;

/// Helper for discovering MCP tools and preparing them for registration.
pub struct McpRegistry;

impl McpRegistry {
    /// Discover all tools from an MCP server and return them as [`McpToolBridge`] instances.
    ///
    /// Each tool can then be registered into a [`ToolRegistry`](agent_harness_core::ToolRegistry).
    pub async fn discover(client: &Arc<McpClient>) -> Result<Vec<McpToolBridge>, McpError> {
        let tools = client.list_tools().await?;

        let bridges: Vec<McpToolBridge> = tools
            .into_iter()
            .map(|t| McpToolBridge::new(client.clone(), t.name, t.description, t.input_schema))
            .collect();

        info!(count = bridges.len(), "created MCP tool bridges");
        Ok(bridges)
    }

    /// Discover tools with a namespace prefix to avoid name collisions.
    ///
    /// Tool names will be formatted as `{prefix}_{original_name}`.
    pub async fn discover_namespaced(
        client: &Arc<McpClient>,
        prefix: &str,
    ) -> Result<Vec<McpToolBridge>, McpError> {
        let tools = client.list_tools().await?;

        let bridges: Vec<McpToolBridge> = tools
            .into_iter()
            .map(|t| {
                let namespaced = format!("{}_{}", prefix, t.name);
                McpToolBridge::new(client.clone(), t.name, t.description, t.input_schema)
                    .with_name(namespaced)
            })
            .collect();

        info!(count = bridges.len(), prefix, "created namespaced MCP tool bridges");
        Ok(bridges)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::McpClient;
    use crate::transport::MockTransport;
    use agent_harness_core::AgentTool;

    const INIT_RESPONSE: &str =
        r#"{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"test"}}}"#;

    fn tools_list_response(id: i64, tools_json: &str) -> String {
        format!(r#"{{"jsonrpc":"2.0","id":{},"result":{{"tools":{}}}}}"#, id, tools_json)
    }

    #[tokio::test]
    async fn discover_creates_bridges() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            tools_list_response(
                2,
                r#"[{"name":"read_file","description":"Read","inputSchema":{"type":"object"}},{"name":"write_file","description":"Write","inputSchema":{"type":"object"}}]"#,
            ),
        ]));
        let client = Arc::new(McpClient::with_transport(transport).await.unwrap());
        let bridges = McpRegistry::discover(&client).await.unwrap();
        assert_eq!(bridges.len(), 2);
        assert_eq!(bridges[0].name(), "read_file");
        assert_eq!(bridges[1].name(), "write_file");
    }

    #[tokio::test]
    async fn discover_namespaced_prefixes() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            tools_list_response(
                2,
                r#"[{"name":"status","description":"Get status","inputSchema":{"type":"object"}},{"name":"commit","description":"Commit","inputSchema":{"type":"object"}}]"#,
            ),
        ]));
        let client = Arc::new(McpClient::with_transport(transport).await.unwrap());
        let bridges = McpRegistry::discover_namespaced(&client, "git").await.unwrap();
        assert_eq!(bridges.len(), 2);
        assert_eq!(bridges[0].name(), "git_status");
        assert_eq!(bridges[1].name(), "git_commit");
    }
}
