use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;
use serde_json::{json, Value};
use tracing::{debug, info, warn};

use crate::error::McpError;
use crate::transport::{StdioTransport, Transport};

/// Discovered MCP tool definition.
#[derive(Debug, Clone)]
pub struct McpToolInfo {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// MCP client — manages a connection to a single MCP server.
///
/// Handles the JSON-RPC protocol lifecycle: `initialize`, `tools/list`, `tools/call`.
pub struct McpClient {
    transport: Arc<dyn Transport>,
    next_id: AtomicI64,
    server_name: Option<String>,
    initialized: bool,
}

impl McpClient {
    /// Create an MCP client over a stdio transport by spawning a child process.
    pub async fn stdio(program: &str, args: &[&str]) -> Result<Self, McpError> {
        let transport = Arc::new(StdioTransport::spawn(program, args).await?);
        let mut client = Self {
            transport,
            next_id: AtomicI64::new(1),
            server_name: None,
            initialized: false,
        };
        client.initialize().await?;
        Ok(client)
    }

    /// Create an MCP client over a stdio transport with custom environment variables.
    pub async fn stdio_with_env(
        program: &str,
        args: &[&str],
        env: &std::collections::HashMap<String, String>,
    ) -> Result<Self, McpError> {
        let transport = Arc::new(StdioTransport::spawn_with_env(program, args, env).await?);
        let mut client = Self {
            transport,
            next_id: AtomicI64::new(1),
            server_name: None,
            initialized: false,
        };
        client.initialize().await?;
        Ok(client)
    }

    /// Create an MCP client with a custom transport.
    pub async fn with_transport(transport: Arc<dyn Transport>) -> Result<Self, McpError> {
        let mut client = Self {
            transport,
            next_id: AtomicI64::new(1),
            server_name: None,
            initialized: false,
        };
        client.initialize().await?;
        Ok(client)
    }

    fn next_id(&self) -> i64 {
        self.next_id.fetch_add(1, Ordering::SeqCst)
    }

    /// Send a JSON-RPC request and wait for the response.
    async fn request(&self, method: &str, params: Value) -> Result<Value, McpError> {
        let id = self.next_id();
        let request = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let msg = serde_json::to_string(&request)?;
        debug!(method, id, "sending MCP request");
        self.transport.send(&msg).await?;

        // Read responses until we find one matching our id.
        // (Servers may send notifications interspersed.)
        loop {
            let line = self.transport.recv().await?;
            let response: Value = serde_json::from_str(&line).map_err(|e| {
                McpError::Protocol(format!("invalid JSON from server: {}: {}", e, line))
            })?;

            // Skip notifications (no id field, or id is null)
            match response.get("id") {
                None | Some(Value::Null) => {
                    debug!("received MCP notification, skipping");
                    continue;
                }
                _ => {}
            }

            // JSON-RPC 2.0 allows id to be a number or string.
            let id_matches = match response.get("id") {
                Some(Value::Number(n)) => n.as_i64() == Some(id),
                Some(Value::String(s)) => s.parse::<i64>().ok() == Some(id),
                _ => false,
            };

            if !id_matches {
                warn!(expected = id, "MCP response id mismatch, skipping");
                continue;
            }

            // Check for error
            if let Some(err) = response.get("error") {
                let code = err.get("code").and_then(|v| v.as_i64()).unwrap_or(-1);
                let message = err
                    .get("message")
                    .and_then(|v| v.as_str())
                    .unwrap_or("unknown error")
                    .to_string();
                return Err(McpError::ServerError { code, message });
            }

            return Ok(response.get("result").cloned().unwrap_or(Value::Null));
        }
    }

    /// Send a JSON-RPC notification (no response expected).
    async fn notify(&self, method: &str, params: Value) -> Result<(), McpError> {
        let notification = json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let msg = serde_json::to_string(&notification)?;
        self.transport.send(&msg).await
    }

    /// Initialize the MCP connection (handshake).
    async fn initialize(&mut self) -> Result<(), McpError> {
        let result = self
            .request(
                "initialize",
                json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "agent-harness-mcp",
                        "version": env!("CARGO_PKG_VERSION"),
                    },
                }),
            )
            .await?;

        self.server_name = result
            .get("serverInfo")
            .and_then(|s| s.get("name"))
            .and_then(|n| n.as_str())
            .map(|s| s.to_string());

        info!(
            server = self.server_name.as_deref().unwrap_or("unknown"),
            "MCP server initialized"
        );

        // Send initialized notification
        self.notify("notifications/initialized", json!({})).await?;
        self.initialized = true;

        Ok(())
    }

    /// List all tools available on the MCP server.
    pub async fn list_tools(&self) -> Result<Vec<McpToolInfo>, McpError> {
        if !self.initialized {
            return Err(McpError::NotInitialized);
        }

        let result = self.request("tools/list", json!({})).await?;
        let tools_array = result
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let mut tools = Vec::new();
        for tool_val in tools_array {
            let name = tool_val
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let description = tool_val
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let input_schema = tool_val
                .get("inputSchema")
                .cloned()
                .unwrap_or(json!({"type": "object"}));

            if !name.is_empty() {
                tools.push(McpToolInfo {
                    name,
                    description,
                    input_schema,
                });
            }
        }

        info!(count = tools.len(), "discovered MCP tools");
        Ok(tools)
    }

    /// Call a tool on the MCP server.
    pub async fn call_tool(
        &self,
        name: &str,
        arguments: Value,
    ) -> Result<Value, McpError> {
        if !self.initialized {
            return Err(McpError::NotInitialized);
        }

        debug!(tool = name, "calling MCP tool");
        let result = self
            .request(
                "tools/call",
                json!({
                    "name": name,
                    "arguments": arguments,
                }),
            )
            .await?;

        // MCP tool results have a `content` array with `{ type: "text", text: "..." }` items
        if let Some(content) = result.get("content") {
            // Try to extract text content
            if let Some(arr) = content.as_array() {
                let texts: Vec<&str> = arr
                    .iter()
                    .filter_map(|item| {
                        let item_type = item.get("type").and_then(|t| t.as_str())?;
                        if item_type == "text" {
                            item.get("text").and_then(|t| t.as_str())
                        } else {
                            None
                        }
                    })
                    .collect();
                if !texts.is_empty() {
                    return Ok(Value::String(texts.join("\n")));
                }
            }
            Ok(content.clone())
        } else {
            Ok(result)
        }
    }

    /// Get the server name (set after initialization).
    pub fn server_name(&self) -> Option<&str> {
        self.server_name.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::MockTransport;

    const INIT_RESPONSE: &str =
        r#"{"jsonrpc":"2.0","id":1,"result":{"serverInfo":{"name":"test-server"}}}"#;

    #[tokio::test]
    async fn initialize_sends_correct_protocol() {
        let transport = Arc::new(MockTransport::new(vec![INIT_RESPONSE.to_string()]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();

        let sent = transport.sent_messages().await;
        assert!(sent.len() >= 2, "expected at least 2 sent messages (init request + notification)");

        // First message: initialize request
        let init_req: Value = serde_json::from_str(&sent[0]).unwrap();
        assert_eq!(init_req["method"], "initialize");
        assert_eq!(init_req["params"]["protocolVersion"], "2024-11-05");
        assert!(init_req["id"].is_number());

        // Second message: notifications/initialized
        let notif: Value = serde_json::from_str(&sent[1]).unwrap();
        assert_eq!(notif["method"], "notifications/initialized");
        assert!(notif.get("id").is_none(), "notification should not have an id");

        assert_eq!(client.server_name(), Some("test-server"));
    }

    #[tokio::test]
    async fn list_tools_parses_tools_array() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"read_file","description":"Read a file","inputSchema":{"type":"object"}},{"name":"write_file","description":"Write a file","inputSchema":{"type":"object"}}]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name, "read_file");
        assert_eq!(tools[0].description, "Read a file");
        assert_eq!(tools[1].name, "write_file");
    }

    #[tokio::test]
    async fn list_tools_empty_name_filtered() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"tools":[{"name":"","description":"bad tool","inputSchema":{"type":"object"}},{"name":"good_tool","description":"ok","inputSchema":{"type":"object"}}]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0].name, "good_tool");
    }

    #[tokio::test]
    async fn call_tool_extracts_text_content() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"hello"}]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        assert_eq!(result, Value::String("hello".to_string()));
    }

    #[tokio::test]
    async fn call_tool_joins_multiple_text() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"line1"},{"type":"text","text":"line2"}]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        assert_eq!(result, Value::String("line1\nline2".to_string()));
    }

    #[tokio::test]
    async fn server_error_mapped() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"error":{"code":-32600,"message":"bad request"}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let err = client.call_tool("test", json!({})).await.unwrap_err();
        match err {
            McpError::ServerError { code, message } => {
                assert_eq!(code, -32600);
                assert_eq!(message, "bad request");
            }
            other => panic!("expected ServerError, got: {:?}", other),
        }
    }

    #[tokio::test]
    async fn request_skips_notifications() {
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            // A notification (no id field) before the real response
            r#"{"jsonrpc":"2.0","method":"some/notification","params":{}}"#.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"text","text":"got it"}]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        assert_eq!(result, Value::String("got it".to_string()));
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[tokio::test]
    async fn response_id_as_string_causes_mismatch() {
        // BUG PROBE: Server returns id as a string "2" instead of number 2.
        // The code uses .as_i64() which returns None for strings.
        // unwrap_or(-1) kicks in, so resp_id = -1, which != our id (2).
        // The response is SKIPPED and the client waits for more data.
        // Since there's no more data, the transport returns an error.
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            // id is a string, not a number
            r#"{"jsonrpc":"2.0","id":"2","result":{"tools":[]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.list_tools().await;
        // Fixed: string id "2" is now matched to numeric id 2.
        assert!(result.is_ok(), "string id should be matched correctly");
        assert_eq!(result.unwrap().len(), 0);
    }

    #[tokio::test]
    async fn tools_list_returns_null_instead_of_array() {
        // BUG PROBE: tools/list result has "tools": null instead of an array.
        // The code does .and_then(|v| v.as_array()) which returns None for null,
        // then .unwrap_or_default() gives an empty vec. Should be fine.
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"tools":null}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 0, "null tools array should be treated as empty");
    }

    #[tokio::test]
    async fn tools_list_missing_tools_field() {
        // BUG PROBE: result has no "tools" field at all
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let tools = client.list_tools().await.unwrap();
        assert_eq!(tools.len(), 0, "missing tools field should be treated as empty");
    }

    #[tokio::test]
    async fn call_tool_no_content_field_returns_raw_result() {
        // BUG PROBE: call_tool result has no "content" field at all.
        // The code falls through to Ok(result), returning the raw JSON.
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"status":"ok","data":42}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        // Returns the entire result object
        assert_eq!(result["status"], "ok");
        assert_eq!(result["data"], 42);
    }

    #[tokio::test]
    async fn call_tool_content_is_string_not_array() {
        // BUG PROBE: content field is a string instead of array.
        // .as_array() returns None, so we fall through to Ok(content.clone())
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"content":"raw string result"}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        assert_eq!(result, Value::String("raw string result".to_string()));
    }

    #[tokio::test]
    async fn call_tool_content_array_with_non_text_types() {
        // BUG PROBE: content array has items with type != "text" (e.g., "image")
        // These are filtered out. If there are NO text items, texts is empty,
        // and we fall through to Ok(content.clone()) returning the whole array.
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"content":[{"type":"image","data":"base64..."}]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        // Returns the raw content array since no text items were found
        assert!(result.is_array());
    }

    #[tokio::test]
    async fn call_tool_result_is_null() {
        // BUG PROBE: Server returns result: null.
        // The request() method does response.get("result").cloned().unwrap_or(Value::Null),
        // so result = Null. Then call_tool checks result.get("content") on Null, which is None.
        // Falls through to Ok(result) = Ok(Null).
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":null}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        assert_eq!(result, Value::Null, "null result returned as-is");
    }

    #[tokio::test]
    async fn server_name_missing_from_init_response() {
        // BUG PROBE: init response with no serverInfo
        let transport = Arc::new(MockTransport::new(vec![
            r#"{"jsonrpc":"2.0","id":1,"result":{}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        assert_eq!(client.server_name(), None);
    }

    #[tokio::test]
    async fn call_tool_content_array_empty() {
        // BUG PROBE: content is an empty array
        let transport = Arc::new(MockTransport::new(vec![
            INIT_RESPONSE.to_string(),
            r#"{"jsonrpc":"2.0","id":2,"result":{"content":[]}}"#.to_string(),
        ]));
        let client = McpClient::with_transport(transport.clone()).await.unwrap();
        let result = client.call_tool("test", json!({})).await.unwrap();
        // Empty array: texts is empty, so falls through to Ok(content.clone())
        assert!(result.is_array());
        assert_eq!(result.as_array().unwrap().len(), 0);
    }
}
