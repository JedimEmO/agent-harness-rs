use reqwest::Client;
use tokio::sync::Mutex;
use tracing::debug;

use crate::error::McpError;
use crate::transport::Transport;

/// HTTP transport for MCP — communicates with an MCP server via Streamable HTTP.
///
/// Implements the request/response pattern: `send()` buffers the outgoing message,
/// `recv()` POSTs it to the server and returns the response body.
pub struct HttpTransport {
    client: Client,
    url: String,
    /// Buffer for the last sent message (POST body for the next recv).
    pending: Mutex<Option<String>>,
    /// Session ID returned by the server during initialization.
    session_id: Mutex<Option<String>>,
}

impl HttpTransport {
    /// Create an HTTP transport targeting the given MCP endpoint URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            client: Client::new(),
            url: url.into(),
            pending: Mutex::new(None),
            session_id: Mutex::new(None),
        }
    }
}

#[async_trait::async_trait]
impl Transport for HttpTransport {
    async fn send(&self, message: &str) -> Result<(), McpError> {
        // Buffer the message — it will be POSTed on the next recv() call.
        // For notifications (no id), POST immediately and discard the response.
        let parsed: serde_json::Value = serde_json::from_str(message)
            .map_err(|e| McpError::Transport(format!("invalid JSON: {e}")))?;

        if parsed.get("id").is_none() {
            // Notification — fire and forget
            debug!(url = %self.url, "sending MCP notification via HTTP");
            let mut req = self
                .client
                .post(&self.url)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json, text/event-stream");

            if let Some(ref sid) = *self.session_id.lock().await {
                req = req.header("Mcp-Session-Id", sid.as_str());
            }

            req.body(message.to_string())
                .send()
                .await
                .map_err(|e| McpError::Transport(format!("HTTP send error: {e}")))?;
            return Ok(());
        }

        // Request — buffer for recv() to POST
        *self.pending.lock().await = Some(message.to_string());
        Ok(())
    }

    async fn recv(&self) -> Result<String, McpError> {
        let body = self
            .pending
            .lock()
            .await
            .take()
            .ok_or_else(|| McpError::Transport("recv called without prior send".into()))?;

        debug!(url = %self.url, "POSTing MCP request via HTTP");

        let mut req = self
            .client
            .post(&self.url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(ref sid) = *self.session_id.lock().await {
            req = req.header("Mcp-Session-Id", sid.as_str());
        }

        let response = req
            .body(body)
            .send()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP request error: {e}")))?;

        // Capture session ID from response headers
        if let Some(sid) = response.headers().get("Mcp-Session-Id") {
            if let Ok(sid_str) = sid.to_str() {
                *self.session_id.lock().await = Some(sid_str.to_string());
            }
        }

        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|e| McpError::Transport(format!("HTTP response read error: {e}")))?;

        if !status.is_success() {
            return Err(McpError::Transport(format!(
                "HTTP {status}: {text}"
            )));
        }

        // The response might be SSE (text/event-stream) or plain JSON.
        // For SSE, extract the data lines. For JSON, return as-is.
        if text.starts_with("data:") || text.contains("\ndata:") {
            // Parse SSE: extract data lines and find the JSON-RPC response
            for line in text.lines() {
                if let Some(data) = line.strip_prefix("data: ").or_else(|| line.strip_prefix("data:")) {
                    let data = data.trim();
                    if !data.is_empty() {
                        // Return the first data line that looks like a JSON-RPC response
                        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                            if parsed.get("id").is_some() || parsed.get("result").is_some() {
                                return Ok(data.to_string());
                            }
                        }
                    }
                }
            }
            Err(McpError::Transport("no JSON-RPC response found in SSE stream".into()))
        } else {
            Ok(text)
        }
    }
}
