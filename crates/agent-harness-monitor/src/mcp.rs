use std::sync::Arc;
use std::time::Instant;

use async_trait::async_trait;

use agent_harness_mcp::{McpError, Transport};

use crate::event::{MonitorEvent, MonitorEventKind};
use crate::sink::MonitorSink;

/// Wraps any MCP [`Transport`] to capture JSON-RPC traffic to the monitor.
pub struct MonitoredMcpTransport {
    inner: Arc<dyn Transport>,
    sink: MonitorSink,
    server_name: String,
    /// Track the last sent request time for duration calculation.
    last_send_time: tokio::sync::Mutex<Option<(i64, Instant)>>,
}

impl MonitoredMcpTransport {
    pub fn new(inner: Arc<dyn Transport>, sink: MonitorSink, server_name: impl Into<String>) -> Self {
        Self {
            inner,
            sink,
            server_name: server_name.into(),
            last_send_time: tokio::sync::Mutex::new(None),
        }
    }
}

#[async_trait]
impl Transport for MonitoredMcpTransport {
    async fn send(&self, message: &str) -> Result<(), McpError> {
        // Try to parse as JSON-RPC to extract metadata
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(message) {
            let method = parsed
                .get("method")
                .and_then(|v| v.as_str())
                .unwrap_or("unknown")
                .to_string();
            let params = parsed
                .get("params")
                .cloned()
                .unwrap_or(serde_json::Value::Null);
            let rpc_id = parsed.get("id").and_then(|v| v.as_i64()).unwrap_or(0);

            // Track timing for this request
            if rpc_id != 0 {
                *self.last_send_time.lock().await = Some((rpc_id, Instant::now()));
            }

            self.sink.emit(MonitorEvent::new(MonitorEventKind::McpRequest {
                server: self.server_name.clone(),
                method,
                params,
                rpc_id,
            }));
        }

        self.inner.send(message).await
    }

    async fn recv(&self) -> Result<String, McpError> {
        let message = self.inner.recv().await?;

        // Try to parse as JSON-RPC response
        if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(&message) {
            let rpc_id = parsed.get("id").and_then(|v| v.as_i64()).unwrap_or(0);
            let result = parsed.get("result").cloned();
            let error = parsed
                .get("error")
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
                .map(String::from);

            // Calculate duration from the matching send
            let duration_ms = if rpc_id != 0 {
                let mut last = self.last_send_time.lock().await;
                if let Some((sent_id, start)) = last.take() {
                    if sent_id == rpc_id {
                        start.elapsed().as_millis() as u64
                    } else {
                        *last = Some((sent_id, start));
                        0
                    }
                } else {
                    0
                }
            } else {
                0
            };

            if rpc_id != 0 || error.is_some() {
                self.sink.emit(MonitorEvent::new(MonitorEventKind::McpResponse {
                    server: self.server_name.clone(),
                    rpc_id,
                    result,
                    error,
                    duration_ms,
                }));
            }
        }

        Ok(message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::MonitorFilter;
    use crate::sink::MonitorSink;

    struct TestTransport {
        responses: tokio::sync::Mutex<std::collections::VecDeque<String>>,
        sent: tokio::sync::Mutex<Vec<String>>,
    }

    impl TestTransport {
        fn new(responses: Vec<String>) -> Self {
            Self {
                responses: tokio::sync::Mutex::new(responses.into()),
                sent: tokio::sync::Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl Transport for TestTransport {
        async fn send(&self, message: &str) -> Result<(), McpError> {
            self.sent.lock().await.push(message.to_string());
            Ok(())
        }

        async fn recv(&self) -> Result<String, McpError> {
            self.responses
                .lock()
                .await
                .pop_front()
                .ok_or_else(|| McpError::Transport("no more responses".into()))
        }
    }

    #[tokio::test]
    async fn monitored_transport_captures_send_recv() {
        let sink = MonitorSink::in_memory().unwrap();

        let response_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"tools": []}
        });
        let transport = Arc::new(TestTransport::new(vec![response_json.to_string()]));
        let monitored = MonitoredMcpTransport::new(transport, sink.clone(), "test-server");

        // Send a JSON-RPC request
        let request_json = serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/list",
            "params": {}
        });
        monitored.send(&request_json.to_string()).await.unwrap();

        // Receive the response
        let resp = monitored.recv().await.unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&resp).unwrap();
        assert_eq!(parsed["id"], 1);

        // Verify events were captured
        let events = sink.query(&MonitorFilter::default()).unwrap();
        assert_eq!(events.len(), 2);

        let tags: Vec<&str> = events.iter().map(|e| e.kind_tag()).collect();
        assert!(tags.contains(&"mcp_request"), "missing mcp_request");
        assert!(tags.contains(&"mcp_response"), "missing mcp_response");

        // Verify request details
        let req_event = events.iter().find(|e| e.kind_tag() == "mcp_request").unwrap();
        match &req_event.kind {
            MonitorEventKind::McpRequest {
                server,
                method,
                rpc_id,
                ..
            } => {
                assert_eq!(server, "test-server");
                assert_eq!(method, "tools/list");
                assert_eq!(*rpc_id, 1);
            }
            _ => panic!("expected McpRequest"),
        }

        // Verify response details
        let resp_event = events.iter().find(|e| e.kind_tag() == "mcp_response").unwrap();
        match &resp_event.kind {
            MonitorEventKind::McpResponse {
                server,
                rpc_id,
                result,
                error,
                duration_ms,
            } => {
                assert_eq!(server, "test-server");
                assert_eq!(*rpc_id, 1);
                assert!(result.is_some());
                assert!(error.is_none());
                // duration_ms should be > 0 (or at least >= 0)
                assert!(*duration_ms < 1000, "duration should be reasonable");
            }
            _ => panic!("expected McpResponse"),
        }

        // All events should have provider="test-server"
        for e in &events {
            assert_eq!(e.provider_name(), Some("test-server"));
        }
    }
}
