use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::Mutex;
use tracing::{debug, error};

use crate::error::McpError;

/// Abstraction over MCP transport (stdio, SSE, etc.).
#[async_trait::async_trait]
pub trait Transport: Send + Sync {
    /// Send a JSON-RPC message (one line of JSON followed by newline).
    async fn send(&self, message: &str) -> Result<(), McpError>;
    /// Receive the next JSON-RPC message line.
    async fn recv(&self) -> Result<String, McpError>;
}

/// Stdio transport — communicates with an MCP server via stdin/stdout of a child process.
pub struct StdioTransport {
    stdin: Mutex<tokio::process::ChildStdin>,
    reader: Mutex<BufReader<tokio::process::ChildStdout>>,
    _child: Mutex<Child>,
}

impl StdioTransport {
    /// Spawn a child process and wrap its stdio as an MCP transport.
    pub async fn spawn(program: &str, args: &[&str]) -> Result<Self, McpError> {
        debug!(program, ?args, "spawning MCP server process");

        let mut child = Command::new(program)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .map_err(|e| McpError::Transport(format!("failed to spawn {}: {}", program, e)))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| McpError::Transport("failed to capture stdin".into()))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| McpError::Transport("failed to capture stdout".into()))?;

        Ok(Self {
            stdin: Mutex::new(stdin),
            reader: Mutex::new(BufReader::new(stdout)),
            _child: Mutex::new(child),
        })
    }
}

#[async_trait::async_trait]
impl Transport for StdioTransport {
    async fn send(&self, message: &str) -> Result<(), McpError> {
        let mut stdin = self.stdin.lock().await;
        stdin
            .write_all(message.as_bytes())
            .await
            .map_err(|e| McpError::Transport(format!("write error: {}", e)))?;
        stdin
            .write_all(b"\n")
            .await
            .map_err(|e| McpError::Transport(format!("write error: {}", e)))?;
        stdin
            .flush()
            .await
            .map_err(|e| McpError::Transport(format!("flush error: {}", e)))?;
        Ok(())
    }

    async fn recv(&self) -> Result<String, McpError> {
        let mut reader = self.reader.lock().await;
        let mut line = String::new();
        let n = reader
            .read_line(&mut line)
            .await
            .map_err(|e| McpError::Transport(format!("read error: {}", e)))?;
        if n == 0 {
            error!("MCP server closed stdout");
            return Err(McpError::Transport("server closed connection".into()));
        }
        Ok(line.trim_end().to_string())
    }
}

#[cfg(test)]
pub(crate) struct MockTransport {
    /// Messages that will be returned by recv() in order
    responses: tokio::sync::Mutex<std::collections::VecDeque<String>>,
    /// Messages that were sent via send()
    sent: tokio::sync::Mutex<Vec<String>>,
}

#[cfg(test)]
impl MockTransport {
    pub fn new(responses: Vec<String>) -> Self {
        Self {
            responses: tokio::sync::Mutex::new(responses.into()),
            sent: tokio::sync::Mutex::new(Vec::new()),
        }
    }

    pub async fn sent_messages(&self) -> Vec<String> {
        self.sent.lock().await.clone()
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl Transport for MockTransport {
    async fn send(&self, message: &str) -> Result<(), McpError> {
        self.sent.lock().await.push(message.to_string());
        Ok(())
    }

    async fn recv(&self) -> Result<String, McpError> {
        self.responses
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| McpError::Transport("no more mock responses".into()))
    }
}
