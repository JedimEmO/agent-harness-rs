/// Errors from MCP client operations.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    #[error("transport error: {0}")]
    Transport(String),
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("server returned error {code}: {message}")]
    ServerError { code: i64, message: String },
    #[error("timeout waiting for response")]
    Timeout,
    #[error("server not initialized")]
    NotInitialized,
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}
