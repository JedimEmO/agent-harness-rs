/// Errors from AI provider operations.
#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("provider error: {0}")]
    ProviderError(String),
    #[error("invalid request: {0}")]
    InvalidRequest(String),
    #[error("not supported: {0}")]
    NotSupported(String),
    #[error("rate limited")]
    RateLimited,
}

impl AiError {
    /// Map any HTTP-like error (e.g. `reqwest::Error`) into an [`AiError::ProviderError`].
    pub fn http_error(e: impl std::fmt::Display) -> Self {
        AiError::ProviderError(format!("HTTP error: {}", e))
    }

    /// Map an API error status code and body into the appropriate [`AiError`] variant.
    ///
    /// Returns [`AiError::RateLimited`] for HTTP 429, otherwise [`AiError::ProviderError`].
    pub fn api_status(status: u16, body: &str, provider: &str) -> Self {
        if status == 429 {
            return AiError::RateLimited;
        }
        AiError::ProviderError(format!("{} API error ({}): {}", provider, status, body))
    }
}

/// Errors from agent operations.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("AI error: {0}")]
    AiError(#[from] AiError),
    #[error("tool '{tool_name}' error: {message}")]
    ToolError { tool_name: String, message: String },
    #[error("session not found: {0}")]
    SessionNotFound(String),
    #[error("storage error: {0}")]
    StorageError(String),
    #[error("exceeded max tool rounds: {0}")]
    MaxRoundsExceeded(usize),
    #[error("channel closed: {0}")]
    ChannelClosed(String),
    #[error("serialization error: {0}")]
    SerializationError(#[from] serde_json::Error),
    /// The turn was cancelled via the cancellation token.
    #[error("turn cancelled")]
    Cancelled,
    /// A tool execution exceeded its timeout.
    #[error("tool '{tool_name}' timed out after {timeout_secs}s")]
    Timeout { tool_name: String, timeout_secs: u64 },
    /// Tool arguments failed validation against the schema.
    #[error("tool '{tool_name}' argument validation failed: {}", errors.join("; "))]
    ValidationError {
        tool_name: String,
        errors: Vec<String>,
    },
}
