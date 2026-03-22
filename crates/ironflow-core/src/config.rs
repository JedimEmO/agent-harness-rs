use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub max_tool_rounds: usize,
    pub max_context_tokens: usize,
    pub chars_per_token_estimate: usize,
    /// Per-tool execution timeout in seconds. None means no timeout.
    pub tool_timeout_secs: Option<u64>,
    /// Maximum tokens for AI provider responses. None lets the provider decide.
    pub max_tokens: Option<u32>,
    /// Maximum number of retries on rate-limit errors.
    pub max_retries: usize,
    /// Base delay in milliseconds for exponential backoff on retries.
    pub retry_base_delay_ms: u64,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            max_tool_rounds: 20,
            max_context_tokens: 100_000,
            chars_per_token_estimate: 4,
            max_tokens: Some(4096),
            tool_timeout_secs: None,
            max_retries: 3,
            retry_base_delay_ms: 1000,
        }
    }
}
