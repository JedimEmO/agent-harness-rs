use serde::{Deserialize, Serialize};

use crate::interaction::InteractionRequest;
use crate::tool::ToolPermission;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "event", content = "data")]
pub enum AgentEvent {
    SessionStarted {
        session_id: String,
    },
    Thinking,
    TextDelta {
        text: String,
    },
    TextComplete {
        message_id: String,
        text: String,
    },
    ToolCallStarted {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
        permission: ToolPermission,
    },
    ToolCallCompleted {
        call_id: String,
        tool_name: String,
        result: serde_json::Value,
        duration_ms: u64,
    },
    ToolApprovalNeeded {
        call_id: String,
        tool_name: String,
        arguments: serde_json::Value,
        description: String,
    },
    InteractionNeeded {
        interaction_id: String,
        request: InteractionRequest,
    },
    TurnComplete,
    Error {
        message: String,
        recoverable: bool,
    },
    MemorySaved {
        key: String,
        summary: String,
    },
    /// Emitted when context truncation drops messages.
    ContextTruncated {
        dropped_messages: usize,
        remaining_messages: usize,
    },
    /// Token usage from the current AI provider call.
    UsageUpdate {
        input_tokens: u32,
        output_tokens: u32,
    },
    /// Emitted when retrying after a rate limit or transient error.
    RetryAttempt {
        round: usize,
        attempt: usize,
        delay_ms: u64,
    },
}
