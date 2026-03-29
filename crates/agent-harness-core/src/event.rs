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
    /// Audio data from a streaming response.
    AudioDelta {
        data: Vec<u8>,
        mime_type: String,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_event_text_delta_serde() {
        let event = AgentEvent::TextDelta { text: "hello".into() };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: AgentEvent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn agent_event_audio_delta_serde() {
        let event = AgentEvent::AudioDelta { data: vec![0, 1, 2], mime_type: "audio/pcm;rate=24000".into() };
        let json = serde_json::to_string(&event).unwrap();
        let deserialized: AgentEvent = serde_json::from_str(&json).unwrap();
        let json2 = serde_json::to_string(&deserialized).unwrap();
        assert_eq!(json, json2);
    }

    #[test]
    fn agent_event_all_variants_serde() {
        let events: Vec<AgentEvent> = vec![
            AgentEvent::SessionStarted { session_id: "s1".into() },
            AgentEvent::Thinking,
            AgentEvent::TextDelta { text: "hi".into() },
            AgentEvent::TextComplete { message_id: "m1".into(), text: "hello".into() },
            AgentEvent::ToolCallStarted {
                call_id: "c1".into(),
                tool_name: "test".into(),
                arguments: serde_json::json!({"a": 1}),
                permission: ToolPermission::AutoExecute,
            },
            AgentEvent::ToolCallCompleted {
                call_id: "c1".into(),
                tool_name: "test".into(),
                result: serde_json::json!("ok"),
                duration_ms: 42,
            },
            AgentEvent::ToolApprovalNeeded {
                call_id: "c2".into(),
                tool_name: "dangerous".into(),
                arguments: serde_json::json!({}),
                description: "needs approval".into(),
            },
            AgentEvent::InteractionNeeded {
                interaction_id: "i1".into(),
                request: InteractionRequest::AskUser { question: "what?".into(), context: None },
            },
            AgentEvent::TurnComplete,
            AgentEvent::Error { message: "boom".into(), recoverable: false },
            AgentEvent::MemorySaved { key: "k1".into(), summary: "saved".into() },
            AgentEvent::AudioDelta { data: vec![0, 1], mime_type: "audio/pcm;rate=24000".into() },
            AgentEvent::ContextTruncated { dropped_messages: 3, remaining_messages: 10 },
            AgentEvent::UsageUpdate { input_tokens: 100, output_tokens: 50 },
            AgentEvent::RetryAttempt { round: 1, attempt: 2, delay_ms: 1000 },
        ];
        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: AgentEvent = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2, "roundtrip failed for: {:?}", event);
        }
    }
}
