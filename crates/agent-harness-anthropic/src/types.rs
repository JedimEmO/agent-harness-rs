#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// --- Request types ---

#[derive(Debug, Serialize)]
pub struct AnthropicRequest {
    pub model: String,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub system: Option<String>,
    pub messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<AnthropicTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolChoice {
    #[serde(rename = "type")]
    pub choice_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    pub content: AnthropicContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AnthropicContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSource {
    #[serde(rename = "type")]
    pub source_type: String,
    pub media_type: String,
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize)]
pub struct AnthropicTool {
    pub name: String,
    pub description: String,
    pub input_schema: serde_json::Value,
}

// --- Response types ---

#[derive(Debug, Deserialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub response_type: String,
    pub role: String,
    pub content: Vec<ResponseContentBlock>,
    pub stop_reason: Option<String>,
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ResponseContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(other)]
    Unknown,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// --- Streaming event types ---

#[derive(Debug, Deserialize)]
pub struct StreamingEvent {
    #[serde(rename = "type")]
    pub event_type: String,
    #[serde(default)]
    pub index: Option<usize>,
    #[serde(default)]
    pub content_block: Option<StreamingContentBlock>,
    #[serde(default)]
    pub delta: Option<StreamingDelta>,
    #[serde(default)]
    pub message: Option<StreamingMessage>,
    #[serde(default)]
    pub usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum StreamingContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum StreamingDelta {
    #[serde(rename = "text_delta")]
    TextDelta { text: String },
    #[serde(rename = "input_json_delta")]
    InputJsonDelta { partial_json: String },
}

#[derive(Debug, Deserialize)]
pub struct StreamingMessage {
    pub usage: Option<AnthropicUsage>,
}

#[cfg(test)]
mod tests {
    use super::*;

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[test]
    fn anthropic_content_untagged_string_vs_blocks_disambiguation() {
        // BUG PROBE: AnthropicContent is #[serde(untagged)] with Text(String) first.
        // A plain JSON string should deserialize as Text.
        let json = r#""hello world""#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        match content {
            AnthropicContent::Text(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected Text variant for plain string"),
        }
    }

    #[test]
    fn anthropic_content_untagged_blocks_array() {
        // A JSON array should deserialize as Blocks
        let json = r#"[{"type":"text","text":"hello"}]"#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        match content {
            AnthropicContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::Text { text } => assert_eq!(text, "hello"),
                    _ => panic!("expected Text block"),
                }
            }
            _ => panic!("expected Blocks variant for array"),
        }
    }

    #[test]
    fn anthropic_content_empty_string() {
        let json = r#""""#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        match content {
            AnthropicContent::Text(s) => assert!(s.is_empty()),
            _ => panic!("expected Text variant for empty string"),
        }
    }

    #[test]
    fn anthropic_content_empty_array() {
        let json = r#"[]"#;
        let content: AnthropicContent = serde_json::from_str(json).unwrap();
        match content {
            AnthropicContent::Blocks(blocks) => assert!(blocks.is_empty()),
            _ => panic!("expected Blocks variant for empty array"),
        }
    }

    #[test]
    fn anthropic_content_null_fails() {
        // null is neither a string nor an array
        let result = serde_json::from_str::<AnthropicContent>("null");
        assert!(result.is_err(), "null should not match any AnthropicContent variant");
    }

    #[test]
    fn response_content_block_unknown_type() {
        // BUG PROBE: ResponseContentBlock has #[serde(other)] for Unknown.
        // An unknown block type should deserialize as Unknown.
        let json = r#"{"type":"thinking","thinking":"I'm pondering..."}"#;
        let block: ResponseContentBlock = serde_json::from_str(json).unwrap();
        assert!(matches!(block, ResponseContentBlock::Unknown));
    }

    #[test]
    fn streaming_event_minimal_fields() {
        // BUG PROBE: StreamingEvent with only required field (event_type)
        let json = r#"{"type":"ping"}"#;
        let event: StreamingEvent = serde_json::from_str(json).unwrap();
        assert_eq!(event.event_type, "ping");
        assert!(event.index.is_none());
        assert!(event.content_block.is_none());
        assert!(event.delta.is_none());
        assert!(event.message.is_none());
        assert!(event.usage.is_none());
    }

    #[test]
    fn content_block_tool_result_deserialization() {
        let json = r#"{"type":"tool_result","tool_use_id":"abc","content":"result text"}"#;
        let block: ContentBlock = serde_json::from_str(json).unwrap();
        match block {
            ContentBlock::ToolResult { tool_use_id, content } => {
                assert_eq!(tool_use_id, "abc");
                assert_eq!(content, "result text");
            }
            _ => panic!("expected ToolResult"),
        }
    }

    #[test]
    fn anthropic_message_serialization_roundtrip() {
        let msg = AnthropicMessage {
            role: "user".to_string(),
            content: AnthropicContent::Text("hello".to_string()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: AnthropicMessage = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role, "user");
        match deserialized.content {
            AnthropicContent::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("expected Text"),
        }
    }
}
