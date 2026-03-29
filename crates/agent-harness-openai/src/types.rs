#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// --- Request types ---

#[derive(Debug, Serialize)]
pub struct OaiRequest {
    pub model: String,
    pub messages: Vec<OaiMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub tools: Vec<OaiTool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<String>,
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub stream: bool,
}

/// OpenAI message content — either a plain string or multi-modal parts.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum OaiContent {
    Text(String),
    Parts(Vec<OaiContentPart>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OaiContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: OaiImageUrl },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiImageUrl {
    pub url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<OaiContent>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OaiToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OaiFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OaiFunction {
    pub name: String,
    #[serde(default)]
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct OaiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OaiToolDef,
}

#[derive(Debug, Clone, Serialize)]
pub struct OaiToolDef {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// --- Response types ---

#[derive(Debug, Deserialize)]
pub struct OaiResponse {
    pub choices: Vec<OaiChoice>,
    pub usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OaiChoice {
    pub message: OaiResponseMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OaiResponseMessage {
    pub role: String,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OaiToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct OaiUsage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
}

// --- Streaming types ---

#[derive(Debug, Deserialize)]
pub struct OaiStreamChunk {
    pub choices: Vec<OaiStreamChoice>,
    pub usage: Option<OaiUsage>,
}

#[derive(Debug, Deserialize)]
pub struct OaiStreamChoice {
    pub delta: OaiStreamDelta,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct OaiStreamDelta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub tool_calls: Option<Vec<OaiStreamToolCall>>,
}

#[derive(Debug, Deserialize)]
pub struct OaiStreamToolCall {
    pub index: usize,
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub call_type: Option<String>,
    pub function: Option<OaiStreamFunction>,
}

#[derive(Debug, Deserialize)]
pub struct OaiStreamFunction {
    pub name: Option<String>,
    pub arguments: Option<String>,
}
