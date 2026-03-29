use std::pin::Pin;

use crate::error::AiError;
use serde::{Deserialize, Serialize};
use tokio_stream::Stream;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenRequest {
    pub prompt: String,
    pub system_prompt: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextGenResponse {
    pub text: String,
    pub tokens_used: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenRequest {
    pub prompt: String,
    pub negative_prompt: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageGenResponse {
    pub image_data: Vec<u8>,
    pub mime_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAnalysisRequest {
    pub image_data: Vec<u8>,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAnalysisResponse {
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderCapabilities {
    pub text_generation: bool,
    pub image_generation: bool,
    pub image_analysis: bool,
    pub streaming: bool,
    pub conversation: bool,
    pub provider_name: String,
}

// Tool/conversation types for agentic AI interactions

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub call_id: String,
    /// Structured tool result. Providers serialize this to the appropriate
    /// format for their API (typically JSON string).
    pub content: serde_json::Value,
}

/// A part of a multi-modal message (text, image, etc.).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ContentPart {
    Text { text: String },
    Image { media_type: String, data: Vec<u8> },
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        ContentPart::Text { text: text.into() }
    }

    pub fn image(media_type: impl Into<String>, data: Vec<u8>) -> Self {
        ContentPart::Image {
            media_type: media_type.into(),
            data,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationMessage {
    /// User message, supporting multi-modal content (text, images).
    User { content: Vec<ContentPart> },
    Assistant { content: String },
    AssistantToolCalls { tool_calls: Vec<ToolCall> },
    ToolResults { results: Vec<ToolResult> },
}

impl ConversationMessage {
    /// Convenience constructor for a text-only user message.
    pub fn user_text(text: impl Into<String>) -> Self {
        ConversationMessage::User {
            content: vec![ContentPart::text(text)],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRequest {
    pub system: Option<String>,
    pub messages: Vec<ConversationMessage>,
    pub tools: Vec<ToolDefinition>,
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConversationResponse {
    Text(String),
    ToolCalls(Vec<ToolCall>),
}

// Streaming types

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StreamEvent {
    TextDelta(String),
    ToolCalls(Vec<ToolCall>),
    TextComplete(String),
    Usage { input_tokens: u32, output_tokens: u32 },
    Done,
}

pub type AiStream = Pin<Box<dyn Stream<Item = Result<StreamEvent, AiError>> + Send>>;

/// Trait for AI provider backends (Anthropic, OpenAI, local LLMs, etc.).
///
/// Implement this trait to add a new LLM backend to agent-harness.
#[async_trait::async_trait]
pub trait AiProvider: Send + Sync {
    async fn converse(&self, request: ConversationRequest) -> Result<ConversationResponse, AiError>;
    fn capabilities(&self) -> ProviderCapabilities;

    /// Simple text generation. Default wraps [`converse`].
    async fn generate_text(&self, request: TextGenRequest) -> Result<TextGenResponse, AiError> {
        let conv_request = ConversationRequest {
            system: request.system_prompt,
            messages: vec![ConversationMessage::user_text(request.prompt)],
            tools: vec![],
            max_tokens: request.max_tokens,
        };
        let response = self.converse(conv_request).await?;
        match response {
            ConversationResponse::Text(text) => Ok(TextGenResponse {
                text,
                tokens_used: None,
            }),
            ConversationResponse::ToolCalls(_) => Err(AiError::ProviderError(
                "unexpected tool calls in text generation".to_string(),
            )),
        }
    }

    /// Image generation. Default returns [`AiError::NotSupported`].
    async fn generate_image(&self, _request: ImageGenRequest) -> Result<ImageGenResponse, AiError> {
        Err(AiError::NotSupported("image generation not supported by this provider".to_string()))
    }

    /// Image analysis. Default returns [`AiError::NotSupported`].
    async fn analyze_image(&self, _request: ImageAnalysisRequest) -> Result<ImageAnalysisResponse, AiError> {
        Err(AiError::NotSupported("image analysis not supported by this provider".to_string()))
    }

    /// Stream a conversation response. Default wraps [`converse`] into a single-item stream.
    async fn converse_stream(&self, request: ConversationRequest) -> Result<AiStream, AiError> {
        let response = self.converse(request).await?;
        let events = match response {
            ConversationResponse::Text(text) => vec![
                Ok(StreamEvent::TextDelta(text.clone())),
                Ok(StreamEvent::TextComplete(text)),
                Ok(StreamEvent::Done),
            ],
            ConversationResponse::ToolCalls(calls) => vec![
                Ok(StreamEvent::ToolCalls(calls)),
                Ok(StreamEvent::Done),
            ],
        };
        Ok(Box::pin(tokio_stream::iter(events)))
    }
}
