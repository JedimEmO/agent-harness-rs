use async_trait::async_trait;
use tracing::{debug, error, info, warn};
use agent_harness_core::*;

use crate::error::{map_api_error, map_reqwest_error};
use crate::streaming::parse_sse_stream;
use crate::types::{
    AnthropicContent, AnthropicMessage, AnthropicRequest, AnthropicResponse, AnthropicTool,
    ContentBlock, ResponseContentBlock, ToolChoice,
};

/// Anthropic Messages API provider.
///
/// Speaks the native Anthropic API at `https://api.anthropic.com/v1/messages`
/// using `x-api-key` authentication.
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl AnthropicProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: "https://api.anthropic.com".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        req.header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
    }

    fn build_messages(&self, messages: &[ConversationMessage]) -> Vec<AnthropicMessage> {
        let mut result = Vec::new();

        for msg in messages {
            match msg {
                ConversationMessage::User { content } => {
                    // Multi-modal: convert ContentParts to Anthropic blocks
                    if content.len() == 1 {
                        if let ContentPart::Text { text } = &content[0] {
                            result.push(AnthropicMessage {
                                role: "user".to_string(),
                                content: AnthropicContent::Text(text.clone()),
                            });
                            continue;
                        }
                    }

                    let blocks: Vec<ContentBlock> = content
                        .iter()
                        .filter_map(|part| match part {
                            ContentPart::Text { text } => {
                                Some(ContentBlock::Text { text: text.clone() })
                            }
                            ContentPart::Image { media_type, data } => {
                                use base64::Engine;
                                let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                                Some(ContentBlock::Image {
                                    source: crate::types::ImageSource {
                                        source_type: "base64".to_string(),
                                        media_type: media_type.clone(),
                                        data: encoded,
                                    },
                                })
                            }
                        })
                        .collect();

                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
                ConversationMessage::Assistant { content } => {
                    result.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Text(content.clone()),
                    });
                }
                ConversationMessage::AssistantToolCalls { tool_calls } => {
                    let blocks: Vec<ContentBlock> = tool_calls
                        .iter()
                        .map(|tc| ContentBlock::ToolUse {
                            id: tc.id.clone(),
                            name: tc.name.clone(),
                            input: tc.arguments.clone(),
                        })
                        .collect();
                    result.push(AnthropicMessage {
                        role: "assistant".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
                ConversationMessage::ToolResults { results } => {
                    let blocks: Vec<ContentBlock> = results
                        .iter()
                        .map(|r| {
                            // Serialize Value to string for the API
                            let content_str = match &r.content {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            ContentBlock::ToolResult {
                                tool_use_id: r.call_id.clone(),
                                content: content_str,
                            }
                        })
                        .collect();
                    result.push(AnthropicMessage {
                        role: "user".to_string(),
                        content: AnthropicContent::Blocks(blocks),
                    });
                }
            }
        }

        result
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<AnthropicTool> {
        tools
            .iter()
            .map(|t| AnthropicTool {
                name: t.name.clone(),
                description: t.description.clone(),
                input_schema: t.parameters.clone(),
            })
            .collect()
    }

    fn parse_response(&self, response: AnthropicResponse) -> ConversationResponse {
        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();

        for block in response.content {
            match block {
                ResponseContentBlock::Text { text } => {
                    text_parts.push(text);
                }
                ResponseContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id,
                        name,
                        arguments: input,
                    });
                }
                ResponseContentBlock::Unknown => {}
            }
        }

        if !tool_calls.is_empty() {
            ConversationResponse::ToolCalls(tool_calls)
        } else {
            ConversationResponse::Text(text_parts.join(""))
        }
    }

    fn extract_tool_calls_from_body(&self, body: &str) -> Vec<ToolCall> {
        let parsed: serde_json::Value = match serde_json::from_str(body) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        let content = match parsed.get("content").and_then(|c| c.as_array()) {
            Some(arr) => arr,
            None => return Vec::new(),
        };

        let mut tool_calls = Vec::new();
        for block in content {
            let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
            if block_type == "tool_use" {
                let id = block.get("id").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let name = block.get("name").and_then(|v| v.as_str()).unwrap_or("").to_string();
                let input = block.get("input").cloned().unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                if !name.is_empty() {
                    debug!(tool_name = %name, tool_id = %id, "extracted tool call from raw body (fallback)");
                    tool_calls.push(ToolCall { id, name, arguments: input });
                }
            }
        }
        tool_calls
    }
}

#[async_trait]
impl AiProvider for AnthropicProvider {
    async fn converse(
        &self,
        request: ConversationRequest,
    ) -> Result<ConversationResponse, AiError> {
        let tools = self.build_tools(&request.tools);
        let tool_choice = if !tools.is_empty() {
            Some(ToolChoice { choice_type: "auto".to_string(), name: None })
        } else {
            None
        };

        let anthropic_request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: request.max_tokens.unwrap_or(4096),
            system: request.system,
            messages: self.build_messages(&request.messages),
            tools,
            tool_choice,
            stream: false,
        };

        let msg_count = anthropic_request.messages.len();
        let tool_count = anthropic_request.tools.len();
        info!(
            model = %self.model,
            base_url = %self.base_url,
            messages = msg_count,
            tools = tool_count,
            max_tokens = anthropic_request.max_tokens,
            "sending request to Anthropic API"
        );

        let req = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("content-type", "application/json")
            .json(&anthropic_request);
        let start = std::time::Instant::now();
        let response = self.apply_auth(req)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let latency_ms = start.elapsed().as_millis();

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            error!(status, latency_ms, body = %body, "Anthropic API error");
            return Err(map_api_error(status, &body));
        }

        let body_text = response.text().await.map_err(map_reqwest_error)?;
        debug!(body = %body_text, "raw API response body");

        let anthropic_response: AnthropicResponse = serde_json::from_str(&body_text)
            .map_err(|e| {
                error!(error = %e, body = %body_text, "failed to parse API response");
                AiError::ProviderError(format!("Response parse error: {}", e))
            })?;

        let tool_blocks = anthropic_response.content.iter().filter(|b| matches!(b, ResponseContentBlock::ToolUse { .. })).count();
        let usage_info = anthropic_response.usage.as_ref().map(|u| format!("in={} out={}", u.input_tokens, u.output_tokens)).unwrap_or_default();
        let stop_reason = anthropic_response.stop_reason.as_deref().unwrap_or("none");
        info!(
            model = %self.model,
            latency_ms,
            tool_blocks,
            stop_reason,
            usage = %usage_info,
            "Anthropic API response received"
        );

        if stop_reason == "tool_use" && tool_blocks == 0 {
            warn!("stop_reason is tool_use but typed parse found 0 tool blocks — trying fallback parser");
            let fallback_tools = self.extract_tool_calls_from_body(&body_text);
            if !fallback_tools.is_empty() {
                info!(count = fallback_tools.len(), "fallback parser recovered tool calls");
                return Ok(ConversationResponse::ToolCalls(fallback_tools));
            }
            warn!("fallback parser also found no tool calls — returning text response");
        }

        Ok(self.parse_response(anthropic_response))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_generation: true,
            image_generation: false,
            image_analysis: false,
            streaming: true,
            conversation: true,
            provider_name: format!("anthropic/{}", self.model),
        }
    }

    async fn converse_stream(
        &self,
        request: ConversationRequest,
    ) -> Result<AiStream, AiError> {
        let tools = self.build_tools(&request.tools);
        let tool_choice = if !tools.is_empty() {
            Some(ToolChoice { choice_type: "auto".to_string(), name: None })
        } else {
            None
        };

        let anthropic_request = AnthropicRequest {
            model: self.model.clone(),
            max_tokens: request.max_tokens.unwrap_or(4096),
            system: request.system,
            messages: self.build_messages(&request.messages),
            tools,
            tool_choice,
            stream: true,
        };

        info!(
            model = %self.model,
            base_url = %self.base_url,
            messages = anthropic_request.messages.len(),
            tools = anthropic_request.tools.len(),
            "sending streaming request to Anthropic API"
        );

        let req = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("content-type", "application/json")
            .json(&anthropic_request);
        let response = self.apply_auth(req)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            error!(status, body = %body, "Anthropic API streaming error");
            return Err(map_api_error(status, &body));
        }

        debug!(model = %self.model, "SSE stream connected");

        let byte_stream = response.bytes_stream();
        let event_stream = parse_sse_stream(byte_stream);

        Ok(Box::pin(event_stream))
    }
}
