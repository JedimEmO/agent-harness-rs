use async_trait::async_trait;
use tracing::{debug, error, info};
use agent_harness_core::*;

use crate::error::{map_api_error, map_reqwest_error};
use crate::streaming::parse_oai_sse_stream;
use crate::types::*;

/// OpenAI Chat Completions API provider.
///
/// Works with OpenAI, OpenRouter, vLLM, Ollama, Azure OpenAI, and any
/// OpenAI-compatible endpoint.
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
    extra_headers: Vec<(String, String)>,
}

impl OpenAiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: "https://api.openai.com".to_string(),
            extra_headers: Vec::new(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    pub fn with_extra_header(mut self, key: String, value: String) -> Self {
        self.extra_headers.push((key, value));
        self
    }

    fn apply_auth(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        let mut req = req.header("Authorization", format!("Bearer {}", self.api_key));
        for (k, v) in &self.extra_headers {
            req = req.header(k.as_str(), v.as_str());
        }
        req
    }

    fn build_messages(
        &self,
        system: &Option<String>,
        messages: &[ConversationMessage],
    ) -> Vec<OaiMessage> {
        let mut result = Vec::new();

        if let Some(sys) = system {
            result.push(OaiMessage {
                role: "system".to_string(),
                content: Some(OaiContent::Text(sys.clone())),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        for msg in messages {
            match msg {
                ConversationMessage::User { content } => {
                    // Multi-modal: check if we have images
                    let has_images = content
                        .iter()
                        .any(|p| matches!(p, ContentPart::Image { .. }));

                    if has_images {
                        let parts: Vec<OaiContentPart> = content
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => {
                                    Some(OaiContentPart::Text { text: text.clone() })
                                }
                                ContentPart::Image { media_type, data } => {
                                    use base64::Engine;
                                    let encoded =
                                        base64::engine::general_purpose::STANDARD.encode(data);
                                    Some(OaiContentPart::ImageUrl {
                                        image_url: OaiImageUrl {
                                            url: format!(
                                                "data:{};base64,{}",
                                                media_type, encoded
                                            ),
                                        },
                                    })
                                }
                            })
                            .collect();
                        result.push(OaiMessage {
                            role: "user".to_string(),
                            content: Some(OaiContent::Parts(parts)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    } else {
                        // Text-only: use simple string content
                        let text: String = content
                            .iter()
                            .filter_map(|p| match p {
                                ContentPart::Text { text } => Some(text.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("");
                        result.push(OaiMessage {
                            role: "user".to_string(),
                            content: Some(OaiContent::Text(text)),
                            tool_calls: None,
                            tool_call_id: None,
                        });
                    }
                }
                ConversationMessage::Assistant { content } => {
                    result.push(OaiMessage {
                        role: "assistant".to_string(),
                        content: Some(OaiContent::Text(content.clone())),
                        tool_calls: None,
                        tool_call_id: None,
                    });
                }
                ConversationMessage::AssistantToolCalls { tool_calls } => {
                    let oai_calls: Vec<OaiToolCall> = tool_calls
                        .iter()
                        .map(|tc| OaiToolCall {
                            id: tc.id.clone(),
                            call_type: "function".to_string(),
                            function: OaiFunction {
                                name: tc.name.clone(),
                                arguments: serde_json::to_string(&tc.arguments)
                                    .unwrap_or_else(|_| "{}".to_string()),
                            },
                        })
                        .collect();
                    result.push(OaiMessage {
                        role: "assistant".to_string(),
                        content: None,
                        tool_calls: Some(oai_calls),
                        tool_call_id: None,
                    });
                }
                ConversationMessage::ToolResults { results } => {
                    for r in results {
                        let content_str = match &r.content {
                            serde_json::Value::String(s) => s.clone(),
                            other => other.to_string(),
                        };
                        result.push(OaiMessage {
                            role: "tool".to_string(),
                            content: Some(OaiContent::Text(content_str)),
                            tool_calls: None,
                            tool_call_id: Some(r.call_id.clone()),
                        });
                    }
                }
            }
        }

        result
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<OaiTool> {
        tools
            .iter()
            .map(|t| OaiTool {
                tool_type: "function".to_string(),
                function: OaiToolDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.parameters.clone(),
                },
            })
            .collect()
    }

    fn parse_response(&self, response: OaiResponse) -> ConversationResponse {
        let choice = match response.choices.into_iter().next() {
            Some(c) => c,
            None => return ConversationResponse::Text(String::new()),
        };

        if let Some(tool_calls) = choice.message.tool_calls {
            if !tool_calls.is_empty() {
                let calls: Vec<ToolCall> = tool_calls
                    .into_iter()
                    .map(|tc| {
                        let arguments = serde_json::from_str(&tc.function.arguments)
                            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                        ToolCall {
                            id: tc.id,
                            name: tc.function.name,
                            arguments,
                        }
                    })
                    .collect();
                return ConversationResponse::ToolCalls(calls);
            }
        }

        ConversationResponse::Text(choice.message.content.unwrap_or_default())
    }
}

#[async_trait]
impl AiProvider for OpenAiProvider {
    async fn converse(
        &self,
        request: ConversationRequest,
    ) -> Result<ConversationResponse, AiError> {
        let tools = self.build_tools(&request.tools);
        let tool_choice = if !tools.is_empty() {
            Some("auto".to_string())
        } else {
            None
        };

        let oai_request = OaiRequest {
            model: self.model.clone(),
            messages: self.build_messages(&request.system, &request.messages),
            tools,
            tool_choice,
            max_tokens: request.max_tokens.unwrap_or(4096),
            stream: false,
        };

        let msg_count = oai_request.messages.len();
        let tool_count = oai_request.tools.len();
        info!(
            model = %self.model,
            base_url = %self.base_url,
            messages = msg_count,
            tools = tool_count,
            "sending request to OpenAI-compatible API"
        );

        let req = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("content-type", "application/json")
            .json(&oai_request);
        let start = std::time::Instant::now();
        let response = self.apply_auth(req)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let latency_ms = start.elapsed().as_millis();

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            error!(status, latency_ms, body = %body, "OpenAI API error");
            return Err(map_api_error(status, &body));
        }

        let body_text = response.text().await.map_err(map_reqwest_error)?;
        debug!(body = %body_text, "raw API response body");

        let oai_response: OaiResponse = serde_json::from_str(&body_text)
            .map_err(|e| {
                error!(error = %e, body = %body_text, "failed to parse API response");
                AiError::ProviderError(format!("Response parse error: {}", e))
            })?;

        let usage_info = oai_response.usage.as_ref()
            .map(|u| format!("in={} out={}", u.prompt_tokens, u.completion_tokens))
            .unwrap_or_default();
        info!(
            model = %self.model,
            latency_ms,
            usage = %usage_info,
            "OpenAI API response received"
        );

        Ok(self.parse_response(oai_response))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_generation: true,
            image_generation: false,
            image_analysis: false,
            streaming: true,
            conversation: true,
            provider_name: format!("openai-compat/{}", self.model),
        }
    }

    async fn converse_stream(
        &self,
        request: ConversationRequest,
    ) -> Result<AiStream, AiError> {
        let tools = self.build_tools(&request.tools);
        let tool_choice = if !tools.is_empty() {
            Some("auto".to_string())
        } else {
            None
        };

        let oai_request = OaiRequest {
            model: self.model.clone(),
            messages: self.build_messages(&request.system, &request.messages),
            tools,
            tool_choice,
            max_tokens: request.max_tokens.unwrap_or(4096),
            stream: true,
        };

        info!(
            model = %self.model,
            base_url = %self.base_url,
            messages = oai_request.messages.len(),
            tools = oai_request.tools.len(),
            "sending streaming request to OpenAI-compatible API"
        );

        let req = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .header("content-type", "application/json")
            .json(&oai_request);
        let response = self.apply_auth(req)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            error!(status, body = %body, "OpenAI API streaming error");
            return Err(map_api_error(status, &body));
        }

        debug!(model = %self.model, "SSE stream connected");

        let byte_stream = response.bytes_stream();
        let event_stream = parse_oai_sse_stream(byte_stream);

        Ok(Box::pin(event_stream))
    }
}
