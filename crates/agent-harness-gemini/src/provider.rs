use async_trait::async_trait;
use tracing::{debug, error, info};
use agent_harness_core::*;

use crate::error::{map_api_error, map_reqwest_error};
use crate::streaming::parse_gemini_sse_stream;
use crate::types::*;

/// Google Gemini API provider.
///
/// Speaks the Gemini `generateContent` and `streamGenerateContent` APIs.
/// Auth is via API key in query parameter.
pub struct GeminiProvider {
    client: reqwest::Client,
    api_key: String,
    model: String,
    base_url: String,
}

impl GeminiProvider {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
            base_url: "https://generativelanguage.googleapis.com".to_string(),
        }
    }

    pub fn with_base_url(mut self, base_url: String) -> Self {
        self.base_url = base_url;
        self
    }

    /// Get the API key (needed by the live provider in the same crate).
    pub(crate) fn api_key(&self) -> &str {
        &self.api_key
    }

    /// Get the model name.
    pub(crate) fn model(&self) -> &str {
        &self.model
    }

    fn api_url(&self, method: &str) -> String {
        format!(
            "{}/v1beta/models/{}:{}?key={}",
            self.base_url, self.model, method, self.api_key
        )
    }

    fn build_contents(&self, messages: &[ConversationMessage]) -> Vec<GeminiContent> {
        let mut contents = Vec::new();

        for msg in messages {
            match msg {
                ConversationMessage::User { content } => {
                    let parts: Vec<GeminiPart> = content
                        .iter()
                        .filter_map(|p| match p {
                            ContentPart::Text { text } => {
                                Some(GeminiPart::Text { text: text.clone() })
                            }
                            ContentPart::Image { media_type, data } => {
                                use base64::Engine;
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                Some(GeminiPart::InlineData {
                                    inline_data: InlineData {
                                        mime_type: media_type.clone(),
                                        data: encoded,
                                    },
                                })
                            }
                            ContentPart::Audio { media_type, data } => {
                                use base64::Engine;
                                let encoded =
                                    base64::engine::general_purpose::STANDARD.encode(data);
                                Some(GeminiPart::InlineData {
                                    inline_data: InlineData {
                                        mime_type: media_type.clone(),
                                        data: encoded,
                                    },
                                })
                            }
                        })
                        .collect();
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts,
                    });
                }
                ConversationMessage::Assistant { content } => {
                    contents.push(GeminiContent {
                        role: Some("model".to_string()),
                        parts: vec![GeminiPart::Text {
                            text: content.clone(),
                        }],
                    });
                }
                ConversationMessage::AssistantToolCalls { tool_calls } => {
                    let parts: Vec<GeminiPart> = tool_calls
                        .iter()
                        .map(|tc| GeminiPart::FunctionCall {
                            function_call: FunctionCall {
                                name: tc.name.clone(),
                                args: tc.arguments.clone(),
                            },
                        })
                        .collect();
                    contents.push(GeminiContent {
                        role: Some("model".to_string()),
                        parts,
                    });
                }
                ConversationMessage::ToolResults { results } => {
                    let parts: Vec<GeminiPart> = results
                        .iter()
                        .map(|r| {
                            // Gemini requires function name in FunctionResponse but we only have call_id.
                            // We use call_id as the name since it's the best we have.
                            GeminiPart::FunctionResponse {
                                function_response: FunctionResponse {
                                    name: r.call_id.clone(),
                                    response: r.content.clone(),
                                },
                            }
                        })
                        .collect();
                    contents.push(GeminiContent {
                        role: Some("user".to_string()),
                        parts,
                    });
                }
            }
        }

        contents
    }

    fn build_tools(&self, tools: &[ToolDefinition]) -> Vec<GeminiToolConfig> {
        if tools.is_empty() {
            return Vec::new();
        }

        let declarations: Vec<FunctionDeclaration> = tools
            .iter()
            .map(|t| FunctionDeclaration {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            })
            .collect();

        vec![GeminiToolConfig {
            function_declarations: declarations,
        }]
    }

    fn parse_response(&self, response: GeminiResponse) -> ConversationResponse {
        let candidate = match response.candidates.into_iter().next() {
            Some(c) => c,
            None => return ConversationResponse::Text(String::new()),
        };

        let content = match candidate.content {
            Some(c) => c,
            None => return ConversationResponse::Text(String::new()),
        };

        let mut text_parts = Vec::new();
        let mut tool_calls = Vec::new();
        let mut tool_call_counter = 0u32;

        for part in content.parts {
            match part {
                GeminiPart::Text { text } => {
                    text_parts.push(text);
                }
                GeminiPart::FunctionCall { function_call } => {
                    tool_call_counter += 1;
                    tool_calls.push(ToolCall {
                        id: format!("call_{}", tool_call_counter),
                        name: function_call.name,
                        arguments: function_call.args,
                    });
                }
                _ => {}
            }
        }

        if !tool_calls.is_empty() {
            ConversationResponse::ToolCalls(tool_calls)
        } else {
            ConversationResponse::Text(text_parts.join(""))
        }
    }
}

#[async_trait]
impl AiProvider for GeminiProvider {
    async fn converse(
        &self,
        request: ConversationRequest,
    ) -> Result<ConversationResponse, AiError> {
        let system_instruction = request.system.map(|sys| GeminiContent {
            role: None,
            parts: vec![GeminiPart::Text { text: sys }],
        });

        let gemini_request = GeminiRequest {
            contents: self.build_contents(&request.messages),
            system_instruction,
            tools: self.build_tools(&request.tools),
            generation_config: Some(GenerationConfig {
                max_output_tokens: request.max_tokens,
                temperature: None,
            }),
        };

        let content_count = gemini_request.contents.len();
        let tool_count = gemini_request
            .tools
            .first()
            .map(|t| t.function_declarations.len())
            .unwrap_or(0);
        info!(
            model = %self.model,
            base_url = %self.base_url,
            contents = content_count,
            tools = tool_count,
            "sending request to Gemini API"
        );

        let start = std::time::Instant::now();
        let response = self
            .client
            .post(self.api_url("generateContent"))
            .header("content-type", "application/json")
            .json(&gemini_request)
            .send()
            .await
            .map_err(map_reqwest_error)?;
        let latency_ms = start.elapsed().as_millis();

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            error!(status, latency_ms, body = %body, "Gemini API error");
            return Err(map_api_error(status, &body));
        }

        let body_text = response.text().await.map_err(map_reqwest_error)?;
        debug!(body = %body_text, "raw API response body");

        let gemini_response: GeminiResponse = serde_json::from_str(&body_text).map_err(|e| {
            error!(error = %e, body = %body_text, "failed to parse Gemini response");
            AiError::ProviderError(format!("Response parse error: {}", e))
        })?;

        let usage_info = gemini_response
            .usage_metadata
            .as_ref()
            .map(|u| {
                format!(
                    "in={} out={}",
                    u.prompt_token_count, u.candidates_token_count
                )
            })
            .unwrap_or_default();
        info!(
            model = %self.model,
            latency_ms,
            usage = %usage_info,
            "Gemini API response received"
        );

        Ok(self.parse_response(gemini_response))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_generation: true,
            image_generation: false,
            image_analysis: true,
            streaming: true,
            conversation: true,
            provider_name: format!("gemini/{}", self.model),
            audio_input: true,
            audio_output: false,
            live_session: cfg!(feature = "live"),
        }
    }

    async fn converse_stream(
        &self,
        request: ConversationRequest,
    ) -> Result<AiStream, AiError> {
        let system_instruction = request.system.map(|sys| GeminiContent {
            role: None,
            parts: vec![GeminiPart::Text { text: sys }],
        });

        let gemini_request = GeminiRequest {
            contents: self.build_contents(&request.messages),
            system_instruction,
            tools: self.build_tools(&request.tools),
            generation_config: Some(GenerationConfig {
                max_output_tokens: request.max_tokens,
                temperature: None,
            }),
        };

        info!(
            model = %self.model,
            base_url = %self.base_url,
            contents = gemini_request.contents.len(),
            "sending streaming request to Gemini API"
        );

        let url = format!("{}&alt=sse", self.api_url("streamGenerateContent"));
        let response = self
            .client
            .post(url)
            .header("content-type", "application/json")
            .json(&gemini_request)
            .send()
            .await
            .map_err(map_reqwest_error)?;

        let status = response.status().as_u16();
        if status != 200 {
            let body = response.text().await.unwrap_or_default();
            error!(status, body = %body, "Gemini API streaming error");
            return Err(map_api_error(status, &body));
        }

        debug!(model = %self.model, "SSE stream connected");

        let byte_stream = response.bytes_stream();
        let event_stream = parse_gemini_sse_stream(byte_stream);

        Ok(Box::pin(event_stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::{ContentPart, ConversationMessage, ToolCall as CoreToolCall, ToolResult};

    fn test_provider() -> GeminiProvider {
        GeminiProvider::new("test-key".to_string(), "gemini-test".to_string())
    }

    #[test]
    fn build_contents_user_text() {
        let provider = test_provider();
        let messages = vec![ConversationMessage::User {
            content: vec![ContentPart::Text {
                text: "Hello".to_string(),
            }],
        }];
        let contents = provider.build_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        match &contents[0].parts[0] {
            GeminiPart::Text { text } => assert_eq!(text, "Hello"),
            _ => panic!("expected Text part"),
        }
    }

    #[test]
    fn build_contents_user_image() {
        let provider = test_provider();
        let messages = vec![ConversationMessage::User {
            content: vec![ContentPart::Image {
                media_type: "image/png".to_string(),
                data: vec![1, 2, 3],
            }],
        }];
        let contents = provider.build_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        match &contents[0].parts[0] {
            GeminiPart::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "image/png");
                // Verify base64 encoding of [1, 2, 3]
                use base64::Engine;
                let expected = base64::engine::general_purpose::STANDARD.encode([1, 2, 3]);
                assert_eq!(inline_data.data, expected);
            }
            _ => panic!("expected InlineData part"),
        }
    }

    #[test]
    fn build_contents_user_audio() {
        let provider = test_provider();
        let messages = vec![ConversationMessage::User {
            content: vec![ContentPart::Audio {
                media_type: "audio/pcm;rate=16000".to_string(),
                data: vec![10, 20, 30],
            }],
        }];
        let contents = provider.build_contents(&messages);
        assert_eq!(contents.len(), 1);
        match &contents[0].parts[0] {
            GeminiPart::InlineData { inline_data } => {
                assert_eq!(inline_data.mime_type, "audio/pcm;rate=16000");
            }
            _ => panic!("expected InlineData part"),
        }
    }

    #[test]
    fn build_contents_assistant_text() {
        let provider = test_provider();
        let messages = vec![ConversationMessage::Assistant {
            content: "Hi there".to_string(),
        }];
        let contents = provider.build_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("model"));
        match &contents[0].parts[0] {
            GeminiPart::Text { text } => assert_eq!(text, "Hi there"),
            _ => panic!("expected Text part"),
        }
    }

    #[test]
    fn build_contents_tool_calls() {
        let provider = test_provider();
        let messages = vec![ConversationMessage::AssistantToolCalls {
            tool_calls: vec![CoreToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                arguments: serde_json::json!({"city": "NYC"}),
            }],
        }];
        let contents = provider.build_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("model"));
        match &contents[0].parts[0] {
            GeminiPart::FunctionCall { function_call } => {
                assert_eq!(function_call.name, "get_weather");
                assert_eq!(function_call.args["city"], "NYC");
            }
            _ => panic!("expected FunctionCall part"),
        }
    }

    #[test]
    fn build_contents_tool_results() {
        let provider = test_provider();
        let messages = vec![ConversationMessage::ToolResults {
            results: vec![ToolResult {
                call_id: "call_1".to_string(),
                content: serde_json::json!({"temp": 72}),
            }],
        }];
        let contents = provider.build_contents(&messages);
        assert_eq!(contents.len(), 1);
        assert_eq!(contents[0].role.as_deref(), Some("user"));
        match &contents[0].parts[0] {
            GeminiPart::FunctionResponse { function_response } => {
                assert_eq!(function_response.name, "call_1");
                assert_eq!(function_response.response["temp"], 72);
            }
            _ => panic!("expected FunctionResponse part"),
        }
    }

    #[test]
    fn parse_response_text() {
        let provider = test_provider();
        let response = GeminiResponse {
            candidates: vec![Candidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart::Text {
                        text: "Hello!".to_string(),
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };
        match provider.parse_response(response) {
            ConversationResponse::Text(text) => assert_eq!(text, "Hello!"),
            _ => panic!("expected Text response"),
        }
    }

    #[test]
    fn parse_response_function_calls() {
        let provider = test_provider();
        let response = GeminiResponse {
            candidates: vec![Candidate {
                content: Some(GeminiContent {
                    role: Some("model".to_string()),
                    parts: vec![GeminiPart::FunctionCall {
                        function_call: FunctionCall {
                            name: "search".to_string(),
                            args: serde_json::json!({"q": "rust"}),
                        },
                    }],
                }),
                finish_reason: Some("STOP".to_string()),
            }],
            usage_metadata: None,
        };
        match provider.parse_response(response) {
            ConversationResponse::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "search");
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].arguments["q"], "rust");
            }
            _ => panic!("expected ToolCalls response"),
        }
    }
}
