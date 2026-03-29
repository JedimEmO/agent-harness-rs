use tracing::{debug, trace, warn};
use tokio_stream::Stream;
use agent_harness_core::{AiError, StreamEvent, ToolCall};

use crate::types::{GeminiPart, GeminiResponse};

/// Parse raw SSE lines from a Gemini streamGenerateContent response into StreamEvents.
pub fn parse_gemini_sse_stream(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<StreamEvent, AiError>> + Send {
    async_stream::stream! {
        use tokio_stream::StreamExt;

        let mut buffer = String::new();
        let mut full_text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut tool_call_counter = 0u32;

        let mut byte_stream = std::pin::pin!(byte_stream);

        while let Some(chunk_result) = byte_stream.next().await {
            let chunk = match chunk_result {
                Ok(bytes) => match String::from_utf8(bytes.to_vec()) {
                    Ok(s) => s,
                    Err(e) => {
                        yield Err(AiError::ProviderError(format!("UTF-8 decode error: {}", e)));
                        return;
                    }
                },
                Err(e) => {
                    yield Err(AiError::ProviderError(format!("Stream error: {}", e)));
                    return;
                }
            };

            buffer.push_str(&chunk);

            // Gemini SSE format: "data: {json}\r\n\r\n"
            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                let response: GeminiResponse = match serde_json::from_str(data) {
                    Ok(r) => r,
                    Err(e) => {
                        trace!(error = %e, "failed to parse Gemini SSE chunk, skipping");
                        continue;
                    }
                };

                // Process usage metadata
                if let Some(usage) = &response.usage_metadata {
                    yield Ok(StreamEvent::Usage {
                        input_tokens: usage.prompt_token_count,
                        output_tokens: usage.candidates_token_count,
                    });
                }

                // Process candidate content
                for candidate in &response.candidates {
                    if let Some(content) = &candidate.content {
                        for part in &content.parts {
                            match part {
                                GeminiPart::Text { text } => {
                                    full_text.push_str(text);
                                    trace!(len = text.len(), "stream: text delta");
                                    yield Ok(StreamEvent::TextDelta(text.clone()));
                                }
                                GeminiPart::FunctionCall { function_call } => {
                                    tool_call_counter += 1;
                                    let call = ToolCall {
                                        id: format!("call_{}", tool_call_counter),
                                        name: function_call.name.clone(),
                                        arguments: function_call.args.clone(),
                                    };
                                    debug!(tool_name = %call.name, "stream: function call");
                                    tool_calls.push(call);
                                }
                                _ => {}
                            }
                        }
                    }

                    // Check finish reason
                    if candidate.finish_reason.as_deref() == Some("STOP")
                        || candidate.finish_reason.as_deref() == Some("MAX_TOKENS")
                    {
                        if !tool_calls.is_empty() {
                            debug!(count = tool_calls.len(), "stream done with tool calls");
                            yield Ok(StreamEvent::ToolCalls(std::mem::take(&mut tool_calls)));
                        } else if !full_text.is_empty() {
                            debug!(text_len = full_text.len(), "stream done with text");
                            yield Ok(StreamEvent::TextComplete(std::mem::take(&mut full_text)));
                        }
                        yield Ok(StreamEvent::Done);
                        return;
                    }
                }
            }
        }

        // Stream ended without explicit STOP — emit what we have
        if !tool_calls.is_empty() {
            warn!(count = tool_calls.len(), "stream ended without STOP, emitting tool calls");
            yield Ok(StreamEvent::ToolCalls(tool_calls));
        } else if !full_text.is_empty() {
            warn!(text_len = full_text.len(), "stream ended without STOP, emitting text");
            yield Ok(StreamEvent::TextComplete(full_text));
        }
        yield Ok(StreamEvent::Done);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::StreamEvent;
    use tokio_stream::StreamExt;

    fn mock_byte_stream(chunks: Vec<&str>) -> impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static {
        let owned: Vec<String> = chunks.into_iter().map(|s| s.to_string()).collect();
        async_stream::stream! {
            for chunk in owned {
                yield Ok(bytes::Bytes::from(chunk));
            }
        }
    }

    #[tokio::test]
    async fn parse_text_response() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"Hello world\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        ];

        let stream = parse_gemini_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hello world"));
        assert!(matches!(&events[1], StreamEvent::TextComplete(t) if t == "Hello world"));
        assert!(matches!(&events[2], StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_function_call() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"functionCall\":{\"name\":\"get_weather\",\"args\":{\"city\":\"NYC\"}}}]},\"finishReason\":\"STOP\"}]}\n\n",
        ];

        let stream = parse_gemini_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let tool_event = events.iter().find(|e| matches!(e, StreamEvent::ToolCalls(_)));
        assert!(tool_event.is_some(), "expected a ToolCalls event");
        match tool_event.unwrap() {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].arguments, serde_json::json!({"city": "NYC"}));
            }
            _ => unreachable!(),
        }
        assert!(matches!(events.last().unwrap(), StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_usage_metadata() {
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"hi\"}]},\"finishReason\":\"STOP\"}],\"usageMetadata\":{\"promptTokenCount\":10,\"candidatesTokenCount\":5,\"totalTokenCount\":15}}\n\n",
        ];

        let stream = parse_gemini_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let usage_event = events.iter().find(|e| matches!(e, StreamEvent::Usage { .. }));
        assert!(usage_event.is_some(), "expected a Usage event");
        assert!(matches!(usage_event.unwrap(), StreamEvent::Usage { input_tokens: 10, output_tokens: 5 }));
    }

    #[tokio::test]
    async fn parse_no_stop_emits_what_we_have() {
        // Stream with text but no finishReason — stream just ends
        let chunks = vec![
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"partial\"}]}}]}\n\n",
        ];

        let stream = parse_gemini_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Should still emit the text delta, TextComplete (from end-of-stream fallback), and Done
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "partial")));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextComplete(t) if t == "partial")));
        assert!(matches!(events.last().unwrap(), StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_invalid_json_skipped() {
        let chunks = vec![
            "data: {this is not valid json}\n\n",
            "data: {\"candidates\":[{\"content\":{\"role\":\"model\",\"parts\":[{\"text\":\"ok\"}]},\"finishReason\":\"STOP\"}]}\n\n",
        ];

        let stream = parse_gemini_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Invalid JSON is skipped; valid line still processed
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "ok")));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextComplete(t) if t == "ok")));
        assert!(matches!(events.last().unwrap(), StreamEvent::Done));
    }
}
