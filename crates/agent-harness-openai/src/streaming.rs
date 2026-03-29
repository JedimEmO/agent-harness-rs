use tracing::{debug, trace, warn};
use tokio_stream::Stream;
use agent_harness_core::{AiError, StreamEvent, ToolCall};

use crate::types::OaiStreamChunk;

struct PendingToolCall {
    id: String,
    name: String,
    arguments: String,
}

/// Parse raw SSE lines from an OpenAI-compatible streaming response into StreamEvents.
pub fn parse_oai_sse_stream(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<StreamEvent, AiError>> + Send {
    async_stream::stream! {
        use tokio_stream::StreamExt;

        let mut buffer = String::new();
        let mut text = String::new();
        let mut pending_tools: Vec<PendingToolCall> = Vec::new();

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

            while let Some(newline_pos) = buffer.find('\n') {
                let line = buffer[..newline_pos].trim_end_matches('\r').to_string();
                buffer = buffer[newline_pos + 1..].to_string();

                if !line.starts_with("data: ") {
                    continue;
                }

                let data = &line[6..];
                if data == "[DONE]" {
                    // Emit TextComplete first if there's accumulated text
                    if !text.is_empty() {
                        debug!(text_len = text.len(), "stream done with text");
                        yield Ok(StreamEvent::TextComplete(std::mem::take(&mut text)));
                    }
                    // Then emit ToolCalls if any
                    if !pending_tools.is_empty() {
                        let tool_calls: Vec<ToolCall> = pending_tools
                            .drain(..)
                            .map(|pt| {
                                let arguments = serde_json::from_str(&pt.arguments)
                                    .unwrap_or_else(|e| {
                                        warn!(tool_name = %pt.name, error = %e, "failed to parse tool JSON");
                                        serde_json::Value::Object(serde_json::Map::new())
                                    });
                                ToolCall {
                                    id: pt.id,
                                    name: pt.name,
                                    arguments,
                                }
                            })
                            .collect();
                        debug!(count = tool_calls.len(), "stream done with tool calls");
                        yield Ok(StreamEvent::ToolCalls(tool_calls));
                    }
                    yield Ok(StreamEvent::Done);
                    return;
                }

                let chunk: OaiStreamChunk = match serde_json::from_str(data) {
                    Ok(c) => c,
                    Err(_) => continue,
                };

                if let Some(usage) = &chunk.usage {
                    yield Ok(StreamEvent::Usage {
                        input_tokens: usage.prompt_tokens,
                        output_tokens: usage.completion_tokens,
                    });
                }

                for choice in &chunk.choices {
                    // Text content
                    if let Some(content) = &choice.delta.content {
                        text.push_str(content);
                        trace!(len = content.len(), "stream: text delta");
                        yield Ok(StreamEvent::TextDelta(content.clone()));
                    }

                    // Tool calls
                    if let Some(tool_calls) = &choice.delta.tool_calls {
                        for tc in tool_calls {
                            // Grow pending_tools to accommodate index
                            while pending_tools.len() <= tc.index {
                                pending_tools.push(PendingToolCall {
                                    id: String::new(),
                                    name: String::new(),
                                    arguments: String::new(),
                                });
                            }

                            let pending = &mut pending_tools[tc.index];
                            if let Some(id) = &tc.id {
                                pending.id = id.clone();
                            }
                            if let Some(func) = &tc.function {
                                if let Some(name) = &func.name {
                                    pending.name = name.clone();
                                }
                                if let Some(args) = &func.arguments {
                                    pending.arguments.push_str(args);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Stream ended without [DONE] — emit what we have (connection drop, timeout, etc.)
        if !text.is_empty() {
            warn!(text_len = text.len(), "stream ended without [DONE], emitting text");
            yield Ok(StreamEvent::TextComplete(text));
        }
        if !pending_tools.is_empty() {
            let tool_calls: Vec<ToolCall> = pending_tools
                .drain(..)
                .map(|pt| {
                    let arguments = serde_json::from_str(&pt.arguments)
                        .unwrap_or_else(|e| {
                            warn!(tool_name = %pt.name, error = %e, "failed to parse tool JSON");
                            serde_json::Value::Object(serde_json::Map::new())
                        });
                    ToolCall {
                        id: pt.id,
                        name: pt.name,
                        arguments,
                    }
                })
                .collect();
            warn!(count = tool_calls.len(), "stream ended without [DONE], emitting tool calls");
            yield Ok(StreamEvent::ToolCalls(tool_calls));
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
    async fn parse_oai_text_stream() {
        let chunks = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n",
            "data: [DONE]\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hello"));
        assert!(matches!(&events[1], StreamEvent::TextDelta(t) if t == " world"));
        assert!(matches!(&events[2], StreamEvent::TextComplete(t) if t == "Hello world"));
        assert!(matches!(&events[3], StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_oai_tool_stream() {
        let chunks = vec![
            // First chunk: tool call start with id and name
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            // Argument fragments
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"city\\\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\": \\\"LA\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Find the ToolCalls event
        let tool_event = events.iter().find(|e| matches!(e, StreamEvent::ToolCalls(_)));
        assert!(tool_event.is_some(), "expected a ToolCalls event");
        match tool_event.unwrap() {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "call_1");
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].arguments, serde_json::json!({"city": "LA"}));
            }
            _ => unreachable!(),
        }
        assert!(matches!(events.last().unwrap(), StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_oai_done_marker() {
        let chunks = vec!["data: [DONE]\n\n"];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_oai_chunked_boundary() {
        // Split an SSE line across two byte chunks at an arbitrary boundary
        let chunks = vec![
            "data: {\"choices\":[{\"delta\":{\"con",
            "tent\":\"Hi\"},\"finish_reason\":null}]}\n\ndata: [DONE]\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 3);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "Hi"));
        assert!(matches!(&events[1], StreamEvent::TextComplete(t) if t == "Hi"));
        assert!(matches!(&events[2], StreamEvent::Done));
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[tokio::test]
    async fn parse_oai_empty_stream() {
        // BUG PROBE: Completely empty stream (no data at all).
        // Stream ends without [DONE], so no events are emitted except possibly from
        // lingering state — but text and pending_tools are empty, so nothing.
        // Actually, we never reach the [DONE] path, and the while loop just ends.
        // No Done event is emitted!
        let chunks: Vec<&str> = vec![];
        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<Result<StreamEvent, _>> = stream.collect::<Vec<_>>().await;
        // Empty stream should still emit Done so caller doesn't hang.
        assert_eq!(events.len(), 1);
        assert!(matches!(events[0].as_ref().unwrap(), StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_oai_stream_ends_without_done_marker() {
        // BUG PROBE: Stream has data but no [DONE] marker.
        // The accumulated text is LOST — no TextComplete is emitted.
        let chunks = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"orphan\"},\"finish_reason\":null}]}\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // TextDelta + TextComplete + Done (from end-of-stream fallback)
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "orphan")));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextComplete(t) if t == "orphan")));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done)));
    }

    #[tokio::test]
    async fn parse_oai_tool_with_empty_arguments() {
        // BUG PROBE: Tool call with empty string arguments ""
        let chunks = vec![
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"no_args\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let tool_event = events.iter().find(|e| matches!(e, StreamEvent::ToolCalls(_)));
        assert!(tool_event.is_some());
        match tool_event.unwrap() {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls[0].name, "no_args");
                // Empty string fails JSON parse, falls back to {}
                assert_eq!(calls[0].arguments, serde_json::json!({}));
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn parse_oai_done_with_no_prior_content() {
        // BUG PROBE: Only [DONE] marker with nothing before it
        let chunks = vec!["data: [DONE]\n\n"];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // Only Done event, no text or tool calls
        assert_eq!(events.len(), 1);
        assert!(matches!(&events[0], StreamEvent::Done));
    }

    #[tokio::test]
    async fn parse_oai_multiple_tool_calls_interleaved() {
        // BUG PROBE: Multiple tool calls arriving with interleaved index fragments
        let chunks = vec![
            // Tool 0 starts
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"type\":\"function\",\"function\":{\"name\":\"tool_a\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            // Tool 1 starts
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"tool_b\",\"arguments\":\"\"}}]},\"finish_reason\":null}]}\n\n",
            // Tool 0 gets arguments
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\":1}\"}}]},\"finish_reason\":null}]}\n\n",
            // Tool 1 gets arguments
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"function\":{\"arguments\":\"{\\\"b\\\":2}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let tool_event = events.iter().find(|e| matches!(e, StreamEvent::ToolCalls(_)));
        assert!(tool_event.is_some());
        match tool_event.unwrap() {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].name, "tool_a");
                assert_eq!(calls[0].arguments, serde_json::json!({"a": 1}));
                assert_eq!(calls[1].name, "tool_b");
                assert_eq!(calls[1].arguments, serde_json::json!({"b": 2}));
            }
            _ => unreachable!(),
        }
    }

    #[tokio::test]
    async fn parse_oai_text_and_tool_calls_same_response() {
        // BUG PROBE: OpenAI can return text content + tool calls in same message.
        // Text deltas come first, then tool calls, then [DONE].
        // On [DONE], pending_tools is non-empty so ToolCalls is emitted
        // but TextComplete is NOT emitted (same bug as Anthropic accumulator).
        let chunks = vec![
            "data: {\"choices\":[{\"delta\":{\"content\":\"thinking...\"},\"finish_reason\":null}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{\\\"q\\\":\\\"test\\\"}\"}}]},\"finish_reason\":null}]}\n\n",
            "data: [DONE]\n\n",
        ];

        let stream = parse_oai_sse_stream(mock_byte_stream(chunks));
        let events: Vec<StreamEvent> = stream
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        let has_text_delta = events.iter().any(|e| matches!(e, StreamEvent::TextDelta(_)));
        let has_text_complete = events.iter().any(|e| matches!(e, StreamEvent::TextComplete(_)));
        let has_tool_calls = events.iter().any(|e| matches!(e, StreamEvent::ToolCalls(_)));

        assert!(has_text_delta, "TextDelta should be emitted");
        assert!(has_tool_calls, "ToolCalls should be emitted");
        assert!(
            has_text_complete,
            "TextComplete should be emitted even when tool calls are also present"
        );
    }
}
