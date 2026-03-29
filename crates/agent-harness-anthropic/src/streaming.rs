use tracing::{debug, trace, warn};
use tokio_stream::Stream;
use agent_harness_core::{AiError, StreamEvent, ToolCall};

use crate::types::{StreamingContentBlock, StreamingDelta, StreamingEvent};

/// State machine for accumulating streaming events from the Anthropic API.
pub struct StreamAccumulator {
    text: String,
    pending_tools: Vec<PendingToolCall>,
}

struct PendingToolCall {
    id: String,
    name: String,
    json_fragments: Vec<String>,
}

impl StreamAccumulator {
    pub fn new() -> Self {
        Self {
            text: String::new(),
            pending_tools: Vec::new(),
        }
    }

    pub fn process_event(&mut self, event: &StreamingEvent) -> Vec<Result<StreamEvent, AiError>> {
        let mut out = Vec::new();

        match event.event_type.as_str() {
            "content_block_start" => {
                if let Some(ref block) = event.content_block {
                    match block {
                        StreamingContentBlock::Text { .. } => {
                            debug!("stream: text block started");
                        }
                        StreamingContentBlock::ToolUse { id, name } => {
                            debug!(tool_id = %id, tool_name = %name, "stream: tool_use block started");
                            self.pending_tools.push(PendingToolCall {
                                id: id.clone(),
                                name: name.clone(),
                                json_fragments: Vec::new(),
                            });
                        }
                    }
                }
            }
            "content_block_delta" => {
                if let Some(ref delta) = event.delta {
                    match delta {
                        StreamingDelta::TextDelta { text } => {
                            self.text.push_str(text);
                            trace!(len = text.len(), total = self.text.len(), "stream: text delta");
                            out.push(Ok(StreamEvent::TextDelta(text.clone())));
                        }
                        StreamingDelta::InputJsonDelta { partial_json } => {
                            trace!(len = partial_json.len(), "stream: input_json_delta");
                            if let Some(tool) = self.pending_tools.last_mut() {
                                tool.json_fragments.push(partial_json.clone());
                            }
                        }
                    }
                }
            }
            "content_block_stop" => {}
            "message_delta" => {
                if let Some(ref usage) = event.usage {
                    out.push(Ok(StreamEvent::Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                    }));
                }
            }
            "message_stop" => {
                // Emit TextComplete first if there's accumulated text
                if !self.text.is_empty() {
                    debug!(text_len = self.text.len(), "stream: message_stop with text");
                    out.push(Ok(StreamEvent::TextComplete(
                        std::mem::take(&mut self.text),
                    )));
                }
                // Then emit ToolCalls if any
                if !self.pending_tools.is_empty() {
                    let tool_names: Vec<&str> = self.pending_tools.iter().map(|pt| pt.name.as_str()).collect();
                    debug!(count = self.pending_tools.len(), tools = ?tool_names, "stream: message_stop with tool calls");
                    let tool_calls: Vec<ToolCall> = self
                        .pending_tools
                        .drain(..)
                        .map(|pt| {
                            let json_str: String = pt.json_fragments.concat();
                            let arguments = serde_json::from_str(&json_str)
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
                    out.push(Ok(StreamEvent::ToolCalls(tool_calls)));
                }
                out.push(Ok(StreamEvent::Done));
            }
            _ => {}
        }

        out
    }
}

/// Parse raw SSE lines from an Anthropic streaming response into StreamEvents.
pub fn parse_sse_stream(
    byte_stream: impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static,
) -> impl Stream<Item = Result<StreamEvent, AiError>> + Send {
    async_stream::stream! {
        use tokio_stream::StreamExt;

        let mut accumulator = StreamAccumulator::new();
        let mut lines = std::pin::pin!(agent_harness_core::sse::parse_sse_lines(byte_stream));

        while let Some(line_result) = lines.next().await {
            let data = match line_result {
                Ok(d) => d,
                Err(e) => {
                    yield Err(e);
                    return;
                }
            };

            if let Ok(event) = serde_json::from_str::<StreamingEvent>(&data) {
                for stream_event in accumulator.process_event(&event) {
                    yield stream_event;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AnthropicUsage, StreamingContentBlock, StreamingDelta, StreamingEvent};
    use agent_harness_core::StreamEvent;

    fn text_block_start(index: usize) -> StreamingEvent {
        StreamingEvent {
            event_type: "content_block_start".to_string(),
            index: Some(index),
            content_block: Some(StreamingContentBlock::Text {
                text: String::new(),
            }),
            delta: None,
            message: None,
            usage: None,
        }
    }

    fn text_delta(text: &str) -> StreamingEvent {
        StreamingEvent {
            event_type: "content_block_delta".to_string(),
            index: None,
            content_block: None,
            delta: Some(StreamingDelta::TextDelta {
                text: text.to_string(),
            }),
            message: None,
            usage: None,
        }
    }

    fn tool_use_block_start(index: usize, id: &str, name: &str) -> StreamingEvent {
        StreamingEvent {
            event_type: "content_block_start".to_string(),
            index: Some(index),
            content_block: Some(StreamingContentBlock::ToolUse {
                id: id.to_string(),
                name: name.to_string(),
            }),
            delta: None,
            message: None,
            usage: None,
        }
    }

    fn input_json_delta(json: &str) -> StreamingEvent {
        StreamingEvent {
            event_type: "content_block_delta".to_string(),
            index: None,
            content_block: None,
            delta: Some(StreamingDelta::InputJsonDelta {
                partial_json: json.to_string(),
            }),
            message: None,
            usage: None,
        }
    }

    fn content_block_stop() -> StreamingEvent {
        StreamingEvent {
            event_type: "content_block_stop".to_string(),
            index: None,
            content_block: None,
            delta: None,
            message: None,
            usage: None,
        }
    }

    fn message_stop() -> StreamingEvent {
        StreamingEvent {
            event_type: "message_stop".to_string(),
            index: None,
            content_block: None,
            delta: None,
            message: None,
            usage: None,
        }
    }

    fn message_delta_with_usage(input: u32, output: u32) -> StreamingEvent {
        StreamingEvent {
            event_type: "message_delta".to_string(),
            index: None,
            content_block: None,
            delta: None,
            message: None,
            usage: Some(AnthropicUsage {
                input_tokens: input,
                output_tokens: output,
            }),
        }
    }

    fn collect_ok(events: Vec<Result<StreamEvent, AiError>>) -> Vec<StreamEvent> {
        events.into_iter().map(|r| r.unwrap()).collect()
    }

    #[test]
    fn accumulator_text_flow() {
        let mut acc = StreamAccumulator::new();

        let mut all = Vec::new();
        all.extend(collect_ok(acc.process_event(&text_block_start(0))));
        all.extend(collect_ok(acc.process_event(&text_delta("hello"))));
        all.extend(collect_ok(acc.process_event(&text_delta(" world"))));
        all.extend(collect_ok(acc.process_event(&content_block_stop())));
        all.extend(collect_ok(acc.process_event(&message_stop())));

        assert_eq!(all.len(), 4);
        assert!(matches!(&all[0], StreamEvent::TextDelta(t) if t == "hello"));
        assert!(matches!(&all[1], StreamEvent::TextDelta(t) if t == " world"));
        assert!(matches!(&all[2], StreamEvent::TextComplete(t) if t == "hello world"));
        assert!(matches!(&all[3], StreamEvent::Done));
    }

    #[test]
    fn accumulator_tool_use_flow() {
        let mut acc = StreamAccumulator::new();

        let mut all = Vec::new();
        all.extend(collect_ok(acc.process_event(&tool_use_block_start(0, "t1", "get_weather"))));
        all.extend(collect_ok(acc.process_event(&input_json_delta(r#"{"loc"#))));
        all.extend(collect_ok(acc.process_event(&input_json_delta(r#"ation":"NYC"}"#))));
        all.extend(collect_ok(acc.process_event(&content_block_stop())));
        all.extend(collect_ok(acc.process_event(&message_stop())));

        assert_eq!(all.len(), 2);
        match &all[0] {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].id, "t1");
                assert_eq!(calls[0].name, "get_weather");
                assert_eq!(calls[0].arguments, serde_json::json!({"location": "NYC"}));
            }
            other => panic!("expected ToolCalls, got {:?}", other),
        }
        assert!(matches!(&all[1], StreamEvent::Done));
    }

    #[test]
    fn accumulator_tool_malformed_json() {
        let mut acc = StreamAccumulator::new();

        let mut all = Vec::new();
        all.extend(collect_ok(acc.process_event(&tool_use_block_start(0, "t1", "broken"))));
        all.extend(collect_ok(acc.process_event(&input_json_delta("{invalid"))));
        all.extend(collect_ok(acc.process_event(&message_stop())));

        // Should not panic; malformed JSON results in empty object
        assert_eq!(all.len(), 2);
        match &all[0] {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 1);
                assert_eq!(calls[0].arguments, serde_json::json!({}));
            }
            other => panic!("expected ToolCalls, got {:?}", other),
        }
        assert!(matches!(&all[1], StreamEvent::Done));
    }

    #[test]
    fn accumulator_usage_on_message_delta() {
        let mut acc = StreamAccumulator::new();

        let all = collect_ok(acc.process_event(&message_delta_with_usage(100, 50)));

        assert_eq!(all.len(), 1);
        assert!(matches!(&all[0], StreamEvent::Usage { input_tokens: 100, output_tokens: 50 }));
    }

    #[test]
    fn accumulator_multiple_tools() {
        let mut acc = StreamAccumulator::new();

        let mut all = Vec::new();
        // First tool
        all.extend(collect_ok(acc.process_event(&tool_use_block_start(0, "t1", "tool_a"))));
        all.extend(collect_ok(acc.process_event(&input_json_delta(r#"{"a":1}"#))));
        all.extend(collect_ok(acc.process_event(&content_block_stop())));
        // Second tool
        all.extend(collect_ok(acc.process_event(&tool_use_block_start(1, "t2", "tool_b"))));
        all.extend(collect_ok(acc.process_event(&input_json_delta(r#"{"b":2}"#))));
        all.extend(collect_ok(acc.process_event(&content_block_stop())));
        // End
        all.extend(collect_ok(acc.process_event(&message_stop())));

        assert_eq!(all.len(), 2);
        match &all[0] {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls.len(), 2);
                assert_eq!(calls[0].id, "t1");
                assert_eq!(calls[0].name, "tool_a");
                assert_eq!(calls[1].id, "t2");
                assert_eq!(calls[1].name, "tool_b");
            }
            other => panic!("expected ToolCalls, got {:?}", other),
        }
        assert!(matches!(&all[1], StreamEvent::Done));
    }

    fn mock_byte_stream(chunks: Vec<&str>) -> impl Stream<Item = Result<bytes::Bytes, reqwest::Error>> + Send + 'static {
        let owned: Vec<String> = chunks.into_iter().map(|s| s.to_string()).collect();
        async_stream::stream! {
            for chunk in owned {
                yield Ok(bytes::Bytes::from(chunk));
            }
        }
    }

    #[tokio::test]
    async fn parse_sse_stream_text() {
        // Build realistic SSE data as Anthropic sends it
        let sse_data = vec![
            "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hello\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\" world\"}}\n\n",
            "event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ];

        let stream = parse_sse_stream(mock_byte_stream(sse_data));
        let events: Vec<StreamEvent> = tokio_stream::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(events.len(), 4);
        assert!(matches!(&events[0], StreamEvent::TextDelta(t) if t == "hello"));
        assert!(matches!(&events[1], StreamEvent::TextDelta(t) if t == " world"));
        assert!(matches!(&events[2], StreamEvent::TextComplete(t) if t == "hello world"));
        assert!(matches!(&events[3], StreamEvent::Done));
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[test]
    fn accumulator_empty_tool_json() {
        // BUG PROBE: Tool call with completely empty JSON string "".
        // serde_json::from_str("") fails, so it falls back to empty object.
        let mut acc = StreamAccumulator::new();
        let mut all = Vec::new();
        all.extend(collect_ok(acc.process_event(&tool_use_block_start(0, "t1", "empty_tool"))));
        // No json deltas at all — so json_fragments is empty, concat = ""
        all.extend(collect_ok(acc.process_event(&content_block_stop())));
        all.extend(collect_ok(acc.process_event(&message_stop())));

        assert_eq!(all.len(), 2);
        match &all[0] {
            StreamEvent::ToolCalls(calls) => {
                assert_eq!(calls[0].name, "empty_tool");
                // Empty string fails to parse, falls back to {}
                assert_eq!(calls[0].arguments, serde_json::json!({}));
            }
            other => panic!("expected ToolCalls, got {:?}", other),
        }
    }

    #[test]
    fn accumulator_text_then_tool_calls_in_same_message() {
        // BUG PROBE: Anthropic can return both text and tool_use blocks in a single message.
        // The accumulator collects text AND tool calls. On message_stop, it checks
        // pending_tools first. If tools are present, it emits ToolCalls but
        // the accumulated text is SILENTLY LOST.
        let mut acc = StreamAccumulator::new();
        let mut all = Vec::new();

        // Text block
        all.extend(collect_ok(acc.process_event(&text_block_start(0))));
        all.extend(collect_ok(acc.process_event(&text_delta("I'll help you. "))));
        all.extend(collect_ok(acc.process_event(&content_block_stop())));

        // Tool use block
        all.extend(collect_ok(acc.process_event(&tool_use_block_start(1, "t1", "search"))));
        all.extend(collect_ok(acc.process_event(&input_json_delta(r#"{"q":"test"}"#))));
        all.extend(collect_ok(acc.process_event(&content_block_stop())));

        // Message stop
        all.extend(collect_ok(acc.process_event(&message_stop())));

        // We should see: TextDelta, TextComplete, ToolCalls, Done
        let has_text_delta = all.iter().any(|e| matches!(e, StreamEvent::TextDelta(_)));
        let has_text_complete = all.iter().any(|e| matches!(e, StreamEvent::TextComplete(_)));
        let has_tool_calls = all.iter().any(|e| matches!(e, StreamEvent::ToolCalls(_)));

        assert!(has_text_delta, "TextDelta should be emitted during streaming");
        assert!(has_tool_calls, "ToolCalls should be emitted");
        assert!(
            has_text_complete,
            "TextComplete should be emitted even when tool calls are also present"
        );
    }

    #[test]
    fn accumulator_message_stop_with_no_content() {
        // BUG PROBE: message_stop with no prior content at all.
        // pending_tools is empty, text is empty, so we only get Done.
        let mut acc = StreamAccumulator::new();
        let all = collect_ok(acc.process_event(&message_stop()));
        assert_eq!(all.len(), 1);
        assert!(matches!(&all[0], StreamEvent::Done));
    }

    #[test]
    fn accumulator_unknown_event_type() {
        // BUG PROBE: Unknown event type should be silently ignored
        let mut acc = StreamAccumulator::new();
        let event = StreamingEvent {
            event_type: "totally_unknown_event".to_string(),
            index: None,
            content_block: None,
            delta: None,
            message: None,
            usage: None,
        };
        let result = acc.process_event(&event);
        assert!(result.is_empty(), "unknown event types should produce no output");
    }

    #[tokio::test]
    async fn parse_sse_stream_split_in_middle_of_data_prefix() {
        // BUG PROBE: SSE data split in the middle of "data: " prefix.
        // First chunk ends with "da", second chunk starts with "ta: {..."
        // The buffer accumulation handles this: it waits for \n before processing.
        let sse_data = vec![
            "event: content_block_start\nda",
            "ta: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
            "event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"split\"}}\n\n",
            "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
        ];

        let stream = parse_sse_stream(mock_byte_stream(sse_data));
        let events: Vec<StreamEvent> = tokio_stream::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        // The split "data: " should be reassembled correctly
        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "split")));
    }

    #[tokio::test]
    async fn parse_sse_stream_crlf_line_endings() {
        // BUG PROBE: \r\n line endings instead of just \n
        let sse_data = vec![
            "event: content_block_start\r\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\r\n\r\n",
            "event: content_block_delta\r\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"crlf\"}}\r\n\r\n",
            "event: content_block_stop\r\ndata: {\"type\":\"content_block_stop\",\"index\":0}\r\n\r\n",
            "event: message_stop\r\ndata: {\"type\":\"message_stop\"}\r\n\r\n",
        ];

        let stream = parse_sse_stream(mock_byte_stream(sse_data));
        let events: Vec<StreamEvent> = tokio_stream::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "crlf")));
        assert!(events.iter().any(|e| matches!(e, StreamEvent::Done)));
    }

    #[tokio::test]
    async fn parse_sse_stream_empty_lines_and_whitespace() {
        // BUG PROBE: Extra empty lines between events
        let sse_data = vec![
            "\n\n\nevent: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\n\n",
            "data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"extra\"}}\n\n",
            "data: {\"type\":\"message_stop\"}\n\n",
        ];

        let stream = parse_sse_stream(mock_byte_stream(sse_data));
        let events: Vec<StreamEvent> = tokio_stream::StreamExt::collect::<Vec<_>>(stream)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert!(events.iter().any(|e| matches!(e, StreamEvent::TextDelta(t) if t == "extra")));
    }

    #[test]
    fn accumulator_delta_before_block_start() {
        // BUG PROBE: What if we get a delta before any block_start?
        // For text delta, it accumulates into self.text (no panic).
        // For json delta, it tries self.pending_tools.last_mut() which returns None, so it's a no-op.
        let mut acc = StreamAccumulator::new();

        // Text delta without block start — accumulates silently
        let all = collect_ok(acc.process_event(&text_delta("orphan")));
        assert_eq!(all.len(), 1);
        assert!(matches!(&all[0], StreamEvent::TextDelta(t) if t == "orphan"));

        // JSON delta without tool block — silently dropped
        let all = collect_ok(acc.process_event(&input_json_delta(r#"{"lost":true}"#)));
        assert!(all.is_empty(), "orphan JSON delta should be silently dropped");
    }
}
