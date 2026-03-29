use tracing::{debug, trace, warn};
use tokio_stream::Stream;
use agent_harness_core::{AiError, StreamEvent, ToolCall};

use crate::types::{StreamingContentBlock, StreamingDelta, StreamingEvent};

/// State machine for accumulating streaming events from the Anthropic API.
pub struct StreamAccumulator {
    text: String,
    pending_tools: Vec<PendingToolCall>,
    current_block_index: Option<usize>,
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
            current_block_index: None,
        }
    }

    pub fn process_event(&mut self, event: &StreamingEvent) -> Vec<Result<StreamEvent, AiError>> {
        let mut out = Vec::new();

        match event.event_type.as_str() {
            "content_block_start" => {
                self.current_block_index = event.index;
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
            "content_block_stop" => {
                self.current_block_index = None;
            }
            "message_delta" => {
                if let Some(ref usage) = event.usage {
                    out.push(Ok(StreamEvent::Usage {
                        input_tokens: usage.input_tokens,
                        output_tokens: usage.output_tokens,
                    }));
                }
            }
            "message_stop" => {
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
                } else if !self.text.is_empty() {
                    debug!(text_len = self.text.len(), "stream: message_stop with text");
                    out.push(Ok(StreamEvent::TextComplete(
                        std::mem::take(&mut self.text),
                    )));
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
        let mut buffer = String::new();

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

                if line.starts_with("data: ") {
                    let data = &line[6..];
                    if let Ok(event) = serde_json::from_str::<StreamingEvent>(data) {
                        for stream_event in accumulator.process_event(&event) {
                            yield stream_event;
                        }
                    }
                }
            }
        }
    }
}
