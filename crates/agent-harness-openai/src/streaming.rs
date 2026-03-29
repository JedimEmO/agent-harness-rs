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
                    // Emit final events
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
                    } else if !text.is_empty() {
                        debug!(text_len = text.len(), "stream done with text");
                        yield Ok(StreamEvent::TextComplete(std::mem::take(&mut text)));
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
    }
}
