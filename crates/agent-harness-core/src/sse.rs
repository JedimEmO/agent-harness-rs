//! Shared SSE (Server-Sent Events) line parsing utilities for streaming providers.

use crate::error::AiError;

/// Parse an SSE byte stream into data payloads.
///
/// Reads bytes, buffers them, splits on newlines, and yields the data portion
/// of lines starting with `"data: "`. This handles chunked boundaries correctly.
pub fn parse_sse_lines<E: std::fmt::Display + Send + 'static>(
    byte_stream: impl tokio_stream::Stream<Item = Result<bytes::Bytes, E>> + Send + 'static,
) -> impl tokio_stream::Stream<Item = Result<String, AiError>> + Send {
    async_stream::stream! {
        use tokio_stream::StreamExt;

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

                if let Some(data) = line.strip_prefix("data: ") {
                    yield Ok(data.to_string());
                }
            }
        }
    }
}
