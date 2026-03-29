use std::pin::Pin;
use std::time::Instant;

use async_trait::async_trait;
use tokio_stream::Stream;

use agent_harness_core::*;

use crate::event::{MonitorEvent, MonitorEventKind};
use crate::sink::MonitorSink;

/// Wraps any [`AiProvider`] to capture all requests and responses to the monitor.
pub struct MonitoredProvider<P: AiProvider> {
    inner: P,
    sink: MonitorSink,
}

impl<P: AiProvider> MonitoredProvider<P> {
    pub fn new(inner: P, sink: MonitorSink) -> Self {
        Self { inner, sink }
    }
}

#[async_trait]
impl<P: AiProvider> AiProvider for MonitoredProvider<P> {
    async fn converse(
        &self,
        request: ConversationRequest,
    ) -> Result<ConversationResponse, AiError> {
        let provider_name = self.inner.capabilities().provider_name;
        let span_id = uuid::Uuid::new_v4().to_string();

        // Emit request event
        self.sink.emit(
            MonitorEvent::new(MonitorEventKind::ProviderRequest {
                provider: provider_name.clone(),
                system_prompt: request.system.clone(),
                message_count: request.messages.len(),
                tool_count: request.tools.len(),
                messages: request.messages.clone(),
                tools: request.tools.clone(),
            })
            .with_span(&span_id),
        );

        let start = Instant::now();
        let result = self.inner.converse(request).await;
        let duration_ms = start.elapsed().as_millis() as u64;

        match &result {
            Ok(response) => {
                self.sink.emit(
                    MonitorEvent::new(MonitorEventKind::ProviderComplete {
                        provider: provider_name,
                        response: response.clone(),
                        duration_ms,
                        input_tokens: None,
                        output_tokens: None,
                    })
                    .with_span(&span_id),
                );
            }
            Err(e) => {
                self.sink.emit(
                    MonitorEvent::new(MonitorEventKind::ProviderError {
                        provider: provider_name,
                        error: e.to_string(),
                    })
                    .with_span(&span_id),
                );
            }
        }

        result
    }

    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }

    async fn converse_stream(
        &self,
        request: ConversationRequest,
    ) -> Result<AiStream, AiError> {
        let provider_name = self.inner.capabilities().provider_name;
        let span_id = uuid::Uuid::new_v4().to_string();

        // Emit request event
        self.sink.emit(
            MonitorEvent::new(MonitorEventKind::ProviderRequest {
                provider: provider_name.clone(),
                system_prompt: request.system.clone(),
                message_count: request.messages.len(),
                tool_count: request.tools.len(),
                messages: request.messages.clone(),
                tools: request.tools.clone(),
            })
            .with_span(&span_id),
        );

        let start = Instant::now();
        let stream_result = self.inner.converse_stream(request).await;

        match stream_result {
            Ok(stream) => {
                let monitored = MonitoredStream {
                    inner: stream,
                    sink: self.sink.clone(),
                    provider: provider_name,
                    span_id,
                    start,
                    last_input_tokens: None,
                    last_output_tokens: None,
                };
                Ok(Box::pin(monitored))
            }
            Err(e) => {
                self.sink.emit(
                    MonitorEvent::new(MonitorEventKind::ProviderError {
                        provider: provider_name,
                        error: e.to_string(),
                    })
                    .with_span(&span_id),
                );
                Err(e)
            }
        }
    }

    async fn generate_text(&self, request: TextGenRequest) -> Result<TextGenResponse, AiError> {
        self.inner.generate_text(request).await
    }

    async fn generate_image(&self, request: ImageGenRequest) -> Result<ImageGenResponse, AiError> {
        self.inner.generate_image(request).await
    }

    async fn analyze_image(
        &self,
        request: ImageAnalysisRequest,
    ) -> Result<ImageAnalysisResponse, AiError> {
        self.inner.analyze_image(request).await
    }
}

/// A stream wrapper that emits monitor events for each streamed item.
struct MonitoredStream {
    inner: AiStream,
    sink: MonitorSink,
    provider: String,
    span_id: String,
    start: Instant,
    last_input_tokens: Option<u32>,
    last_output_tokens: Option<u32>,
}

impl Stream for MonitoredStream {
    type Item = Result<StreamEvent, AiError>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        let inner = unsafe { self.as_mut().map_unchecked_mut(|s| &mut s.inner) };
        match inner.poll_next(cx) {
            std::task::Poll::Ready(Some(Ok(event))) => {
                // Track usage for the final Complete event
                if let StreamEvent::Usage {
                    input_tokens,
                    output_tokens,
                } = &event
                {
                    self.last_input_tokens = Some(*input_tokens);
                    self.last_output_tokens = Some(*output_tokens);
                }

                // On Done, emit ProviderComplete
                if matches!(event, StreamEvent::Done) {
                    let duration_ms = self.start.elapsed().as_millis() as u64;
                    self.sink.emit(
                        MonitorEvent::new(MonitorEventKind::ProviderComplete {
                            provider: self.provider.clone(),
                            response: ConversationResponse::Text(String::new()), // placeholder
                            duration_ms,
                            input_tokens: self.last_input_tokens,
                            output_tokens: self.last_output_tokens,
                        })
                        .with_span(&self.span_id),
                    );
                } else {
                    // Emit stream event
                    self.sink.emit(
                        MonitorEvent::new(MonitorEventKind::ProviderStreamEvent {
                            provider: self.provider.clone(),
                            event: event.clone(),
                        })
                        .with_span(&self.span_id),
                    );
                }

                std::task::Poll::Ready(Some(Ok(event)))
            }
            std::task::Poll::Ready(Some(Err(e))) => {
                self.sink.emit(
                    MonitorEvent::new(MonitorEventKind::ProviderError {
                        provider: self.provider.clone(),
                        error: e.to_string(),
                    })
                    .with_span(&self.span_id),
                );
                std::task::Poll::Ready(Some(Err(e)))
            }
            other => other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::MonitorFilter;
    use crate::sink::MonitorSink;
    use tokio_stream::StreamExt;

    struct TestProvider {
        response: ConversationResponse,
    }

    #[async_trait]
    impl AiProvider for TestProvider {
        async fn converse(
            &self,
            _: ConversationRequest,
        ) -> Result<ConversationResponse, AiError> {
            Ok(self.response.clone())
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider_name: "test-provider".into(),
                text_generation: true,
                image_generation: false,
                image_analysis: false,
                streaming: true,
                conversation: true,
                audio_input: false,
                audio_output: false,
                live_session: false,
            }
        }
    }

    struct ErrorProvider;

    #[async_trait]
    impl AiProvider for ErrorProvider {
        async fn converse(
            &self,
            _: ConversationRequest,
        ) -> Result<ConversationResponse, AiError> {
            Err(AiError::ProviderError("test error".into()))
        }

        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                provider_name: "error-provider".into(),
                text_generation: true,
                image_generation: false,
                image_analysis: false,
                streaming: true,
                conversation: true,
                audio_input: false,
                audio_output: false,
                live_session: false,
            }
        }
    }

    fn make_request() -> ConversationRequest {
        ConversationRequest {
            system: Some("You are helpful".into()),
            messages: vec![ConversationMessage::user_text("Hello")],
            tools: vec![],
            max_tokens: None,
        }
    }

    #[tokio::test]
    async fn monitored_provider_captures_converse() {
        let sink = MonitorSink::in_memory().unwrap();
        let provider = TestProvider {
            response: ConversationResponse::Text("Hi there!".into()),
        };
        let monitored = MonitoredProvider::new(provider, sink.clone());

        let result = monitored.converse(make_request()).await.unwrap();
        assert!(matches!(result, ConversationResponse::Text(ref t) if t == "Hi there!"));

        let events = sink.query(&MonitorFilter::default()).unwrap();
        assert_eq!(events.len(), 2);

        // Events are ordered desc by timestamp, so complete comes first (or same ts)
        // Check that we have one request and one complete
        let tags: Vec<&str> = events.iter().map(|e| e.kind_tag()).collect();
        assert!(tags.contains(&"provider_request"));
        assert!(tags.contains(&"provider_complete"));

        // Verify they share the same span_id
        let span_ids: std::collections::HashSet<_> = events
            .iter()
            .filter_map(|e| e.span_id.as_deref())
            .collect();
        assert_eq!(span_ids.len(), 1, "request and complete should share a span_id");

        // Verify provider name is captured
        for e in &events {
            assert_eq!(e.provider_name(), Some("test-provider"));
        }
    }

    #[tokio::test]
    async fn monitored_provider_captures_stream() {
        let sink = MonitorSink::in_memory().unwrap();
        let provider = TestProvider {
            response: ConversationResponse::Text("streamed response".into()),
        };
        let monitored = MonitoredProvider::new(provider, sink.clone());

        let mut stream = monitored.converse_stream(make_request()).await.unwrap();

        // Consume the stream fully
        let mut stream_events = Vec::new();
        while let Some(item) = stream.next().await {
            stream_events.push(item.unwrap());
        }
        assert!(!stream_events.is_empty());
        assert!(matches!(stream_events.last(), Some(StreamEvent::Done)));

        let events = sink.query(&MonitorFilter::default()).unwrap();

        // Should have: ProviderRequest, ProviderStreamEvent(s), ProviderComplete
        let tags: Vec<&str> = events.iter().map(|e| e.kind_tag()).collect();
        assert!(tags.contains(&"provider_request"), "missing provider_request");
        assert!(
            tags.contains(&"provider_stream"),
            "missing provider_stream events"
        );
        assert!(
            tags.contains(&"provider_complete"),
            "missing provider_complete"
        );

        // All should share the same span
        let span_ids: std::collections::HashSet<_> = events
            .iter()
            .filter_map(|e| e.span_id.as_deref())
            .collect();
        assert_eq!(span_ids.len(), 1);
    }

    #[tokio::test]
    async fn monitored_provider_captures_error() {
        let sink = MonitorSink::in_memory().unwrap();
        let monitored = MonitoredProvider::new(ErrorProvider, sink.clone());

        let result = monitored.converse(make_request()).await;
        assert!(result.is_err());

        let events = sink.query(&MonitorFilter::default()).unwrap();
        let tags: Vec<&str> = events.iter().map(|e| e.kind_tag()).collect();
        assert!(tags.contains(&"provider_request"));
        assert!(tags.contains(&"provider_error"));

        // Verify error message is captured
        let error_event = events.iter().find(|e| e.kind_tag() == "provider_error").unwrap();
        match &error_event.kind {
            MonitorEventKind::ProviderError { error, .. } => {
                assert!(error.contains("test error"));
            }
            _ => panic!("expected ProviderError kind"),
        }
    }
}
