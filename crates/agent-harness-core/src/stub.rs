use crate::error::AiError;
use crate::provider::*;

pub struct StubProvider;

#[async_trait::async_trait]
impl AiProvider for StubProvider {
    async fn converse(&self, request: ConversationRequest) -> Result<ConversationResponse, AiError> {
        let context = request.messages.iter().find_map(|m| {
            if let ConversationMessage::User { content } = m {
                // Extract text from content parts
                let text: String = content
                    .iter()
                    .filter_map(|p| {
                        if let ContentPart::Text { text } = p {
                            Some(text.as_str())
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .join(" ");
                Some(text)
            } else {
                None
            }
        }).unwrap_or_default();

        Ok(ConversationResponse::Text(format!(
            "[STUB] Response to: {}",
            &context[..context.len().min(200)]
        )))
    }

    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            text_generation: true,
            image_generation: false,
            image_analysis: false,
            streaming: false,
            conversation: true,
            provider_name: "stub".to_string(),
        }
    }
}
