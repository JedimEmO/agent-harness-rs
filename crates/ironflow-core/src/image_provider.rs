use crate::error::AiError;

#[derive(Debug, Clone)]
pub struct ImageProviderConfig {
    pub aspect_ratio: Option<String>,
    pub image_size: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ImageProviderRequest {
    pub prompt: String,
    pub config: Option<ImageProviderConfig>,
}

#[derive(Debug, Clone)]
pub struct ImageProviderResponse {
    pub image_data: Vec<u8>,
    pub mime_type: String,
}

/// Trait for dedicated image generation providers (ComfyUI, DALL-E, etc.).
#[async_trait::async_trait]
pub trait ImageProvider: Send + Sync {
    async fn generate(&self, request: ImageProviderRequest) -> Result<ImageProviderResponse, AiError>;
    fn provider_name(&self) -> &str;
    fn model_name(&self) -> &str;
}

pub struct StubImageProvider;

#[async_trait::async_trait]
impl ImageProvider for StubImageProvider {
    async fn generate(&self, _request: ImageProviderRequest) -> Result<ImageProviderResponse, AiError> {
        let png_data = vec![
            0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D, 0x49, 0x48,
            0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x06, 0x00, 0x00,
            0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00, 0x0A, 0x49, 0x44, 0x41, 0x54, 0x78,
            0x9C, 0x62, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xE5, 0x27, 0xDE, 0xFC, 0x00, 0x00,
            0x00, 0x00, 0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82,
        ];

        Ok(ImageProviderResponse {
            image_data: png_data,
            mime_type: "image/png".to_string(),
        })
    }

    fn provider_name(&self) -> &str {
        "stub"
    }

    fn model_name(&self) -> &str {
        "stub"
    }
}
