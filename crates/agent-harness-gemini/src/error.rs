use agent_harness_core::AiError;

pub fn map_reqwest_error(e: reqwest::Error) -> AiError {
    AiError::ProviderError(format!("HTTP error: {}", e))
}

pub fn map_api_error(status: u16, body: &str) -> AiError {
    if status == 429 {
        return AiError::RateLimited;
    }
    AiError::ProviderError(format!("Gemini API error ({}): {}", status, body))
}
