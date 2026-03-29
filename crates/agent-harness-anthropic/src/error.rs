use agent_harness_core::AiError;

pub fn map_reqwest_error(e: reqwest::Error) -> AiError {
    AiError::ProviderError(format!("HTTP error: {}", e))
}

pub fn map_api_error(status: u16, body: &str) -> AiError {
    if status == 429 {
        AiError::RateLimited
    } else {
        AiError::ProviderError(format!("Anthropic API error ({}): {}", status, body))
    }
}
