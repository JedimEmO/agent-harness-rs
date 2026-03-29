use agent_harness_core::AiError;

pub fn map_reqwest_error(e: reqwest::Error) -> AiError {
    AiError::http_error(e)
}

pub fn map_api_error(status: u16, body: &str) -> AiError {
    AiError::api_status(status, body, "OpenAI")
}
