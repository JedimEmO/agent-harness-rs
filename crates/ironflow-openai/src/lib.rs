//! # ironflow-openai
//!
//! OpenAI Chat Completions API provider for the ironflow agent framework.
//!
//! Speaks the OpenAI Chat Completions format (`/v1/chat/completions`, Bearer auth).
//! Works with OpenAI, OpenRouter, vLLM, Ollama, Azure OpenAI, and any
//! OpenAI-compatible endpoint.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ironflow_openai::OpenAiProvider;
//!
//! // OpenRouter
//! let provider = OpenAiProvider::new(
//!     "your-api-key".to_string(),
//!     "anthropic/claude-sonnet-4-6".to_string(),
//! ).with_base_url("https://openrouter.ai/api".to_string());
//!
//! // Local Ollama
//! let provider = OpenAiProvider::new(
//!     String::new(),
//!     "llama3".to_string(),
//! ).with_base_url("http://localhost:11434".to_string());
//! ```

mod error;
mod provider;
mod streaming;
mod types;

pub use provider::OpenAiProvider;
