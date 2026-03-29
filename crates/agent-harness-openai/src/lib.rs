//! # agent-harness-openai
//!
//! OpenAI Chat Completions API provider for the agent-harness agent framework.
//!
//! Speaks the OpenAI Chat Completions format (`/v1/chat/completions`, Bearer auth).
//! Works with OpenAI, OpenRouter, vLLM, Ollama, Azure OpenAI, and any
//! OpenAI-compatible endpoint.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use agent_harness_openai::OpenAiProvider;
//!
//! // From environment variables (recommended):
//! let provider = OpenAiProvider::from_env()
//!     .expect("OPENAI_API_KEY must be set");
//!
//! // Or explicit:
//! let provider = OpenAiProvider::new(
//!     "your-api-key".to_string(),
//!     "gpt-4o".to_string(),
//! );
//!
//! // OpenRouter (via env: OPENAI_BASE_URL=https://openrouter.ai/api)
//! let provider = OpenAiProvider::new(
//!     "your-api-key".to_string(),
//!     "anthropic/claude-sonnet-4-6".to_string(),
//! ).with_base_url("https://openrouter.ai/api".to_string());
//!
//! // Local Ollama (via env: OPENAI_BASE_URL=http://localhost:11434)
//! let provider = OpenAiProvider::new(
//!     String::new(),
//!     "llama3".to_string(),
//! ).with_base_url("http://localhost:11434".to_string());
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Required | Default |
//! |----------|----------|---------|
//! | `OPENAI_API_KEY` | Yes | — |
//! | `OPENAI_MODEL` | No | `gpt-4o` |
//! | `OPENAI_BASE_URL` | No | `https://api.openai.com` |

mod error;
mod provider;
mod streaming;
mod types;

pub use provider::OpenAiProvider;
