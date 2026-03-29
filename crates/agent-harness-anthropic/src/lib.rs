//! # agent-harness-anthropic
//!
//! Anthropic Messages API provider for the agent-harness agent framework.
//!
//! Speaks the native Anthropic Messages API (`/v1/messages`, `x-api-key` auth).
//! Supports both synchronous and streaming conversation modes.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use agent_harness_anthropic::AnthropicProvider;
//!
//! // From environment variables (recommended):
//! let provider = AnthropicProvider::from_env()
//!     .expect("ANTHROPIC_API_KEY must be set");
//!
//! // Or explicit:
//! let provider = AnthropicProvider::new(
//!     "your-api-key".to_string(),
//!     "claude-sonnet-4-6".to_string(),
//! );
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Required | Default |
//! |----------|----------|---------|
//! | `ANTHROPIC_API_KEY` | Yes | — |
//! | `ANTHROPIC_MODEL` | No | `claude-sonnet-4-6` |
//! | `ANTHROPIC_BASE_URL` | No | `https://api.anthropic.com` |

mod error;
mod provider;
mod streaming;
mod types;

pub use provider::AnthropicProvider;
