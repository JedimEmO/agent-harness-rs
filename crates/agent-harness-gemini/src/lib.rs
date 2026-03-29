//! # agent-harness-gemini
//!
//! Google Gemini API provider for the agent-harness agent framework.
//!
//! Supports both the standard `generateContent` / `streamGenerateContent` API
//! (via [`GeminiProvider`] implementing [`AiProvider`](agent_harness_core::AiProvider))
//! and the real-time Live API over WebSocket (via [`LiveProvider`](agent_harness_core::LiveProvider),
//! feature-gated behind `live`).
//!
//! ## Usage
//!
//! ```rust,no_run
//! use agent_harness_gemini::GeminiProvider;
//!
//! // From environment variables (recommended):
//! let provider = GeminiProvider::from_env()
//!     .expect("GEMINI_API_KEY or GOOGLE_API_KEY must be set");
//!
//! // Or explicit:
//! let provider = GeminiProvider::new(
//!     "your-api-key".to_string(),
//!     "gemini-2.5-flash".to_string(),
//! );
//! ```
//!
//! ## Environment Variables
//!
//! | Variable | Required | Default |
//! |----------|----------|---------|
//! | `GEMINI_API_KEY` / `GOOGLE_API_KEY` | Yes | — |
//! | `GEMINI_MODEL` | No | `gemini-2.5-flash` |
//! | `GEMINI_BASE_URL` | No | `https://generativelanguage.googleapis.com` |

mod error;
mod provider;
mod streaming;
mod types;
#[cfg(feature = "live")]
mod live;

pub use provider::GeminiProvider;
