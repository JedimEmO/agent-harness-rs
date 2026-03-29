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
//! let provider = GeminiProvider::new(
//!     "your-api-key".to_string(),
//!     "gemini-2.5-flash".to_string(),
//! );
//! ```

mod error;
mod provider;
mod streaming;
mod types;
#[cfg(feature = "live")]
mod live;

pub use provider::GeminiProvider;
