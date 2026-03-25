//! # ironflow-core
//!
//! Core types, traits, and agent runner for the ironflow LLM agent framework.
//!
//! This crate provides:
//! - **AI provider abstraction** ([`AiProvider`]) for pluggable LLM backends
//! - **Agent runner** ([`AgentRunner`]) implementing a multi-turn tool-calling loop with streaming
//! - **Agent composability** ([`Agent`], [`SubAgentTool`]) for multi-agent workflows
//! - **Tool system** ([`AgentTool`], [`ToolRegistry`]) with validation and dynamic registration
//! - **Storage traits** ([`SessionStore`], [`MemoryStore`]) for persistence
//! - **Event streaming** ([`AgentEvent`]) for real-time UI updates
//! - **Interaction protocol** ([`InteractionRequest`], [`InteractionResponse`]) for user input
//! - **Turn request builder** ([`TurnRequest`], [`TurnChannels`]) for ergonomic API

mod provider;
mod error;
mod stub;
mod image_provider;
mod runner;
mod tool;
mod event;
mod session;
mod memory;
mod interaction;
mod context;
#[cfg(feature = "context-pipeline")]
mod context_bridge;
mod config;
mod agent;
mod guardrail;

// --- Errors ---
pub use error::{AgentError, AiError};

// --- AI provider ---
pub use provider::{
    AiProvider, AiStream, ContentPart, ConversationMessage, ConversationRequest,
    ConversationResponse, ImageAnalysisRequest, ImageAnalysisResponse, ImageGenRequest,
    ImageGenResponse, ProviderCapabilities, StreamEvent, TextGenRequest, TextGenResponse,
    ToolCall, ToolDefinition, ToolResult,
};

// --- Stub provider (for testing) ---
pub use stub::StubProvider;

// --- Image provider ---
pub use image_provider::{
    ImageProvider, ImageProviderConfig, ImageProviderRequest, ImageProviderResponse,
    StubImageProvider,
};

// --- Agent runner ---
pub use runner::{AgentRunner, TurnChannels, TurnRequest};

// --- Tool system ---
pub use tool::{AgentTool, ToolExecResult, ToolPermission, ToolRegistry};

// --- Events ---
pub use event::AgentEvent;

// --- Session storage ---
pub use session::{
    MessageContent, MessageRole, Session, SessionMessage, SessionStore, ToolCallRecord,
    ToolResultRecord,
};

// --- Memory storage ---
pub use memory::{Memory, MemoryCategory, MemoryStore, SemanticMemoryStore};

// --- Interaction protocol ---
pub use interaction::{
    InteractionOption, InteractionRequest, InteractionResponse, PlanStep,
};

// --- Configuration ---
pub use config::AgentConfig;

// --- Agent composability ---
pub use agent::{Agent, SubAgentTool};

// --- Output guardrails ---
pub use guardrail::{GuardrailContext, GuardrailResult, OutputGuardrail};
