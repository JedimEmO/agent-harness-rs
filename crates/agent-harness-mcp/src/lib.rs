//! # agent-harness-mcp
//!
//! MCP (Model Context Protocol) client for the agent-harness agent framework.
//!
//! Bridges MCP tool servers into the agent-harness [`ToolRegistry`](agent_harness_core::ToolRegistry),
//! allowing agents to call tools exposed by any MCP-compliant server.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use agent_harness_mcp::{McpClient, McpRegistry};
//! use agent_harness_core::ToolRegistry;
//! use std::sync::Arc;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let client = Arc::new(McpClient::stdio("npx", &["-y", "@anthropic/mcp-server-filesystem"]).await?);
//! let tools = McpRegistry::discover(&client).await?;
//!
//! let registry = ToolRegistry::new();
//! for tool in tools {
//!     registry.register(Arc::new(tool)).await;
//! }
//! # Ok(())
//! # }
//! ```

mod client;
mod bridge;
mod registry;
mod error;
mod transport;

pub use client::McpClient;
pub use bridge::McpToolBridge;
pub use registry::McpRegistry;
pub use error::McpError;
pub use transport::{StdioTransport, Transport};
