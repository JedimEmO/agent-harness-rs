//! # agent-harness-monitor
//!
//! Observability and monitoring for the agent-harness framework.
//!
//! Captures all LLM provider requests/responses, MCP JSON-RPC traffic, and live
//! session events to a SQLite database with real-time streaming via broadcast channels.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use agent_harness_monitor::{MonitorSink, MonitoredProvider};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let sink = MonitorSink::new("data/monitor.db")?;
//! // Wrap any AiProvider:
//! // let monitored = MonitoredProvider::new(my_provider, sink.clone());
//! # Ok(())
//! # }
//! ```

mod event;
mod filter;
mod sink;
mod store;
mod provider;
mod mcp;
mod live;

pub use event::{MonitorEvent, MonitorEventKind};
pub use filter::{Direction, EventKindTag, MonitorFilter};
pub use sink::MonitorSink;
pub use store::{MonitorStore, MonitorStoreError, RetentionPolicy};
pub use provider::MonitoredProvider;
pub use mcp::MonitoredMcpTransport;
pub use live::MonitoredLiveProvider;
