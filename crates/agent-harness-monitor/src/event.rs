use serde::{Deserialize, Serialize};

use agent_harness_core::{
    AgentEvent, ConversationMessage, ConversationResponse, LiveClientEvent, LiveServerEvent,
    StreamEvent, ToolDefinition,
};

/// A captured monitoring event with metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorEvent {
    pub id: String,
    pub timestamp: String,
    pub span_id: Option<String>,
    pub session_id: Option<String>,
    pub kind: MonitorEventKind,
}

impl MonitorEvent {
    pub fn new(kind: MonitorEventKind) -> Self {
        Self {
            id: uuid::Uuid::new_v4().to_string(),
            timestamp: chrono::Utc::now().to_rfc3339(),
            span_id: None,
            session_id: None,
            kind,
        }
    }

    pub fn with_span(mut self, span_id: impl Into<String>) -> Self {
        self.span_id = Some(span_id.into());
        self
    }

    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// The kind tag as a string for indexing.
    pub fn kind_tag(&self) -> &'static str {
        self.kind.tag()
    }

    /// The provider name, if applicable.
    pub fn provider_name(&self) -> Option<&str> {
        self.kind.provider()
    }

    /// The direction: "request", "response", "stream", or None.
    pub fn direction(&self) -> Option<&'static str> {
        self.kind.direction()
    }

    /// Duration in ms, if applicable.
    pub fn duration_ms(&self) -> Option<u64> {
        self.kind.duration_ms()
    }

    /// Token counts, if applicable.
    pub fn tokens(&self) -> (Option<u32>, Option<u32>) {
        self.kind.tokens()
    }
}

/// The payload of a monitoring event.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum MonitorEventKind {
    // --- LLM Provider Traffic ---
    ProviderRequest {
        provider: String,
        system_prompt: Option<String>,
        message_count: usize,
        tool_count: usize,
        messages: Vec<ConversationMessage>,
        tools: Vec<ToolDefinition>,
    },
    ProviderStreamEvent {
        provider: String,
        event: StreamEvent,
    },
    ProviderComplete {
        provider: String,
        response: ConversationResponse,
        duration_ms: u64,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },
    ProviderError {
        provider: String,
        error: String,
    },

    // --- MCP JSON-RPC Traffic ---
    McpRequest {
        server: String,
        method: String,
        params: serde_json::Value,
        rpc_id: i64,
    },
    McpResponse {
        server: String,
        rpc_id: i64,
        result: Option<serde_json::Value>,
        error: Option<String>,
        duration_ms: u64,
    },

    // --- Live Session Traffic ---
    LiveClientMessage {
        event: LiveClientEvent,
    },
    LiveServerMessage {
        event: LiveServerEvent,
    },

    // --- Agent Events (forwarded) ---
    Agent(AgentEvent),
}

impl MonitorEventKind {
    pub fn tag(&self) -> &'static str {
        match self {
            Self::ProviderRequest { .. } => "provider_request",
            Self::ProviderStreamEvent { .. } => "provider_stream",
            Self::ProviderComplete { .. } => "provider_complete",
            Self::ProviderError { .. } => "provider_error",
            Self::McpRequest { .. } => "mcp_request",
            Self::McpResponse { .. } => "mcp_response",
            Self::LiveClientMessage { .. } => "live_client",
            Self::LiveServerMessage { .. } => "live_server",
            Self::Agent(_) => "agent",
        }
    }

    pub fn provider(&self) -> Option<&str> {
        match self {
            Self::ProviderRequest { provider, .. }
            | Self::ProviderStreamEvent { provider, .. }
            | Self::ProviderComplete { provider, .. }
            | Self::ProviderError { provider, .. } => Some(provider),
            Self::McpRequest { server, .. } | Self::McpResponse { server, .. } => Some(server),
            _ => None,
        }
    }

    pub fn direction(&self) -> Option<&'static str> {
        match self {
            Self::ProviderRequest { .. } | Self::McpRequest { .. } | Self::LiveClientMessage { .. } => {
                Some("request")
            }
            Self::ProviderComplete { .. } | Self::McpResponse { .. } | Self::LiveServerMessage { .. } => {
                Some("response")
            }
            Self::ProviderStreamEvent { .. } => Some("stream"),
            _ => None,
        }
    }

    pub fn duration_ms(&self) -> Option<u64> {
        match self {
            Self::ProviderComplete { duration_ms, .. } => Some(*duration_ms),
            Self::McpResponse { duration_ms, .. } => Some(*duration_ms),
            _ => None,
        }
    }

    pub fn tokens(&self) -> (Option<u32>, Option<u32>) {
        match self {
            Self::ProviderComplete {
                input_tokens,
                output_tokens,
                ..
            } => (*input_tokens, *output_tokens),
            _ => (None, None),
        }
    }
}
