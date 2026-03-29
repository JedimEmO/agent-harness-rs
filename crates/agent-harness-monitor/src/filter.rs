use serde::{Deserialize, Serialize};

/// Event kind tag for filtering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EventKindTag {
    ProviderRequest,
    ProviderStream,
    ProviderComplete,
    ProviderError,
    McpRequest,
    McpResponse,
    LiveClient,
    LiveServer,
    Agent,
}

impl EventKindTag {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::ProviderRequest => "provider_request",
            Self::ProviderStream => "provider_stream",
            Self::ProviderComplete => "provider_complete",
            Self::ProviderError => "provider_error",
            Self::McpRequest => "mcp_request",
            Self::McpResponse => "mcp_response",
            Self::LiveClient => "live_client",
            Self::LiveServer => "live_server",
            Self::Agent => "agent",
        }
    }
}

impl std::fmt::Display for EventKindTag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Direction of a monitored event.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Request,
    Response,
    Stream,
}

impl Direction {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Request => "request",
            Self::Response => "response",
            Self::Stream => "stream",
        }
    }
}

impl std::fmt::Display for Direction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Filter criteria for querying monitor events.
///
/// All fields are optional — unset fields don't constrain the query.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitorFilter {
    /// Filter by event kind tags.
    pub kinds: Option<Vec<EventKindTag>>,
    /// Filter by provider/server name.
    pub provider: Option<String>,
    /// Filter by session ID.
    pub session_id: Option<String>,
    /// Full-text search on the JSON content column.
    pub text_search: Option<String>,
    /// Events after this timestamp (RFC3339).
    pub after: Option<String>,
    /// Events before this timestamp (RFC3339).
    pub before: Option<String>,
    /// Filter by span ID (find all events in a request-response span).
    pub span_id: Option<String>,
    /// Filter by direction.
    pub direction: Option<Direction>,
    /// Maximum number of results (default: 100).
    pub limit: Option<usize>,
    /// Offset for pagination.
    pub offset: Option<usize>,
}

impl MonitorFilter {
    pub fn effective_limit(&self) -> i64 {
        self.limit.unwrap_or(100) as i64
    }

    pub fn effective_offset(&self) -> i64 {
        self.offset.unwrap_or(0) as i64
    }
}
