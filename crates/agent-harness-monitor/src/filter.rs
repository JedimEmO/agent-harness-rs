use serde::{Deserialize, Serialize};

/// Filter criteria for querying monitor events.
///
/// All fields are optional — unset fields don't constrain the query.
/// Filters translate directly to SQL WHERE clauses.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MonitorFilter {
    /// Filter by event kind tags (e.g., "provider_request", "mcp_response").
    pub kinds: Option<Vec<String>>,
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
    /// Filter by direction ("request", "response", "stream").
    pub direction: Option<String>,
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
