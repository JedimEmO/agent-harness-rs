use std::sync::Arc;
use std::time::Duration;

use diesel::prelude::*;
use diesel::r2d2::{ConnectionManager, Pool};
use diesel_migrations::{embed_migrations, EmbeddedMigrations, MigrationHarness};
use tracing::{debug, warn};

use crate::event::{MonitorEvent, MonitorEventKind};
use crate::filter::MonitorFilter;

pub type MonitorPool = Pool<ConnectionManager<SqliteConnection>>;

const MIGRATIONS: EmbeddedMigrations = embed_migrations!("migrations");

diesel::table! {
    monitor_events (id) {
        id -> Text,
        timestamp -> Text,
        span_id -> Nullable<Text>,
        session_id -> Nullable<Text>,
        kind -> Text,
        provider -> Nullable<Text>,
        direction -> Nullable<Text>,
        content -> Text,
        duration_ms -> Nullable<Integer>,
        input_tokens -> Nullable<Integer>,
        output_tokens -> Nullable<Integer>,
    }
}

#[allow(dead_code)] // Fields are populated by Diesel queries
#[derive(Queryable, Selectable, Clone)]
#[diesel(table_name = monitor_events)]
struct EventRow {
    id: String,
    timestamp: String,
    span_id: Option<String>,
    session_id: Option<String>,
    kind: String,
    provider: Option<String>,
    direction: Option<String>,
    content: String,
    duration_ms: Option<i32>,
    input_tokens: Option<i32>,
    output_tokens: Option<i32>,
}

/// Retention policy for automatic cleanup of old events.
#[derive(Debug, Clone)]
pub struct RetentionPolicy {
    pub max_age: Option<Duration>,
    pub max_events: Option<usize>,
    pub cleanup_interval: Duration,
}

impl Default for RetentionPolicy {
    fn default() -> Self {
        Self {
            max_age: None,
            max_events: None,
            cleanup_interval: Duration::from_secs(300),
        }
    }
}

/// SQLite-backed monitor event store.
pub struct MonitorStore {
    pool: MonitorPool,
}

impl MonitorStore {
    /// Create a new store, running migrations.
    pub fn new(db_path: &str) -> Result<Self, MonitorStoreError> {
        if let Some(parent) = std::path::Path::new(db_path).parent() {
            std::fs::create_dir_all(parent).ok();
        }

        let manager = ConnectionManager::<SqliteConnection>::new(db_path);
        let pool = Pool::builder()
            .build(manager)
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;

        let mut conn = pool
            .get()
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;
        conn.run_pending_migrations(MIGRATIONS)
            .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        Ok(Self { pool })
    }

    /// Create an in-memory store (for testing).
    pub fn in_memory() -> Result<Self, MonitorStoreError> {
        Self::new(":memory:")
    }

    /// Insert a monitor event.
    pub fn insert(&self, event: &MonitorEvent) -> Result<(), MonitorStoreError> {
        let content =
            serde_json::to_string(&event.kind).map_err(|e| MonitorStoreError::Query(e.to_string()))?;
        let (input_tok, output_tok) = event.tokens();

        let mut conn = self
            .pool
            .get()
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;

        diesel::insert_into(monitor_events::table)
            .values((
                monitor_events::id.eq(&event.id),
                monitor_events::timestamp.eq(&event.timestamp),
                monitor_events::span_id.eq(&event.span_id),
                monitor_events::session_id.eq(&event.session_id),
                monitor_events::kind.eq(event.kind_tag()),
                monitor_events::provider.eq(event.provider_name()),
                monitor_events::direction.eq(event.direction()),
                monitor_events::content.eq(&content),
                monitor_events::duration_ms.eq(event.duration_ms().map(|d| d as i32)),
                monitor_events::input_tokens.eq(input_tok.map(|t| t as i32)),
                monitor_events::output_tokens.eq(output_tok.map(|t| t as i32)),
            ))
            .execute(&mut conn)
            .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        Ok(())
    }

    /// Query events matching a filter.
    pub fn query(&self, filter: &MonitorFilter) -> Result<Vec<MonitorEvent>, MonitorStoreError> {
        let mut conn = self
            .pool
            .get()
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;

        let query = apply_filters(monitor_events::table.into_boxed(), filter);

        let rows: Vec<EventRow> = query
            .order(monitor_events::timestamp.desc())
            .limit(filter.effective_limit())
            .offset(filter.effective_offset())
            .select(EventRow::as_select())
            .load(&mut conn)
            .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        rows.into_iter().map(row_to_event).collect()
    }

    /// Count events matching a filter (applies all the same filters as `query()`).
    pub fn count(&self, filter: &MonitorFilter) -> Result<usize, MonitorStoreError> {
        let mut conn = self
            .pool
            .get()
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;

        let query = apply_filters(monitor_events::table.into_boxed(), filter);

        let count: i64 = query
            .count()
            .get_result(&mut conn)
            .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        Ok(count as usize)
    }

    /// Delete events older than the given timestamp.
    pub fn cleanup_before(&self, before: &str) -> Result<usize, MonitorStoreError> {
        let mut conn = self
            .pool
            .get()
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;

        let deleted = diesel::delete(
            monitor_events::table.filter(monitor_events::timestamp.lt(before)),
        )
        .execute(&mut conn)
        .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        if deleted > 0 {
            debug!(deleted, "cleaned up old monitor events");
        }
        Ok(deleted)
    }

    /// Delete events beyond the max count, keeping the newest.
    pub fn cleanup_excess(&self, max_events: usize) -> Result<usize, MonitorStoreError> {
        let mut conn = self
            .pool
            .get()
            .map_err(|e| MonitorStoreError::Connection(e.to_string()))?;

        let total: i64 = monitor_events::table
            .count()
            .get_result(&mut conn)
            .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        if (total as usize) <= max_events {
            return Ok(0);
        }

        let to_delete = total as usize - max_events;

        let deleted = diesel::sql_query(format!(
            "DELETE FROM monitor_events WHERE id IN (SELECT id FROM monitor_events ORDER BY timestamp ASC LIMIT {})",
            to_delete
        ))
        .execute(&mut conn)
        .map_err(|e| MonitorStoreError::Query(e.to_string()))?;

        if deleted > 0 {
            debug!(deleted, max_events, "cleaned up excess monitor events");
        }
        Ok(deleted)
    }

    /// Start a background retention cleanup task.
    pub fn start_retention(self: &Arc<Self>, policy: RetentionPolicy) {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(policy.cleanup_interval);
            loop {
                interval.tick().await;

                if let Some(max_age) = policy.max_age {
                    let cutoff = (chrono::Utc::now() - max_age).to_rfc3339();
                    if let Err(e) = store.cleanup_before(&cutoff) {
                        warn!(error = %e, "monitor retention cleanup (age) failed");
                    }
                }

                if let Some(max_events) = policy.max_events {
                    if let Err(e) = store.cleanup_excess(max_events) {
                        warn!(error = %e, "monitor retention cleanup (count) failed");
                    }
                }
            }
        });
    }
}

fn apply_filters<'a>(
    mut query: monitor_events::BoxedQuery<'a, diesel::sqlite::Sqlite>,
    filter: &'a MonitorFilter,
) -> monitor_events::BoxedQuery<'a, diesel::sqlite::Sqlite> {
    if let Some(ref kinds) = filter.kinds {
        if !kinds.is_empty() {
            let kind_strs: Vec<&str> = kinds.iter().map(|k| k.as_str()).collect();
            query = query.filter(monitor_events::kind.eq_any(kind_strs));
        }
    }
    if let Some(ref provider) = filter.provider {
        query = query.filter(monitor_events::provider.eq(provider));
    }
    if let Some(ref session_id) = filter.session_id {
        query = query.filter(monitor_events::session_id.eq(session_id));
    }
    if let Some(ref span_id) = filter.span_id {
        query = query.filter(monitor_events::span_id.eq(span_id));
    }
    if let Some(ref direction) = filter.direction {
        query = query.filter(monitor_events::direction.eq(direction.as_str()));
    }
    if let Some(ref after) = filter.after {
        query = query.filter(monitor_events::timestamp.gt(after));
    }
    if let Some(ref before) = filter.before {
        query = query.filter(monitor_events::timestamp.lt(before));
    }
    if let Some(ref text) = filter.text_search {
        let pattern = format!("%{}%", text);
        query = query.filter(monitor_events::content.like(pattern));
    }
    query
}

fn row_to_event(row: EventRow) -> Result<MonitorEvent, MonitorStoreError> {
    let kind: MonitorEventKind =
        serde_json::from_str(&row.content).map_err(|e| MonitorStoreError::Query(e.to_string()))?;

    Ok(MonitorEvent {
        id: row.id,
        timestamp: row.timestamp,
        span_id: row.span_id,
        session_id: row.session_id,
        kind,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum MonitorStoreError {
    #[error("connection error: {0}")]
    Connection(String),
    #[error("query error: {0}")]
    Query(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{MonitorEvent, MonitorEventKind};
    use crate::filter::{EventKindTag, MonitorFilter};

    fn make_provider_request_event(provider: &str) -> MonitorEvent {
        MonitorEvent::new(MonitorEventKind::ProviderRequest {
            provider: provider.into(),
            system_prompt: Some("You are helpful".into()),
            message_count: 1,
            tool_count: 0,
            messages: vec![],
            tools: vec![],
        })
    }

    fn make_provider_complete_event(provider: &str) -> MonitorEvent {
        MonitorEvent::new(MonitorEventKind::ProviderComplete {
            provider: provider.into(),
            response: agent_harness_core::ConversationResponse::Text("hello".into()),
            duration_ms: 42,
            input_tokens: Some(10),
            output_tokens: Some(20),
        })
    }

    fn make_mcp_request_event(server: &str) -> MonitorEvent {
        MonitorEvent::new(MonitorEventKind::McpRequest {
            server: server.into(),
            method: "tools/list".into(),
            params: serde_json::Value::Null,
            rpc_id: 1,
        })
    }

    #[test]
    fn insert_and_query_basic() {
        let store = MonitorStore::in_memory().unwrap();
        let event = make_provider_request_event("anthropic");

        store.insert(&event).unwrap();

        let results = store.query(&MonitorFilter::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, event.id);
        assert_eq!(results[0].kind_tag(), "provider_request");
        assert_eq!(results[0].provider_name(), Some("anthropic"));
    }

    #[test]
    fn query_filter_by_kind() {
        let store = MonitorStore::in_memory().unwrap();

        store.insert(&make_provider_request_event("anthropic")).unwrap();
        store.insert(&make_provider_complete_event("anthropic")).unwrap();
        store.insert(&make_mcp_request_event("filesystem")).unwrap();

        let filter = MonitorFilter {
            kinds: Some(vec![EventKindTag::ProviderRequest]),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind_tag(), "provider_request");

        // Test multiple kinds
        let filter = MonitorFilter {
            kinds: Some(vec![EventKindTag::ProviderRequest, EventKindTag::McpRequest]),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn query_filter_by_provider() {
        let store = MonitorStore::in_memory().unwrap();

        store.insert(&make_provider_request_event("anthropic")).unwrap();
        store.insert(&make_provider_request_event("openai")).unwrap();
        store.insert(&make_provider_request_event("anthropic")).unwrap();

        let filter = MonitorFilter {
            provider: Some("openai".into()),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].provider_name(), Some("openai"));
    }

    #[test]
    fn query_filter_by_span_id() {
        let store = MonitorStore::in_memory().unwrap();

        let span = "span-abc-123";
        let req = make_provider_request_event("anthropic").with_span(span);
        let resp = make_provider_complete_event("anthropic").with_span(span);
        let other = make_provider_request_event("anthropic").with_span("other-span");

        store.insert(&req).unwrap();
        store.insert(&resp).unwrap();
        store.insert(&other).unwrap();

        let filter = MonitorFilter {
            span_id: Some(span.into()),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.span_id.as_deref(), Some(span));
        }
    }

    #[test]
    fn query_filter_by_session_id() {
        let store = MonitorStore::in_memory().unwrap();

        let e1 = make_provider_request_event("anthropic").with_session("session-1");
        let e2 = make_provider_request_event("anthropic").with_session("session-2");
        let e3 = make_provider_request_event("anthropic").with_session("session-1");

        store.insert(&e1).unwrap();
        store.insert(&e2).unwrap();
        store.insert(&e3).unwrap();

        let filter = MonitorFilter {
            session_id: Some("session-1".into()),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 2);
        for r in &results {
            assert_eq!(r.session_id.as_deref(), Some("session-1"));
        }
    }

    #[test]
    fn query_text_search() {
        let store = MonitorStore::in_memory().unwrap();

        store.insert(&make_provider_request_event("anthropic")).unwrap();
        store.insert(&make_mcp_request_event("filesystem")).unwrap();

        // Search for "tools/list" which appears in the MCP request content JSON
        let filter = MonitorFilter {
            text_search: Some("tools/list".into()),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind_tag(), "mcp_request");

        // Search for something that appears in the provider request
        let filter = MonitorFilter {
            text_search: Some("You are helpful".into()),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].kind_tag(), "provider_request");
    }

    #[test]
    fn query_pagination() {
        let store = MonitorStore::in_memory().unwrap();

        // Insert 10 events with deterministic timestamps so ordering is predictable
        for i in 0..10 {
            let mut event = make_provider_request_event("anthropic");
            // Use a timestamp that sorts in ascending order
            event.timestamp = format!("2025-01-01T00:00:{:02}Z", i);
            store.insert(&event).unwrap();
        }

        // Query ordered desc by timestamp, limit=3, offset=2
        let filter = MonitorFilter {
            limit: Some(3),
            offset: Some(2),
            ..Default::default()
        };
        let results = store.query(&filter).unwrap();
        assert_eq!(results.len(), 3);
        // Results are desc, so full order is :09, :08, :07, :06, :05, :04, :03, :02, :01, :00
        // offset=2, limit=3 should give :07, :06, :05
        assert!(results[0].timestamp.contains(":07"));
        assert!(results[1].timestamp.contains(":06"));
        assert!(results[2].timestamp.contains(":05"));
    }

    #[test]
    fn count_events() {
        let store = MonitorStore::in_memory().unwrap();

        for _ in 0..5 {
            store.insert(&make_provider_request_event("anthropic")).unwrap();
        }

        let count = store.count(&MonitorFilter::default()).unwrap();
        assert_eq!(count, 5);

        // Count with a filter
        store.insert(&make_mcp_request_event("filesystem")).unwrap();
        let filter = MonitorFilter {
            kinds: Some(vec![EventKindTag::ProviderRequest]),
            ..Default::default()
        };
        let count = store.count(&filter).unwrap();
        assert_eq!(count, 5);
    }

    #[test]
    fn cleanup_before() {
        let store = MonitorStore::in_memory().unwrap();

        // Insert old events
        for i in 0..3 {
            let mut event = make_provider_request_event("anthropic");
            event.timestamp = format!("2024-01-01T00:00:{:02}Z", i);
            store.insert(&event).unwrap();
        }
        // Insert new events
        for i in 0..2 {
            let mut event = make_provider_request_event("anthropic");
            event.timestamp = format!("2025-06-01T00:00:{:02}Z", i);
            store.insert(&event).unwrap();
        }

        let deleted = store.cleanup_before("2025-01-01T00:00:00Z").unwrap();
        assert_eq!(deleted, 3);

        let remaining = store.count(&MonitorFilter::default()).unwrap();
        assert_eq!(remaining, 2);
    }

    #[test]
    fn cleanup_excess() {
        let store = MonitorStore::in_memory().unwrap();

        for i in 0..10 {
            let mut event = make_provider_request_event("anthropic");
            event.timestamp = format!("2025-01-01T00:00:{:02}Z", i);
            store.insert(&event).unwrap();
        }

        let deleted = store.cleanup_excess(5).unwrap();
        assert_eq!(deleted, 5);

        let remaining = store.count(&MonitorFilter::default()).unwrap();
        assert_eq!(remaining, 5);

        // Verify the newest events survived (the ones with higher timestamps)
        let results = store.query(&MonitorFilter::default()).unwrap();
        for r in &results {
            // Should only have timestamps :05 through :09
            let ts = &r.timestamp;
            assert!(ts.as_str() >= "2025-01-01T00:00:05Z", "unexpected old event survived: {}", ts);
        }

        // Calling again with same max should delete nothing
        let deleted = store.cleanup_excess(5).unwrap();
        assert_eq!(deleted, 0);
    }
}
