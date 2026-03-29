use std::sync::Arc;

use tokio::sync::broadcast;
use tracing::warn;

use crate::event::MonitorEvent;
use crate::filter::MonitorFilter;
use crate::store::{MonitorStore, MonitorStoreError, RetentionPolicy};

/// Central hub for monitoring events.
///
/// Events are both persisted to SQLite and broadcast to real-time subscribers.
#[derive(Clone)]
pub struct MonitorSink {
    tx: broadcast::Sender<MonitorEvent>,
    store: Arc<MonitorStore>,
}

impl MonitorSink {
    /// Create a new sink with SQLite persistence at the given path.
    pub fn new(db_path: &str) -> Result<Self, MonitorStoreError> {
        let store = Arc::new(MonitorStore::new(db_path)?);
        let (tx, _) = broadcast::channel(1024);
        Ok(Self { tx, store })
    }

    /// Create a sink with in-memory SQLite (for testing).
    pub fn in_memory() -> Result<Self, MonitorStoreError> {
        let store = Arc::new(MonitorStore::in_memory()?);
        let (tx, _) = broadcast::channel(1024);
        Ok(Self { tx, store })
    }

    /// Emit a monitor event — stores to SQLite and broadcasts to subscribers.
    pub fn emit(&self, event: MonitorEvent) {
        // Store persistently (sync, but fast for SQLite)
        if let Err(e) = self.store.insert(&event) {
            warn!(error = %e, "failed to store monitor event");
        }

        // Broadcast to real-time consumers (ignore if no receivers)
        let _ = self.tx.send(event);
    }

    /// Subscribe to real-time event stream.
    pub fn subscribe(&self) -> broadcast::Receiver<MonitorEvent> {
        self.tx.subscribe()
    }

    /// Query persisted events.
    pub fn query(&self, filter: &MonitorFilter) -> Result<Vec<MonitorEvent>, MonitorStoreError> {
        self.store.query(filter)
    }

    /// Count events matching a filter.
    pub fn count(&self, filter: &MonitorFilter) -> Result<usize, MonitorStoreError> {
        self.store.count(filter)
    }

    /// Access the underlying store (for retention config, etc.).
    pub fn store(&self) -> &Arc<MonitorStore> {
        &self.store
    }

    /// Start background retention cleanup.
    pub fn start_retention(&self, policy: RetentionPolicy) {
        self.store.start_retention(policy);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{MonitorEvent, MonitorEventKind};
    use crate::filter::MonitorFilter;

    fn make_event(provider: &str) -> MonitorEvent {
        MonitorEvent::new(MonitorEventKind::ProviderRequest {
            provider: provider.into(),
            system_prompt: None,
            message_count: 1,
            tool_count: 0,
            messages: vec![],
            tools: vec![],
        })
    }

    #[tokio::test]
    async fn emit_stores_and_broadcasts() {
        let sink = MonitorSink::in_memory().unwrap();
        let mut rx = sink.subscribe();

        let event = make_event("anthropic");
        let event_id = event.id.clone();
        sink.emit(event);

        // Verify broadcast received
        let received = rx.recv().await.unwrap();
        assert_eq!(received.id, event_id);

        // Verify persisted and queryable
        let results = sink.query(&MonitorFilter::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, event_id);
    }

    #[tokio::test]
    async fn query_through_sink() {
        let sink = MonitorSink::in_memory().unwrap();

        sink.emit(make_event("anthropic"));
        sink.emit(make_event("openai"));
        sink.emit(make_event("anthropic"));

        // Query all
        let all = sink.query(&MonitorFilter::default()).unwrap();
        assert_eq!(all.len(), 3);

        // Count all
        let count = sink.count(&MonitorFilter::default()).unwrap();
        assert_eq!(count, 3);

        // Filter by provider
        let filter = MonitorFilter {
            provider: Some("openai".into()),
            ..Default::default()
        };
        let filtered = sink.query(&filter).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].provider_name(), Some("openai"));
    }
}
