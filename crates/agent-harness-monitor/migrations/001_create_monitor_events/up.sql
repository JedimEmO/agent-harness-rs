CREATE TABLE monitor_events (
    id TEXT PRIMARY KEY NOT NULL,
    timestamp TEXT NOT NULL,
    span_id TEXT,
    session_id TEXT,
    kind TEXT NOT NULL,
    provider TEXT,
    direction TEXT,
    content TEXT NOT NULL,
    duration_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER
);

CREATE INDEX idx_monitor_timestamp ON monitor_events(timestamp);
CREATE INDEX idx_monitor_kind ON monitor_events(kind);
CREATE INDEX idx_monitor_span ON monitor_events(span_id);
CREATE INDEX idx_monitor_session ON monitor_events(session_id);
CREATE INDEX idx_monitor_provider ON monitor_events(provider);
