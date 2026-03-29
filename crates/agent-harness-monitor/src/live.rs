use async_trait::async_trait;
use tokio::sync::mpsc;

use agent_harness_core::*;

use crate::event::{MonitorEvent, MonitorEventKind};
use crate::sink::MonitorSink;

/// Wraps any [`LiveProvider`] to capture bidirectional traffic to the monitor.
pub struct MonitoredLiveProvider<P: LiveProvider> {
    inner: P,
    sink: MonitorSink,
}

impl<P: LiveProvider> MonitoredLiveProvider<P> {
    pub fn new(inner: P, sink: MonitorSink) -> Self {
        Self { inner, sink }
    }
}

#[async_trait]
impl<P: LiveProvider> LiveProvider for MonitoredLiveProvider<P> {
    async fn connect_live(
        &self,
        config: LiveSessionConfig,
    ) -> Result<LiveSession, AiError> {
        let session = self.inner.connect_live(config).await?;

        // Create intercepting channels
        let (client_tx, mut client_rx_inner) = mpsc::channel::<LiveClientEvent>(256);
        let (server_tx_inner, server_rx) = mpsc::channel::<LiveServerEvent>(256);

        let original_tx = session.tx;
        let mut original_rx = session.rx;

        // Relay client events: app → monitor → provider
        let sink_out = self.sink.clone();
        tokio::spawn(async move {
            while let Some(event) = client_rx_inner.recv().await {
                sink_out.emit(MonitorEvent::new(MonitorEventKind::LiveClientMessage {
                    event: event.clone(),
                }));
                if original_tx.send(event).await.is_err() {
                    break;
                }
            }
        });

        // Relay server events: provider → monitor → app
        let sink_in = self.sink.clone();
        tokio::spawn(async move {
            while let Some(event) = original_rx.recv().await {
                sink_in.emit(MonitorEvent::new(MonitorEventKind::LiveServerMessage {
                    event: event.clone(),
                }));
                if server_tx_inner.send(event).await.is_err() {
                    break;
                }
            }
        });

        Ok(LiveSession {
            tx: client_tx,
            rx: server_rx,
        })
    }
}
