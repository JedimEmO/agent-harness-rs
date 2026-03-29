//! Bidirectional live streaming traits and types for real-time AI sessions.
//!
//! This module provides the [`LiveProvider`] trait for providers that support
//! persistent, bidirectional connections (e.g., Gemini Live API over WebSocket).
//! Unlike [`AiProvider`](crate::AiProvider) which is request-response, live
//! sessions allow concurrent audio/text input and output with mid-stream tool calls.

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::AiError;
use crate::provider::{ConversationMessage, ToolCall, ToolDefinition, ToolResult};

/// Events flowing from the live provider to the application.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiveServerEvent {
    /// Audio data from the model (e.g., PCM at 24kHz).
    AudioDelta { data: Vec<u8>, mime_type: String },
    /// Text transcript of the model's speech.
    TranscriptDelta(String),
    /// Transcript of the user's speech (from provider-side STT).
    InputTranscript(String),
    /// Model wants to call tools.
    ToolCalls(Vec<ToolCall>),
    /// Model finished its current turn.
    TurnComplete,
    /// User interrupted the model via barge-in; ongoing generation was cancelled.
    Interrupted,
    /// Server is shutting down soon — reconnect with a resume token.
    GoingAway {
        time_left: Option<std::time::Duration>,
    },
    /// Session setup confirmed by the provider.
    SetupComplete,
    /// Token usage update.
    Usage {
        input_tokens: u32,
        output_tokens: u32,
    },
    /// Updated session resumption handle (store for reconnection).
    SessionResumeToken(String),
    /// Session closed.
    Closed,
    /// Error from provider.
    Error(String),
}

/// Events flowing from the application to the live provider.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LiveClientEvent {
    /// Audio chunk from the user's microphone (e.g., PCM at 16kHz).
    AudioChunk { data: Vec<u8>, mime_type: String },
    /// Text message from the user.
    TextMessage(String),
    /// Image or video frame (e.g., JPEG at ≤1fps).
    VideoFrame { data: Vec<u8>, mime_type: String },
    /// Tool execution results sent back to the model.
    ToolResults(Vec<ToolResult>),
    /// Interrupt the model's current response.
    Interrupt,
    /// End the session.
    Close,
}

/// Voice Activity Detection configuration.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VadConfig {
    pub enabled: bool,
    /// e.g., "START_SENSITIVITY_LOW", "START_SENSITIVITY_HIGH"
    pub start_sensitivity: Option<String>,
    /// e.g., "END_SENSITIVITY_LOW", "END_SENSITIVITY_HIGH"
    pub end_sensitivity: Option<String>,
    pub prefix_padding_ms: Option<u32>,
    pub silence_duration_ms: Option<u32>,
}

/// Thinking/reasoning configuration for live sessions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveThinkingConfig {
    /// Thinking level: "minimal", "low", "medium", "high".
    pub level: String,
    /// Whether to include thought summaries in responses.
    pub include_thoughts: bool,
}

/// Configuration for a live bidirectional session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveSessionConfig {
    /// System instruction for the session.
    pub system_prompt: Option<String>,
    /// Tools available to the model during the session.
    pub tools: Vec<ToolDefinition>,
    /// Voice name for TTS (provider-specific, e.g., "Kore").
    pub voice: Option<String>,
    /// Response modalities: e.g., \["AUDIO"\], \["TEXT"\].
    pub response_modalities: Vec<String>,
    /// MIME type for input audio (e.g., "audio/pcm;rate=16000").
    pub input_audio_mime: String,
    /// MIME type for output audio (e.g., "audio/pcm;rate=24000").
    pub output_audio_mime: String,
    /// Voice Activity Detection settings.
    pub vad: Option<VadConfig>,
    /// Enable transcription of user's speech.
    pub enable_input_transcription: bool,
    /// Enable transcription of model's speech.
    pub enable_output_transcription: bool,
    /// Thinking/reasoning configuration.
    pub thinking: Option<LiveThinkingConfig>,
    /// Enable context window compression for unlimited session duration.
    pub context_window_compression: bool,
    /// Session resumption handle from a previous session.
    pub resume_handle: Option<String>,
    /// Initial conversation history to seed the session.
    pub initial_history: Vec<ConversationMessage>,
}

impl Default for LiveSessionConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            tools: Vec::new(),
            voice: None,
            response_modalities: vec!["AUDIO".to_string()],
            input_audio_mime: "audio/pcm;rate=16000".to_string(),
            output_audio_mime: "audio/pcm;rate=24000".to_string(),
            vad: None,
            enable_input_transcription: false,
            enable_output_transcription: false,
            thinking: None,
            context_window_compression: false,
            resume_handle: None,
            initial_history: Vec::new(),
        }
    }
}

/// A live bidirectional session handle.
///
/// Send [`LiveClientEvent`]s on `tx` and receive [`LiveServerEvent`]s from `rx`.
/// The session runs in a background task that bridges the WebSocket connection.
pub struct LiveSession {
    /// Send events to the live provider (audio chunks, text, tool results, etc.).
    pub tx: mpsc::Sender<LiveClientEvent>,
    /// Receive events from the live provider (audio, transcripts, tool calls, etc.).
    pub rx: mpsc::Receiver<LiveServerEvent>,
}

/// Trait for AI providers that support bidirectional live sessions.
///
/// Implementors open a persistent connection (typically WebSocket) and return
/// a [`LiveSession`] with channels for bidirectional communication.
#[async_trait::async_trait]
pub trait LiveProvider: Send + Sync {
    /// Open a live session with the given configuration.
    async fn connect_live(&self, config: LiveSessionConfig) -> Result<LiveSession, AiError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn live_session_config_default() {
        let config = LiveSessionConfig::default();
        assert_eq!(config.response_modalities, vec!["AUDIO".to_string()]);
        assert_eq!(config.input_audio_mime, "audio/pcm;rate=16000");
        assert_eq!(config.output_audio_mime, "audio/pcm;rate=24000");
        assert!(!config.enable_input_transcription);
        assert!(!config.enable_output_transcription);
        assert!(!config.context_window_compression);
        assert!(config.system_prompt.is_none());
        assert!(config.tools.is_empty());
        assert!(config.voice.is_none());
        assert!(config.vad.is_none());
        assert!(config.thinking.is_none());
        assert!(config.resume_handle.is_none());
        assert!(config.initial_history.is_empty());
    }

    #[test]
    fn live_server_event_serde_roundtrip() {
        let events = vec![
            LiveServerEvent::AudioDelta { data: vec![1, 2, 3], mime_type: "audio/pcm;rate=24000".into() },
            LiveServerEvent::TranscriptDelta("hello".into()),
            LiveServerEvent::ToolCalls(vec![ToolCall { id: "t1".into(), name: "test".into(), arguments: serde_json::json!({}) }]),
            LiveServerEvent::TurnComplete,
            LiveServerEvent::GoingAway { time_left: None },
            LiveServerEvent::SetupComplete,
            LiveServerEvent::Usage { input_tokens: 10, output_tokens: 20 },
            LiveServerEvent::InputTranscript("hi there".into()),
            LiveServerEvent::Interrupted,
            LiveServerEvent::SessionResumeToken("token123".into()),
            LiveServerEvent::Closed,
            LiveServerEvent::Error("bad thing".into()),
        ];
        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: LiveServerEvent = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2, "roundtrip failed for: {:?}", event);
        }
    }

    #[test]
    fn live_client_event_serde_roundtrip() {
        let events = vec![
            LiveClientEvent::AudioChunk { data: vec![4, 5, 6], mime_type: "audio/pcm;rate=16000".into() },
            LiveClientEvent::TextMessage("hello".into()),
            LiveClientEvent::ToolResults(vec![ToolResult { call_id: "c1".into(), content: serde_json::json!("done") }]),
            LiveClientEvent::Close,
            LiveClientEvent::VideoFrame { data: vec![0xFF, 0xD8], mime_type: "image/jpeg".into() },
            LiveClientEvent::Interrupt,
        ];
        for event in &events {
            let json = serde_json::to_string(event).unwrap();
            let deserialized: LiveClientEvent = serde_json::from_str(&json).unwrap();
            let json2 = serde_json::to_string(&deserialized).unwrap();
            assert_eq!(json, json2, "roundtrip failed for: {:?}", event);
        }
    }

    #[test]
    fn vad_config_default() {
        let vad = VadConfig::default();
        assert!(!vad.enabled);
        assert!(vad.start_sensitivity.is_none());
        assert!(vad.end_sensitivity.is_none());
        assert!(vad.prefix_padding_ms.is_none());
        assert!(vad.silence_duration_ms.is_none());
    }
}
