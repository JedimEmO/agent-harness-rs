use async_trait::async_trait;
use base64::Engine;
use futures_util::{SinkExt, StreamExt};
use serde_json::json;
use tokio::sync::mpsc;
use tokio_tungstenite::tungstenite::Message;
use tracing::{debug, error, info, warn};

use agent_harness_core::{
    AiError, LiveClientEvent, LiveProvider, LiveServerEvent, LiveSession, LiveSessionConfig,
    ToolCall,
};

use crate::provider::GeminiProvider;

#[async_trait]
impl LiveProvider for GeminiProvider {
    async fn connect_live(
        &self,
        config: LiveSessionConfig,
    ) -> Result<LiveSession, AiError> {
        let url = format!(
            "wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key={}",
            self.api_key()
        );

        info!(model = %self.model(), "connecting to Gemini Live API");

        let (ws_stream, _response) = tokio_tungstenite::connect_async(&url)
            .await
            .map_err(|e| AiError::ProviderError(format!("WebSocket connect error: {}", e)))?;

        let (mut ws_sink, mut ws_source) = ws_stream.split();

        // Send setup message
        let setup = build_setup_message(self.model(), &config);
        let setup_str = serde_json::to_string(&setup)
            .map_err(|e| AiError::ProviderError(format!("setup serialization error: {}", e)))?;

        debug!("sending BidiGenerateContentSetup");
        ws_sink
            .send(Message::Text(setup_str.into()))
            .await
            .map_err(|e| AiError::ProviderError(format!("WebSocket send error: {}", e)))?;

        // Wait for setup complete
        loop {
            match ws_source.next().await {
                Some(Ok(Message::Text(text))) => {
                    let msg: serde_json::Value =
                        serde_json::from_str(&text).unwrap_or_default();
                    if msg.get("setupComplete").is_some() {
                        info!("Gemini Live session setup complete");
                        break;
                    }
                    debug!(?msg, "received pre-setup message");
                }
                Some(Ok(_)) => continue,
                Some(Err(e)) => {
                    return Err(AiError::ProviderError(format!(
                        "WebSocket error during setup: {}",
                        e
                    )));
                }
                None => {
                    return Err(AiError::ProviderError(
                        "WebSocket closed during setup".into(),
                    ));
                }
            }
        }

        // Create channels
        let (client_tx, mut client_rx) = mpsc::channel::<LiveClientEvent>(256);
        let (server_tx, server_rx) = mpsc::channel::<LiveServerEvent>(256);

        let server_tx_clone = server_tx.clone();

        // Send SetupComplete event
        let _ = server_tx.send(LiveServerEvent::SetupComplete).await;

        // Spawn outbound task: client events -> WebSocket
        let _outbound_handle = tokio::spawn(async move {
            while let Some(event) = client_rx.recv().await {
                let msg = match &event {
                    LiveClientEvent::AudioChunk { data, mime_type } => {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                        json!({
                            "realtimeInput": {
                                "audio": {
                                    "data": encoded,
                                    "mimeType": mime_type,
                                }
                            }
                        })
                    }
                    LiveClientEvent::TextMessage(text) => {
                        json!({
                            "realtimeInput": {
                                "text": text,
                            }
                        })
                    }
                    LiveClientEvent::VideoFrame { data, mime_type } => {
                        let encoded = base64::engine::general_purpose::STANDARD.encode(data);
                        json!({
                            "realtimeInput": {
                                "video": {
                                    "data": encoded,
                                    "mimeType": mime_type,
                                }
                            }
                        })
                    }
                    LiveClientEvent::ToolResults(results) => {
                        let responses: Vec<serde_json::Value> = results
                            .iter()
                            .map(|r| {
                                json!({
                                    "id": r.call_id,
                                    "name": r.call_id,
                                    "response": r.content,
                                })
                            })
                            .collect();
                        json!({
                            "toolResponse": {
                                "functionResponses": responses,
                            }
                        })
                    }
                    LiveClientEvent::Interrupt => {
                        // Gemini doesn't have an explicit interrupt message;
                        // interruption happens automatically via VAD when user speaks.
                        // We could close and reconnect, but for now just skip.
                        warn!("explicit interrupt not directly supported in Gemini Live protocol");
                        continue;
                    }
                    LiveClientEvent::Close => {
                        debug!("closing WebSocket");
                        let _ = ws_sink.close().await;
                        return;
                    }
                };

                let msg_str = match serde_json::to_string(&msg) {
                    Ok(s) => s,
                    Err(e) => {
                        error!(error = %e, "failed to serialize client event");
                        continue;
                    }
                };

                if let Err(e) = ws_sink.send(Message::Text(msg_str.into())).await {
                    error!(error = %e, "WebSocket send error");
                    return;
                }
            }

            // Client channel closed — close WebSocket
            let _ = ws_sink.close().await;
        });

        // Spawn inbound task: WebSocket -> server events
        tokio::spawn(async move {
            let mut tool_call_counter = 0u32;

            while let Some(ws_msg) = ws_source.next().await {
                let text = match ws_msg {
                    Ok(Message::Text(t)) => t.to_string(),
                    Ok(Message::Close(_)) => {
                        let _ = server_tx_clone.send(LiveServerEvent::Closed).await;
                        return;
                    }
                    Ok(_) => continue,
                    Err(e) => {
                        let _ = server_tx_clone
                            .send(LiveServerEvent::Error(format!("WebSocket error: {}", e)))
                            .await;
                        return;
                    }
                };

                let msg: serde_json::Value = match serde_json::from_str(&text) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error = %e, "failed to parse WebSocket message");
                        continue;
                    }
                };

                // Handle serverContent
                if let Some(server_content) = msg.get("serverContent") {
                    // Model turn — audio and text
                    if let Some(model_turn) = server_content.get("modelTurn") {
                        if let Some(parts) = model_turn.get("parts").and_then(|p| p.as_array()) {
                            for part in parts {
                                if let Some(inline_data) = part.get("inlineData") {
                                    if let Some(data_str) =
                                        inline_data.get("data").and_then(|d| d.as_str())
                                    {
                                        let mime = inline_data
                                            .get("mimeType")
                                            .and_then(|m| m.as_str())
                                            .unwrap_or("audio/pcm;rate=24000")
                                            .to_string();
                                        if let Ok(decoded) =
                                            base64::engine::general_purpose::STANDARD
                                                .decode(data_str)
                                        {
                                            let _ = server_tx_clone
                                                .send(LiveServerEvent::AudioDelta {
                                                    data: decoded,
                                                    mime_type: mime,
                                                })
                                                .await;
                                        }
                                    }
                                }
                                if let Some(text) = part.get("text").and_then(|t| t.as_str()) {
                                    let _ = server_tx_clone
                                        .send(LiveServerEvent::TranscriptDelta(
                                            text.to_string(),
                                        ))
                                        .await;
                                }
                            }
                        }
                    }

                    // Input transcription
                    if let Some(input_t) = server_content.get("inputTranscription") {
                        if let Some(text) = input_t.get("text").and_then(|t| t.as_str()) {
                            let _ = server_tx_clone
                                .send(LiveServerEvent::InputTranscript(text.to_string()))
                                .await;
                        }
                    }

                    // Output transcription
                    if let Some(output_t) = server_content.get("outputTranscription") {
                        if let Some(text) = output_t.get("text").and_then(|t| t.as_str()) {
                            let _ = server_tx_clone
                                .send(LiveServerEvent::TranscriptDelta(text.to_string()))
                                .await;
                        }
                    }

                    // Interrupted
                    if server_content
                        .get("interrupted")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        let _ = server_tx_clone.send(LiveServerEvent::Interrupted).await;
                    }

                    // Generation complete
                    if server_content
                        .get("generationComplete")
                        .and_then(|v| v.as_bool())
                        .unwrap_or(false)
                    {
                        let _ = server_tx_clone.send(LiveServerEvent::TurnComplete).await;
                    }
                }

                // Handle toolCall
                if let Some(tool_call) = msg.get("toolCall") {
                    if let Some(fn_calls) =
                        tool_call.get("functionCalls").and_then(|f| f.as_array())
                    {
                        let calls: Vec<ToolCall> = fn_calls
                            .iter()
                            .filter_map(|fc| {
                                let name =
                                    fc.get("name").and_then(|n| n.as_str())?.to_string();
                                let id = fc
                                    .get("id")
                                    .and_then(|i| i.as_str())
                                    .map(|s| s.to_string())
                                    .unwrap_or_else(|| {
                                        tool_call_counter += 1;
                                        format!("call_{}", tool_call_counter)
                                    });
                                let args =
                                    fc.get("args").cloned().unwrap_or(serde_json::Value::Object(
                                        serde_json::Map::new(),
                                    ));
                                Some(ToolCall {
                                    id,
                                    name,
                                    arguments: args,
                                })
                            })
                            .collect();

                        if !calls.is_empty() {
                            let _ = server_tx_clone
                                .send(LiveServerEvent::ToolCalls(calls))
                                .await;
                        }
                    }
                }

                // Handle usageMetadata
                if let Some(usage) = msg.get("usageMetadata") {
                    let input = usage
                        .get("promptTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    let output = usage
                        .get("candidatesTokenCount")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as u32;
                    if input > 0 || output > 0 {
                        let _ = server_tx_clone
                            .send(LiveServerEvent::Usage {
                                input_tokens: input,
                                output_tokens: output,
                            })
                            .await;
                    }
                }

                // Handle goAway
                if let Some(go_away) = msg.get("goAway") {
                    let time_left = go_away
                        .get("timeLeft")
                        .and_then(|v| v.as_str())
                        .and_then(|s| {
                            // Parse duration string like "30s"
                            s.trim_end_matches('s')
                                .parse::<u64>()
                                .ok()
                                .map(std::time::Duration::from_secs)
                        });
                    let _ = server_tx_clone
                        .send(LiveServerEvent::GoingAway { time_left })
                        .await;
                }

                // Handle sessionResumptionUpdate
                if let Some(resume) = msg.get("sessionResumptionUpdate") {
                    if let Some(handle) = resume.get("handle").and_then(|h| h.as_str()) {
                        let _ = server_tx_clone
                            .send(LiveServerEvent::SessionResumeToken(handle.to_string()))
                            .await;
                    }
                }
            }

            // WebSocket closed
            let _ = server_tx_clone.send(LiveServerEvent::Closed).await;
        });

        Ok(LiveSession {
            tx: client_tx,
            rx: server_rx,
        })
    }
}

/// Build the BidiGenerateContentSetup JSON message.
fn build_setup_message(model: &str, config: &LiveSessionConfig) -> serde_json::Value {
    let mut setup = json!({
        "model": format!("models/{}", model),
    });

    let setup_obj = setup.as_object_mut().unwrap();

    // Generation config
    let mut gen_config = json!({
        "responseModalities": config.response_modalities,
    });
    if let Some(ref thinking) = config.thinking {
        gen_config["thinkingConfig"] = json!({
            "thinkingLevel": thinking.level,
            "includeThoughts": thinking.include_thoughts,
        });
    }
    setup_obj.insert("generationConfig".to_string(), gen_config);

    // System instruction (skip empty strings)
    if let Some(ref prompt) = config.system_prompt {
        if !prompt.is_empty() {
            setup_obj.insert(
                "systemInstruction".to_string(),
                json!({
                    "parts": [{"text": prompt}],
                }),
            );
        }
    }

    // Tools
    if !config.tools.is_empty() {
        let declarations: Vec<serde_json::Value> = config
            .tools
            .iter()
            .map(|t| {
                json!({
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                })
            })
            .collect();
        setup_obj.insert(
            "tools".to_string(),
            json!([{"functionDeclarations": declarations}]),
        );
    }

    // Speech config (voice)
    if let Some(ref voice) = config.voice {
        setup_obj.insert(
            "speechConfig".to_string(),
            json!({
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": voice,
                    }
                }
            }),
        );
    }

    // VAD config
    if let Some(ref vad) = config.vad {
        let mut vad_config = json!({
            "disabled": !vad.enabled,
        });
        let vad_obj = vad_config.as_object_mut().unwrap();
        if let Some(ref s) = vad.start_sensitivity {
            vad_obj.insert("startOfSpeechSensitivity".to_string(), json!(s));
        }
        if let Some(ref s) = vad.end_sensitivity {
            vad_obj.insert("endOfSpeechSensitivity".to_string(), json!(s));
        }
        if let Some(ms) = vad.prefix_padding_ms {
            vad_obj.insert("prefixPaddingMs".to_string(), json!(ms));
        }
        if let Some(ms) = vad.silence_duration_ms {
            vad_obj.insert("silenceDurationMs".to_string(), json!(ms));
        }
        setup_obj.insert(
            "realtimeInputConfig".to_string(),
            json!({
                "automaticActivityDetection": vad_config,
            }),
        );
    }

    // Transcription
    if config.enable_input_transcription {
        setup_obj.insert("inputAudioTranscription".to_string(), json!({}));
    }
    if config.enable_output_transcription {
        setup_obj.insert("outputAudioTranscription".to_string(), json!({}));
    }

    // Context window compression
    if config.context_window_compression {
        setup_obj.insert("contextWindowCompression".to_string(), json!({}));
    }

    // Session resumption
    if let Some(ref handle) = config.resume_handle {
        setup_obj.insert(
            "sessionResumption".to_string(),
            json!({"handle": handle}),
        );
    }

    json!({"setup": setup})
}

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::{LiveSessionConfig, ToolDefinition, VadConfig, LiveThinkingConfig};

    #[test]
    fn setup_message_basic() {
        let config = LiveSessionConfig::default();
        let msg = build_setup_message("gemini-2.0-flash", &config);

        let setup = &msg["setup"];
        assert_eq!(setup["model"], "models/gemini-2.0-flash");
        let modalities = setup["generationConfig"]["responseModalities"]
            .as_array()
            .unwrap();
        assert_eq!(modalities.len(), 1);
        assert_eq!(modalities[0], "AUDIO");
        // No optional fields
        assert!(setup.get("systemInstruction").is_none());
        assert!(setup.get("tools").is_none());
        assert!(setup.get("speechConfig").is_none());
    }

    #[test]
    fn setup_message_with_system_prompt() {
        let config = LiveSessionConfig {
            system_prompt: Some("You are helpful.".to_string()),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let sys = &msg["setup"]["systemInstruction"];
        assert_eq!(sys["parts"][0]["text"], "You are helpful.");
    }

    #[test]
    fn setup_message_with_tools() {
        let config = LiveSessionConfig {
            tools: vec![ToolDefinition {
                name: "get_weather".to_string(),
                description: "Get the weather".to_string(),
                parameters: serde_json::json!({"type": "object", "properties": {"city": {"type": "string"}}}),
            }],
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let tools = msg["setup"]["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 1);
        let declarations = tools[0]["functionDeclarations"].as_array().unwrap();
        assert_eq!(declarations.len(), 1);
        assert_eq!(declarations[0]["name"], "get_weather");
        assert_eq!(declarations[0]["description"], "Get the weather");
    }

    #[test]
    fn setup_message_with_voice() {
        let config = LiveSessionConfig {
            voice: Some("Kore".to_string()),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let voice_name = &msg["setup"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"]["voiceName"];
        assert_eq!(voice_name, "Kore");
    }

    #[test]
    fn setup_message_with_vad() {
        let config = LiveSessionConfig {
            vad: Some(VadConfig {
                enabled: true,
                start_sensitivity: Some("START_SENSITIVITY_HIGH".to_string()),
                end_sensitivity: Some("END_SENSITIVITY_LOW".to_string()),
                prefix_padding_ms: Some(200),
                silence_duration_ms: Some(500),
            }),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let vad = &msg["setup"]["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(vad["disabled"], false);
        assert_eq!(vad["startOfSpeechSensitivity"], "START_SENSITIVITY_HIGH");
        assert_eq!(vad["endOfSpeechSensitivity"], "END_SENSITIVITY_LOW");
        assert_eq!(vad["prefixPaddingMs"], 200);
        assert_eq!(vad["silenceDurationMs"], 500);
    }

    #[test]
    fn setup_message_with_thinking() {
        let config = LiveSessionConfig {
            thinking: Some(LiveThinkingConfig {
                level: "medium".to_string(),
                include_thoughts: true,
            }),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let thinking = &msg["setup"]["generationConfig"]["thinkingConfig"];
        assert_eq!(thinking["thinkingLevel"], "medium");
        assert_eq!(thinking["includeThoughts"], true);
    }

    #[test]
    fn setup_message_with_transcription() {
        let config = LiveSessionConfig {
            enable_input_transcription: true,
            enable_output_transcription: true,
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        assert!(msg["setup"].get("inputAudioTranscription").is_some());
        assert!(msg["setup"].get("outputAudioTranscription").is_some());
    }

    #[test]
    fn setup_message_with_resume_handle() {
        let config = LiveSessionConfig {
            resume_handle: Some("abc-123-handle".to_string()),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        assert_eq!(msg["setup"]["sessionResumption"]["handle"], "abc-123-handle");
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[test]
    fn setup_message_empty_tools_list_omitted() {
        // BUG PROBE: When tools list is empty, the code checks !config.tools.is_empty()
        // and skips adding tools. Verify no empty tools array is sent.
        let config = LiveSessionConfig {
            tools: vec![],
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        assert!(
            msg["setup"].get("tools").is_none(),
            "Empty tools list should NOT produce a tools field in setup message"
        );
    }

    #[test]
    fn setup_message_vad_disabled() {
        // BUG PROBE: VAD with enabled=false should produce "disabled": true.
        let config = LiveSessionConfig {
            vad: Some(VadConfig {
                enabled: false,
                start_sensitivity: None,
                end_sensitivity: None,
                prefix_padding_ms: None,
                silence_duration_ms: None,
            }),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let vad = &msg["setup"]["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(
            vad["disabled"], true,
            "VAD with enabled=false should set disabled=true"
        );
    }

    #[test]
    fn setup_message_vad_enabled_sets_disabled_false() {
        // Confirm that enabled=true sets "disabled": false
        let config = LiveSessionConfig {
            vad: Some(VadConfig {
                enabled: true,
                ..Default::default()
            }),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let vad = &msg["setup"]["realtimeInputConfig"]["automaticActivityDetection"];
        assert_eq!(vad["disabled"], false);
    }

    #[test]
    fn setup_message_empty_system_prompt_omitted() {
        // BUG PROBE: system_prompt=Some("") — empty string still creates the field
        let config = LiveSessionConfig {
            system_prompt: Some("".to_string()),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        // Empty string should be skipped — no systemInstruction field.
        assert!(
            msg["setup"].get("systemInstruction").is_none(),
            "empty system prompt should not create systemInstruction field"
        );
    }

    #[test]
    fn setup_message_context_window_compression() {
        let config = LiveSessionConfig {
            context_window_compression: true,
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        assert!(msg["setup"].get("contextWindowCompression").is_some());
    }

    #[test]
    fn setup_message_all_options_combined() {
        // Stress test: every option enabled at once
        let config = LiveSessionConfig {
            system_prompt: Some("Be helpful".to_string()),
            tools: vec![ToolDefinition {
                name: "tool1".to_string(),
                description: "desc".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }],
            voice: Some("Aoede".to_string()),
            vad: Some(VadConfig {
                enabled: true,
                start_sensitivity: Some("HIGH".to_string()),
                end_sensitivity: Some("LOW".to_string()),
                prefix_padding_ms: Some(100),
                silence_duration_ms: Some(200),
            }),
            thinking: Some(LiveThinkingConfig {
                level: "high".to_string(),
                include_thoughts: true,
            }),
            enable_input_transcription: true,
            enable_output_transcription: true,
            context_window_compression: true,
            resume_handle: Some("handle-xyz".to_string()),
            ..Default::default()
        };
        let msg = build_setup_message("gemini-test", &config);
        let setup = &msg["setup"];

        // All fields should be present
        assert!(setup.get("systemInstruction").is_some());
        assert!(setup.get("tools").is_some());
        assert!(setup.get("speechConfig").is_some());
        assert!(setup.get("realtimeInputConfig").is_some());
        assert!(setup.get("inputAudioTranscription").is_some());
        assert!(setup.get("outputAudioTranscription").is_some());
        assert!(setup.get("contextWindowCompression").is_some());
        assert!(setup.get("sessionResumption").is_some());
        assert!(setup["generationConfig"].get("thinkingConfig").is_some());
    }
}
