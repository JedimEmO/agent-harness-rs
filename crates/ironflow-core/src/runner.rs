use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::agent::Agent;
use crate::config::AgentConfig;
use crate::context::ContextManager;
use crate::error::{AgentError, AiError};
use crate::event::AgentEvent;
use crate::guardrail::{GuardrailContext, GuardrailResult, OutputGuardrail};
use crate::interaction::InteractionResponse;
use crate::memory::MemoryStore;
use crate::provider::{AiProvider, ConversationRequest, StreamEvent};
use crate::session::{
    MessageContent, MessageRole, Session, SessionMessage, SessionStore, ToolCallRecord,
    ToolResultRecord,
};
use crate::tool::{validate_arguments, ToolExecResult, ToolPermission, ToolRegistry};

// ---------------------------------------------------------------------------
// TurnRequest — replaces the 8-parameter run_turn() signature
// ---------------------------------------------------------------------------

/// Request to execute a single agent turn.
///
/// Use [`TurnRequest::new`] to create a request along with the [`TurnChannels`]
/// needed for event handling and interaction. This replaces the previous
/// 8-parameter `run_turn()` method.
///
/// # Example
///
/// ```rust,no_run
/// # use ironflow_core::*;
/// let (request, channels) = TurnRequest::new(
///     "session-id",
///     "scope-id",
///     "Hello, agent!",
///     "You are a helpful assistant.",
/// );
/// // Spawn a task to consume channels.event_rx
/// // Then: runner.run_turn(request).await
/// ```
pub struct TurnRequest {
    pub session_id: String,
    pub scope_id: String,
    pub user_message: String,
    pub system_prompt: String,
    pub event_tx: mpsc::Sender<AgentEvent>,
    pub interaction_rx: mpsc::Receiver<(String, InteractionResponse)>,
    pub approval_rx: mpsc::Receiver<(String, bool)>,
    pub auto_approve: bool,
    /// Optional cancellation token. If the receiver gets a message, the turn stops.
    pub cancel_rx: Option<mpsc::Receiver<()>>,
    /// Optional filter for which tools are available this turn.
    pub tool_filter: Option<Box<dyn Fn(&str) -> bool + Send + Sync>>,
}

/// The caller-side channels returned by [`TurnRequest::new`].
pub struct TurnChannels {
    pub event_rx: mpsc::Receiver<AgentEvent>,
    pub interaction_tx: mpsc::Sender<(String, InteractionResponse)>,
    pub approval_tx: mpsc::Sender<(String, bool)>,
}

impl TurnRequest {
    /// Create a new turn request and its associated channels.
    pub fn new(
        session_id: impl Into<String>,
        scope_id: impl Into<String>,
        user_message: impl Into<String>,
        system_prompt: impl Into<String>,
    ) -> (Self, TurnChannels) {
        let (event_tx, event_rx) = mpsc::channel(100);
        let (interaction_tx, interaction_rx) = mpsc::channel(10);
        let (approval_tx, approval_rx) = mpsc::channel(10);

        (
            TurnRequest {
                session_id: session_id.into(),
                scope_id: scope_id.into(),
                user_message: user_message.into(),
                system_prompt: system_prompt.into(),
                event_tx,
                interaction_rx,
                approval_rx,
                auto_approve: false,
                cancel_rx: None,
                tool_filter: None,
            },
            TurnChannels {
                event_rx,
                interaction_tx,
                approval_tx,
            },
        )
    }

    pub fn auto_approve(mut self, auto_approve: bool) -> Self {
        self.auto_approve = auto_approve;
        self
    }

    /// Provide a cancellation receiver. Send `()` to cancel the turn.
    pub fn cancel_rx(mut self, rx: mpsc::Receiver<()>) -> Self {
        self.cancel_rx = Some(rx);
        self
    }

    /// Filter which tools are available for this turn.
    pub fn tool_filter(mut self, filter: impl Fn(&str) -> bool + Send + Sync + 'static) -> Self {
        self.tool_filter = Some(Box::new(filter));
        self
    }
}

// ---------------------------------------------------------------------------
// AgentRunner
// ---------------------------------------------------------------------------

pub struct AgentRunner {
    ai_provider: Arc<dyn AiProvider>,
    tool_registry: Arc<ToolRegistry>,
    session_store: Arc<dyn SessionStore>,
    memory_store: Arc<dyn MemoryStore>,
    config: AgentConfig,
    context_manager: ContextManager,
    guardrails: Vec<Arc<dyn OutputGuardrail>>,
    #[cfg(feature = "context-pipeline")]
    context_pipeline: Option<Arc<ironflow_context::ContextPipeline>>,
}

impl AgentRunner {
    pub fn new(
        ai_provider: Arc<dyn AiProvider>,
        tool_registry: Arc<ToolRegistry>,
        session_store: Arc<dyn SessionStore>,
        memory_store: Arc<dyn MemoryStore>,
        config: AgentConfig,
    ) -> Self {
        let context_manager =
            ContextManager::new(config.max_context_tokens, config.chars_per_token_estimate);
        Self {
            ai_provider,
            tool_registry,
            session_store,
            memory_store,
            config,
            context_manager,
            guardrails: Vec::new(),
            #[cfg(feature = "context-pipeline")]
            context_pipeline: None,
        }
    }

    /// Add an output guardrail that validates LLM text responses.
    pub fn add_guardrail(&mut self, guardrail: Arc<dyn OutputGuardrail>) {
        self.guardrails.push(guardrail);
    }

    /// Use a custom [`ContextPipeline`](ironflow_context::ContextPipeline) instead of the
    /// built-in truncation-based context manager.
    ///
    /// When set, the pipeline processes `SessionMessage`s (converted to `ContextBlock`s)
    /// through its composable strategies before sending to the AI provider.
    #[cfg(feature = "context-pipeline")]
    pub fn with_context_pipeline(
        mut self,
        pipeline: Arc<ironflow_context::ContextPipeline>,
    ) -> Self {
        self.context_pipeline = Some(pipeline);
        self
    }

    pub fn session_store(&self) -> &Arc<dyn SessionStore> {
        &self.session_store
    }

    pub fn memory_store(&self) -> &Arc<dyn MemoryStore> {
        &self.memory_store
    }

    pub fn tool_registry(&self) -> &Arc<ToolRegistry> {
        &self.tool_registry
    }

    pub async fn create_session(&self, scope_id: &str) -> Result<Session, AgentError> {
        let session = self.session_store.create_session(scope_id).await?;
        info!(session_id = %session.id, scope_id, "created agent session");
        Ok(session)
    }

    /// Run one agent turn using the new [`TurnRequest`] API.
    ///
    /// This method uses streaming to emit `TextDelta` events in real-time,
    /// validates tool arguments before execution, supports cancellation and
    /// timeouts, and retries on rate-limit errors with exponential backoff.
    pub async fn run_turn(&self, mut request: TurnRequest) -> Result<(), AgentError> {
        let session_id = &request.session_id;
        let scope_id = &request.scope_id;
        info!(session_id = %session_id, scope_id, auto_approve = request.auto_approve, "starting agent turn");
        debug!(session_id = %session_id, user_message = %request.user_message, "user message");

        let mut plan_approved = request.auto_approve;

        // 1. Persist the user message
        let user_msg = SessionMessage {
            id: Uuid::new_v4().to_string(),
            session_id: session_id.to_string(),
            role: MessageRole::User,
            content: MessageContent::Text(request.user_message.clone()),
            created_at: chrono::Utc::now().to_rfc3339(),
        };
        self.session_store.append_message(&user_msg).await?;

        // 2. Load history once and maintain in-memory
        let mut local_messages: Vec<SessionMessage> =
            self.session_store.get_messages(session_id).await?;

        // 3. Get tool definitions (optionally filtered)
        let tools = if let Some(ref filter) = request.tool_filter {
            self.tool_registry.definitions_filtered(filter).await
        } else {
            self.tool_registry.definitions().await
        };

        // 4. Tool loop
        let mut tools_used_this_turn: Vec<String> = Vec::new();
        for round in 0..self.config.max_tool_rounds {
            // Check cancellation
            if let Some(ref mut cancel_rx) = request.cancel_rx {
                if cancel_rx.try_recv().is_ok() {
                    info!(session_id = %session_id, round, "turn cancelled");
                    let _ = request.event_tx.send(AgentEvent::TurnComplete).await;
                    return Err(AgentError::Cancelled);
                }
            }

            info!(session_id = %session_id, round, max_rounds = self.config.max_tool_rounds, "agent loop round");
            let _ = request.event_tx.send(AgentEvent::Thinking).await;

            // Build conversation from in-memory history
            #[cfg(feature = "context-pipeline")]
            let (messages, truncation_info) = if let Some(ref pipeline) = self.context_pipeline {
                let blocks = crate::context_bridge::session_messages_to_blocks(&local_messages);
                let (processed, report) = pipeline
                    .process(blocks, self.config.max_context_tokens)
                    .await
                    .map_err(|e| AgentError::StorageError(format!("context pipeline: {}", e)))?;
                let msgs = crate::context_bridge::blocks_to_conversation(&processed);
                let trunc = if report.original_blocks != report.final_blocks {
                    Some(crate::context::TruncationInfo {
                        dropped_messages: report.original_blocks - report.final_blocks,
                        remaining_messages: report.final_blocks,
                    })
                } else {
                    None
                };
                (msgs, trunc)
            } else {
                self.context_manager.prepare_messages(&local_messages)
            };

            #[cfg(not(feature = "context-pipeline"))]
            let (messages, truncation_info) =
                self.context_manager.prepare_messages(&local_messages);

            if let Some(ref info) = truncation_info {
                warn!(
                    session_id = %session_id,
                    dropped = info.dropped_messages,
                    remaining = info.remaining_messages,
                    "context truncated"
                );
                let _ = request
                    .event_tx
                    .send(AgentEvent::ContextTruncated {
                        dropped_messages: info.dropped_messages,
                        remaining_messages: info.remaining_messages,
                    })
                    .await;
            }

            let msg_count = messages.len();
            debug!(session_id = %session_id, context_messages = msg_count, tools = tools.len(), "calling AI provider (streaming)");

            let conv_request = ConversationRequest {
                system: Some(request.system_prompt.clone()),
                messages,
                tools: tools.clone(),
                max_tokens: self.config.max_tokens,
            };

            // Call AI provider with streaming + retry on rate limits
            let (response_text, response_tool_calls) = self
                .call_provider_with_retry(
                    conv_request,
                    round,
                    &request.event_tx,
                    &mut request.cancel_rx,
                )
                .await?;

            if !response_tool_calls.is_empty() {
                // ---- Handle tool calls ----
                let tool_names: Vec<&str> =
                    response_tool_calls.iter().map(|tc| tc.name.as_str()).collect();
                tools_used_this_turn.extend(tool_names.iter().map(|s| s.to_string()));
                info!(session_id = %session_id, round, count = response_tool_calls.len(), tools = ?tool_names, "agent requested tool calls");

                let call_records: Vec<ToolCallRecord> = response_tool_calls
                    .iter()
                    .map(ToolCallRecord::from)
                    .collect();

                let tc_msg = SessionMessage {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: MessageRole::Assistant,
                    content: MessageContent::ToolCalls(call_records),
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                self.session_store.append_message(&tc_msg).await?;
                local_messages.push(tc_msg);

                let mut result_records = Vec::new();

                for tc in &response_tool_calls {
                    let record = self
                        .execute_single_tool(
                            tc,
                            scope_id,
                            session_id,
                            &request.event_tx,
                            &mut request.approval_rx,
                            &mut request.interaction_rx,
                            &mut plan_approved,
                        )
                        .await?;
                    result_records.push(record);
                }

                let tr_msg = SessionMessage {
                    id: Uuid::new_v4().to_string(),
                    session_id: session_id.to_string(),
                    role: MessageRole::Tool,
                    content: MessageContent::ToolResults(result_records),
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                self.session_store.append_message(&tr_msg).await?;
                local_messages.push(tr_msg);
            } else {
                // ---- Handle text response ----
                let mut text = response_text;
                info!(session_id = %session_id, round, text_len = text.len(), "agent produced text response");

                // Run output guardrails
                if !self.guardrails.is_empty() {
                    let guardrail_ctx = GuardrailContext {
                        session_id: session_id.to_string(),
                        scope_id: scope_id.to_string(),
                        tool_names_used: tools_used_this_turn.clone(),
                        round,
                    };

                    for guardrail in &self.guardrails {
                        match guardrail.validate(&text, &guardrail_ctx).await {
                            GuardrailResult::Pass => {}
                            GuardrailResult::Modify(modified) => {
                                info!(guardrail = guardrail.name(), "guardrail modified output");
                                text = modified;
                            }
                            GuardrailResult::Block(reason) => {
                                warn!(guardrail = guardrail.name(), reason = %reason, "guardrail blocked output");
                                let _ = request
                                    .event_tx
                                    .send(AgentEvent::Error {
                                        message: format!(
                                            "Output blocked by guardrail '{}': {}",
                                            guardrail.name(),
                                            reason
                                        ),
                                        recoverable: false,
                                    })
                                    .await;
                                let _ = request
                                    .event_tx
                                    .send(AgentEvent::TurnComplete)
                                    .await;
                                return Ok(());
                            }
                        }
                    }
                }

                let msg_id = Uuid::new_v4().to_string();
                let assistant_msg = SessionMessage {
                    id: msg_id.clone(),
                    session_id: session_id.to_string(),
                    role: MessageRole::Assistant,
                    content: MessageContent::Text(text.clone()),
                    created_at: chrono::Utc::now().to_rfc3339(),
                };
                self.session_store.append_message(&assistant_msg).await?;

                let _ = request
                    .event_tx
                    .send(AgentEvent::TextComplete {
                        message_id: msg_id,
                        text,
                    })
                    .await;

                let _ = request.event_tx.send(AgentEvent::TurnComplete).await;
                return Ok(());
            }
        }

        // Exceeded max rounds
        warn!(session_id = %session_id, max_rounds = self.config.max_tool_rounds, plan_approved, "agent exceeded max tool rounds");
        if !plan_approved {
            let _ = request
                .event_tx
                .send(AgentEvent::Error {
                    message: format!(
                        "Exceeded maximum of {} tool rounds",
                        self.config.max_tool_rounds
                    ),
                    recoverable: false,
                })
                .await;
        }
        let _ = request.event_tx.send(AgentEvent::TurnComplete).await;

        if plan_approved {
            Ok(())
        } else {
            Err(AgentError::MaxRoundsExceeded(self.config.max_tool_rounds))
        }
    }

    /// Call the AI provider via streaming, with automatic retry on rate limits.
    ///
    /// Returns `(accumulated_text, tool_calls)`. Emits `TextDelta` and
    /// `UsageUpdate` events as they arrive from the stream.
    async fn call_provider_with_retry(
        &self,
        request: ConversationRequest,
        round: usize,
        event_tx: &mpsc::Sender<AgentEvent>,
        cancel_rx: &mut Option<mpsc::Receiver<()>>,
    ) -> Result<(String, Vec<crate::provider::ToolCall>), AgentError> {
        let mut attempt = 0;

        loop {
            // Check cancellation before each attempt
            if let Some(ref mut rx) = cancel_rx {
                if rx.try_recv().is_ok() {
                    return Err(AgentError::Cancelled);
                }
            }

            let ai_start = Instant::now();

            match self
                .ai_provider
                .converse_stream(request.clone())
                .await
            {
                Ok(stream) => {
                    let ai_ms = ai_start.elapsed().as_millis();
                    debug!(round, attempt, ai_latency_ms = ai_ms, "streaming response started");

                    // Consume the stream
                    let mut full_text = String::new();
                    let mut tool_calls = Vec::new();
                    let mut stream = std::pin::pin!(stream);

                    while let Some(event_result) = stream.next().await {
                        match event_result {
                            Ok(StreamEvent::TextDelta(delta)) => {
                                let _ = event_tx
                                    .send(AgentEvent::TextDelta {
                                        text: delta.clone(),
                                    })
                                    .await;
                                full_text.push_str(&delta);
                            }
                            Ok(StreamEvent::TextComplete(text)) => {
                                full_text = text;
                            }
                            Ok(StreamEvent::ToolCalls(calls)) => {
                                tool_calls = calls;
                            }
                            Ok(StreamEvent::Usage {
                                input_tokens,
                                output_tokens,
                            }) => {
                                let _ = event_tx
                                    .send(AgentEvent::UsageUpdate {
                                        input_tokens,
                                        output_tokens,
                                    })
                                    .await;
                            }
                            Ok(StreamEvent::Done) => break,
                            Err(AiError::RateLimited) => {
                                // Rate limit during stream — treat as retryable
                                warn!(round, attempt, "rate limited during stream");
                                break;
                            }
                            Err(e) => {
                                return Err(AgentError::AiError(e));
                            }
                        }

                        // Check cancellation between stream events
                        if let Some(ref mut rx) = cancel_rx {
                            if rx.try_recv().is_ok() {
                                return Err(AgentError::Cancelled);
                            }
                        }
                    }

                    return Ok((full_text, tool_calls));
                }
                Err(AiError::RateLimited) => {
                    if attempt >= self.config.max_retries {
                        error!(round, attempt, "rate limit retries exhausted");
                        return Err(AgentError::AiError(AiError::RateLimited));
                    }

                    let delay_ms =
                        self.config.retry_base_delay_ms * 2u64.pow(attempt as u32);
                    warn!(round, attempt, delay_ms, "rate limited, retrying");

                    let _ = event_tx
                        .send(AgentEvent::RetryAttempt {
                            round,
                            attempt,
                            delay_ms,
                        })
                        .await;

                    tokio::time::sleep(Duration::from_millis(delay_ms)).await;
                    attempt += 1;
                }
                Err(e) => {
                    return Err(AgentError::AiError(e));
                }
            }
        }
    }

    /// Execute a single tool call: validate, approve, run, handle interaction.
    ///
    /// Returns `Ok(Some(record))` with the result, or `Ok(None)` if the tool
    /// was unknown or validation failed (error already pushed to `result_records`).
    /// The `plan_approved` flag may be set to `true` if a plan is approved via interaction.
    async fn execute_single_tool(
        &self,
        tc: &crate::provider::ToolCall,
        scope_id: &str,
        session_id: &str,
        event_tx: &mpsc::Sender<AgentEvent>,
        approval_rx: &mut mpsc::Receiver<(String, bool)>,
        interaction_rx: &mut mpsc::Receiver<(String, InteractionResponse)>,
        plan_approved: &mut bool,
    ) -> Result<ToolResultRecord, AgentError> {
        // Look up tool
        let tool = match self.tool_registry.get(&tc.name).await {
            Some(t) => t,
            None => {
                let err_msg = format!("Unknown tool: {}", tc.name);
                warn!(session_id = %session_id, tool_name = %tc.name, "unknown tool requested by AI");
                return Ok(ToolResultRecord {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    content: serde_json::Value::String(err_msg),
                });
            }
        };

        // Validate arguments
        let definition = tool.definition();
        if let Err(validation_errors) = validate_arguments(&definition.parameters, &tc.arguments) {
            let err_msg = format!(
                "Argument validation failed: {}",
                validation_errors.join("; ")
            );
            warn!(session_id = %session_id, tool_name = %tc.name, errors = ?validation_errors, "tool argument validation failed");
            let _ = event_tx
                .send(AgentEvent::Error {
                    message: format!("Tool '{}' argument validation failed", tc.name),
                    recoverable: true,
                })
                .await;
            return Ok(ToolResultRecord {
                call_id: tc.id.clone(),
                tool_name: tc.name.clone(),
                content: serde_json::Value::String(err_msg),
            });
        }

        let permission = tool.permission();

        let _ = event_tx
            .send(AgentEvent::ToolCallStarted {
                call_id: tc.id.clone(),
                tool_name: tc.name.clone(),
                arguments: tc.arguments.clone(),
                permission,
            })
            .await;

        // Handle approval
        if permission == ToolPermission::RequiresApproval && !*plan_approved {
            info!(session_id = %session_id, tool_name = %tc.name, call_id = %tc.id, "waiting for user approval");
            let _ = event_tx
                .send(AgentEvent::ToolApprovalNeeded {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    arguments: tc.arguments.clone(),
                    description: definition.description.clone(),
                })
                .await;

            let approved = loop {
                match approval_rx.recv().await {
                    Some((id, approved)) if id == tc.id => break approved,
                    Some(_) => continue,
                    None => {
                        return Err(AgentError::ChannelClosed(
                            "approval channel".to_string(),
                        ))
                    }
                }
            };

            info!(session_id = %session_id, tool_name = %tc.name, approved, "received approval decision");

            if !approved {
                let _ = event_tx
                    .send(AgentEvent::ToolCallCompleted {
                        call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        result: serde_json::Value::String("Denied by user".to_string()),
                        duration_ms: 0,
                    })
                    .await;
                return Ok(ToolResultRecord {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    content: serde_json::Value::String(
                        "Tool execution was denied by the user.".to_string(),
                    ),
                });
            }
        } else if permission == ToolPermission::RequiresApproval {
            info!(session_id = %session_id, tool_name = %tc.name, call_id = %tc.id, "auto-approving tool (plan approved)");
        }

        // Execute with optional timeout
        debug!(session_id = %session_id, tool_name = %tc.name, args = %tc.arguments, "executing tool");
        let start = Instant::now();

        let exec_result = if let Some(timeout_secs) = self.config.tool_timeout_secs {
            match tokio::time::timeout(
                Duration::from_secs(timeout_secs),
                tool.execute(scope_id, tc.arguments.clone()),
            )
            .await
            {
                Ok(result) => result,
                Err(_) => {
                    warn!(session_id = %session_id, tool_name = %tc.name, timeout_secs, "tool execution timed out");
                    Err(AgentError::Timeout {
                        tool_name: tc.name.clone(),
                        timeout_secs,
                    })
                }
            }
        } else {
            tool.execute(scope_id, tc.arguments.clone()).await
        };

        let duration_ms = start.elapsed().as_millis() as u64;

        match exec_result {
            Ok(ToolExecResult::Completed(content)) => {
                info!(session_id = %session_id, tool_name = %tc.name, duration_ms, "tool completed");
                let _ = event_tx
                    .send(AgentEvent::ToolCallCompleted {
                        call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        result: content.clone(),
                        duration_ms,
                    })
                    .await;
                Ok(ToolResultRecord {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    content,
                })
            }
            Ok(ToolExecResult::NeedsInteraction(interaction_request)) => {
                let interaction_id = Uuid::new_v4().to_string();
                info!(session_id = %session_id, tool_name = %tc.name, interaction_id = %interaction_id, "tool needs user interaction");

                let _ = event_tx
                    .send(AgentEvent::InteractionNeeded {
                        interaction_id: interaction_id.clone(),
                        request: interaction_request,
                    })
                    .await;

                let response_text = loop {
                    match interaction_rx.recv().await {
                        Some((id, resp)) if id == interaction_id => {
                            if let InteractionResponse::PlanApproved {
                                approved: true,
                                ..
                            } = &resp
                            {
                                *plan_approved = true;
                                info!(session_id = %session_id, "plan approved — enabling auto-approve for this turn");
                            }
                            break format_interaction_response(&resp);
                        }
                        Some(_) => continue,
                        None => {
                            return Err(AgentError::ChannelClosed(
                                "interaction channel".to_string(),
                            ))
                        }
                    }
                };

                let _ = event_tx
                    .send(AgentEvent::ToolCallCompleted {
                        call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        result: serde_json::Value::String(response_text.clone()),
                        duration_ms: start.elapsed().as_millis() as u64,
                    })
                    .await;

                Ok(ToolResultRecord {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    content: serde_json::Value::String(response_text),
                })
            }
            Err(e) => {
                let err_msg = format!("Tool error: {}", e);
                error!(session_id = %session_id, tool_name = %tc.name, duration_ms, error = %e, "tool execution failed");
                let _ = event_tx
                    .send(AgentEvent::ToolCallCompleted {
                        call_id: tc.id.clone(),
                        tool_name: tc.name.clone(),
                        result: serde_json::Value::String(err_msg.clone()),
                        duration_ms,
                    })
                    .await;
                Ok(ToolResultRecord {
                    call_id: tc.id.clone(),
                    tool_name: tc.name.clone(),
                    content: serde_json::Value::String(err_msg),
                })
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Agent trait implementation for AgentRunner
// ---------------------------------------------------------------------------

#[async_trait::async_trait]
impl Agent for AgentRunner {
    fn name(&self) -> &str {
        "agent_runner"
    }

    fn description(&self) -> &str {
        "The main agent runner with tool calling, streaming, and approval gates"
    }

    async fn run(
        &self,
        input: &str,
        scope_id: &str,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Result<String, AgentError> {
        // Create a session for this sub-agent run
        let session = self.create_session(scope_id).await?;

        let (mut request, mut channels) = TurnRequest::new(
            &session.id,
            scope_id,
            input,
            "", // Sub-agents typically get their system prompt from the parent
        );
        request.auto_approve = true; // Sub-agents auto-approve by default

        // If caller provided an event_tx, forward events to it
        let forward_tx = event_tx.clone();

        // Spawn event forwarder
        let collector_handle = tokio::spawn(async move {
            let mut final_text = String::new();
            while let Some(event) = channels.event_rx.recv().await {
                if let AgentEvent::TextComplete { ref text, .. } = event {
                    final_text = text.clone();
                }
                if let Some(ref tx) = forward_tx {
                    let _ = tx.send(event).await;
                }
            }
            final_text
        });

        self.run_turn(request).await?;

        let result = collector_handle
            .await
            .map_err(|e| AgentError::StorageError(format!("event collector panicked: {}", e)))?;

        Ok(result)
    }
}

fn format_interaction_response(resp: &InteractionResponse) -> String {
    match resp {
        InteractionResponse::Text { text } => format!("User responded: {}", text),
        InteractionResponse::SelectedOptions { ids } => {
            format!("User selected options: {}", ids.join(", "))
        }
        InteractionResponse::Confirmed { approved } => {
            if *approved {
                "User confirmed the action.".to_string()
            } else {
                "User denied the action.".to_string()
            }
        }
        InteractionResponse::PlanApproved {
            approved,
            modifications,
        } => {
            if *approved {
                match modifications {
                    Some(mods) => format!("User approved the plan with modifications: {}", mods),
                    None => "User approved the plan.".to_string(),
                }
            } else {
                "User rejected the plan.".to_string()
            }
        }
    }
}
