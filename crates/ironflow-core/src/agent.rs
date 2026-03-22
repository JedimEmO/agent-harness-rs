use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::error::AgentError;
use crate::event::AgentEvent;
use crate::provider::ToolDefinition;
use crate::tool::{AgentTool, ToolExecResult, ToolPermission};

/// Core agent trait for composability.
///
/// An agent can process input and produce output. Agents can be nested —
/// a "supervisor" agent can delegate to "worker" agents by wrapping them
/// as tools via [`SubAgentTool`].
#[async_trait]
pub trait Agent: Send + Sync {
    /// A short name identifying this agent.
    fn name(&self) -> &str;

    /// Human-readable description of this agent's capabilities.
    fn description(&self) -> &str;

    /// Run the agent with the given input and return a text result.
    ///
    /// The `scope_id` groups related data (sessions, memories, tasks).
    /// Events are optionally sent to `event_tx` for real-time UI updates.
    async fn run(
        &self,
        input: &str,
        scope_id: &str,
        event_tx: Option<mpsc::Sender<AgentEvent>>,
    ) -> Result<String, AgentError>;
}

/// Wraps an [`Agent`] as an [`AgentTool`], enabling agent-to-agent delegation.
///
/// When this tool is called by a parent agent, it runs the sub-agent with
/// the provided input and returns the sub-agent's output as the tool result.
///
/// # Example
///
/// ```rust,no_run
/// use ironflow_core::*;
/// use std::sync::Arc;
///
/// // Create a sub-agent (e.g., a specialist for code review)
/// // let code_reviewer: Arc<dyn Agent> = ...;
///
/// // Wrap it as a tool and register with the parent agent
/// // let tool = SubAgentTool::new(code_reviewer);
/// // registry.register(Arc::new(tool)).await;
/// ```
pub struct SubAgentTool {
    agent: std::sync::Arc<dyn Agent>,
    parameter_description: String,
}

impl SubAgentTool {
    pub fn new(agent: std::sync::Arc<dyn Agent>) -> Self {
        Self {
            parameter_description: format!(
                "The input/instructions to send to the '{}' agent",
                agent.name()
            ),
            agent,
        }
    }

    /// Override the default parameter description for the "input" field.
    pub fn with_parameter_description(mut self, desc: impl Into<String>) -> Self {
        self.parameter_description = desc.into();
        self
    }
}

#[async_trait]
impl AgentTool for SubAgentTool {
    fn name(&self) -> &str {
        self.agent.name()
    }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.agent.name().to_string(),
            description: self.agent.description().to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": self.parameter_description
                    }
                },
                "required": ["input"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission {
        ToolPermission::AutoExecute
    }

    async fn execute(
        &self,
        scope_id: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecResult, AgentError> {
        let input = arguments
            .get("input")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let result = self.agent.run(input, scope_id, None).await?;
        Ok(ToolExecResult::Completed(serde_json::Value::String(result)))
    }
}
