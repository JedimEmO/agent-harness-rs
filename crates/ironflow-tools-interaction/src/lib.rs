//! # ironflow-tools-interaction
//!
//! User interaction tools for ironflow agents.
//!
//! Provides four tools that enable agents to communicate with users:
//! - [`AskUserTool`] — ask the user a free-form question
//! - [`OfferOptionsTool`] — present options for the user to choose from
//! - [`ConfirmActionTool`] — ask for confirmation before a significant action
//! - [`ShowPlanTool`] — present a multi-step plan for approval

use async_trait::async_trait;
use ironflow_core::{
    AgentError, AgentTool, InteractionOption, InteractionRequest, PlanStep, ToolDefinition,
    ToolExecResult, ToolPermission,
};

// --- Ask User ---

pub struct AskUserTool;

#[async_trait]
impl AgentTool for AskUserTool {
    fn name(&self) -> &str { "ask_user" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "ask_user".to_string(),
            description: "Ask the user a question and wait for their response. Use this when you need clarification or input.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "The question to ask the user" },
                    "context": { "type": "string", "description": "Optional context to help the user understand why you're asking" }
                },
                "required": ["question"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(&self, _scope_id: &str, arguments: serde_json::Value) -> Result<ToolExecResult, AgentError> {
        let question = arguments.get("question").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let context = arguments.get("context").and_then(|v| v.as_str()).map(String::from);
        Ok(ToolExecResult::NeedsInteraction(InteractionRequest::AskUser { question, context }))
    }
}

// --- Offer Options ---

pub struct OfferOptionsTool;

#[async_trait]
impl AgentTool for OfferOptionsTool {
    fn name(&self) -> &str { "offer_options" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "offer_options".to_string(),
            description: "Present the user with a set of options to choose from.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "question": { "type": "string", "description": "The question or prompt" },
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "label": { "type": "string" },
                                "description": { "type": "string" }
                            },
                            "required": ["id", "label"]
                        },
                        "description": "The options to present"
                    },
                    "allow_multiple": { "type": "boolean", "description": "Whether the user can select multiple options" }
                },
                "required": ["question", "options"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(&self, _scope_id: &str, arguments: serde_json::Value) -> Result<ToolExecResult, AgentError> {
        let question = arguments.get("question").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let allow_multiple = arguments.get("allow_multiple").and_then(|v| v.as_bool()).unwrap_or(false);
        let options: Vec<InteractionOption> = arguments.get("options")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(ToolExecResult::NeedsInteraction(InteractionRequest::OfferOptions {
            question,
            options,
            allow_multiple,
        }))
    }
}

// --- Confirm Action ---

pub struct ConfirmActionTool;

#[async_trait]
impl AgentTool for ConfirmActionTool {
    fn name(&self) -> &str { "confirm_action" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "confirm_action".to_string(),
            description: "Ask the user to confirm before proceeding with a significant action.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": { "type": "string", "description": "What action you want to take" },
                    "details": { "type": "string", "description": "Additional details about the action" }
                },
                "required": ["description"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(&self, _scope_id: &str, arguments: serde_json::Value) -> Result<ToolExecResult, AgentError> {
        let description = arguments.get("description").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let details = arguments.get("details").and_then(|v| v.as_str()).map(String::from);
        Ok(ToolExecResult::NeedsInteraction(InteractionRequest::ConfirmAction { description, details }))
    }
}

// --- Show Plan ---

pub struct ShowPlanTool;

#[async_trait]
impl AgentTool for ShowPlanTool {
    fn name(&self) -> &str { "show_plan" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "show_plan".to_string(),
            description: "Present a detailed execution plan to the user for approval before carrying out a complex multi-step task.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "title": { "type": "string", "description": "A short title for the plan" },
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "description": { "type": "string", "description": "What this step will do" },
                                "tool_name": { "type": "string", "description": "The tool that will be used for this step, if applicable" }
                            },
                            "required": ["description"]
                        },
                        "description": "The ordered steps in the plan"
                    }
                },
                "required": ["title", "steps"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(&self, _scope_id: &str, arguments: serde_json::Value) -> Result<ToolExecResult, AgentError> {
        let title = arguments.get("title").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let steps: Vec<PlanStep> = arguments.get("steps")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        Ok(ToolExecResult::NeedsInteraction(InteractionRequest::ShowPlan { title, steps }))
    }
}
