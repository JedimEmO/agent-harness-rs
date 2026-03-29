//! # agent-harness-tools-interaction
//!
//! User interaction tools for agent-harness agents.
//!
//! Provides four tools that enable agents to communicate with users:
//! - [`AskUserTool`] — ask the user a free-form question
//! - [`OfferOptionsTool`] — present options for the user to choose from
//! - [`ConfirmActionTool`] — ask for confirmation before a significant action
//! - [`ShowPlanTool`] — present a multi-step plan for approval

use async_trait::async_trait;
use agent_harness_core::{
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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::AgentTool;

    #[tokio::test]
    async fn ask_user_basic() {
        let tool = AskUserTool;
        let result = tool.execute("s", serde_json::json!({"question": "what color?"})).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::AskUser { question, context }) => {
                assert_eq!(question, "what color?");
                assert!(context.is_none());
            }
            other => panic!("expected NeedsInteraction(AskUser), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn ask_user_with_context() {
        let tool = AskUserTool;
        let result = tool.execute("s", serde_json::json!({
            "question": "what color?",
            "context": "for the background"
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::AskUser { question, context }) => {
                assert_eq!(question, "what color?");
                assert_eq!(context, Some("for the background".to_string()));
            }
            other => panic!("expected NeedsInteraction(AskUser), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn ask_user_missing_question() {
        let tool = AskUserTool;
        let result = tool.execute("s", serde_json::json!({})).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::AskUser { question, .. }) => {
                assert_eq!(question, "");
            }
            other => panic!("expected NeedsInteraction(AskUser), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn offer_options_basic() {
        let tool = OfferOptionsTool;
        let result = tool.execute("s", serde_json::json!({
            "question": "pick one",
            "options": [
                {"id": "a", "label": "Option A"},
                {"id": "b", "label": "Option B"}
            ]
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::OfferOptions { question, options, allow_multiple }) => {
                assert_eq!(question, "pick one");
                assert_eq!(options.len(), 2);
                assert_eq!(options[0].id, "a");
                assert_eq!(options[1].label, "Option B");
                assert!(!allow_multiple);
            }
            other => panic!("expected NeedsInteraction(OfferOptions), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn offer_options_allow_multiple() {
        let tool = OfferOptionsTool;
        let result = tool.execute("s", serde_json::json!({
            "question": "pick",
            "options": [{"id": "a", "label": "A"}],
            "allow_multiple": true
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::OfferOptions { allow_multiple, .. }) => {
                assert!(allow_multiple);
            }
            other => panic!("expected NeedsInteraction(OfferOptions), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn offer_options_empty_options() {
        let tool = OfferOptionsTool;
        let result = tool.execute("s", serde_json::json!({
            "question": "pick",
            "options": []
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::OfferOptions { options, .. }) => {
                assert!(options.is_empty());
            }
            other => panic!("expected NeedsInteraction(OfferOptions), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn offer_options_malformed_options() {
        let tool = OfferOptionsTool;
        let result = tool.execute("s", serde_json::json!({
            "question": "pick",
            "options": "not an array"
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::OfferOptions { options, .. }) => {
                assert!(options.is_empty());
            }
            other => panic!("expected NeedsInteraction(OfferOptions), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn confirm_action_basic() {
        let tool = ConfirmActionTool;
        let result = tool.execute("s", serde_json::json!({
            "description": "delete everything"
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::ConfirmAction { description, details }) => {
                assert_eq!(description, "delete everything");
                assert!(details.is_none());
            }
            other => panic!("expected NeedsInteraction(ConfirmAction), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn confirm_action_with_details() {
        let tool = ConfirmActionTool;
        let result = tool.execute("s", serde_json::json!({
            "description": "delete everything",
            "details": "this is irreversible"
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::ConfirmAction { description, details }) => {
                assert_eq!(description, "delete everything");
                assert_eq!(details, Some("this is irreversible".to_string()));
            }
            other => panic!("expected NeedsInteraction(ConfirmAction), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn show_plan_basic() {
        let tool = ShowPlanTool;
        let result = tool.execute("s", serde_json::json!({
            "title": "Deploy plan",
            "steps": [
                {"description": "Build"},
                {"description": "Test", "tool_name": "run_tests"}
            ]
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::ShowPlan { title, steps }) => {
                assert_eq!(title, "Deploy plan");
                assert_eq!(steps.len(), 2);
                assert_eq!(steps[0].description, "Build");
                assert!(steps[0].tool_name.is_none());
                assert_eq!(steps[1].tool_name, Some("run_tests".to_string()));
            }
            other => panic!("expected NeedsInteraction(ShowPlan), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn show_plan_empty_steps() {
        let tool = ShowPlanTool;
        let result = tool.execute("s", serde_json::json!({
            "title": "Empty plan",
            "steps": []
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::ShowPlan { steps, .. }) => {
                assert!(steps.is_empty());
            }
            other => panic!("expected NeedsInteraction(ShowPlan), got {:?}", other),
        }
    }

    #[tokio::test]
    async fn show_plan_malformed_steps() {
        let tool = ShowPlanTool;
        let result = tool.execute("s", serde_json::json!({
            "title": "Bad plan",
            "steps": "not an array"
        })).await.unwrap();
        match result {
            ToolExecResult::NeedsInteraction(InteractionRequest::ShowPlan { steps, .. }) => {
                assert!(steps.is_empty());
            }
            other => panic!("expected NeedsInteraction(ShowPlan), got {:?}", other),
        }
    }

    #[test]
    fn all_tools_have_correct_names() {
        let tools: Vec<Box<dyn AgentTool>> = vec![
            Box::new(AskUserTool),
            Box::new(OfferOptionsTool),
            Box::new(ConfirmActionTool),
            Box::new(ShowPlanTool),
        ];
        for tool in &tools {
            assert_eq!(tool.name(), tool.definition().name);
        }
    }

    #[test]
    fn all_tools_are_auto_execute() {
        let tools: Vec<Box<dyn AgentTool>> = vec![
            Box::new(AskUserTool),
            Box::new(OfferOptionsTool),
            Box::new(ConfirmActionTool),
            Box::new(ShowPlanTool),
        ];
        for tool in &tools {
            assert_eq!(tool.permission(), ToolPermission::AutoExecute, "tool {} should be AutoExecute", tool.name());
        }
    }
}
