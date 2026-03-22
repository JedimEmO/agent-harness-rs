use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InteractionRequest {
    AskUser {
        question: String,
        context: Option<String>,
    },
    OfferOptions {
        question: String,
        options: Vec<InteractionOption>,
        allow_multiple: bool,
    },
    ConfirmAction {
        description: String,
        details: Option<String>,
    },
    ShowPlan {
        title: String,
        steps: Vec<PlanStep>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionOption {
    pub id: String,
    pub label: String,
    pub description: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub description: String,
    pub tool_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum InteractionResponse {
    Text { text: String },
    SelectedOptions { ids: Vec<String> },
    Confirmed { approved: bool },
    PlanApproved { approved: bool, modifications: Option<String> },
}
