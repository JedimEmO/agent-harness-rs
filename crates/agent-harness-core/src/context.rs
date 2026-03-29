use crate::provider::{ContentPart, ConversationMessage, ToolCall, ToolResult};
use crate::session::{MessageContent, MessageRole, SessionMessage};

pub(crate) struct ContextManager {
    max_context_tokens: usize,
    chars_per_token: usize,
}

/// Information about a context truncation operation.
#[derive(Debug, Clone)]
pub(crate) struct TruncationInfo {
    pub dropped_messages: usize,
    pub remaining_messages: usize,
}

impl ContextManager {
    pub(crate) fn new(max_context_tokens: usize, chars_per_token: usize) -> Self {
        Self {
            max_context_tokens,
            chars_per_token,
        }
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        text.len() / self.chars_per_token
    }

    fn estimate_content_tokens(&self, parts: &[ContentPart]) -> usize {
        parts
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => self.estimate_tokens(text),
                // Images are roughly 1000 tokens each (conservative estimate)
                ContentPart::Image { data, .. } => {
                    let _ = data;
                    1000
                }
            })
            .sum()
    }

    fn estimate_value_tokens(&self, value: &serde_json::Value) -> usize {
        match value {
            serde_json::Value::String(s) => self.estimate_tokens(s),
            other => self.estimate_tokens(&other.to_string()),
        }
    }

    fn message_to_conversation(&self, msg: &SessionMessage) -> Option<ConversationMessage> {
        match (&msg.role, &msg.content) {
            (MessageRole::User, MessageContent::Text(text)) => {
                Some(ConversationMessage::user_text(text.clone()))
            }
            (MessageRole::Assistant, MessageContent::Text(text)) => {
                Some(ConversationMessage::Assistant {
                    content: text.clone(),
                })
            }
            (MessageRole::Assistant, MessageContent::ToolCalls(calls)) => {
                let tool_calls = calls.iter().map(ToolCall::from).collect();
                Some(ConversationMessage::AssistantToolCalls { tool_calls })
            }
            (MessageRole::Tool, MessageContent::ToolResults(results)) => {
                let tool_results = results.iter().map(ToolResult::from).collect();
                Some(ConversationMessage::ToolResults {
                    results: tool_results,
                })
            }
            (_, MessageContent::Summary(text)) => {
                Some(ConversationMessage::user_text(format!(
                    "[Previous conversation summary]: {}",
                    text
                )))
            }
            _ => None,
        }
    }

    fn estimate_message_tokens(&self, msg: &ConversationMessage) -> usize {
        match msg {
            ConversationMessage::User { content } => self.estimate_content_tokens(content),
            ConversationMessage::Assistant { content } => self.estimate_tokens(content),
            ConversationMessage::AssistantToolCalls { tool_calls } => tool_calls
                .iter()
                .map(|tc| {
                    self.estimate_tokens(&tc.name)
                        + self.estimate_tokens(&tc.arguments.to_string())
                })
                .sum(),
            ConversationMessage::ToolResults { results } => results
                .iter()
                .map(|r| self.estimate_value_tokens(&r.content))
                .sum(),
        }
    }

    /// Convert stored messages to conversation format, truncating from the beginning
    /// if total token count exceeds the limit.
    ///
    /// **Key fix:** Tool call/result pairs are treated as atomic units — if an
    /// `AssistantToolCalls` message is included, its following `ToolResults` message
    /// is always included too (and vice versa). This prevents structurally invalid
    /// conversations that would be rejected by LLM APIs.
    ///
    /// Returns the messages and optional truncation info.
    pub(crate) fn prepare_messages(
        &self,
        messages: &[SessionMessage],
    ) -> (Vec<ConversationMessage>, Option<TruncationInfo>) {
        // 1. Convert all messages
        let converted: Vec<ConversationMessage> = messages
            .iter()
            .filter_map(|msg| self.message_to_conversation(msg))
            .collect();

        // 2. Group into "units" — standalone messages or (ToolCalls, ToolResults) pairs
        let units = self.group_into_units(&converted);

        // 3. Estimate tokens per unit
        let unit_tokens: Vec<usize> = units
            .iter()
            .map(|unit| unit.iter().map(|msg| self.estimate_message_tokens(msg)).sum())
            .collect();
        let total_tokens: usize = unit_tokens.iter().sum();

        if total_tokens <= self.max_context_tokens {
            return (converted, None);
        }

        // 4. Truncate from the beginning, keeping most recent units
        let mut remaining_budget = self.max_context_tokens;
        let mut kept_units: Vec<&Vec<&ConversationMessage>> = Vec::new();

        for (unit, tokens) in units.iter().zip(unit_tokens.iter()).rev() {
            if *tokens > remaining_budget {
                break;
            }
            remaining_budget -= tokens;
            kept_units.push(unit);
        }

        kept_units.reverse();

        let result: Vec<ConversationMessage> = kept_units
            .into_iter()
            .flat_map(|unit| unit.iter().map(|msg| (*msg).clone()))
            .collect();

        let dropped = converted.len() - result.len();
        let info = TruncationInfo {
            dropped_messages: dropped,
            remaining_messages: result.len(),
        };

        (result, Some(info))
    }

    /// Group messages into atomic units. An `AssistantToolCalls` immediately
    /// followed by `ToolResults` forms a single unit.
    fn group_into_units<'a>(
        &self,
        messages: &'a [ConversationMessage],
    ) -> Vec<Vec<&'a ConversationMessage>> {
        let mut units: Vec<Vec<&'a ConversationMessage>> = Vec::new();
        let mut i = 0;

        while i < messages.len() {
            if matches!(&messages[i], ConversationMessage::AssistantToolCalls { .. })
                && i + 1 < messages.len()
                && matches!(&messages[i + 1], ConversationMessage::ToolResults { .. })
            {
                // Group tool call + result as one unit
                units.push(vec![&messages[i], &messages[i + 1]]);
                i += 2;
            } else {
                units.push(vec![&messages[i]]);
                i += 1;
            }
        }

        units
    }
}
