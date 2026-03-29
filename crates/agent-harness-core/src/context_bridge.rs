//! Bridge between `agent-harness-context` block types and agent-harness-core's session/provider types.
//!
//! Only compiled when the `context-pipeline` feature is enabled.

use agent_harness_context::{BlockKind, ContextBlock, Priority};

use crate::provider::{ConversationMessage, ToolCall, ToolResult};
use crate::session::{MessageContent, MessageRole, SessionMessage, ToolCallRecord, ToolResultRecord};

/// Convert `SessionMessage`s into `ContextBlock`s.
///
/// Tool-call / tool-result pairs are grouped atomically.
pub(crate) fn session_messages_to_blocks(messages: &[SessionMessage]) -> Vec<ContextBlock> {
    let mut blocks = Vec::with_capacity(messages.len());
    let mut i = 0;

    while i < messages.len() {
        let msg = &messages[i];
        match (&msg.role, &msg.content) {
            (MessageRole::User, MessageContent::Text(text)) => {
                blocks.push(ContextBlock::new(&msg.id, BlockKind::UserMessage, text.clone()));
                i += 1;
            }
            (MessageRole::Assistant, MessageContent::Text(text)) => {
                blocks.push(ContextBlock::new(
                    &msg.id,
                    BlockKind::AssistantMessage,
                    text.clone(),
                ));
                i += 1;
            }
            (MessageRole::Assistant, MessageContent::ToolCalls(calls)) => {
                let group_id = format!("tc-{}", msg.id);
                let json = serde_json::to_string(calls).unwrap_or_default();
                let block = ContextBlock::new(&msg.id, BlockKind::AssistantToolCalls, json)
                    .with_group(&group_id);

                if i + 1 < messages.len() {
                    if let (MessageRole::Tool, MessageContent::ToolResults(results)) =
                        (&messages[i + 1].role, &messages[i + 1].content)
                    {
                        blocks.push(block);
                        let result_json = serde_json::to_string(results).unwrap_or_default();
                        blocks.push(
                            ContextBlock::new(
                                &messages[i + 1].id,
                                BlockKind::ToolResults,
                                result_json,
                            )
                            .with_group(&group_id),
                        );
                        i += 2;
                        continue;
                    }
                }
                blocks.push(block);
                i += 1;
            }
            (MessageRole::Tool, MessageContent::ToolResults(results)) => {
                let json = serde_json::to_string(results).unwrap_or_default();
                blocks.push(ContextBlock::new(&msg.id, BlockKind::ToolResults, json));
                i += 1;
            }
            (_, MessageContent::Summary(text)) => {
                blocks.push(
                    ContextBlock::new(&msg.id, BlockKind::Summary, text.clone())
                        .with_priority(Priority::High),
                );
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }

    blocks
}

/// Convert processed `ContextBlock`s back to `ConversationMessage`s.
pub(crate) fn blocks_to_conversation(blocks: &[ContextBlock]) -> Vec<ConversationMessage> {
    blocks
        .iter()
        .filter_map(|block| match &block.kind {
            BlockKind::UserMessage => {
                Some(ConversationMessage::user_text(block.content.as_text()))
            }
            BlockKind::AssistantMessage => Some(ConversationMessage::Assistant {
                content: block.content.as_text(),
            }),
            BlockKind::AssistantToolCalls => {
                let text = block.content.as_text();
                let calls: Vec<ToolCallRecord> =
                    serde_json::from_str(&text).unwrap_or_default();
                let tool_calls = calls.iter().map(ToolCall::from).collect();
                Some(ConversationMessage::AssistantToolCalls { tool_calls })
            }
            BlockKind::ToolResults => {
                let text = block.content.as_text();
                let records: Vec<ToolResultRecord> =
                    serde_json::from_str(&text).unwrap_or_default();
                let results = records.iter().map(ToolResult::from).collect();
                Some(ConversationMessage::ToolResults { results })
            }
            BlockKind::Summary => Some(ConversationMessage::user_text(format!(
                "[Previous conversation summary]: {}",
                block.content.as_text()
            ))),
            BlockKind::UserProfile | BlockKind::FileContent | BlockKind::RetrievalChunk => {
                Some(ConversationMessage::user_text(block.content.as_text()))
            }
            _ => None,
        })
        .collect()
}
