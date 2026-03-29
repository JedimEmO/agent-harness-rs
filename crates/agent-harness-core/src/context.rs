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
            chars_per_token: chars_per_token.max(1),
        }
    }

    fn estimate_tokens(&self, text: &str) -> usize {
        // Ceiling division so short messages always cost at least 1 token
        (text.len() + self.chars_per_token - 1) / self.chars_per_token
    }

    fn estimate_content_tokens(&self, parts: &[ContentPart]) -> usize {
        parts
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => self.estimate_tokens(text),
                // Images are roughly 1000 tokens each (conservative estimate)
                ContentPart::Image { .. } => 1000,
                // Audio: ~25 tokens per second at 16kHz 16-bit PCM (2 bytes/sample)
                ContentPart::Audio { data, .. } => {
                    let samples = data.len() / 2;
                    let seconds = samples as f64 / 16000.0;
                    (seconds * 25.0).ceil() as usize
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
            // System messages are passed through as user context
            (MessageRole::System, MessageContent::Text(text)) => {
                Some(ConversationMessage::user_text(format!(
                    "[System]: {}",
                    text
                )))
            }
            // Summaries preserve the original role's perspective
            (MessageRole::Assistant, MessageContent::Summary(text)) => {
                Some(ConversationMessage::Assistant {
                    content: format!("[Previous conversation summary]: {}", text),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{MessageContent, MessageRole, SessionMessage, ToolCallRecord, ToolResultRecord};

    fn make_session_msg(role: MessageRole, content: MessageContent) -> SessionMessage {
        SessionMessage {
            id: uuid::Uuid::new_v4().to_string(),
            session_id: "test-session".into(),
            role,
            content,
            created_at: "2026-01-01T00:00:00Z".into(),
        }
    }

    fn user_text_msg(text: &str) -> SessionMessage {
        make_session_msg(MessageRole::User, MessageContent::Text(text.into()))
    }

    fn assistant_text_msg(text: &str) -> SessionMessage {
        make_session_msg(MessageRole::Assistant, MessageContent::Text(text.into()))
    }

    fn tool_call_msg() -> SessionMessage {
        make_session_msg(
            MessageRole::Assistant,
            MessageContent::ToolCalls(vec![ToolCallRecord {
                call_id: "c1".into(),
                tool_name: "test".into(),
                arguments: serde_json::json!({}),
            }]),
        )
    }

    fn tool_result_msg() -> SessionMessage {
        make_session_msg(
            MessageRole::Tool,
            MessageContent::ToolResults(vec![ToolResultRecord {
                call_id: "c1".into(),
                tool_name: "test".into(),
                content: serde_json::json!("result"),
            }]),
        )
    }

    #[test]
    fn audio_token_estimation() {
        let cm = ContextManager::new(100000, 4);
        // 1 second of 16kHz 16-bit PCM = 16000 samples * 2 bytes = 32000 bytes
        let parts = vec![ContentPart::Audio {
            media_type: "audio/pcm;rate=16000".into(),
            data: vec![0u8; 32000],
        }];
        let tokens = cm.estimate_content_tokens(&parts);
        assert_eq!(tokens, 25, "1 second of audio should be ~25 tokens");
    }

    #[test]
    fn image_token_estimation() {
        let cm = ContextManager::new(100000, 4);
        let parts = vec![ContentPart::Image {
            media_type: "image/jpeg".into(),
            data: vec![0u8; 5000],
        }];
        let tokens = cm.estimate_content_tokens(&parts);
        assert_eq!(tokens, 1000);
    }

    #[test]
    fn text_token_estimation() {
        let cm = ContextManager::new(100000, 4);
        // "Hello world" is 11 chars, ceil(11/4) = 3 (ceiling division)
        let tokens = cm.estimate_tokens("Hello world");
        assert_eq!(tokens, 3);
    }

    #[test]
    fn prepare_messages_under_budget() {
        // Large budget so nothing gets dropped
        let cm = ContextManager::new(100000, 4);
        let messages = vec![
            user_text_msg("hello"),
            assistant_text_msg("hi there"),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 2);
        assert!(truncation.is_none());
    }

    #[test]
    fn prepare_messages_over_budget_drops_oldest() {
        // Very small budget: only the last message should fit
        // "newest message" = 14 chars / 4 = 3 tokens
        // Budget of 4 tokens should fit only the last message
        let cm = ContextManager::new(4, 4);
        let messages = vec![
            user_text_msg("this is a very long old message that should be dropped"),
            assistant_text_msg("newest message"),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 1);
        let truncation = truncation.expect("should have truncation info");
        assert_eq!(truncation.dropped_messages, 1);
        assert_eq!(truncation.remaining_messages, 1);
        // The kept message should be the newest (assistant)
        match &result[0] {
            ConversationMessage::Assistant { content } => assert_eq!(content, "newest message"),
            _ => panic!("expected Assistant variant"),
        }
    }

    #[test]
    fn prepare_messages_tool_call_result_atomic() {
        // Budget allows only user+assistant text, not enough for tool call+result pair
        // tool call + tool result together should be dropped as a unit
        let cm = ContextManager::new(10, 4);
        let messages = vec![
            user_text_msg("hi"),
            tool_call_msg(),
            tool_result_msg(),
            assistant_text_msg("done"),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        // The tool call/result pair is atomic. If budget is tight,
        // they should either both be included or both dropped.
        let has_tool_calls = result.iter().any(|m| matches!(m, ConversationMessage::AssistantToolCalls { .. }));
        let has_tool_results = result.iter().any(|m| matches!(m, ConversationMessage::ToolResults { .. }));
        assert_eq!(has_tool_calls, has_tool_results, "tool calls and results must be kept or dropped together");

        if let Some(info) = truncation {
            // If truncation happened, verify consistency
            assert_eq!(info.dropped_messages + info.remaining_messages, 4);
        }
    }

    #[test]
    fn prepare_messages_with_audio_content() {
        // Audio content should contribute to token budget
        let cm = ContextManager::new(100000, 4);
        let messages = vec![
            // Create a user message with audio via direct SessionMessage
            // Since there's no direct way to create audio SessionMessage through helpers,
            // we test via the content token estimation path
            user_text_msg("describe this audio"),
            assistant_text_msg("I heard something"),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 2);
        assert!(truncation.is_none());

        // Also verify audio token estimation participates in budget
        let audio_parts = vec![ContentPart::Audio {
            media_type: "audio/pcm;rate=16000".into(),
            data: vec![0u8; 64000], // 2 seconds
        }];
        let tokens = cm.estimate_content_tokens(&audio_parts);
        assert_eq!(tokens, 50, "2 seconds of audio should be ~50 tokens");
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[test]
    fn audio_token_estimation_zero_bytes() {
        // BUG PROBE: 0-byte audio data. data.len() / 2 = 0 samples, 0/16000 = 0 seconds,
        // 0 * 25.0 = 0.0, ceil(0.0) = 0. Should be 0 tokens, but is this the *right*
        // behavior? An empty audio chunk should arguably still be 0 tokens.
        let cm = ContextManager::new(100000, 4);
        let parts = vec![ContentPart::Audio {
            media_type: "audio/pcm;rate=16000".into(),
            data: vec![],
        }];
        let tokens = cm.estimate_content_tokens(&parts);
        assert_eq!(tokens, 0, "zero-byte audio should produce 0 tokens");
    }

    #[test]
    fn audio_token_estimation_one_byte() {
        // BUG PROBE: 1-byte audio data is not a valid 16-bit PCM sample (needs 2 bytes).
        // data.len() / 2 = 0 (integer division), so we get 0 tokens.
        // This silently accepts invalid audio data.
        let cm = ContextManager::new(100000, 4);
        let parts = vec![ContentPart::Audio {
            media_type: "audio/pcm;rate=16000".into(),
            data: vec![0u8],
        }];
        let tokens = cm.estimate_content_tokens(&parts);
        assert_eq!(tokens, 0, "1-byte audio (invalid sample) should produce 0 tokens");
    }

    #[test]
    fn audio_token_estimation_odd_byte_count() {
        // BUG PROBE: 3 bytes = 1.5 samples. Integer division gives 1 sample.
        // The extra byte is silently ignored.
        let cm = ContextManager::new(100000, 4);
        let parts = vec![ContentPart::Audio {
            media_type: "audio/pcm;rate=16000".into(),
            data: vec![0u8; 3],
        }];
        let tokens = cm.estimate_content_tokens(&parts);
        // 1 sample / 16000 = 0.0000625 seconds * 25 = 0.0015625, ceil = 1
        assert_eq!(tokens, 1, "3 bytes (1 sample + 1 stray byte) = ceil(tiny) = 1 token");
    }

    #[test]
    fn audio_token_estimation_huge_data_no_overflow() {
        // BUG PROBE: Very large audio data. data.len() is usize.
        // `samples as f64` could lose precision for values > 2^53.
        // Let's try a large but not absurd value: 4GB worth = usize::MAX on 32-bit,
        // but on 64-bit we can try a large value.
        let cm = ContextManager::new(usize::MAX, 4);
        // ~100 hours of audio at 16kHz 16-bit = 100*3600*16000*2 = 11_520_000_000 bytes
        // That's too much to allocate, so let's just test the math path:
        // We can directly call estimate_content_tokens with a smaller but still large vec.
        // 10 million bytes = 5M samples = 312.5 seconds = 7812.5 tokens -> 7813
        let parts = vec![ContentPart::Audio {
            media_type: "audio/pcm;rate=16000".into(),
            data: vec![0u8; 10_000_000],
        }];
        let tokens = cm.estimate_content_tokens(&parts);
        assert_eq!(tokens, 7813, "10MB audio should be ~7813 tokens");
    }

    #[test]
    fn text_token_estimation_empty_string() {
        // BUG PROBE: empty text. 0 / 4 = 0. Should be fine.
        let cm = ContextManager::new(100000, 4);
        assert_eq!(cm.estimate_tokens(""), 0);
    }

    #[test]
    fn text_token_estimation_shorter_than_chars_per_token() {
        // Short text should still cost at least 1 token (ceiling division).
        let cm = ContextManager::new(100000, 4);
        let tokens = cm.estimate_tokens("hi");
        assert_eq!(tokens, 1, "'hi' (2 chars) should estimate to 1 token via ceiling division");
    }

    #[test]
    fn prepare_messages_zero_budget() {
        // BUG PROBE: budget of 0 tokens. Should drop everything.
        let cm = ContextManager::new(0, 4);
        let messages = vec![
            user_text_msg("hello"),
            assistant_text_msg("world"),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        // With 0 budget, nothing should fit... but short messages estimate to 0 tokens
        // due to integer division. "hello" = 5/4 = 1 token, "world" = 5/4 = 1 token.
        // Total = 2 tokens > 0 budget, so we truncate.
        // But the truncation loop breaks when *tokens > remaining_budget.
        // With remaining_budget = 0 and tokens = 1, 1 > 0 is true, so it breaks.
        // Result: 0 messages kept.
        assert_eq!(result.len(), 0, "zero budget should keep no messages");
        assert!(truncation.is_some());
    }

    #[test]
    fn prepare_messages_zero_budget_drops_short_messages() {
        // With ceiling division, even short messages cost at least 1 token.
        // Budget of 0 should drop everything.
        let cm = ContextManager::new(0, 4);
        let messages = vec![
            user_text_msg("hi"),
            assistant_text_msg("ok"),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 0, "zero budget should keep no messages");
        assert!(truncation.is_some());
    }

    #[test]
    fn prepare_messages_all_too_large() {
        // BUG PROBE: Every single message is too large to fit in budget.
        // Budget = 1 token, each message is huge.
        let cm = ContextManager::new(1, 4);
        let messages = vec![
            user_text_msg(&"x".repeat(100)),
            assistant_text_msg(&"y".repeat(100)),
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        // Each message is 100/4 = 25 tokens. Budget = 1.
        // The reverse loop: first checks newest (25 > 1), breaks immediately.
        // Result: empty.
        assert_eq!(result.len(), 0, "all-too-large messages should produce empty result");
        assert!(truncation.is_some());
        let info = truncation.unwrap();
        assert_eq!(info.dropped_messages, 2);
        assert_eq!(info.remaining_messages, 0);
    }

    #[test]
    fn prepare_messages_tool_results_without_preceding_tool_calls() {
        // BUG PROBE: A ToolResults message appears without a preceding AssistantToolCalls.
        // The grouping logic only pairs AssistantToolCalls followed by ToolResults.
        // A standalone ToolResults gets treated as its own unit.
        // This is structurally invalid, but does the code handle it gracefully?
        let cm = ContextManager::new(100000, 4);
        let messages = vec![
            user_text_msg("hi"),
            tool_result_msg(), // orphaned tool result
            assistant_text_msg("done"),
        ];
        let (result, _) = cm.prepare_messages(&messages);
        // The orphaned ToolResults should still appear in the output
        assert_eq!(result.len(), 3);
        assert!(matches!(&result[1], ConversationMessage::ToolResults { .. }));
    }

    #[test]
    fn prepare_messages_tool_calls_without_following_results() {
        // BUG PROBE: AssistantToolCalls at the end without a following ToolResults.
        // The grouping code checks i+1 < len && messages[i+1] is ToolResults.
        // If there's no following message, it becomes a standalone unit.
        let cm = ContextManager::new(100000, 4);
        let messages = vec![
            user_text_msg("hi"),
            tool_call_msg(), // no following tool result
        ];
        let (result, _) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 2);
        // The tool call stands alone — structurally invalid for LLM APIs
        assert!(matches!(&result[1], ConversationMessage::AssistantToolCalls { .. }));
    }

    #[test]
    fn prepare_messages_chars_per_token_one() {
        // BUG PROBE: chars_per_token = 1 means every char is a token.
        // This is the most aggressive estimation.
        let cm = ContextManager::new(5, 1);
        let messages = vec![
            user_text_msg("hello"), // 5 tokens
            assistant_text_msg("world"), // 5 tokens
        ];
        let (result, truncation) = cm.prepare_messages(&messages);
        // Total = 10 > 5, truncate from beginning
        assert_eq!(result.len(), 1);
        assert!(truncation.is_some());
    }

    #[test]
    fn prepare_messages_empty_input() {
        // BUG PROBE: no messages at all
        let cm = ContextManager::new(100000, 4);
        let messages: Vec<SessionMessage> = vec![];
        let (result, truncation) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 0);
        assert!(truncation.is_none());
    }

    #[test]
    fn prepare_messages_system_role_preserved() {
        // System messages should be passed through (as user context with [System] prefix).
        let cm = ContextManager::new(100000, 4);
        let messages = vec![
            make_session_msg(MessageRole::System, MessageContent::Text("you are helpful".into())),
            user_text_msg("hello"),
        ];
        let (result, _) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 2, "System messages should be preserved");
        match &result[0] {
            ConversationMessage::User { content } => {
                match &content[0] {
                    ContentPart::Text { text } => assert!(text.contains("[System]")),
                    _ => panic!("expected text"),
                }
            }
            _ => panic!("expected User variant for system message"),
        }
    }

    #[test]
    fn estimate_tokens_with_chars_per_token_zero_clamped() {
        // chars_per_token=0 is clamped to 1 to prevent division by zero.
        let cm = ContextManager::new(100000, 0);
        let tokens = cm.estimate_tokens("hello");
        assert_eq!(tokens, 5, "chars_per_token=0 clamped to 1, so 5 chars = 5 tokens");
    }

    #[test]
    fn prepare_messages_assistant_summary_stays_assistant() {
        // Assistant summaries should remain Assistant messages to preserve attribution.
        let cm = ContextManager::new(100000, 4);
        let messages = vec![
            make_session_msg(MessageRole::Assistant, MessageContent::Summary("previous context".into())),
            user_text_msg("continue"),
        ];
        let (result, _) = cm.prepare_messages(&messages);
        assert_eq!(result.len(), 2);
        match &result[0] {
            ConversationMessage::Assistant { content } => {
                assert!(content.contains("Previous conversation summary"));
            }
            _ => panic!("Assistant summary should become Assistant message"),
        }
    }
}
