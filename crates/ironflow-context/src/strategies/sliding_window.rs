use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::error::ContextError;
use crate::strategy::ContextStrategy;
use crate::token::TokenCounter;
use crate::types::{BlockKind, ContextBlock, ContextReport};

/// Keeps only the N most recent message blocks.
///
/// Pinned blocks and `SystemPrompt` blocks are always retained regardless of
/// window size. Non-message blocks (e.g., `RetrievalChunk`, `ToolDefinition`)
/// are also retained — only message-role blocks count toward the window.
pub struct SlidingWindow {
    window_size: usize,
}

impl SlidingWindow {
    pub fn new(window_size: usize) -> Self {
        Self { window_size }
    }
}

fn is_message_block(block: &ContextBlock) -> bool {
    matches!(
        block.kind,
        BlockKind::UserMessage
            | BlockKind::AssistantMessage
            | BlockKind::AssistantToolCalls
            | BlockKind::ToolResults
    )
}

#[async_trait]
impl ContextStrategy for SlidingWindow {
    fn name(&self) -> &str {
        "sliding_window"
    }

    async fn apply(
        &self,
        mut blocks: Vec<ContextBlock>,
        _token_budget: usize,
        counter: &dyn TokenCounter,
    ) -> Result<(Vec<ContextBlock>, ContextReport), ContextError> {
        counter.cache_blocks(&mut blocks);

        let original_count = blocks.len();
        let original_tokens: usize = blocks.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        let message_indices: Vec<usize> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| is_message_block(b) && !b.pinned)
            .map(|(i, _)| i)
            .collect();

        if message_indices.len() <= self.window_size {
            return Ok((blocks, ContextReport::no_change(original_count, original_tokens)));
        }

        let cutoff = message_indices.len() - self.window_size;
        let evict_indices: Vec<usize> = message_indices[..cutoff].to_vec();

        // Pre-build group index for O(1) group member lookups
        let group_index = build_group_index(&blocks);

        let mut evict_set: HashSet<usize> = evict_indices.iter().copied().collect();
        for &idx in &evict_indices {
            if let Some(ref group) = blocks[idx].group {
                if let Some(members) = group_index.get(group.as_str()) {
                    evict_set.extend(members);
                }
            }
        }

        let evicted_ids: Vec<String> = evict_set.iter().map(|&i| blocks[i].id.clone()).collect();

        let result: Vec<ContextBlock> = blocks
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !evict_set.contains(i))
            .map(|(_, b)| b)
            .collect();

        let final_tokens: usize = result.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        Ok((
            result,
            ContextReport {
                original_blocks: original_count,
                final_blocks: original_count - evict_set.len(),
                original_tokens,
                final_tokens,
                evicted_block_ids: evicted_ids,
                strategy_logs: vec![format!(
                    "sliding_window({}): kept {} of {} message blocks",
                    self.window_size,
                    self.window_size,
                    message_indices.len(),
                )],
                ..Default::default()
            },
        ))
    }
}

/// Build a map from group ID to the set of block indices in that group.
fn build_group_index(blocks: &[ContextBlock]) -> HashMap<&str, Vec<usize>> {
    let mut index: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, block) in blocks.iter().enumerate() {
        if let Some(ref g) = block.group {
            index.entry(g.as_str()).or_default().push(i);
        }
    }
    index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharEstimateCounter;

    fn msg(id: &str, text: &str) -> ContextBlock {
        ContextBlock::new(id, BlockKind::UserMessage, text)
    }

    fn sys(id: &str, text: &str) -> ContextBlock {
        ContextBlock::new(id, BlockKind::SystemPrompt, text).pinned()
    }

    #[tokio::test]
    async fn keeps_n_most_recent_messages() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("1", "a"),
            msg("2", "b"),
            msg("3", "c"),
            msg("4", "d"),
        ];
        let (result, report) = SlidingWindow::new(2)
            .apply(blocks, 1000, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "3");
        assert_eq!(result[1].id, "4");
        assert!(report.evicted_block_ids.contains(&"1".to_string()));
        assert!(report.evicted_block_ids.contains(&"2".to_string()));
    }

    #[tokio::test]
    async fn retains_system_and_pinned() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            sys("sys", "system prompt"),
            msg("1", "a"),
            msg("2", "b"),
            msg("3", "c"),
        ];
        let (result, _) = SlidingWindow::new(2)
            .apply(blocks, 1000, &counter)
            .await
            .unwrap();
        assert!(result.iter().any(|b| b.id == "sys"));
        assert!(result.iter().any(|b| b.id == "2"));
        assert!(result.iter().any(|b| b.id == "3"));
        assert!(!result.iter().any(|b| b.id == "1"));
    }

    #[tokio::test]
    async fn retains_non_message_blocks() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            ContextBlock::new("rag", BlockKind::RetrievalChunk, "context"),
            msg("1", "a"),
            msg("2", "b"),
            msg("3", "c"),
        ];
        let (result, _) = SlidingWindow::new(2)
            .apply(blocks, 1000, &counter)
            .await
            .unwrap();
        assert!(result.iter().any(|b| b.id == "rag"));
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn under_window_keeps_all() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![msg("1", "a"), msg("2", "b")];
        let (result, _) = SlidingWindow::new(5)
            .apply(blocks, 1000, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
    }
}
