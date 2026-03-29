use std::collections::HashSet;

use async_trait::async_trait;

use crate::error::ContextError;
use crate::strategy::ContextStrategy;
use crate::token::TokenCounter;
use crate::types::{BlockContent, BlockKind, ContextBlock, ContextReport, Priority};

/// Backend for producing summaries from evicted blocks.
///
/// Implement this with your LLM provider to enable context compression.
/// This trait keeps `agent-harness-context` decoupled from any specific provider.
#[async_trait]
pub trait Summarizer: Send + Sync {
    /// Summarize a list of text blocks into a single condensed string.
    async fn summarize(&self, texts: Vec<String>) -> Result<String, ContextError>;
}

/// Instead of dropping evicted blocks, summarizes them into a single `Summary`
/// block that preserves the gist of the conversation.
///
/// This strategy first identifies blocks that would be evicted (oldest,
/// non-pinned, lowest priority) and replaces them with a summary. The summary
/// block is inserted at the position of the first evicted block.
pub struct Summarize {
    summarizer: Box<dyn Summarizer>,
    /// Maximum tokens to evict before summarizing. If exceeded, the strategy
    /// summarizes in one batch. Default: usize::MAX (summarize everything).
    max_evict_tokens: usize,
}

impl Summarize {
    pub fn new(summarizer: impl Summarizer + 'static) -> Self {
        Self {
            summarizer: Box::new(summarizer),
            max_evict_tokens: usize::MAX,
        }
    }

    pub fn with_max_evict_tokens(mut self, max: usize) -> Self {
        self.max_evict_tokens = max;
        self
    }
}

#[async_trait]
impl ContextStrategy for Summarize {
    fn name(&self) -> &str {
        "summarize"
    }

    async fn apply(
        &self,
        mut blocks: Vec<ContextBlock>,
        token_budget: usize,
        counter: &dyn TokenCounter,
    ) -> Result<(Vec<ContextBlock>, ContextReport), ContextError> {
        counter.cache_blocks(&mut blocks);

        let original_count = blocks.len();
        let total_tokens: usize = blocks.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        if total_tokens <= token_budget {
            return Ok((blocks, ContextReport::no_change(original_count, total_tokens)));
        }

        let excess = total_tokens - token_budget;
        let mut evict_tokens = 0usize;
        let mut evict_indices = Vec::new();

        for (i, block) in blocks.iter().enumerate() {
            if evict_tokens >= excess || evict_tokens >= self.max_evict_tokens {
                break;
            }
            if block.pinned {
                continue;
            }
            evict_tokens += block.token_count.unwrap_or(0);
            evict_indices.push(i);
        }

        if evict_indices.is_empty() {
            return Ok((blocks, ContextReport::no_change(original_count, total_tokens)));
        }

        let texts: Vec<String> = evict_indices
            .iter()
            .map(|&i| blocks[i].content.as_text())
            .collect();

        let summarized_ids: Vec<String> =
            evict_indices.iter().map(|&i| blocks[i].id.clone()).collect();

        let summary_text = self.summarizer.summarize(texts).await?;
        let summary_tokens = counter.count(&summary_text);

        let summary_block = ContextBlock {
            id: format!("summary-{}", uuid::Uuid::new_v4()),
            kind: BlockKind::Summary,
            content: BlockContent::Text(summary_text),
            priority: Priority::High,
            pinned: false,
            token_count: Some(summary_tokens),
            metadata: Default::default(),
            group: None,
        };

        let insert_pos = evict_indices[0];
        let evict_set: HashSet<usize> = evict_indices.iter().copied().collect();
        let mut result: Vec<ContextBlock> = Vec::with_capacity(blocks.len());
        let mut inserted = false;

        for (i, block) in blocks.into_iter().enumerate() {
            if evict_set.contains(&i) {
                if !inserted && i == insert_pos {
                    result.push(summary_block.clone());
                    inserted = true;
                }
                continue;
            }
            result.push(block);
        }

        let final_blocks = result.len();
        let final_tokens: usize = result.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        Ok((
            result,
            ContextReport {
                original_blocks: original_count,
                final_blocks,
                original_tokens: total_tokens,
                final_tokens,
                evicted_block_ids: Vec::new(),
                summarized_block_ids: summarized_ids,
                strategy_logs: vec![format!(
                    "summarize: compressed {} blocks ({} tokens) into summary ({} tokens)",
                    evict_indices.len(),
                    evict_tokens,
                    summary_tokens,
                )],
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharEstimateCounter;

    struct MockSummarizer;

    #[async_trait]
    impl Summarizer for MockSummarizer {
        async fn summarize(&self, texts: Vec<String>) -> Result<String, ContextError> {
            Ok(format!("Summary of {} items", texts.len()))
        }
    }

    fn msg(id: &str, text: &str) -> ContextBlock {
        ContextBlock::new(id, BlockKind::UserMessage, text)
    }

    #[tokio::test]
    async fn summarizes_evicted_blocks() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("1", "aaaaaaaaaa"),
            msg("2", "bbbbbbbbbb"),
            msg("3", "cccc"),
        ];
        let (result, report) = Summarize::new(MockSummarizer)
            .apply(blocks, 15, &counter)
            .await
            .unwrap();

        assert!(report.summarized_block_ids.contains(&"1".to_string()));
        assert!(result.iter().any(|b| b.kind == BlockKind::Summary));
        assert!(result.iter().any(|b| b.id == "2"));
        assert!(result.iter().any(|b| b.id == "3"));
    }

    #[tokio::test]
    async fn pinned_blocks_not_summarized() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("pin", "aaaaaaaaaa").pinned(),
            msg("1", "bbbbbbbbbb"),
            msg("2", "cccc"),
        ];
        let (result, report) = Summarize::new(MockSummarizer)
            .apply(blocks, 15, &counter)
            .await
            .unwrap();
        assert!(result.iter().any(|b| b.id == "pin"));
        assert!(report.summarized_block_ids.contains(&"1".to_string()));
    }

    #[tokio::test]
    async fn under_budget_no_summarization() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![msg("1", "aa"), msg("2", "bb")];
        let (result, report) = Summarize::new(MockSummarizer)
            .apply(blocks, 100, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert!(report.summarized_block_ids.is_empty());
    }
}
