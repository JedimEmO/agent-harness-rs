use std::collections::{HashMap, HashSet};

use async_trait::async_trait;

use crate::error::ContextError;
use crate::strategy::ContextStrategy;
use crate::token::TokenCounter;
use crate::types::{ContextBlock, ContextReport};

/// Evicts the lowest-priority non-pinned blocks first until under token budget.
///
/// Within the same priority level, older blocks (earlier in the list) are
/// evicted first. Blocks sharing a `group` are treated atomically.
pub struct PriorityRetention;

impl PriorityRetention {
    pub fn new() -> Self {
        Self
    }
}

impl Default for PriorityRetention {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ContextStrategy for PriorityRetention {
    fn name(&self) -> &str {
        "priority_retention"
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

        // Sorted by priority ascending (Low first), then by index ascending (oldest first)
        let mut candidates: Vec<(usize, u8, usize)> = blocks
            .iter()
            .enumerate()
            .filter(|(_, b)| !b.pinned)
            .map(|(i, b)| (i, b.priority as u8, b.token_count.unwrap_or(0)))
            .collect();

        candidates.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

        // Pre-build group index for O(1) lookups
        let group_index: HashMap<&str, Vec<usize>> = {
            let mut idx: HashMap<&str, Vec<usize>> = HashMap::new();
            for (i, b) in blocks.iter().enumerate() {
                if let Some(ref g) = b.group {
                    if !b.pinned {
                        idx.entry(g.as_str()).or_default().push(i);
                    }
                }
            }
            idx
        };

        let mut current_tokens = total_tokens;
        let mut evict_set: HashSet<usize> = HashSet::new();
        let mut evicted_ids = Vec::new();

        for (idx, _, tokens) in &candidates {
            if current_tokens <= token_budget {
                break;
            }

            if let Some(ref group) = blocks[*idx].group {
                if let Some(members) = group_index.get(group.as_str()) {
                    let group_tokens: usize = members
                        .iter()
                        .filter(|i| !evict_set.contains(i))
                        .map(|&i| blocks[i].token_count.unwrap_or(0))
                        .sum();

                    for &gi in members {
                        if evict_set.insert(gi) {
                            evicted_ids.push(blocks[gi].id.clone());
                        }
                    }
                    current_tokens = current_tokens.saturating_sub(group_tokens);
                }
            } else if evict_set.insert(*idx) {
                evicted_ids.push(blocks[*idx].id.clone());
                current_tokens = current_tokens.saturating_sub(*tokens);
            }
        }

        let result: Vec<ContextBlock> = blocks
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !evict_set.contains(i))
            .map(|(_, b)| b)
            .collect();

        let final_blocks = result.len();
        let final_tokens: usize = result.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        Ok((
            result,
            ContextReport {
                original_blocks: original_count,
                final_blocks,
                original_tokens: total_tokens,
                final_tokens,
                evicted_block_ids: evicted_ids,
                strategy_logs: vec![format!(
                    "priority_retention: evicted {} blocks, {} -> {} tokens",
                    original_count - final_blocks,
                    total_tokens,
                    final_tokens,
                )],
                ..Default::default()
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharEstimateCounter;
    use crate::types::{BlockKind, Priority};

    fn msg(id: &str, text: &str, priority: Priority) -> ContextBlock {
        ContextBlock::new(id, BlockKind::UserMessage, text).with_priority(priority)
    }

    #[tokio::test]
    async fn evicts_lowest_priority_first() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("high", "aaaa", Priority::High),
            msg("low", "bbbb", Priority::Low),
            msg("normal", "cccc", Priority::Normal),
        ];
        let (result, report) = PriorityRetention::new()
            .apply(blocks, 8, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert!(report.evicted_block_ids.contains(&"low".to_string()));
        assert!(result.iter().any(|b| b.id == "high"));
        assert!(result.iter().any(|b| b.id == "normal"));
    }

    #[tokio::test]
    async fn pinned_never_evicted() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("pinned", "aaaa", Priority::Low).pinned(),
            msg("low", "bbbb", Priority::Low),
            msg("high", "cccc", Priority::High),
        ];
        let (result, _) = PriorityRetention::new()
            .apply(blocks, 8, &counter)
            .await
            .unwrap();
        assert!(result.iter().any(|b| b.id == "pinned"));
    }

    #[tokio::test]
    async fn same_priority_evicts_oldest_first() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("1", "aaaa", Priority::Normal),
            msg("2", "bbbb", Priority::Normal),
            msg("3", "cccc", Priority::Normal),
        ];
        let (result, report) = PriorityRetention::new()
            .apply(blocks, 8, &counter)
            .await
            .unwrap();
        assert_eq!(report.evicted_block_ids, vec!["1"]);
        assert!(result.iter().any(|b| b.id == "2"));
        assert!(result.iter().any(|b| b.id == "3"));
    }

    #[tokio::test]
    async fn under_budget_keeps_all() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            msg("1", "aa", Priority::Low),
            msg("2", "bb", Priority::High),
        ];
        let (result, _) = PriorityRetention::new()
            .apply(blocks, 100, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
    }
}
