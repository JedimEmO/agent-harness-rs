use std::collections::HashSet;

use async_trait::async_trait;

use crate::error::ContextError;
use crate::strategy::ContextStrategy;
use crate::token::TokenCounter;
use crate::types::{ContextBlock, ContextReport};

/// Drops the oldest non-pinned blocks until the total token count fits within budget.
///
/// Blocks with the same `group` are treated as atomic units — they are evicted
/// or retained together. This preserves structural invariants like tool-call /
/// tool-result pairs.
pub struct TruncateOldest;

impl TruncateOldest {
    pub fn new() -> Self {
        Self
    }
}

impl Default for TruncateOldest {
    fn default() -> Self {
        Self::new()
    }
}

struct Unit {
    indices: Vec<usize>,
    tokens: usize,
    pinned: bool,
}

#[async_trait]
impl ContextStrategy for TruncateOldest {
    fn name(&self) -> &str {
        "truncate_oldest"
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

        let units = group_blocks(&blocks);

        // Reserve pinned tokens first, then fill remaining budget from most recent
        let pinned_tokens: usize = units.iter().filter(|u| u.pinned).map(|u| u.tokens).sum();
        let mut remaining_budget = token_budget.saturating_sub(pinned_tokens);

        let mut kept_unit_indices: Vec<usize> = Vec::new();
        for (ui, unit) in units.iter().enumerate().rev() {
            if unit.pinned {
                kept_unit_indices.push(ui);
            } else if unit.tokens <= remaining_budget {
                remaining_budget = remaining_budget.saturating_sub(unit.tokens);
                kept_unit_indices.push(ui);
            }
        }

        kept_unit_indices.sort();

        let kept_set: HashSet<usize> = kept_unit_indices
            .iter()
            .flat_map(|&ui| &units[ui].indices)
            .copied()
            .collect();

        let evicted_ids: Vec<String> = blocks
            .iter()
            .enumerate()
            .filter(|(i, _)| !kept_set.contains(i))
            .map(|(_, b)| b.id.clone())
            .collect();

        let result: Vec<ContextBlock> = blocks
            .into_iter()
            .enumerate()
            .filter(|(i, _)| kept_set.contains(i))
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
                    "truncate_oldest: dropped {} blocks ({} -> {} tokens)",
                    original_count - final_blocks,
                    total_tokens,
                    final_tokens,
                )],
                ..Default::default()
            },
        ))
    }
}

fn group_blocks(blocks: &[ContextBlock]) -> Vec<Unit> {
    let mut units: Vec<Unit> = Vec::new();
    let mut i = 0;

    while i < blocks.len() {
        if let Some(ref group) = blocks[i].group {
            let mut indices = vec![i];
            let mut tokens = blocks[i].token_count.unwrap_or(0);
            let mut pinned = blocks[i].pinned;
            let mut j = i + 1;
            while j < blocks.len() && blocks[j].group.as_deref() == Some(group) {
                indices.push(j);
                tokens += blocks[j].token_count.unwrap_or(0);
                pinned = pinned || blocks[j].pinned;
                j += 1;
            }
            units.push(Unit {
                indices,
                tokens,
                pinned,
            });
            i = j;
        } else {
            units.push(Unit {
                indices: vec![i],
                tokens: blocks[i].token_count.unwrap_or(0),
                pinned: blocks[i].pinned,
            });
            i += 1;
        }
    }

    units
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::token::CharEstimateCounter;
    use crate::types::{BlockKind, Priority};

    fn make_block(id: &str, text: &str) -> ContextBlock {
        ContextBlock::new(id, BlockKind::UserMessage, text)
    }

    #[tokio::test]
    async fn under_budget_keeps_all() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("1", "hello"),
            make_block("2", "world"),
        ];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 100, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert!(report.evicted_block_ids.is_empty());
    }

    #[tokio::test]
    async fn over_budget_drops_oldest() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("1", "aaaa"),
            make_block("2", "bbbb"),
            make_block("3", "cccc"),
        ];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 8, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].id, "2");
        assert_eq!(result[1].id, "3");
        assert_eq!(report.evicted_block_ids, vec!["1"]);
    }

    #[tokio::test]
    async fn pinned_blocks_never_evicted() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("sys", "system prompt text").pinned(),
            make_block("1", "aaaa"),
            make_block("2", "bbbb"),
        ];
        let (result, _) = TruncateOldest::new()
            .apply(blocks, 22, &counter)
            .await
            .unwrap();
        assert!(result.iter().any(|b| b.id == "sys"));
        assert!(result.iter().any(|b| b.id == "2"));
    }

    #[tokio::test]
    async fn grouped_blocks_evicted_together() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("tc1", "call").with_group("g1"),
            make_block("tr1", "result").with_group("g1"),
            make_block("2", "recent"),
        ];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 7, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "2");
        assert!(report.evicted_block_ids.contains(&"tc1".to_string()));
        assert!(report.evicted_block_ids.contains(&"tr1".to_string()));
    }

    #[tokio::test]
    async fn priority_does_not_affect_truncation() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("1", "aaaa").with_priority(Priority::Critical),
            make_block("2", "bbbb").with_priority(Priority::Low),
        ];
        let (result, _) = TruncateOldest::new()
            .apply(blocks, 4, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "2");
    }

    // =========================================================================
    // ADVERSARIAL TESTS
    // =========================================================================

    #[tokio::test]
    async fn zero_budget_drops_everything() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("1", "hello"),
            make_block("2", "world"),
        ];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 0, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 0, "zero budget should drop all blocks");
        assert_eq!(report.evicted_block_ids.len(), 2);
    }

    #[tokio::test]
    async fn zero_budget_but_pinned_blocks_kept() {
        // BUG PROBE: Pinned blocks with budget=0.
        // pinned_tokens = 5, remaining_budget = 0.saturating_sub(5) = 0.
        // Pinned blocks are always kept. Non-pinned blocks need 0 remaining budget.
        // But the pinned block itself has 5 tokens exceeding budget of 0.
        // The implementation always keeps pinned blocks regardless of budget.
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("pinned", "hello").pinned(),
            make_block("normal", "world"),
        ];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 0, &counter)
            .await
            .unwrap();
        // BUG: Pinned block is kept even though budget is 0,
        // meaning output exceeds budget
        assert_eq!(result.len(), 1, "only pinned block should remain");
        assert_eq!(result[0].id, "pinned");
        assert_eq!(
            report.final_tokens, 5,
            "BUG: final tokens (5) EXCEEDS budget (0) because pinned blocks are always kept"
        );
    }

    #[tokio::test]
    async fn empty_blocks_list() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 100, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 0);
        assert_eq!(report.original_blocks, 0);
        assert_eq!(report.final_blocks, 0);
    }

    #[tokio::test]
    async fn all_blocks_too_large_for_budget() {
        // Every block individually exceeds budget
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("1", &"a".repeat(100)),
            make_block("2", &"b".repeat(100)),
        ];
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 5, &counter)
            .await
            .unwrap();
        // Each block is 100 tokens, budget is 5. Neither fits.
        assert_eq!(result.len(), 0, "all blocks too large should produce empty result");
        assert_eq!(report.evicted_block_ids.len(), 2);
    }

    #[tokio::test]
    async fn grouped_blocks_one_pinned_keeps_whole_group() {
        // BUG PROBE: If one block in a group is pinned, the whole group is pinned.
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            make_block("tc1", "call").with_group("g1"),
            make_block("tr1", "result that is long").with_group("g1").pinned(),
            make_block("other", "x"),
        ];
        let (result, _) = TruncateOldest::new()
            .apply(blocks, 5, &counter)
            .await
            .unwrap();
        // Group g1 is pinned because tr1 is pinned. Total group tokens = 4 + 20 = 24.
        // Budget = 5, but pinned blocks are always kept.
        // remaining_budget = 5 - 24 = 0 (saturating). "x" (1 token) fits in 0? No.
        let has_tc1 = result.iter().any(|b| b.id == "tc1");
        let has_tr1 = result.iter().any(|b| b.id == "tr1");
        assert!(has_tc1 && has_tr1, "pinned group members keep whole group");
    }

    #[tokio::test]
    async fn pre_cached_token_counts_used() {
        // BUG PROBE: If token_count is already set, it should be used as-is
        // even if it doesn't match the actual content length.
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![
            // Claim this block is 1000 tokens even though content is 5 chars
            make_block("1", "hello").with_token_count(1000),
            make_block("2", "world"),
        ];
        let (result, _) = TruncateOldest::new()
            .apply(blocks, 100, &counter)
            .await
            .unwrap();
        // Block 1 claims 1000 tokens, block 2 is 5 tokens. Total = 1005 > 100.
        // Reverse: block 2 (5) fits in 100, block 1 (1000) doesn't fit in 95.
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].id, "2");
    }

    #[tokio::test]
    async fn chars_per_token_zero_handled() {
        // BUG PROBE: CharEstimateCounter with chars_per_token=0.
        // The implementation uses .max(1) to prevent division by zero.
        let counter = CharEstimateCounter::new(0);
        let blocks = vec![make_block("1", "hello")];
        let (result, _) = TruncateOldest::new()
            .apply(blocks, 100, &counter)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
    }

    #[tokio::test]
    async fn single_block_exactly_at_budget() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![make_block("1", "hello")]; // 5 tokens
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 5, &counter)
            .await
            .unwrap();
        // 5 <= 5, so under budget — no truncation
        assert_eq!(result.len(), 1);
        assert!(report.evicted_block_ids.is_empty());
    }

    #[tokio::test]
    async fn single_block_one_over_budget() {
        let counter = CharEstimateCounter::new(1);
        let blocks = vec![make_block("1", "hello")]; // 5 tokens
        let (result, report) = TruncateOldest::new()
            .apply(blocks, 4, &counter)
            .await
            .unwrap();
        // 5 > 4, and it's the only block so it gets dropped
        assert_eq!(result.len(), 0);
        assert_eq!(report.evicted_block_ids, vec!["1"]);
    }
}
