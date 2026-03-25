use std::sync::Arc;

use tracing::debug;

use crate::error::ContextError;
use crate::strategy::ContextStrategy;
use crate::token::TokenCounter;
use crate::types::{ContextBlock, ContextReport};

/// Chains multiple [`ContextStrategy`] implementations in sequence.
///
/// Each strategy's output feeds the next. The pipeline tracks a merged
/// [`ContextReport`] across all stages.
///
/// # Example
///
/// ```rust
/// use ironflow_context::*;
///
/// # async fn example() -> Result<(), ContextError> {
/// let pipeline = ContextPipeline::builder()
///     .token_counter(CharEstimateCounter::new(4))
///     .strategy(PriorityRetention::new())
///     .strategy(TruncateOldest::new())
///     .build();
///
/// let blocks = vec![
///     ContextBlock::new("1", BlockKind::UserMessage, "Hello"),
/// ];
///
/// let (result, report) = pipeline.process(blocks, 1000).await?;
/// # Ok(())
/// # }
/// ```
pub struct ContextPipeline {
    pub(crate) strategies: Vec<Box<dyn ContextStrategy>>,
    pub(crate) counter: Arc<dyn TokenCounter>,
}

impl ContextPipeline {
    /// Create a new pipeline with the given counter and strategies.
    pub fn new(counter: Arc<dyn TokenCounter>, strategies: Vec<Box<dyn ContextStrategy>>) -> Self {
        Self {
            strategies,
            counter,
        }
    }

    /// Start building a pipeline.
    pub fn builder() -> crate::builder::PipelineBuilder {
        crate::builder::PipelineBuilder::new()
    }

    /// Process blocks through all strategies in sequence.
    pub async fn process(
        &self,
        mut blocks: Vec<ContextBlock>,
        token_budget: usize,
    ) -> Result<(Vec<ContextBlock>, ContextReport), ContextError> {
        self.counter.cache_blocks(&mut blocks);

        let original_blocks = blocks.len();
        let original_tokens: usize = blocks.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        let mut merged_report = ContextReport {
            original_blocks,
            original_tokens,
            ..Default::default()
        };

        for strategy in &self.strategies {
            debug!(strategy = strategy.name(), blocks = blocks.len(), "applying context strategy");
            let (new_blocks, report) = strategy
                .apply(blocks, token_budget, self.counter.as_ref())
                .await?;
            merged_report.merge(&report);
            blocks = new_blocks;
        }

        merged_report.final_blocks = blocks.len();
        merged_report.final_tokens =
            blocks.iter().map(|b| b.token_count.unwrap_or(0)).sum();

        Ok((blocks, merged_report))
    }
}
