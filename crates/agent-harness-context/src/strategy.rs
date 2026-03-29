use async_trait::async_trait;

use crate::error::ContextError;
use crate::token::TokenCounter;
use crate::types::{ContextBlock, ContextReport};

/// A composable strategy for managing context blocks within a token budget.
///
/// Strategies can evict, reorder, merge, or insert blocks. They are chained
/// in a [`ContextPipeline`](crate::pipeline::ContextPipeline) where each
/// strategy's output feeds the next.
#[async_trait]
pub trait ContextStrategy: Send + Sync {
    /// Name for logging and reporting.
    fn name(&self) -> &str;

    /// Transform the list of context blocks.
    ///
    /// The strategy receives the current blocks and token budget, and returns
    /// the (possibly modified) blocks along with a report of what changed.
    async fn apply(
        &self,
        blocks: Vec<ContextBlock>,
        token_budget: usize,
        counter: &dyn TokenCounter,
    ) -> Result<(Vec<ContextBlock>, ContextReport), ContextError>;
}
