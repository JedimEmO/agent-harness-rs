use std::sync::Arc;

use crate::pipeline::ContextPipeline;
use crate::strategy::ContextStrategy;
use crate::token::{CharEstimateCounter, TokenCounter};

/// Builder for constructing a [`ContextPipeline`].
pub struct PipelineBuilder {
    strategies: Vec<Box<dyn ContextStrategy>>,
    counter: Option<Arc<dyn TokenCounter>>,
}

impl PipelineBuilder {
    pub fn new() -> Self {
        Self {
            strategies: Vec::new(),
            counter: None,
        }
    }

    /// Set the token counter. If not set, defaults to [`CharEstimateCounter`] with 4 chars/token.
    pub fn token_counter(mut self, counter: impl TokenCounter + 'static) -> Self {
        self.counter = Some(Arc::new(counter));
        self
    }

    /// Append a strategy to the pipeline. Strategies are applied in the order they are added.
    pub fn strategy(mut self, strategy: impl ContextStrategy + 'static) -> Self {
        self.strategies.push(Box::new(strategy));
        self
    }

    /// Build the pipeline.
    pub fn build(self) -> ContextPipeline {
        let counter = self
            .counter
            .unwrap_or_else(|| Arc::new(CharEstimateCounter::default()));
        ContextPipeline::new(counter, self.strategies)
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}
