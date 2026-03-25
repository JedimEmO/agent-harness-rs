use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContextError {
    #[error("summarization failed: {0}")]
    SummarizationFailed(String),

    #[error("token budget exceeded: need {needed}, budget {budget}")]
    BudgetExceeded { needed: usize, budget: usize },

    #[error("strategy error in '{strategy}': {message}")]
    StrategyError { strategy: String, message: String },

    #[error("{0}")]
    Other(String),
}
