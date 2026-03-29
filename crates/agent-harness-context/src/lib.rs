pub mod error;
pub mod types;
pub mod token;
pub mod strategy;
pub mod strategies;
pub mod pipeline;
pub mod builder;

// Re-exports for convenience
pub use error::ContextError;
pub use types::{BlockContent, BlockKind, ContentPart, ContextBlock, ContextReport, Priority};
pub use token::{CharEstimateCounter, TokenCounter};
pub use strategy::ContextStrategy;
pub use pipeline::ContextPipeline;
pub use builder::PipelineBuilder;

pub use strategies::truncate::TruncateOldest;
pub use strategies::sliding_window::SlidingWindow;
pub use strategies::priority::PriorityRetention;
pub use strategies::summarize::{Summarize, Summarizer};
