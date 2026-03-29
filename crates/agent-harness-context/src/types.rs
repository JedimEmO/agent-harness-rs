use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// A labeled block of context with priority and metadata.
/// This is the atomic unit that context strategies operate on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBlock {
    pub id: String,
    pub kind: BlockKind,
    pub content: BlockContent,
    pub priority: Priority,
    /// Pinned blocks are never evicted by any strategy.
    pub pinned: bool,
    /// Cached token count. Strategies populate this via the `TokenCounter`.
    pub token_count: Option<usize>,
    /// Arbitrary key-value metadata for consumer use.
    pub metadata: HashMap<String, String>,
    /// Optional group ID for atomic eviction (e.g., tool-call + tool-result pairs).
    /// Blocks sharing the same `group` are evicted or retained together.
    pub group: Option<String>,
}

impl ContextBlock {
    /// Create a new block with the given kind and text content at Normal priority.
    pub fn new(id: impl Into<String>, kind: BlockKind, text: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            kind,
            content: BlockContent::Text(text.into()),
            priority: Priority::Normal,
            pinned: false,
            token_count: None,
            metadata: HashMap::new(),
            group: None,
        }
    }

    pub fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn pinned(mut self) -> Self {
        self.pinned = true;
        self
    }

    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    pub fn with_content(mut self, content: BlockContent) -> Self {
        self.content = content;
        self
    }

    pub fn with_metadata(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    pub fn with_token_count(mut self, count: usize) -> Self {
        self.token_count = Some(count);
        self
    }
}

/// The semantic kind of a context block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockKind {
    SystemPrompt,
    UserMessage,
    AssistantMessage,
    AssistantToolCalls,
    ToolResults,
    ToolDefinition,
    RetrievalChunk,
    FileContent,
    UserProfile,
    Summary,
    Custom(String),
}

/// The content payload of a context block.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlockContent {
    Text(String),
    Structured(serde_json::Value),
    Parts(Vec<ContentPart>),
}

impl BlockContent {
    /// Extract the text representation for token counting.
    pub fn as_text(&self) -> String {
        match self {
            BlockContent::Text(t) => t.clone(),
            BlockContent::Structured(v) => v.to_string(),
            BlockContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match p {
                    ContentPart::Text(t) => Some(t.as_str()),
                    ContentPart::Image { .. } => None,
                })
                .collect::<Vec<_>>()
                .join(""),
        }
    }

    /// Count the number of image parts.
    pub fn image_count(&self) -> usize {
        match self {
            BlockContent::Parts(parts) => parts
                .iter()
                .filter(|p| matches!(p, ContentPart::Image { .. }))
                .count(),
            _ => 0,
        }
    }
}

/// A part of multi-modal content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text(String),
    Image { media_type: String, data: Vec<u8> },
}

/// Priority levels for eviction ordering.
/// Higher priority blocks are retained longer when the context must be trimmed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Priority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// The result of running a context pipeline.
#[derive(Debug, Clone, Default)]
pub struct ContextReport {
    pub original_blocks: usize,
    pub final_blocks: usize,
    pub original_tokens: usize,
    pub final_tokens: usize,
    pub evicted_block_ids: Vec<String>,
    pub summarized_block_ids: Vec<String>,
    pub strategy_logs: Vec<String>,
}

impl ContextReport {
    /// Create a report indicating no changes were made.
    pub fn no_change(block_count: usize, total_tokens: usize) -> Self {
        Self {
            original_blocks: block_count,
            final_blocks: block_count,
            original_tokens: total_tokens,
            final_tokens: total_tokens,
            ..Default::default()
        }
    }

    /// Merge another report into this one (used when chaining strategies).
    pub fn merge(&mut self, other: &ContextReport) {
        self.evicted_block_ids
            .extend(other.evicted_block_ids.iter().cloned());
        self.summarized_block_ids
            .extend(other.summarized_block_ids.iter().cloned());
        self.strategy_logs
            .extend(other.strategy_logs.iter().cloned());
    }
}
