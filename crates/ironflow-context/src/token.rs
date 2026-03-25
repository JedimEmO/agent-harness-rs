use crate::types::{BlockContent, ContentPart, ContextBlock};

/// Trait for counting tokens in text and images.
///
/// Implement this to plug in an accurate tokenizer (tiktoken, Anthropic's
/// tokenizer, etc.). The default [`CharEstimateCounter`] divides character
/// count by a configurable ratio.
pub trait TokenCounter: Send + Sync {
    /// Count tokens in a text string.
    fn count(&self, text: &str) -> usize;

    /// Estimate tokens for an image. Default: 1000 tokens per image.
    fn count_image(&self, _media_type: &str, _data: &[u8]) -> usize {
        1000
    }

    /// Count tokens for an entire block's content.
    fn count_block(&self, block: &ContextBlock) -> usize {
        if let Some(cached) = block.token_count {
            return cached;
        }
        self.count_content(&block.content)
    }

    /// Ensure all blocks have cached token counts.
    fn cache_blocks(&self, blocks: &mut [ContextBlock]) {
        for block in blocks {
            if block.token_count.is_none() {
                block.token_count = Some(self.count_block(block));
            }
        }
    }

    /// Count tokens for block content.
    fn count_content(&self, content: &BlockContent) -> usize {
        match content {
            BlockContent::Text(t) => self.count(t),
            BlockContent::Structured(v) => self.count(&v.to_string()),
            BlockContent::Parts(parts) => parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text(t) => self.count(t),
                    ContentPart::Image { media_type, data } => {
                        self.count_image(media_type, data)
                    }
                })
                .sum(),
        }
    }
}

/// Simple token counter that estimates tokens as `chars / chars_per_token`.
#[derive(Debug, Clone)]
pub struct CharEstimateCounter {
    pub chars_per_token: usize,
}

impl CharEstimateCounter {
    pub fn new(chars_per_token: usize) -> Self {
        Self { chars_per_token }
    }
}

impl Default for CharEstimateCounter {
    fn default() -> Self {
        Self { chars_per_token: 4 }
    }
}

impl TokenCounter for CharEstimateCounter {
    fn count(&self, text: &str) -> usize {
        text.len() / self.chars_per_token.max(1)
    }
}
