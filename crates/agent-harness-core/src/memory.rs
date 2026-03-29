use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::AgentError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryCategory {
    Fact,
    Preference,
    Instruction,
    Context,
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryCategory::Fact => write!(f, "fact"),
            MemoryCategory::Preference => write!(f, "preference"),
            MemoryCategory::Instruction => write!(f, "instruction"),
            MemoryCategory::Context => write!(f, "context"),
        }
    }
}

impl std::str::FromStr for MemoryCategory {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fact" => Ok(MemoryCategory::Fact),
            "preference" => Ok(MemoryCategory::Preference),
            "instruction" => Ok(MemoryCategory::Instruction),
            "context" => Ok(MemoryCategory::Context),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memory {
    pub id: String,
    pub scope_id: String,
    pub key: String,
    pub content: String,
    pub category: MemoryCategory,
    pub created_at: String,
    pub updated_at: String,
}

/// Trait for persisting agent memories.
///
/// Implement this to store memories in your preferred backend.
///
/// The base trait provides substring-based search via [`search`].
/// For semantic similarity search using embeddings, implement
/// [`SemanticMemoryStore`] as well.
#[async_trait]
pub trait MemoryStore: Send + Sync {
    async fn save(
        &self,
        scope_id: &str,
        key: &str,
        content: &str,
        category: MemoryCategory,
    ) -> Result<Memory, AgentError>;

    /// Search memories by substring matching on key and content.
    async fn search(
        &self,
        scope_id: &str,
        query: &str,
        limit: usize,
    ) -> Result<Vec<Memory>, AgentError>;

    async fn list(
        &self,
        scope_id: &str,
        category: Option<MemoryCategory>,
    ) -> Result<Vec<Memory>, AgentError>;

    async fn delete(&self, id: &str) -> Result<bool, AgentError>;
}

/// Extended memory store with semantic (embedding-based) search.
///
/// Implement this alongside [`MemoryStore`] to enable similarity search
/// using vector embeddings. Requires an embedding provider to generate
/// vectors from text.
#[async_trait]
pub trait SemanticMemoryStore: MemoryStore {
    /// Search memories by semantic similarity using embeddings.
    ///
    /// The `query_embedding` is a pre-computed vector for the search query.
    /// Returns memories sorted by similarity (most similar first).
    async fn search_semantic(
        &self,
        scope_id: &str,
        query_embedding: &[f32],
        limit: usize,
    ) -> Result<Vec<Memory>, AgentError>;

    /// Save a memory with its embedding vector for later semantic search.
    async fn save_with_embedding(
        &self,
        scope_id: &str,
        key: &str,
        content: &str,
        category: MemoryCategory,
        embedding: Vec<f32>,
    ) -> Result<Memory, AgentError>;
}
