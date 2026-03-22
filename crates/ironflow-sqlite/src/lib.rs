//! # ironflow-sqlite
//!
//! SQLite-backed [`SessionStore`](ironflow_core::SessionStore) and
//! [`MemoryStore`](ironflow_core::MemoryStore) implementations for ironflow,
//! powered by Diesel.
//!
//! ## Usage
//!
//! ```rust,no_run
//! use ironflow_sqlite::{init_db, SqliteSessionStore, SqliteMemoryStore};
//!
//! let pool = init_db("data/ironflow.db").expect("failed to init DB");
//! let session_store = SqliteSessionStore::new(pool.clone());
//! let memory_store = SqliteMemoryStore::new(pool);
//! ```

mod error;
mod pool;
mod schema;
mod session_store;
mod memory_store;

pub use error::DbError;
pub use pool::{DbPool, init_db};
pub use session_store::SqliteSessionStore;
pub use memory_store::SqliteMemoryStore;
