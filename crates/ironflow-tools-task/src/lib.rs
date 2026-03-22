//! # ironflow-tools-task
//!
//! Task tracking tools for ironflow agents.
//!
//! Provides an in-memory [`TaskStore`] and three tools:
//! - [`CreateTaskTool`] — create a new task
//! - [`UpdateTaskTool`] — update task status (pending/in_progress/completed)
//! - [`ListTasksTool`] — list all tasks with progress

mod store;
mod tools;

pub use store::{AgentTask, TaskStatus, TaskStore};
pub use tools::{CreateTaskTool, UpdateTaskTool, ListTasksTool};
