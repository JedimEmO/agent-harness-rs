use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    pub id: String,
    pub description: String,
    pub status: TaskStatus,
    /// Task IDs that must complete before this task can start.
    pub depends_on: Vec<String>,
    /// Which agent or sub-agent owns this task.
    pub assigned_to: Option<String>,
    /// Output/artifact from completing this task.
    pub result: Option<String>,
    /// Parent task ID for sub-tasks.
    pub parent_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TaskStatus::Pending => write!(f, "pending"),
            TaskStatus::InProgress => write!(f, "in_progress"),
            TaskStatus::Completed => write!(f, "completed"),
            TaskStatus::Blocked => write!(f, "blocked"),
        }
    }
}

/// In-memory task store, scoped by `scope_id`.
///
/// Now supports task dependencies, assignment, results, and sub-tasks.
#[derive(Clone)]
pub struct TaskStore {
    tasks: Arc<RwLock<HashMap<String, Vec<AgentTask>>>>,
}

impl TaskStore {
    pub fn new() -> Self {
        Self {
            tasks: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub async fn create_task(&self, scope_id: &str, description: &str) -> AgentTask {
        let mut tasks = self.tasks.write().await;
        let list = tasks.entry(scope_id.to_string()).or_default();
        let id = format!("task_{}", list.len() + 1);
        let task = AgentTask {
            id: id.clone(),
            description: description.to_string(),
            status: TaskStatus::Pending,
            depends_on: Vec::new(),
            assigned_to: None,
            result: None,
            parent_id: None,
        };
        list.push(task.clone());
        task
    }

    pub async fn create_subtask(
        &self,
        scope_id: &str,
        parent_id: &str,
        description: &str,
    ) -> AgentTask {
        let mut tasks = self.tasks.write().await;
        let list = tasks.entry(scope_id.to_string()).or_default();
        let id = format!("task_{}", list.len() + 1);
        let task = AgentTask {
            id: id.clone(),
            description: description.to_string(),
            status: TaskStatus::Pending,
            depends_on: Vec::new(),
            assigned_to: None,
            result: None,
            parent_id: Some(parent_id.to_string()),
        };
        list.push(task.clone());
        task
    }

    pub async fn update_task(
        &self,
        scope_id: &str,
        task_id: &str,
        status: TaskStatus,
    ) -> Option<AgentTask> {
        let mut tasks = self.tasks.write().await;
        let list = tasks.get_mut(scope_id)?;
        let task = list.iter_mut().find(|t| t.id == task_id)?;
        task.status = status;
        Some(task.clone())
    }

    pub async fn set_result(
        &self,
        scope_id: &str,
        task_id: &str,
        result: &str,
    ) -> Option<AgentTask> {
        let mut tasks = self.tasks.write().await;
        let list = tasks.get_mut(scope_id)?;
        let task = list.iter_mut().find(|t| t.id == task_id)?;
        task.result = Some(result.to_string());
        Some(task.clone())
    }

    pub async fn assign_task(
        &self,
        scope_id: &str,
        task_id: &str,
        assignee: &str,
    ) -> Option<AgentTask> {
        let mut tasks = self.tasks.write().await;
        let list = tasks.get_mut(scope_id)?;
        let task = list.iter_mut().find(|t| t.id == task_id)?;
        task.assigned_to = Some(assignee.to_string());
        Some(task.clone())
    }

    pub async fn add_dependency(
        &self,
        scope_id: &str,
        task_id: &str,
        depends_on_id: &str,
    ) -> Option<AgentTask> {
        let mut tasks = self.tasks.write().await;
        let list = tasks.get_mut(scope_id)?;
        let task = list.iter_mut().find(|t| t.id == task_id)?;
        if !task.depends_on.contains(&depends_on_id.to_string()) {
            task.depends_on.push(depends_on_id.to_string());
        }
        Some(task.clone())
    }

    pub async fn list_tasks(&self, scope_id: &str) -> Vec<AgentTask> {
        let tasks = self.tasks.read().await;
        tasks.get(scope_id).cloned().unwrap_or_default()
    }

    pub async fn get_task(&self, scope_id: &str, task_id: &str) -> Option<AgentTask> {
        let tasks = self.tasks.read().await;
        tasks
            .get(scope_id)?
            .iter()
            .find(|t| t.id == task_id)
            .cloned()
    }

    /// Check if all dependencies of a task are completed.
    pub async fn is_ready(&self, scope_id: &str, task_id: &str) -> bool {
        let tasks = self.tasks.read().await;
        let list = match tasks.get(scope_id) {
            Some(l) => l,
            None => return false,
        };
        let task = match list.iter().find(|t| t.id == task_id) {
            Some(t) => t,
            None => return false,
        };
        task.depends_on.iter().all(|dep_id| {
            list.iter()
                .find(|t| t.id == *dep_id)
                .map_or(false, |t| t.status == TaskStatus::Completed)
        })
    }

    pub async fn clear_tasks(&self, scope_id: &str) {
        let mut tasks = self.tasks.write().await;
        tasks.remove(scope_id);
    }
}

impl Default for TaskStore {
    fn default() -> Self {
        Self::new()
    }
}
