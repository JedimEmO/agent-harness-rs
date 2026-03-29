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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn create_task_basic() {
        let store = TaskStore::new();
        let task = store.create_task("s", "do something").await;
        assert_eq!(task.id, "task_1");
        assert_eq!(task.description, "do something");
        assert_eq!(task.status, TaskStatus::Pending);
        assert!(task.depends_on.is_empty());
        assert!(task.assigned_to.is_none());
        assert!(task.result.is_none());
        assert!(task.parent_id.is_none());
    }

    #[tokio::test]
    async fn create_task_increments_id() {
        let store = TaskStore::new();
        let t1 = store.create_task("s", "first").await;
        let t2 = store.create_task("s", "second").await;
        assert_eq!(t1.id, "task_1");
        assert_eq!(t2.id, "task_2");
    }

    #[tokio::test]
    async fn create_subtask() {
        let store = TaskStore::new();
        let parent = store.create_task("s", "parent").await;
        let sub = store.create_subtask("s", &parent.id, "child").await;
        assert_eq!(sub.parent_id, Some("task_1".to_string()));
        assert_eq!(sub.id, "task_2");
    }

    #[tokio::test]
    async fn update_task_status() {
        let store = TaskStore::new();
        store.create_task("s", "task").await;
        let updated = store.update_task("s", "task_1", TaskStatus::InProgress).await;
        assert!(updated.is_some());
        assert_eq!(updated.unwrap().status, TaskStatus::InProgress);
    }

    #[tokio::test]
    async fn update_task_not_found() {
        let store = TaskStore::new();
        let result = store.update_task("s", "nonexistent", TaskStatus::Completed).await;
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn set_result() {
        let store = TaskStore::new();
        store.create_task("s", "task").await;
        let updated = store.set_result("s", "task_1", "done with output").await;
        assert!(updated.is_some());
        assert_eq!(updated.unwrap().result, Some("done with output".to_string()));
    }

    #[tokio::test]
    async fn assign_task() {
        let store = TaskStore::new();
        store.create_task("s", "task").await;
        let updated = store.assign_task("s", "task_1", "agent-alpha").await;
        assert!(updated.is_some());
        assert_eq!(updated.unwrap().assigned_to, Some("agent-alpha".to_string()));
    }

    #[tokio::test]
    async fn add_dependency() {
        let store = TaskStore::new();
        store.create_task("s", "first").await;
        store.create_task("s", "second").await;
        let updated = store.add_dependency("s", "task_2", "task_1").await;
        assert!(updated.is_some());
        assert_eq!(updated.unwrap().depends_on, vec!["task_1".to_string()]);
    }

    #[tokio::test]
    async fn add_duplicate_dependency() {
        let store = TaskStore::new();
        store.create_task("s", "first").await;
        store.create_task("s", "second").await;
        store.add_dependency("s", "task_2", "task_1").await;
        store.add_dependency("s", "task_2", "task_1").await;
        let task = store.get_task("s", "task_2").await.unwrap();
        assert_eq!(task.depends_on.len(), 1);
    }

    #[tokio::test]
    async fn list_tasks_empty() {
        let store = TaskStore::new();
        let tasks = store.list_tasks("s").await;
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn list_tasks_scope_isolation() {
        let store = TaskStore::new();
        store.create_task("scope_a", "task in A").await;
        let tasks = store.list_tasks("scope_b").await;
        assert!(tasks.is_empty());
    }

    #[tokio::test]
    async fn is_ready_no_deps() {
        let store = TaskStore::new();
        store.create_task("s", "independent").await;
        assert!(store.is_ready("s", "task_1").await);
    }

    #[tokio::test]
    async fn is_ready_deps_not_met() {
        let store = TaskStore::new();
        store.create_task("s", "first").await;
        store.create_task("s", "second").await;
        store.add_dependency("s", "task_2", "task_1").await;
        assert!(!store.is_ready("s", "task_2").await);
    }

    #[tokio::test]
    async fn is_ready_deps_met() {
        let store = TaskStore::new();
        store.create_task("s", "first").await;
        store.create_task("s", "second").await;
        store.add_dependency("s", "task_2", "task_1").await;
        store.update_task("s", "task_1", TaskStatus::Completed).await;
        assert!(store.is_ready("s", "task_2").await);
    }

    #[tokio::test]
    async fn clear_tasks() {
        let store = TaskStore::new();
        store.create_task("s", "task").await;
        assert!(!store.list_tasks("s").await.is_empty());
        store.clear_tasks("s").await;
        assert!(store.list_tasks("s").await.is_empty());
    }
}
