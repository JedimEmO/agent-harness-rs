use async_trait::async_trait;
use agent_harness_core::{AgentError, AgentTool, ToolDefinition, ToolExecResult, ToolPermission};

use crate::store::{TaskStatus, TaskStore};

// --- Create Task ---

pub struct CreateTaskTool {
    store: TaskStore,
}

impl CreateTaskTool {
    pub fn new(store: TaskStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl AgentTool for CreateTaskTool {
    fn name(&self) -> &str { "create_task" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "create_task".to_string(),
            description: "Create a task to track progress on a plan step. Optionally specify dependencies and assignment.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A concise description of the task"
                    },
                    "depends_on": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "IDs of tasks that must complete before this task can start"
                    },
                    "assigned_to": {
                        "type": "string",
                        "description": "Name of the agent or sub-agent to assign this task to"
                    },
                    "parent_id": {
                        "type": "string",
                        "description": "Parent task ID if this is a sub-task"
                    }
                },
                "required": ["description"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(
        &self,
        scope_id: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecResult, AgentError> {
        let description = arguments
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let parent_id = arguments
            .get("parent_id")
            .and_then(|v| v.as_str());

        let task = if let Some(pid) = parent_id {
            self.store.create_subtask(scope_id, pid, &description).await
        } else {
            self.store.create_task(scope_id, &description).await
        };

        // Add dependencies if specified
        if let Some(deps) = arguments.get("depends_on").and_then(|v| v.as_array()) {
            for dep in deps {
                if let Some(dep_id) = dep.as_str() {
                    self.store.add_dependency(scope_id, &task.id, dep_id).await;
                }
            }
        }

        // Assign if specified
        if let Some(assignee) = arguments.get("assigned_to").and_then(|v| v.as_str()) {
            self.store.assign_task(scope_id, &task.id, assignee).await;
        }

        let result = serde_json::json!({
            "task_id": task.id,
            "description": task.description,
            "status": task.status.to_string(),
            "depends_on": task.depends_on,
            "assigned_to": task.assigned_to,
            "parent_id": task.parent_id,
        });
        Ok(ToolExecResult::Completed(result))
    }
}

// --- Update Task ---

pub struct UpdateTaskTool {
    store: TaskStore,
}

impl UpdateTaskTool {
    pub fn new(store: TaskStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl AgentTool for UpdateTaskTool {
    fn name(&self) -> &str { "update_task" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "update_task".to_string(),
            description: "Update a task's status or set its result. Use this to mark tasks as in_progress, completed, or to record output.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "The ID of the task to update"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "completed", "blocked"],
                        "description": "The new status for the task"
                    },
                    "result": {
                        "type": "string",
                        "description": "The output or artifact produced by completing this task"
                    }
                },
                "required": ["task_id"]
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(
        &self,
        scope_id: &str,
        arguments: serde_json::Value,
    ) -> Result<ToolExecResult, AgentError> {
        let task_id = arguments
            .get("task_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Update status if provided
        if let Some(status_str) = arguments.get("status").and_then(|v| v.as_str()) {
            let status = match status_str {
                "in_progress" => TaskStatus::InProgress,
                "completed" => TaskStatus::Completed,
                "blocked" => TaskStatus::Blocked,
                _ => TaskStatus::Pending,
            };

            if self.store.update_task(scope_id, task_id, status).await.is_none() {
                return Ok(ToolExecResult::text(format!("Task '{}' not found", task_id)));
            }
        }

        // Set result if provided
        if let Some(result) = arguments.get("result").and_then(|v| v.as_str()) {
            self.store.set_result(scope_id, task_id, result).await;
        }

        match self.store.get_task(scope_id, task_id).await {
            Some(task) => {
                let result = serde_json::json!({
                    "task_id": task.id,
                    "description": task.description,
                    "status": task.status.to_string(),
                    "result": task.result,
                    "depends_on": task.depends_on,
                    "assigned_to": task.assigned_to,
                });
                Ok(ToolExecResult::Completed(result))
            }
            None => Ok(ToolExecResult::text(format!("Task '{}' not found", task_id))),
        }
    }
}

// --- List Tasks ---

pub struct ListTasksTool {
    store: TaskStore,
}

impl ListTasksTool {
    pub fn new(store: TaskStore) -> Self {
        Self { store }
    }
}

#[async_trait]
impl AgentTool for ListTasksTool {
    fn name(&self) -> &str { "list_tasks" }

    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: "list_tasks".to_string(),
            description: "List all tasks and their current status, dependencies, and assignments.".to_string(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {},
            }),
        }
    }

    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }

    async fn execute(
        &self,
        scope_id: &str,
        _arguments: serde_json::Value,
    ) -> Result<ToolExecResult, AgentError> {
        let tasks = self.store.list_tasks(scope_id).await;

        if tasks.is_empty() {
            return Ok(ToolExecResult::text("No tasks created yet."));
        }

        let completed = tasks.iter().filter(|t| t.status == TaskStatus::Completed).count();
        let total = tasks.len();

        let mut output = format!("Progress: {}/{} completed\n\n", completed, total);
        for task in &tasks {
            let icon = match task.status {
                TaskStatus::Pending => "[ ]",
                TaskStatus::InProgress => "[~]",
                TaskStatus::Completed => "[x]",
                TaskStatus::Blocked => "[!]",
            };
            output.push_str(&format!("{} {} — {}", icon, task.id, task.description));

            if !task.depends_on.is_empty() {
                output.push_str(&format!(" (depends: {})", task.depends_on.join(", ")));
            }
            if let Some(ref assignee) = task.assigned_to {
                output.push_str(&format!(" [{}]", assignee));
            }
            if let Some(ref parent) = task.parent_id {
                output.push_str(&format!(" (sub of {})", parent));
            }
            output.push('\n');
        }

        Ok(ToolExecResult::text(output))
    }
}
