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

#[cfg(test)]
mod tests {
    use super::*;
    use agent_harness_core::AgentTool;

    fn make_store() -> TaskStore {
        TaskStore::new()
    }

    #[tokio::test]
    async fn create_task_tool_basic() {
        let store = make_store();
        let tool = CreateTaskTool::new(store);
        let result = tool.execute("s", serde_json::json!({"description": "build widget"})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                assert_eq!(v["task_id"], "task_1");
                assert_eq!(v["status"], "pending");
                assert_eq!(v["description"], "build widget");
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn create_task_tool_with_parent() {
        let store = make_store();
        let tool = CreateTaskTool::new(store);
        // Create parent first
        tool.execute("s", serde_json::json!({"description": "parent"})).await.unwrap();
        let result = tool.execute("s", serde_json::json!({
            "description": "child",
            "parent_id": "task_1"
        })).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                assert_eq!(v["parent_id"], "task_1");
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn create_task_tool_with_dependencies() {
        let store = make_store();
        let tool = CreateTaskTool::new(store.clone());
        tool.execute("s", serde_json::json!({"description": "dep1"})).await.unwrap();
        tool.execute("s", serde_json::json!({
            "description": "main task",
            "depends_on": ["task_1"]
        })).await.unwrap();
        let task = store.get_task("s", "task_2").await.unwrap();
        assert_eq!(task.depends_on, vec!["task_1".to_string()]);
    }

    #[tokio::test]
    async fn create_task_tool_with_assignment() {
        let store = make_store();
        let tool = CreateTaskTool::new(store.clone());
        tool.execute("s", serde_json::json!({
            "description": "assigned task",
            "assigned_to": "agent-beta"
        })).await.unwrap();
        let task = store.get_task("s", "task_1").await.unwrap();
        assert_eq!(task.assigned_to, Some("agent-beta".to_string()));
    }

    #[tokio::test]
    async fn update_task_tool_status() {
        let store = make_store();
        let create = CreateTaskTool::new(store.clone());
        let update = UpdateTaskTool::new(store.clone());
        create.execute("s", serde_json::json!({"description": "task"})).await.unwrap();
        let result = update.execute("s", serde_json::json!({
            "task_id": "task_1",
            "status": "completed"
        })).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                assert_eq!(v["status"], "completed");
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn update_task_tool_not_found() {
        let store = make_store();
        let tool = UpdateTaskTool::new(store);
        let result = tool.execute("s", serde_json::json!({
            "task_id": "nonexistent",
            "status": "completed"
        })).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                let text = v.as_str().unwrap();
                assert!(text.contains("not found"), "got: {}", text);
            }
            other => panic!("expected Completed text, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn update_task_tool_with_result() {
        let store = make_store();
        let create = CreateTaskTool::new(store.clone());
        let update = UpdateTaskTool::new(store.clone());
        create.execute("s", serde_json::json!({"description": "task"})).await.unwrap();
        let result = update.execute("s", serde_json::json!({
            "task_id": "task_1",
            "status": "completed",
            "result": "all tests passed"
        })).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                assert_eq!(v["result"], "all tests passed");
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn list_tasks_tool_empty() {
        let store = make_store();
        let tool = ListTasksTool::new(store);
        let result = tool.execute("s", serde_json::json!({})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                assert_eq!(v.as_str().unwrap(), "No tasks created yet.");
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn list_tasks_tool_with_tasks() {
        let store = make_store();
        let create = CreateTaskTool::new(store.clone());
        let update = UpdateTaskTool::new(store.clone());
        let list = ListTasksTool::new(store.clone());

        create.execute("s", serde_json::json!({"description": "first"})).await.unwrap();
        create.execute("s", serde_json::json!({"description": "second"})).await.unwrap();
        update.execute("s", serde_json::json!({"task_id": "task_1", "status": "completed"})).await.unwrap();

        let result = list.execute("s", serde_json::json!({})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                let text = v.as_str().unwrap();
                assert!(text.contains("1/2 completed"), "got: {}", text);
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn list_tasks_tool_shows_dependencies() {
        let store = make_store();
        let create = CreateTaskTool::new(store.clone());
        let list = ListTasksTool::new(store.clone());

        create.execute("s", serde_json::json!({"description": "first"})).await.unwrap();
        create.execute("s", serde_json::json!({
            "description": "second",
            "depends_on": ["task_1"]
        })).await.unwrap();

        let result = list.execute("s", serde_json::json!({})).await.unwrap();
        match result {
            ToolExecResult::Completed(v) => {
                let text = v.as_str().unwrap();
                assert!(text.contains("depends:"), "got: {}", text);
            }
            other => panic!("expected Completed, got {:?}", other),
        }
    }

    #[test]
    fn all_task_tools_correct_names() {
        let store = make_store();
        let tools: Vec<Box<dyn AgentTool>> = vec![
            Box::new(CreateTaskTool::new(store.clone())),
            Box::new(UpdateTaskTool::new(store.clone())),
            Box::new(ListTasksTool::new(store)),
        ];
        for tool in &tools {
            assert_eq!(tool.name(), tool.definition().name);
        }
    }

    #[test]
    fn all_task_tools_auto_execute() {
        let store = make_store();
        let tools: Vec<Box<dyn AgentTool>> = vec![
            Box::new(CreateTaskTool::new(store.clone())),
            Box::new(UpdateTaskTool::new(store.clone())),
            Box::new(ListTasksTool::new(store)),
        ];
        for tool in &tools {
            assert_eq!(tool.permission(), ToolPermission::AutoExecute, "tool {} should be AutoExecute", tool.name());
        }
    }
}
