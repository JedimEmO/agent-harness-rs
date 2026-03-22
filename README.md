# ironflow

A Rust framework for building LLM-powered agents with tool calling, approval gates, and streaming events.

## Architecture

```
ironflow-anthropic ─────────> ironflow-core
ironflow-openai ────────────> ironflow-core
ironflow-tools-interaction ──> ironflow-core
ironflow-tools-memory ──────> ironflow-core
ironflow-tools-task ────────> ironflow-core
ironflow-sqlite ────────────> ironflow-core + diesel
```

## Crates

| Crate | Description |
|-------|-------------|
| `ironflow-core` | AI provider trait, agent runner, tool system, events, session/memory store traits |
| `ironflow-anthropic` | Anthropic Messages API provider (`/v1/messages`, `x-api-key` auth) |
| `ironflow-openai` | OpenAI Chat Completions provider (OpenRouter, vLLM, Ollama, Azure, OpenAI) |
| `ironflow-tools-interaction` | User interaction tools: ask_user, offer_options, confirm_action, show_plan |
| `ironflow-tools-memory` | Memory tools: save_memory, recall_memories (uses `MemoryStore` trait) |
| `ironflow-tools-task` | Task tracking: create_task, update_task, list_tasks + in-memory TaskStore |
| `ironflow-sqlite` | SQLite-backed `SessionStore` + `MemoryStore` via Diesel |

## Quick Start

```rust
use std::sync::Arc;
use ironflow_core::*;
use ironflow_anthropic::AnthropicProvider;
use ironflow_tools_interaction::*;
use ironflow_tools_task::*;

// 1. Create a provider
let provider: Arc<dyn AiProvider> = Arc::new(
    AnthropicProvider::new("sk-...".into(), "claude-sonnet-4-6".into())
);

// 2. Register tools
let task_store = TaskStore::new();
let mut registry = ToolRegistry::new();
registry.register(Box::new(AskUserTool));
registry.register(Box::new(ShowPlanTool));
registry.register(Box::new(CreateTaskTool::new(task_store.clone())));
// ... more tools

// 3. Provide storage (implement SessionStore + MemoryStore, or use ironflow-sqlite)
let session_store: Arc<dyn SessionStore> = /* your impl */;
let memory_store: Arc<dyn MemoryStore> = /* your impl */;

// 4. Create runner and go
let runner = AgentRunner::new(provider, Arc::new(registry), session_store, memory_store, AgentConfig::default());
let session = runner.create_session("my-scope").await?;

// 5. Run turns with event streaming
let (event_tx, mut event_rx) = tokio::sync::mpsc::channel(100);
let (_, mut interaction_rx) = tokio::sync::mpsc::channel(10);
let (_, mut approval_rx) = tokio::sync::mpsc::channel(10);

runner.run_turn(&session.id, "my-scope", "Hello!", "You are helpful.", event_tx, &mut interaction_rx, &mut approval_rx, false).await?;
```

## Key Concepts

### scope_id

An opaque string that groups sessions, memories, and tasks. Your application decides what it represents — a project ID, user ID, workspace ID, etc.

### Tool System

Implement `AgentTool` for custom tools:

```rust
#[async_trait]
impl AgentTool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn definition(&self) -> ToolDefinition { /* JSON schema */ }
    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }
    async fn execute(&self, scope_id: &str, args: Value) -> Result<ToolExecResult, AgentError> {
        Ok(ToolExecResult::Completed("done".into()))
    }
}
```

Tools return either `Completed(result)` or `NeedsInteraction(request)`.

### Approval Flow

Tools with `RequiresApproval` block until the caller sends an approval via the `approval_rx` channel. When a plan is approved (via `ShowPlanTool`), subsequent `RequiresApproval` tools auto-approve for that turn.

### Event Streaming

The runner emits `AgentEvent`s via the `event_tx` channel:
`Thinking` → `ToolCallStarted` → `ToolCallCompleted` → ... → `TextComplete` → `TurnComplete`

## Running the Example

```sh
ANTHROPIC_API_KEY=sk-... cargo run -p simple-chatbot
```

## Building

```sh
cargo check --workspace
cargo doc --workspace --no-deps
```
