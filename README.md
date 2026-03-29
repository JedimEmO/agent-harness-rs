# agent-harness

A Rust framework for building LLM-powered agents with tool calling, approval gates, and streaming events.

## Architecture

```
agent-harness-anthropic ─────────> agent-harness-core
agent-harness-openai ────────────> agent-harness-core
agent-harness-tools-interaction ──> agent-harness-core
agent-harness-tools-memory ──────> agent-harness-core
agent-harness-tools-task ────────> agent-harness-core
agent-harness-sqlite ────────────> agent-harness-core + diesel
```

## Crates

| Crate | Description |
|-------|-------------|
| `agent-harness-core` | AI provider trait, agent runner, tool system, events, session/memory store traits |
| `agent-harness-anthropic` | Anthropic Messages API provider (`/v1/messages`, `x-api-key` auth) |
| `agent-harness-openai` | OpenAI Chat Completions provider (OpenRouter, vLLM, Ollama, Azure, OpenAI) |
| `agent-harness-tools-interaction` | User interaction tools: ask_user, offer_options, confirm_action, show_plan |
| `agent-harness-tools-memory` | Memory tools: save_memory, recall_memories (uses `MemoryStore` trait) |
| `agent-harness-tools-task` | Task tracking: create_task, update_task, list_tasks + in-memory TaskStore |
| `agent-harness-sqlite` | SQLite-backed `SessionStore` + `MemoryStore` via Diesel |

## Quick Start

```rust
use std::sync::Arc;
use agent_harness_core::*;
use agent_harness_anthropic::AnthropicProvider;
use agent_harness_tools_interaction::*;
use agent_harness_tools_task::*;

#[tokio::main]
async fn main() {
    // 1. Create a provider
    let provider: Arc<dyn AiProvider> = Arc::new(
        AnthropicProvider::new("sk-...".into(), "claude-sonnet-4-6".into())
    );

    // 2. Register tools
    let task_store = TaskStore::new();
    let registry = ToolRegistry::new();
    registry.register(Arc::new(AskUserTool)).await;
    registry.register(Arc::new(ShowPlanTool)).await;
    registry.register(Arc::new(CreateTaskTool::new(task_store.clone()))).await;

    // 3. Provide storage (implement SessionStore + MemoryStore, or use agent-harness-sqlite)
    let session_store: Arc<dyn SessionStore> = /* your impl */;
    let memory_store: Arc<dyn MemoryStore> = /* your impl */;

    // 4. Create runner
    let runner = AgentRunner::new(
        provider,
        Arc::new(registry),
        session_store,
        memory_store,
        AgentConfig::default(),
    );
    let session = runner.create_session("my-scope").await.unwrap();

    // 5. Run a turn with event streaming
    let (request, channels) = TurnRequest::new(
        &session.id,
        "my-scope",
        "Hello!",
        "You are a helpful assistant.",
    );

    // Consume events in a separate task
    tokio::spawn(async move {
        let mut event_rx = channels.event_rx;
        while let Some(event) = event_rx.recv().await {
            match event {
                AgentEvent::TextDelta { text } => print!("{}", text),
                AgentEvent::TextComplete { text, .. } => println!("\n{}", text),
                AgentEvent::TurnComplete => break,
                _ => {}
            }
        }
    });

    runner.run_turn(request).await.unwrap();
}
```

## Key Concepts

### scope_id

An opaque string that groups sessions, memories, and tasks. Your application decides what it represents — a project ID, user ID, workspace ID, etc.

### AiProvider

Implement `AiProvider` to add a new LLM backend. Only two methods are required:

- `converse()` — send a conversation and get a response
- `capabilities()` — report what the provider supports

Default implementations are provided for `generate_text` (wraps `converse`), `generate_image`, `analyze_image` (both return `NotSupported`), and `converse_stream` (wraps `converse` into a single-item stream).

### Tool System

Implement `AgentTool` for custom tools:

```rust
#[async_trait]
impl AgentTool for MyTool {
    fn name(&self) -> &str { "my_tool" }
    fn definition(&self) -> ToolDefinition { /* JSON schema */ }
    fn permission(&self) -> ToolPermission { ToolPermission::AutoExecute }
    async fn execute(&self, scope_id: &str, args: Value) -> Result<ToolExecResult, AgentError> {
        Ok(ToolExecResult::text("done"))
    }
}
```

Tools return either `Completed(result)` or `NeedsInteraction(request)`.

### Approval Flow

Tools with `RequiresApproval` block until the caller sends an approval via `TurnChannels.approval_tx`. When a plan is approved (via `ShowPlanTool`), subsequent `RequiresApproval` tools auto-approve for that turn.

### Event Streaming

`TurnRequest::new()` returns `(request, TurnChannels)`. Consume `TurnChannels.event_rx` to receive events in real-time:

`Thinking` → `ToolCallStarted` → `ToolCallCompleted` → ... → `TextComplete` → `TurnComplete`

### Configuration

`AgentConfig` controls runtime behavior:

| Field | Default | Description |
|-------|---------|-------------|
| `max_tool_rounds` | 20 | Max tool-calling rounds per turn |
| `max_tokens` | 4096 | Max tokens for AI provider responses |
| `max_context_tokens` | 100,000 | Token budget for conversation context |
| `tool_timeout_secs` | None | Per-tool execution timeout |
| `max_retries` | 3 | Retries on rate-limit errors |
| `retry_base_delay_ms` | 1000 | Base delay for exponential backoff |

## Running the Example

```sh
ANTHROPIC_API_KEY=sk-... cargo run -p simple-chatbot
```

## Building

```sh
cargo check --workspace
cargo doc --workspace --no-deps
```
