//! Simple CLI chatbot demonstrating the ironflow agent framework.
//!
//! Run with: `ANTHROPIC_API_KEY=... cargo run -p simple-chatbot`

mod stores;

use std::io::{self, Write};
use std::sync::Arc;

use ironflow_core::*;
use ironflow_anthropic::AnthropicProvider;
use ironflow_tools_interaction::*;
use ironflow_tools_task::*;

use stores::{InMemorySessionStore, InMemoryMemoryStore};

const SYSTEM_PROMPT: &str = "\
You are a helpful assistant. You can use tools to interact with the user \
and track tasks. Be concise and direct in your responses.";

#[tokio::main]
async fn main() {
    let api_key = std::env::var("ANTHROPIC_API_KEY")
        .expect("Set ANTHROPIC_API_KEY environment variable");
    let model = std::env::var("IRONFLOW_MODEL")
        .unwrap_or_else(|_| "claude-sonnet-4-6".to_string());

    // 1. Create provider
    let provider: Arc<dyn AiProvider> = Arc::new(
        AnthropicProvider::new(api_key, model)
    );

    // 2. Build tool registry
    let task_store = TaskStore::new();
    let registry = ToolRegistry::new();
    registry.register(Arc::new(AskUserTool)).await;
    registry.register(Arc::new(OfferOptionsTool)).await;
    registry.register(Arc::new(ConfirmActionTool)).await;
    registry.register(Arc::new(ShowPlanTool)).await;
    registry.register(Arc::new(CreateTaskTool::new(task_store.clone()))).await;
    registry.register(Arc::new(UpdateTaskTool::new(task_store.clone()))).await;
    registry.register(Arc::new(ListTasksTool::new(task_store))).await;

    // 3. Create stores
    let session_store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::new());
    let memory_store: Arc<dyn MemoryStore> = Arc::new(InMemoryMemoryStore::new());

    // 4. Create runner
    let runner = AgentRunner::new(
        provider,
        Arc::new(registry),
        session_store,
        memory_store,
        AgentConfig::default(),
    );

    // 5. Create a session
    let scope_id = "chatbot";
    let session = runner.create_session(scope_id).await.expect("failed to create session");
    println!("Session created: {}", session.id);
    println!("Type your messages (Ctrl+C to quit):\n");

    // 6. REPL loop
    loop {
        print!("> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        if io::stdin().read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }
        let input = input.trim();
        if input.is_empty() {
            continue;
        }

        // Create TurnRequest with channels — much cleaner than 8 parameters
        let (request, channels) = TurnRequest::new(
            &session.id,
            scope_id,
            input,
            SYSTEM_PROMPT,
        );

        let interaction_tx = channels.interaction_tx;

        // Spawn event consumer
        let event_handle = tokio::spawn(async move {
            let mut event_rx = channels.event_rx;
            while let Some(event) = event_rx.recv().await {
                match event {
                    AgentEvent::TextDelta { text } => {
                        // Stream tokens in real-time
                        eprint!("{}", text);
                    }
                    AgentEvent::TextComplete { text, .. } => {
                        // Clear the streamed text and show final version
                        eprintln!();
                        println!("\nAssistant: {}\n", text);
                    }
                    AgentEvent::ToolCallStarted { tool_name, .. } => {
                        eprint!("[tool: {}] ", tool_name);
                    }
                    AgentEvent::ToolCallCompleted { tool_name, result, .. } => {
                        eprintln!("[{}: done]", tool_name);
                        if tool_name == "list_tasks" || tool_name == "create_task" || tool_name == "update_task" {
                            let display = match &result {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            eprintln!("  {}", display.replace('\n', "\n  "));
                        }
                    }
                    AgentEvent::InteractionNeeded { interaction_id, request } => {
                        let response = handle_interaction(&request);
                        let _ = interaction_tx.send((interaction_id, response)).await;
                    }
                    AgentEvent::ContextTruncated { dropped_messages, remaining_messages } => {
                        eprintln!("[context truncated: {} dropped, {} remaining]",
                            dropped_messages, remaining_messages);
                    }
                    AgentEvent::UsageUpdate { input_tokens, output_tokens } => {
                        eprintln!("[usage: in={} out={}]", input_tokens, output_tokens);
                    }
                    AgentEvent::RetryAttempt { attempt, delay_ms, .. } => {
                        eprintln!("[rate limited, retry #{} in {}ms]", attempt + 1, delay_ms);
                    }
                    AgentEvent::TurnComplete => break,
                    AgentEvent::Error { message, .. } => {
                        eprintln!("Error: {}", message);
                        break;
                    }
                    _ => {} // Thinking, SessionStarted, ToolApprovalNeeded, MemorySaved
                }
            }
        });

        // Run the agent turn
        let result = runner.run_turn(request).await;

        let _ = event_handle.await;

        if let Err(e) = result {
            eprintln!("Agent error: {}", e);
        }
    }
}

fn handle_interaction(request: &InteractionRequest) -> InteractionResponse {
    match request {
        InteractionRequest::AskUser { question, context } => {
            if let Some(ctx) = context {
                println!("\n[Context: {}]", ctx);
            }
            println!("\nAgent asks: {}", question);
            print!("Your answer: ");
            io::stdout().flush().unwrap();
            let mut answer = String::new();
            io::stdin().read_line(&mut answer).unwrap();
            InteractionResponse::Text { text: answer.trim().to_string() }
        }
        InteractionRequest::OfferOptions { question, options, .. } => {
            println!("\n{}", question);
            for (i, opt) in options.iter().enumerate() {
                println!("  {}. {} — {}", i + 1, opt.label,
                    opt.description.as_deref().unwrap_or(""));
            }
            print!("Choose (number): ");
            io::stdout().flush().unwrap();
            let mut choice = String::new();
            io::stdin().read_line(&mut choice).unwrap();
            let idx: usize = choice.trim().parse().unwrap_or(1);
            let id = options.get(idx.saturating_sub(1))
                .map(|o| o.id.clone())
                .unwrap_or_default();
            InteractionResponse::SelectedOptions { ids: vec![id] }
        }
        InteractionRequest::ConfirmAction { description, details } => {
            println!("\nConfirm: {}", description);
            if let Some(d) = details {
                println!("  Details: {}", d);
            }
            print!("Approve? (y/n): ");
            io::stdout().flush().unwrap();
            let mut answer = String::new();
            io::stdin().read_line(&mut answer).unwrap();
            InteractionResponse::Confirmed { approved: answer.trim().to_lowercase().starts_with('y') }
        }
        InteractionRequest::ShowPlan { title, steps } => {
            println!("\nPlan: {}", title);
            for (i, step) in steps.iter().enumerate() {
                println!("  {}. {}", i + 1, step.description);
            }
            print!("Approve plan? (y/n): ");
            io::stdout().flush().unwrap();
            let mut answer = String::new();
            io::stdin().read_line(&mut answer).unwrap();
            InteractionResponse::PlanApproved {
                approved: answer.trim().to_lowercase().starts_with('y'),
                modifications: None,
            }
        }
    }
}
