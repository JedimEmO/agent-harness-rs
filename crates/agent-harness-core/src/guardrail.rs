use async_trait::async_trait;

/// Context provided to guardrails for validation decisions.
#[derive(Debug, Clone)]
pub struct GuardrailContext {
    pub session_id: String,
    pub scope_id: String,
    /// The tool calls that were made in the current round, if any.
    pub tool_names_used: Vec<String>,
    /// Current round number in the tool loop.
    pub round: usize,
}

/// Outcome of a guardrail validation.
#[derive(Debug, Clone)]
pub enum GuardrailResult {
    /// Output is acceptable.
    Pass,
    /// Output should be modified (e.g., redacted).
    Modify(String),
    /// Output should be blocked entirely, with an explanation.
    Block(String),
}

/// Trait for validating LLM outputs before they reach the user.
///
/// Guardrails are checked on text responses before they are stored and emitted.
/// They can pass, modify, or block outputs.
///
/// # Example
///
/// ```rust,no_run
/// use agent_harness_core::*;
/// use async_trait::async_trait;
///
/// struct PiiGuardrail;
///
/// #[async_trait]
/// impl OutputGuardrail for PiiGuardrail {
///     fn name(&self) -> &str { "pii_filter" }
///
///     async fn validate(&self, output: &str, _ctx: &GuardrailContext) -> GuardrailResult {
///         // Check for potential PII patterns
///         if output.contains("SSN") || output.contains("social security") {
///             GuardrailResult::Block("Response may contain PII".to_string())
///         } else {
///             GuardrailResult::Pass
///         }
///     }
/// }
/// ```
#[async_trait]
pub trait OutputGuardrail: Send + Sync {
    fn name(&self) -> &str;
    async fn validate(&self, output: &str, context: &GuardrailContext) -> GuardrailResult;
}
