//! Pluggable cost-saving model routers for goose.
//!
//! A [`Router`] scores the current conversation and decides whether a turn can
//! be served by a cheaper "fast" model instead of the main frontier model. This
//! is the engine behind goose's *cost savings mode*.
//!
//! Two routing strategies are supported behind the same trait:
//!
//! * [`EmbeddingRouter`] — the simple, self-contained approach ported from the
//!   `nvidia-router/llm-router` toolkit. It embeds the rendered conversation
//!   with a small ONNX encoder and runs an MLP head to produce a complexity
//!   score in `[0, 1]`. Works fully offline from a local bundle.
//! * [`HiddenStateRouter`] — the richer prefill/hidden-state approach (NVIDIA
//!   LLM Router v3 style) that predicts per-model success probabilities. This
//!   is harder to run locally today, so it is exposed as a pluggable extension
//!   point and is selected only when its bundle is present.
//!
//! Callers normally use [`load_default_router`], which picks the best available
//! strategy based on which bundles exist on disk, honouring the
//! `GOOSE_ROUTER_STRATEGY` override.

mod embedding;
mod hidden_state;
mod render;

pub use embedding::EmbeddingRouter;
pub use hidden_state::HiddenStateRouter;
pub use render::render_for_routing;

use goose_providers::conversation::Conversation;

/// What the agent loop needs to know after routing a turn.
#[derive(Debug, Clone, Copy)]
pub struct RouteDecision {
    /// Estimated cognitive complexity of the turn in `[0, 1]`.
    pub complexity: f32,
    /// Whether the turn should be served by the cheaper fast model.
    pub use_fast: bool,
    /// How long the routing decision took.
    pub elapsed_ms: u64,
}

/// A pluggable strategy that decides whether a turn can use the fast model.
///
/// Implementations are expected to be cheap to clone (wrap heavy state in an
/// `Arc`) and safe to call from the async agent loop on a blocking section.
pub trait Router: Send + Sync {
    /// Human-readable name of the strategy, for logs and diagnostics.
    fn name(&self) -> &'static str;

    /// Score `conversation` and decide whether to route to the fast model.
    ///
    /// Returns `None` when no decision can be made (e.g. the conversation has
    /// no user turn). Callers should treat `None` as "use the main model".
    fn route(&self, conversation: &Conversation) -> Option<RouteDecision>;
}

/// Which routing strategy to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterStrategy {
    /// Embedding + MLP complexity scorer (simple, offline).
    Embedding,
    /// Prefill hidden-state success predictor (richer, may not be available).
    HiddenState,
}

impl RouterStrategy {
    fn from_env() -> Option<Self> {
        match std::env::var("GOOSE_ROUTER_STRATEGY").ok()?.as_str() {
            "embedding" => Some(Self::Embedding),
            "hidden_state" | "hidden-state" | "prefill" => Some(Self::HiddenState),
            other => {
                tracing::warn!(
                    target: "goose::router",
                    value = other,
                    "unknown GOOSE_ROUTER_STRATEGY; ignoring",
                );
                None
            }
        }
    }
}

/// Load the best available router for cost savings mode.
///
/// Strategy selection:
///   1. If `GOOSE_ROUTER_STRATEGY` is set, honour it (returns `None` if that
///      strategy's bundle is missing).
///   2. Otherwise prefer the hidden-state router when its bundle is present,
///      falling back to the embedding router.
///
/// Returns `None` (not an error) when no router bundle is available — callers
/// should treat that as "cost savings routing disabled".
pub fn load_default_router() -> Option<Box<dyn Router>> {
    match RouterStrategy::from_env() {
        Some(RouterStrategy::Embedding) => {
            EmbeddingRouter::try_load_default().map(|r| Box::new(r) as Box<dyn Router>)
        }
        Some(RouterStrategy::HiddenState) => {
            HiddenStateRouter::try_load_default().map(|r| Box::new(r) as Box<dyn Router>)
        }
        None => {
            if let Some(r) = HiddenStateRouter::try_load_default() {
                Some(Box::new(r))
            } else {
                EmbeddingRouter::try_load_default().map(|r| Box::new(r) as Box<dyn Router>)
            }
        }
    }
}
