//! Hidden-state (prefill) router.
//!
//! The prefill approach predicts per-model success probabilities from the
//! hidden states of a small local encoder, then selects the cheapest model
//! inside a quality tolerance band. Running that encoder natively in Rust is
//! not easy yet, so this router is pluggable: it delegates the decision to a
//! running `model-router` sidecar over its documented `POST /v1/route` HTTP
//! API (from the `nvidia-router/llm-router` toolkit).
//!
//! Enable it by pointing goose at a sidecar:
//!
//! ```text
//! export GOOSE_ROUTER_STRATEGY=hidden_state
//! export GOOSE_ROUTER_SIDECAR_URL=http://localhost:8079
//! ```
//!
//! Because everything goes through the [`Router`] trait, a future native
//! in-process encoder can replace this implementation without touching the
//! agent loop.

use std::time::{Duration, Instant};

use serde::Deserialize;

use goose_providers::conversation::Conversation;

use crate::{render::render_for_routing, RouteDecision, Router};

const DEFAULT_TIMEOUT_SECS: u64 = 5;
const DEFAULT_TOLERANCE: f32 = 0.20;

/// Routes by consulting a running prefill router sidecar.
pub struct HiddenStateRouter {
    sidecar_url: String,
    tolerance: f32,
    timeout: Duration,
}

#[derive(Debug, Deserialize)]
struct RouteResponse {
    selected_model: String,
    #[serde(default)]
    model_names: Vec<String>,
    #[serde(default)]
    costs: Vec<CostEntry>,
}

#[derive(Debug, Deserialize)]
struct CostEntry {
    #[serde(default)]
    cost_per_m_output_tokens: f32,
}

impl HiddenStateRouter {
    /// Construct from the environment. Returns `None` when no sidecar URL is
    /// configured (`GOOSE_ROUTER_SIDECAR_URL`).
    pub fn try_load_default() -> Option<Self> {
        let sidecar_url = std::env::var("GOOSE_ROUTER_SIDECAR_URL")
            .ok()
            .map(|s| s.trim().trim_end_matches('/').to_string())
            .filter(|s| !s.is_empty())?;

        let tolerance = std::env::var("GOOSE_ROUTER_TOLERANCE")
            .ok()
            .and_then(|v| v.trim().parse::<f32>().ok())
            .filter(|v| (0.0..=1.0).contains(v))
            .unwrap_or(DEFAULT_TOLERANCE);

        let timeout = std::env::var("GOOSE_ROUTER_SIDECAR_TIMEOUT_S")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .map(Duration::from_secs)
            .unwrap_or_else(|| Duration::from_secs(DEFAULT_TIMEOUT_SECS));

        tracing::info!(
            target: "goose::router",
            url = %sidecar_url,
            tolerance,
            "hidden-state router loaded (sidecar delegation)",
        );

        Some(Self {
            sidecar_url,
            tolerance,
            timeout,
        })
    }

    fn query(&self, question: &str) -> anyhow::Result<RouteResponse> {
        let body = serde_json::json!({
            "question": question,
            "tolerance": self.tolerance,
        });
        let resp: RouteResponse = ureq::AgentBuilder::new()
            .timeout(self.timeout)
            .build()
            .post(&format!("{}/v1/route", self.sidecar_url))
            .send_json(body)?
            .into_json()?;
        Ok(resp)
    }
}

/// A turn uses the fast model when the sidecar selected anything other than
/// the single most expensive model in the pool. That mirrors goose's binary
/// main-vs-fast choice: the router decided a cheaper model is good enough.
fn decide_use_fast(resp: &RouteResponse) -> (bool, f32) {
    if resp.model_names.is_empty() || resp.costs.len() != resp.model_names.len() {
        // Without a comparable pool, treat any non-empty selection as "fast"
        // only if we can't prove it is the top model. Be conservative: main.
        return (false, 1.0);
    }

    let selected_idx = resp
        .model_names
        .iter()
        .position(|m| m == &resp.selected_model);

    let max_cost = resp
        .costs
        .iter()
        .map(|c| c.cost_per_m_output_tokens)
        .fold(f32::MIN, f32::max);

    match selected_idx {
        Some(idx) => {
            let selected_cost = resp.costs[idx].cost_per_m_output_tokens;
            let use_fast = selected_cost < max_cost;
            // Rough complexity proxy: how close the selected model's cost is to
            // the top of the ladder (higher cost ⇒ harder turn).
            let complexity = if max_cost > 0.0 {
                (selected_cost / max_cost).clamp(0.0, 1.0)
            } else {
                1.0
            };
            (use_fast, complexity)
        }
        None => (false, 1.0),
    }
}

impl Router for HiddenStateRouter {
    fn name(&self) -> &'static str {
        "hidden_state"
    }

    fn route(&self, conversation: &Conversation) -> Option<RouteDecision> {
        let rendered = render_for_routing(conversation)?;
        let started = Instant::now();
        match self.query(&rendered) {
            Ok(resp) => {
                let (use_fast, complexity) = decide_use_fast(&resp);
                Some(RouteDecision {
                    complexity,
                    use_fast,
                    elapsed_ms: started.elapsed().as_millis() as u64,
                })
            }
            Err(e) => {
                tracing::warn!(
                    target: "goose::router",
                    "hidden-state sidecar query failed, defaulting to main model: {:#}",
                    e
                );
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resp(selected: &str, names: &[&str], costs: &[f32]) -> RouteResponse {
        RouteResponse {
            selected_model: selected.to_string(),
            model_names: names.iter().map(|s| s.to_string()).collect(),
            costs: costs
                .iter()
                .map(|&c| CostEntry {
                    cost_per_m_output_tokens: c,
                })
                .collect(),
        }
    }

    #[test]
    fn top_model_means_main() {
        let r = resp("opus", &["nano", "mini", "opus"], &[0.2, 0.4, 25.0]);
        let (use_fast, complexity) = decide_use_fast(&r);
        assert!(!use_fast);
        assert_eq!(complexity, 1.0);
    }

    #[test]
    fn cheap_model_means_fast() {
        let r = resp("nano", &["nano", "mini", "opus"], &[0.2, 0.4, 25.0]);
        let (use_fast, complexity) = decide_use_fast(&r);
        assert!(use_fast);
        assert!(complexity < 0.5);
    }

    #[test]
    fn missing_costs_is_conservative() {
        let r = resp("nano", &["nano", "mini"], &[]);
        let (use_fast, _) = decide_use_fast(&r);
        assert!(!use_fast);
    }
}
