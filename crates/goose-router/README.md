# goose-router

Pluggable cost-saving model routers for goose.

A **router** scores the current conversation and decides whether a turn can be
served by a cheaper *fast* model instead of the main frontier model. This is the
engine behind goose's **cost savings mode**: simple turns go to a small/cheap
model, hard turns stay on the frontier model.

The crate is built around a single trait so routing strategies are swappable:

```rust
pub trait Router: Send + Sync {
    fn name(&self) -> &'static str;
    fn route(&self, conversation: &Conversation) -> Option<RouteDecision>;
}
```

## Strategies

### `EmbeddingRouter` (default, offline)

The simple, self-contained approach ported from the
[`nvidia-router/llm-router`](https://github.com/) toolkit. It renders the
conversation, embeds it with a small ONNX encoder, and runs an MLP head to
produce a complexity score in `[0, 1]`. Below a threshold the turn is routed to
the fast model.

It loads a bundle from `~/.goose/complexity_model/`:

```
config.json            # bundle metadata (embedder + head architecture)
model.onnx             # ONNX encoder (e.g. multilingual-e5)
tokenizer.json         # HF tokenizer
weights.safetensors    # MLP head weights
```

If the bundle is absent, the router is simply disabled (no error).

### `HiddenStateRouter` (richer, pluggable)

The prefill / hidden-state approach (NVIDIA LLM Router v3 style) predicts
per-model success probabilities and selects the cheapest model inside a quality
tolerance band. Running that encoder natively in Rust isn't easy yet, so this
router delegates to a running `model-router` **sidecar** over its documented
`POST /v1/route` HTTP API. The `Router` trait boundary means a future native
in-process encoder can drop in without touching the agent loop.

## Enabling in goose

Cost savings mode is off by default. Turn it on with config/env:

```bash
export GOOSE_COST_SAVINGS_MODE=true
```

Then provide a router:

- **Embedding** (default): drop a bundle into `~/.goose/complexity_model/`.
- **Hidden-state sidecar**:
  ```bash
  export GOOSE_ROUTER_STRATEGY=hidden_state
  export GOOSE_ROUTER_SIDECAR_URL=http://localhost:8079
  ```

When enabled, the agent scores each turn and swaps to the provider's fast model
(`GOOSE_FAST_MODEL`, or the provider default) for low-complexity turns.

## Environment variables

| Variable | Strategy | Purpose |
|---|---|---|
| `GOOSE_COST_SAVINGS_MODE` | all | Master on/off switch (default `false`) |
| `GOOSE_ROUTER_STRATEGY` | all | `embedding` or `hidden_state` (auto-detect if unset) |
| `GOOSE_ROUTER_THRESHOLD` | embedding | Fast-model complexity cutoff in `[0,1]` (default `0.30`) |
| `GOOSE_ROUTER_SIDECAR_URL` | hidden_state | Base URL of the running router sidecar |
| `GOOSE_ROUTER_TOLERANCE` | hidden_state | Quality tolerance band (default `0.20`) |
| `GOOSE_ROUTER_SIDECAR_TIMEOUT_S` | hidden_state | HTTP timeout for sidecar calls (default `5`) |
