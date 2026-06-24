//! Embedding + MLP complexity router.
//!
//! Loads a self-contained bundle from `~/.goose/complexity_model/` containing a
//! fastembed-style ONNX embedder, an HF tokenizer, an MLP head exported as
//! safetensors, and a `config.json` describing the architecture. The bundle is
//! produced by the `nvidia-router/llm-router` training pipeline.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::TensorRef;
use safetensors::SafeTensors;
use serde::Deserialize;
use tokenizers::{Tokenizer, TruncationDirection, TruncationParams, TruncationStrategy};

use goose_providers::conversation::Conversation;

use crate::{render::render_for_routing, RouteDecision, Router};

const DEFAULT_BUNDLE_SUBDIR: &str = "complexity_model";
const MAX_SEQ_LEN: usize = 512;

/// Complexity threshold under which a turn is routed to the fast model.
/// Tuned empirically against the WildChat distribution; overridable via
/// `GOOSE_ROUTER_THRESHOLD`.
pub const DEFAULT_FAST_MODEL_THRESHOLD: f32 = 0.30;

#[derive(Debug, Deserialize)]
struct BundleConfig {
    format_version: u32,
    embedder: EmbedderConfig,
    head: HeadConfig,
}

#[derive(Debug, Deserialize)]
struct EmbedderConfig {
    repo_id: String,
    output_dim: usize,
    onnx_file: String,
    tokenizer_file: String,
}

#[derive(Debug, Deserialize)]
struct HeadConfig {
    input_dim: usize,
    hidden_dims: Vec<usize>,
    output_dim: usize,
    #[serde(default)]
    activation: String,
    #[serde(default)]
    output_activation: String,
}

struct LinearLayer {
    weight: Array2<f32>, // (out_dim, in_dim) — matches torch.nn.Linear convention
    bias: Array1<f32>,   // (out_dim,)
}

impl LinearLayer {
    fn apply(&self, x: &Array1<f32>) -> Array1<f32> {
        self.weight.dot(x) + &self.bias
    }
}

/// Output of a single complexity scoring call.
#[derive(Debug, Clone, Copy)]
pub struct ComplexityScore {
    pub complexity: f32,
    pub tool_calls_norm: f32,
    pub elapsed_ms: u64,
}

/// Embedding-based complexity router. Cheap to clone (state behind an `Arc`).
#[derive(Clone)]
pub struct EmbeddingRouter {
    inner: Arc<Inner>,
    threshold: f32,
}

struct Inner {
    embedder_dim: usize,
    tokenizer: Tokenizer,
    session: Mutex<Session>,
    needs_token_type_ids: bool,
    head: Vec<LinearLayer>,
    head_out_dim: usize,
    repo_id: String,
}

impl EmbeddingRouter {
    /// Default location: `~/.goose/complexity_model/`. Returns `None` (not an
    /// error) if the bundle is missing — callers treat that as "disabled".
    pub fn try_load_default() -> Option<Self> {
        let dir = default_bundle_dir()?;
        if !dir.join("config.json").exists() {
            tracing::info!(
                target: "goose::router",
                path = %dir.display(),
                "no embedding router bundle at default path; embedding routing disabled",
            );
            return None;
        }
        match Self::load_from_dir(&dir) {
            Ok(m) => {
                tracing::info!(
                    target: "goose::router",
                    path = %dir.display(),
                    threshold = m.threshold,
                    "embedding router loaded",
                );
                Some(m)
            }
            Err(e) => {
                tracing::warn!(
                    target: "goose::router",
                    path = %dir.display(),
                    error = %format!("{:#}", e),
                    "failed to load embedding router",
                );
                None
            }
        }
    }

    pub fn load_from_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("config.json");
        let cfg_text = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("reading {}", cfg_path.display()))?;
        let cfg: BundleConfig = serde_json::from_str(&cfg_text)
            .with_context(|| format!("parsing {}", cfg_path.display()))?;

        if cfg.format_version != 1 {
            bail!("unsupported bundle format_version {}", cfg.format_version);
        }
        if cfg.head.input_dim != cfg.embedder.output_dim {
            bail!(
                "head input_dim {} != embedder output_dim {}",
                cfg.head.input_dim,
                cfg.embedder.output_dim
            );
        }
        if !cfg.head.activation.is_empty() && cfg.head.activation != "relu" {
            bail!(
                "unsupported head activation {:?} (only \"relu\" is implemented)",
                cfg.head.activation
            );
        }
        if !cfg.head.output_activation.is_empty() && cfg.head.output_activation != "sigmoid" {
            bail!(
                "unsupported head output_activation {:?} (only \"sigmoid\" is implemented)",
                cfg.head.output_activation
            );
        }

        let tokenizer_path = dir.join(&cfg.embedder.tokenizer_file);
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("loading tokenizer at {}: {}", tokenizer_path.display(), e))?;
        // Respect the bundle's own truncation config (fastembed does the same).
        // BERT-family MiniLM bundles set 128; XLM-R-family e5 bundles set 512.
        // Only enforce a default when the tokenizer declares none, so special
        // tokens are still placed correctly when an input overflows.
        if tokenizer.get_truncation().is_none() {
            tokenizer
                .with_truncation(Some(TruncationParams {
                    direction: TruncationDirection::Right,
                    max_length: MAX_SEQ_LEN,
                    strategy: TruncationStrategy::LongestFirst,
                    stride: 0,
                }))
                .map_err(|e| anyhow!("enable truncation: {}", e))?;
        }

        let onnx_path = dir.join(&cfg.embedder.onnx_file);
        let session = Session::builder()?
            .commit_from_file(&onnx_path)
            .with_context(|| format!("loading ONNX from {}", onnx_path.display()))?;

        // BERT-family encoders (e.g. MiniLM) require a `token_type_ids` input;
        // XLM-R-family ones (e.g. e5) do not. Detect which by inspecting the
        // graph's declared inputs so the same loader handles both.
        let needs_token_type_ids = session
            .inputs()
            .iter()
            .any(|input| input.name() == "token_type_ids");

        let weights_path = dir.join("weights.safetensors");
        let weights_bytes = std::fs::read(&weights_path)
            .with_context(|| format!("reading {}", weights_path.display()))?;
        let head = load_head_weights(&weights_bytes, &cfg.head)?;

        Ok(Self {
            inner: Arc::new(Inner {
                embedder_dim: cfg.embedder.output_dim,
                tokenizer,
                session: Mutex::new(session),
                needs_token_type_ids,
                head,
                head_out_dim: cfg.head.output_dim,
                repo_id: cfg.embedder.repo_id,
            }),
            threshold: threshold_from_env(),
        })
    }

    /// Score one rendered conversation. ~25-50ms on CPU for typical inputs.
    pub fn score(&self, text: &str) -> Result<ComplexityScore> {
        let started = Instant::now();
        let embedding = self.embed(text)?;
        let mut activations = embedding;
        let last = self.inner.head.len() - 1;
        for (i, layer) in self.inner.head.iter().enumerate() {
            activations = layer.apply(&activations);
            if i != last {
                activations.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
        }
        activations.mapv_inplace(sigmoid);

        if activations.len() < self.inner.head_out_dim {
            bail!(
                "head output has {} values, expected {}",
                activations.len(),
                self.inner.head_out_dim
            );
        }

        Ok(ComplexityScore {
            complexity: activations[0],
            tool_calls_norm: if self.inner.head_out_dim >= 2 {
                activations[1]
            } else {
                0.0
            },
            elapsed_ms: started.elapsed().as_millis() as u64,
        })
    }

    pub fn embedder_repo(&self) -> &str {
        &self.inner.repo_id
    }

    fn embed(&self, text: &str) -> Result<Array1<f32>> {
        let encoding = self
            .inner
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenize: {}", e))?;
        let ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let seq_len = ids.len();
        let ids_arr = Array2::from_shape_vec((1, seq_len), ids)?;
        let mask_arr = Array2::from_shape_vec((1, seq_len), mask)?;
        let token_type_arr = Array2::<i64>::zeros((1, seq_len));

        let mut session = self
            .inner
            .session
            .lock()
            .map_err(|_| anyhow!("embedding router session mutex poisoned"))?;
        let mut inputs = ort::inputs![
            "input_ids" => TensorRef::from_array_view(ids_arr.view())?,
            "attention_mask" => TensorRef::from_array_view(mask_arr.view())?,
        ];
        if self.inner.needs_token_type_ids {
            inputs.push((
                "token_type_ids".into(),
                TensorRef::from_array_view(token_type_arr.view())?.into(),
            ));
        }
        let outputs = session.run(inputs)?;

        // The ONNX embedder outputs `last_hidden_state` of shape (1, seq_len, hidden).
        // For multilingual-e5-large (and most XLM-R-based models) fastembed does
        // attention-masked mean pooling, NO L2 normalization. We mirror that.
        let last_hidden = outputs
            .get("last_hidden_state")
            .ok_or_else(|| anyhow!("ONNX output is missing 'last_hidden_state'"))?;
        let tensor_view = last_hidden.try_extract_array::<f32>()?;
        let shape = tensor_view.shape();
        if shape.len() != 3 {
            bail!("expected 3D embedder output, got shape {:?}", shape);
        }
        let hidden = shape[2];
        if hidden != self.inner.embedder_dim {
            bail!(
                "embedder hidden dim {} != config output_dim {}",
                hidden,
                self.inner.embedder_dim
            );
        }

        let hidden_slice = tensor_view.slice(ndarray::s![0, .., ..]);
        let mut sum = Array1::<f32>::zeros(hidden);
        let mut mask_sum: f32 = 0.0;
        for (t, &m) in mask_arr.row(0).iter().enumerate() {
            if m == 0 {
                continue;
            }
            let mf = m as f32;
            mask_sum += mf;
            for (h, &v) in hidden_slice.slice(ndarray::s![t, ..]).iter().enumerate() {
                sum[h] += v * mf;
            }
        }
        if mask_sum <= 0.0 {
            bail!("attention mask is all zeros");
        }
        sum.mapv_inplace(|v| v / mask_sum);
        Ok(sum)
    }
}

impl Router for EmbeddingRouter {
    fn name(&self) -> &'static str {
        "embedding"
    }

    fn route(&self, conversation: &Conversation) -> Option<RouteDecision> {
        let rendered = render_for_routing(conversation)?;
        match self.score(&rendered) {
            Ok(score) => Some(RouteDecision {
                complexity: score.complexity,
                use_fast: score.complexity < self.threshold,
                elapsed_ms: score.elapsed_ms,
            }),
            Err(e) => {
                tracing::warn!(
                    target: "goose::router",
                    "embedding scoring failed, defaulting to main model: {:#}",
                    e
                );
                None
            }
        }
    }
}

fn threshold_from_env() -> f32 {
    std::env::var("GOOSE_ROUTER_THRESHOLD")
        .ok()
        .and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|v| (0.0..=1.0).contains(v))
        .unwrap_or(DEFAULT_FAST_MODEL_THRESHOLD)
}

fn default_bundle_dir() -> Option<PathBuf> {
    let home = dirs::home_dir()?;
    Some(home.join(".goose").join(DEFAULT_BUNDLE_SUBDIR))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn load_head_weights(bytes: &[u8], cfg: &HeadConfig) -> Result<Vec<LinearLayer>> {
    let st = SafeTensors::deserialize(bytes).context("parse safetensors")?;

    // The Python head is `Sequential(Linear, ReLU, Dropout, Linear, ReLU, Dropout, ...)`
    // followed by `Linear` named `out`. Inside `Sequential` the linears are at
    // indices 0, 3, 6, … which corresponds to `trunk.{0,3,6,…}` in the
    // state_dict. We materialize them in order plus the final `out` linear.
    let mut layers = Vec::with_capacity(cfg.hidden_dims.len() + 1);
    let mut sequential_idx = 0usize;
    let mut prev_dim = cfg.input_dim;
    for &h in &cfg.hidden_dims {
        let w_name = format!("trunk.{}.weight", sequential_idx);
        let b_name = format!("trunk.{}.bias", sequential_idx);
        layers.push(read_linear(&st, &w_name, &b_name, h, prev_dim)?);
        prev_dim = h;
        sequential_idx += 3; // skip ReLU + Dropout
    }
    layers.push(read_linear(
        &st,
        "out.weight",
        "out.bias",
        cfg.output_dim,
        prev_dim,
    )?);
    Ok(layers)
}

fn read_linear(
    st: &SafeTensors,
    weight_name: &str,
    bias_name: &str,
    expected_out: usize,
    expected_in: usize,
) -> Result<LinearLayer> {
    let w_tensor = st
        .tensor(weight_name)
        .with_context(|| format!("missing tensor {}", weight_name))?;
    let b_tensor = st
        .tensor(bias_name)
        .with_context(|| format!("missing tensor {}", bias_name))?;

    let w_shape = w_tensor.shape();
    if w_shape != [expected_out, expected_in] {
        bail!(
            "{}: shape {:?} != expected [{}, {}]",
            weight_name,
            w_shape,
            expected_out,
            expected_in
        );
    }
    let b_shape = b_tensor.shape();
    if b_shape != [expected_out] {
        bail!(
            "{}: shape {:?} != expected [{}]",
            bias_name,
            b_shape,
            expected_out
        );
    }

    let w_data = bytes_to_f32(w_tensor.data())?;
    let b_data = bytes_to_f32(b_tensor.data())?;
    let weight = Array2::from_shape_vec((expected_out, expected_in), w_data)?;
    let bias = Array1::from_vec(b_data);
    Ok(LinearLayer { weight, bias })
}

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if !bytes.len().is_multiple_of(4) {
        bail!("tensor byte length {} is not a multiple of 4", bytes.len());
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}
