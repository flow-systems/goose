//! Parity test for the embedding router. Loads `~/.goose/complexity_model/`
//! and its `parity_fixture.jsonl` (produced by the llm-router training
//! pipeline), then asserts the Rust scoring pipeline matches the Python
//! reference within tolerance on every row.
//!
//! Skipped (not failed) if the bundle is absent — CI without local weights
//! just won't exercise this.

use std::path::PathBuf;

use goose_router::EmbeddingRouter;
use serde::Deserialize;

const TOLERANCE: f32 = 1e-3;

#[derive(Debug, Deserialize)]
struct ParityRow {
    text: String,
    expected_complexity: f32,
    expected_tool_calls_norm: f32,
}

fn bundle_dir() -> PathBuf {
    dirs::home_dir()
        .expect("home dir")
        .join(".goose")
        .join("complexity_model")
}

#[test]
fn parity_against_python_reference() {
    let dir = bundle_dir();
    let cfg_path = dir.join("config.json");
    let fixture_path = dir.join("parity_fixture.jsonl");
    if !cfg_path.exists() || !fixture_path.exists() {
        eprintln!(
            "skipping: missing {} or {}",
            cfg_path.display(),
            fixture_path.display()
        );
        return;
    }

    let router = EmbeddingRouter::load_from_dir(&dir).expect("load embedding bundle");

    let fixture = std::fs::read_to_string(&fixture_path).expect("read fixture");
    let mut rows = 0usize;
    for line in fixture.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let row: ParityRow = serde_json::from_str(line).expect("parse fixture row");
        let score = router.score(&row.text).expect("score row");

        let dc = (score.complexity - row.expected_complexity).abs();
        let dt = (score.tool_calls_norm - row.expected_tool_calls_norm).abs();
        assert!(
            dc <= TOLERANCE,
            "complexity mismatch: got {}, expected {} (|Δ|={}) for text starting {:?}",
            score.complexity,
            row.expected_complexity,
            dc,
            &row.text.chars().take(40).collect::<String>(),
        );
        assert!(
            dt <= TOLERANCE,
            "tool_calls_norm mismatch: got {}, expected {} (|Δ|={})",
            score.tool_calls_norm,
            row.expected_tool_calls_norm,
            dt,
        );
        rows += 1;
    }

    assert!(rows > 0, "fixture had no rows");
    eprintln!("parity verified on {rows} rows");
}
