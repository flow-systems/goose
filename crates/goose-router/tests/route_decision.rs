//! End-to-end decision test for the embedding router on real goose
//! `Conversation` objects, using the installed bundle at
//! `~/.goose/complexity_model/`.
//!
//! Proves the full `Router::route` path (render → score → threshold) yields
//! sensible cost-saving decisions: a trivial turn routes to the fast model, a
//! demanding multi-step turn stays on the main model.
//!
//! Skipped (not failed) if the bundle is absent.

use std::path::PathBuf;

use goose_providers::conversation::{message::Message, Conversation};
use goose_router::{EmbeddingRouter, Router};

fn bundle_dir() -> PathBuf {
    dirs::home_dir()
        .expect("home dir")
        .join(".goose")
        .join("complexity_model")
}

#[test]
fn simple_turn_routes_to_fast_complex_stays_on_main() {
    let dir = bundle_dir();
    if !dir.join("config.json").exists() {
        eprintln!("skipping: no bundle at {}", dir.display());
        return;
    }

    let router = EmbeddingRouter::load_from_dir(&dir).expect("load bundle");

    let simple = Conversation::new_unvalidated(vec![Message::user().with_text("hi, how are you?")]);
    let complex = Conversation::new_unvalidated(vec![Message::user().with_text(
        "Refactor this multi-threaded Rust web server to use a bounded work-stealing \
         scheduler, prove the absence of data races, add graceful shutdown with \
         in-flight request draining, instrument it with OpenTelemetry spans, and write \
         property-based tests covering backpressure under load.",
    )]);

    let simple_decision = router.route(&simple).expect("route simple");
    let complex_decision = router.route(&complex).expect("route complex");

    eprintln!(
        "simple: complexity={:.4} use_fast={} ({}ms)",
        simple_decision.complexity, simple_decision.use_fast, simple_decision.elapsed_ms
    );
    eprintln!(
        "complex: complexity={:.4} use_fast={} ({}ms)",
        complex_decision.complexity, complex_decision.use_fast, complex_decision.elapsed_ms
    );

    assert!(
        simple_decision.complexity < complex_decision.complexity,
        "simple turn should score lower complexity than complex turn",
    );
    assert!(
        simple_decision.use_fast,
        "trivial greeting should route to the fast model (complexity {:.4})",
        simple_decision.complexity,
    );
    assert!(
        !complex_decision.use_fast,
        "demanding engineering task should stay on the main model (complexity {:.4})",
        complex_decision.complexity,
    );
}
