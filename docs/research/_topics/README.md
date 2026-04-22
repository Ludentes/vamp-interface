## Living research indexes

One file per **active research thread**. Each index curates the dated
evidence docs in `docs/research/` into a current-belief summary plus an
ordered reading list.

**Read these first** on any fresh session before diving into dated docs.
A dated doc (`YYYY-MM-DD-*.md`) is append-only evidence; a topic index is
the *mutable interpretation layer* — it changes when our understanding
changes, and it tells you which dated docs are load-bearing vs
superseded.

### Index

| Topic | What it covers | Status |
|-------|----------------|--------|
| [manifold-geometry](manifold-geometry.md) | FluxSpace linearity in attention-cache vs image space, α-interp phase boundary, Hessian-geometry / RJF framing | live, active |
| [metrics-and-direction-quality](metrics-and-direction-quality.md) | Mahalanobis → `max_env`, ridge vs prompt-pair direction extraction, FluxSpace as winning coarse baseline; Mahalanobis-vs-Riemannian critique extension | live |
| [demographic-pc-pipeline](demographic-pc-pipeline.md) | Stage 0-4.5 staged extraction pipeline; paused at Stage 5 behind FluxSpace | live, paused |
| [archived-threads](archived-threads.md) | Superseded dated docs with reasons — *read to decide what NOT to mine for current truth* | archive |

### Maintenance rule

When you add a dated research doc under `docs/research/`, update the
matching topic index *in the same commit*. New thread → new topic file
and a row in this table. Superseded thread → mark status and leave the
index as provenance.
