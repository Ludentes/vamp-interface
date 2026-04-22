# Local paper archive

Canonical PDFs of papers we rely on, so future sessions don't have to re-fetch
and so our reading of the method survives arXiv outages / paper revisions.

| File | Paper | Why it's here |
|------|-------|---------------|
| `fluxspace-2412.09611.pdf` | Dalva et al., *FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers*, CVPR 2025 (arXiv 2412.09611) | Training-free DiT-internal intervention on Flux. Our Stage 4.5 baseline. Our prompt-pair "FluxSpace-coarse" is a strawman — the real coarse variant operates on pooled adaLN features inside each joint DiT block; the real fine variant projects joint-attention outputs ℓθ orthogonally to a prior/null condition. See `docs/research/2026-04-14-rectifid-fluxspace-flowchef-verification.md` §2. |
| `hessian-geometry-2506.10632.pdf` | Lobashev, Guskov, Larchenko, Tamm, *Hessian Geometry of Latent Space in Generative Models*, ICML 2025 (arXiv 2506.10632) | Predicts the α≈0.45 phase boundary we observed in the Mona-Lisa→Joker `mix_b` sweep (Lipschitz divergence at Fisher-metric phase transition). Theoretical frame for `docs/research/2026-04-22-alpha-interp-phase-boundary.md`. |
| `learning-on-manifold-rjf-2602.10099.pdf` | Kumar & Patel, *Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders* (arXiv 2602.10099) | Diagnoses our non-monotonic `mouthSmile` under Euclidean `mix_b` as geometric-interference failure on hyperspherical features; proposes Riemannian Flow Matching with Jacobi regularisation (RJF) as the corrected alternative. |
| `diffusion-string-method-2602.22122.pdf` | Moreau et al., *Probing the Geometry of Diffusion Models with the String Method* (arXiv 2602.22122) | Likelihood-vs-realism paradox explains why linear attention-cache paths produce off-manifold images; provides Minimum-Energy-Path and Principal-Curve tooling for entropy-aware interpolation between endpoint prompts. |

When adding: save to `fluxspace-<arxiv_id>.pdf` or `<firstauthor>-<year>-<slug>.pdf`,
and add a row above with a one-line note on what load-bearing claim we take from it.
