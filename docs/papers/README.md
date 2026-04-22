# Local paper archive

Canonical PDFs of papers we rely on, so future sessions don't have to re-fetch
and so our reading of the method survives arXiv outages / paper revisions.

| File | Paper | Why it's here |
|------|-------|---------------|
| `fluxspace-2412.09611.pdf` | Dalva et al., *FluxSpace: Disentangled Semantic Editing in Rectified Flow Transformers*, CVPR 2025 (arXiv 2412.09611) | Training-free DiT-internal intervention on Flux. Our Stage 4.5 baseline. Our prompt-pair "FluxSpace-coarse" is a strawman — the real coarse variant operates on pooled adaLN features inside each joint DiT block; the real fine variant projects joint-attention outputs ℓθ orthogonally to a prior/null condition. See `docs/research/2026-04-14-rectifid-fluxspace-flowchef-verification.md` §2. |
| `hessian-geometry-2506.10632.pdf` | Lobashev, Guskov, Larchenko, Tamm, *Hessian Geometry of Latent Space in Generative Models*, ICML 2025 (arXiv 2506.10632) | Proposition 4.1 (p.9) is a toy model of exactly the α≈0.45 cliff phenomenon: diverging Lyapunov exponent at a bimodal phase boundary. Fisher metric here is the Hessian of fit log-Z over a 2D latent slice, not a sample covariance. Validated only on SD 1.5 / Ising / TASEP — not DiT / Flux. See `docs/research/2026-04-22-manifold-papers-distillation.md`. |
| `learning-on-manifold-rjf-2602.10099.pdf` | Kumar & Patel, *Learning on the Manifold: Unlocking Standard Diffusion Transformers with Representation Encoders* (arXiv 2602.10099) | Training-time only — does not cover test-time editing. Requires analytically known hypersphere manifold (LayerNorm-enforced); Flux attention cache has no such structure. Useful as conceptual framing ("chord vs geodesic") but not a usable method for our setting. |
| `diffusion-string-method-2602.22122.pdf` | Moreau et al., *Probing the Geometry of Diffusion Models with the String Method* (arXiv 2602.22122) | Fig. 4 directly shows non-monotonic log-likelihood along linear-initialised strings — closest match to our non-monotonic smile phenomenology. Principal-Curve regime flattens it. Operates on state space / VAE latent, not attention caches; requires score + velocity (Flux gives only velocity). Reparametrisation idea is portable as a cheap scheduling heuristic on `mix_b`. |

When adding: save to `fluxspace-<arxiv_id>.pdf` or `<firstauthor>-<year>-<slug>.pdf`,
and add a row above with a one-line note on what load-bearing claim we take from it.
