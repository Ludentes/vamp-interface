---
status: live
topic: lam-chibi-recipe
---

# Research: Gaussian-splat editing best practices vs. our chibi scale-ratio approach

**Date:** 2026-05-15
**Sources:** 9 sources — GaussianAvatars (CVPR 2024), CAGE-GS (arXiv 2504.12800), GSDeformer (arXiv 2405.15491), SpLap (arXiv 2511.19542), SuGaR (CVPR 2024), GS-editing survey (arXiv 2508.09977), GaussianEditor (CVPR 2024), 3DGS survey (arXiv 2401.03890), CAGE-GS sketch variant (arXiv 2411.12168).

---

## Executive Summary

We are **not** doing something fundamentally wrong. Our two core moves — isotropic per-vertex `_gm.scaling` ratio (v2) and full anisotropic `Σ' = JΣJᵀ` with SVD recovery (v3) — are exactly what the canonical mesh-bound and cage-based methods do. GaussianAvatars (the reference FLAME-splat method) transforms splat scale by a single isotropic scalar `k = mean edge length`; CAGE-GS does `Σ' = JRSSᵀRᵀJᵀ` then SVD-recovers `(R', S')`. The important finding is that **the iris-through-lid leak was never a covariance problem**, so no covariance fix can close it — which is exactly why v3 came out visually ≈ v2. The literature is unanimous that covariance-update methods (CAGE-GS, SpLap, J·SVD) **do not handle the density/occlusion gap that opens when a surface stretches**. The only family that does is **densification-on-deform** (GSDeformer splits Gaussians to handle bending). Our already-proposed `LAM_AUX_SPLATS_PLY` aux-splat path is the literature-aligned fix; the lid-boost α-hack is a workaround for a missing densification step.

## Key Findings

### Our isotropic scale ratio matches the canonical FLAME-splat method

GaussianAvatars — the CVPR 2024 method that LAM's per-vertex binding descends from — rigs each Gaussian to a triangle frame and transforms it by `μ' = kRμ + T`, `r' = Rr`, `s' = ks`, where `k` is **a scalar derived from mean edge length** [1]. Scale is updated *isotropically* by a single per-triangle scalar. Our v2 `LAM_CHIBI_SCALE_RATIO` (per-vertex `mean_edge_chibi / mean_edge_canon`) is the same construction lifted to LAM's per-vertex topology. So the "scale-ratio hack" is not a hack — it is the standard mesh-bound transform. The one place we diverge from GaussianAvatars is that they also scale *position* by `k`; LAM bakes splat offsets differently, but the scaling channel is identical in spirit.

GaussianAvatars also reports a regularization detail relevant to our smear-at-large-s failures: they apply a **scaling loss with threshold ε=0.6** specifically to stop "magnified jitter from small triangle rotations" [1]. Without it, splats either blow up or collapse. Our large-s smear is the same instability — it argues for clamping the boosted scale-ratio rather than letting α push it arbitrarily high.

### The "correct" anisotropic update (v3) provably cannot fix occlusion

CAGE-GS implements the full anisotropic path: decompose `Σ = RSSᵀRᵀ`, apply the deformation-gradient Jacobian as `Σ' = JRSSᵀRᵀJᵀ`, then SVD-recover `(R', S')` [2]. This is our v3 J·SVD, and it is the field's accepted best practice for covariance under deformation. But CAGE-GS's own paper notes it **"does not explicitly address density/occlusion adjustments when surface area changes — opacity α is preserved unchanged"** [2]. SpLap (Nov 2025) likewise adapts kernels to "maintain coverage on the deformed surface" but **explicitly does not densify or split Gaussians when surfaces stretch** [4]. The GS-editing survey confirms the gap: it has no section on densification-during-deformation and lists "geometric artifacts / loss of structural integrity during manipulation" as an open failure mode [3].

This triangulates cleanly with our own diagnosis (`project_chibi_iris_lid_mask_v3_falsification`): the lid leak is alpha-accumulation through a sparse splat sheet that got stretched, not a covariance error. v3 restored covariance *correctly* and the leak stayed — because correct covariance on a too-sparse sheet is still a too-sparse sheet. **No method in the covariance family would have fixed it.** This is not a bug in our math; it is a known boundary of the whole approach.

### The literature-aligned fix is densification, and we already proposed it

The one method that handles surface stretch by adding primitives is **GSDeformer**: it builds a proxy point cloud, deforms it cage-style, transfers the deformation back, and **splits the relevant Gaussians to handle bending** [5]. GaussianAvatars handles it at training time — splats are "added and removed adaptively, with binding relation to triangles inherited" so coverage stays dense as the mesh moves [1]. The principle both encode: **when a bound surface expands, you add splats; you do not just enlarge the ones you have.** Enlarging (our boost) trades a coverage gap for a smear, which is why every α we sweep is a compromise.

Our `LAM_AUX_SPLATS_PLY` hook + the proposed aux-splat builder (stamp new splats with count ∝ local `|det J| − 1`, surface-aligned orientation from FLAME tangent basis) is precisely the GSDeformer/GaussianAvatars move. It is the recommended path, not a fallback. It also dodges the over-occlusion regression at small `s` that a global boost causes, because it adds nothing where `|det J| ≤ 1`.

### Editing-tool landscape — what is and isn't relevant to us

GaussianEditor and the text-guided family (GSEditPro, GaussCtrl, 3DGS-Drag) edit *appearance and content* via 2D-diffusion guidance with semantic tracing [3][7]. Not relevant to a geometric chibi deform, but the **semantic-tracing / 3D-mask** idea is relevant to our mask derivation — it is the same problem as our blink-displacement lid mask, and confirms animation-derived masks over topological regions.

SuGaR extracts an editable mesh from a splat scene so you can rig/sculpt in Blender, then re-binds splats [6]. We already *have* the mesh (FLAME) and the binding (LAM), so SuGaR's pipeline collapses to "edit the FLAME mesh, let LAM re-bind" — which is exactly `LAM_EDIT_XYZ_OBJ`. Confirms our architecture; offers nothing new.

Cage-based tools (GSDeformer, CAGE-GS, SpLap, sketch-guided) are for deforming *unstructured* trained splat scenes that have no mesh. We have a mesh, so we get the deformation gradient `J` analytically from the FLAME→chibi vertex correspondence instead of from a fitted cage — strictly better-conditioned than what these papers work with. The one transferable piece is GSDeformer's split-on-stretch.

## Comparison

| Method | Scale/covariance update | Densifies on stretch? | Needs a mesh? | Relevance to us |
|---|---|---|---|---|
| GaussianAvatars [1] | isotropic scalar `k` (mean edge) | yes — train-time adaptive | yes (FLAME) | our v2 = their `s'=ks`; copy their scaling-loss clamp |
| CAGE-GS [2] | full `JΣJᵀ` + SVD recovery | **no** (opacity fixed) | no (cage) | our v3; confirms v3 is correct *and* insufficient |
| SpLap [4] | surface-preserving kernel adaptation | **no** | no | confirms covariance-only family can't fix occlusion |
| GSDeformer [5] | cage transform | **yes — splits Gaussians** | no | the fix: split/aux-splat on stretch |
| SuGaR [6] | edit mesh in Blender, re-bind | n/a | yes (extracted) | = our `LAM_EDIT_XYZ_OBJ`; no new capability |

## Open Questions

- **Aux-splat orientation.** GSDeformer splits along the bend; we would stamp fresh splats on the FLAME tangent basis. Whether LAM's identity-only Gaussian net tolerates externally-stamped splats (SH colour, opacity) without an OOD shift is unverified — needs a smoke render. The neural-renderer-override-audit rule applies: stamping splats is another `_gm.*` injection.
- **No source gives a closed-form density target.** "Add splats ∝ `|det J| − 1`" is our heuristic; no paper publishes a stretch→count law. GaussianAvatars densifies by gradient, not by analytic `J`. Single-source / unverified.
- **Scaling-loss clamp value.** GaussianAvatars' ε=0.6 is tuned for animation jitter on a fixed identity, not for a deliberate 2× chibi deform. The clamp magnitude does not transfer directly; it argues for *a* clamp, not for that number.

## Bottom line for the chibi thread

1. Stop treating the iris leak as a scale/covariance problem — the literature proves that family tops out exactly where v3 did.
2. The lid-boost α-sweep is a stopgap; pick the least-bad α to unblock renders, but log it as a workaround.
3. Promote the aux-splat densification path (`LAM_AUX_SPLATS_PLY`) from "fallback" to the primary fix — it is what GSDeformer/GaussianAvatars do, and it is the only thing that addresses the actual mechanism.
4. Add a clamp on the boosted scale-ratio (GaussianAvatars scaling-loss analogue) to kill the large-`s` smear.

## Sources

[1] Qian et al. "GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians." CVPR 2024. https://arxiv.org/html/2312.02069v2 (Retrieved 2026-05-15)
[2] "CAGE-GS: High-fidelity Cage Based 3D Gaussian Splatting Deformation." arXiv:2504.12800. https://arxiv.org/html/2504.12800v1 (Retrieved 2026-05-15)
[3] "A Survey on 3D Gaussian Splatting Applications: Segmentation, Editing, and Generation." arXiv:2508.09977v4. https://arxiv.org/html/2508.09977v4 (Retrieved 2026-05-15)
[4] "SpLap: Proxy-Free Gaussian Splats Deformation with Splat-Based Surface Estimation." arXiv:2511.19542. https://arxiv.org/html/2511.19542 (Retrieved 2026-05-15)
[5] Huang et al. "GSDeformer: Direct, Real-time and Extensible Cage-based Deformation for 3D Gaussian Splatting." arXiv:2405.15491. https://jhuangbu.github.io/gsdeformer/ (Retrieved 2026-05-15)
[6] Guédon & Lepetit. "SuGaR: Surface-Aligned Gaussian Splatting." CVPR 2024. https://github.com/Anttwo/SuGaR (Retrieved 2026-05-15)
[7] Chen et al. "GaussianEditor: Swift and Controllable 3D Editing with Gaussian Splatting." CVPR 2024. https://buaacyw.github.io/gaussian-editor/ (Retrieved 2026-05-15)
[8] "Sketch-guided Cage-based 3D Gaussian Splatting Deformation." arXiv:2411.12168. https://arxiv.org/abs/2411.12168 (Retrieved 2026-05-15)
[9] Chen et al. "A Survey on 3D Gaussian Splatting." arXiv:2401.03890v8. https://arxiv.org/html/2401.03890v8 (Retrieved 2026-05-15)
