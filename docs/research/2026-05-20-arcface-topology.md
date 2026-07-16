---
status: live
topic: photobooth-sweep
---

# ArcFace embedding topology — manifold, linearity, paths, OOD, operators

**Date:** 2026-05-20
**Sources:** 8 sources retrieved (ArcFace CVPR 2019, SphereFace CVPR 2017, Arc2Face ECCV 2024, Arc2Morph 2026 preprint, SlerpFace, "Attributes Shape FR Embeddings" 2025, StyleID 2026 preprint, KappaFace 2022, vMF-bias 2022, InterFaceGAN CVPR 2020).
**Why this exists:** Our HyperSwap pipeline does `embedding = (1−α)·source + α·target` then re-normalizes (`scripts/swap_core.py`, Phase 3 sweep). That is a *straight chord* through the ArcFace 512-d hypersphere — projected back to the surface. The matryoshka target face is OOD for ArcFace's training prior. This doc collects what is actually known about whether that path makes sense, what could replace it, and what semantic operators (if any) exist in the space.

---

## Executive summary

ArcFace embeddings live on the unit hypersphere `S^511 ⊂ ℝ^512`; angular distance equals geodesic distance by construction [1][2]. Linear interpolation between two identity vectors, followed by L2-renormalization (lerp+renorm), traces a chord projected onto the sphere — equivalent to slerp to second order in the angle but *not* identical when the two vectors are far apart. Independent recent generative work confirms straight paths through this space produce plausible intermediate identities — Arc2Face explicitly shows "linearly blending ArcFace vectors" yields recognizable in-between faces [3, Fig. 10], and Arc2Morph builds an identity-morphing system on top of this property [4]. **So for nearby photoreal endpoints, the chord is fine.** The two caveats that matter for our use case: (a) the embedding is high intrinsic dimensionality (~300–400 of 512 PCs needed to reconstruct identity [3, Fig. 9]) so don't expect a clean low-rank "identity basis"; (b) ArcFace is severely OOD on stylized portraits — TPR collapses from 0.765 → 0.372 on unseen-method stylization [5], with two opposing failure modes (texture-sensitive, geometry-insensitive). The matryoshka doll's painted face sits squarely in that OOD region. Our `α·target` term is therefore pulling toward a *noisy, unreliably-positioned* point on the hypersphere, not toward a clean stylization-axis offset. **There is no published "photo→portrait direction" you can subtract in ArcFace space.** Practical implication: weight sweeps `w → 0` will reduce the magnitude of the perturbation, but the *direction* of perturbation is not a clean style axis — it's "wherever ArcFace happens to project this doll."

---

## Key findings

### The manifold: unit hypersphere with geodesic = angular metric

ArcFace's defining property is that all features are L2-normalized to the unit hypersphere `S^511` *and* class weight vectors are normalized, so the logit reduces to `cos θ` and the margin in the loss is an *additive angle* added to the target class angle [1]. Deng et al. state this explicitly: "the additive angular margin penalty is equal to the geodesic distance margin penalty in the normalized hypersphere" [1]. SphereFace [2] introduced the spherical embedding view; ArcFace tightened the geodesic correspondence.

Two consequences:

- The *natural* distance is cosine / angle, not Euclidean. Two embeddings `e₁, e₂` have `‖e₁ − e₂‖² = 2 − 2·cos θ`, so cosine ranks identically to chord length on the sphere.
- The *natural* path between two embeddings is the great-circle arc (slerp), not the straight chord. But for small angles the chord and arc coincide to O(θ²) — at θ ≈ 30° (cos ≈ 0.866, a strong match by ArcFace standards) the chord error is < 4%.

### Distribution on the sphere: vMF, not isotropic — κ varies per identity

The von Mises–Fisher (vMF) distribution is the natural "Gaussian on the sphere" and several FR works have modeled per-identity ArcFace clusters as vMF [6][7]. The concentration parameter `κ` measures how tightly an identity's samples cluster around its centroid; low-quality / high-variance identities have lower κ. KappaFace [7] explicitly uses this to set per-class margin: under-represented or hard identities get higher margins to compensate for lower κ.

For our case, this matters in one way: when we sweep `face_swapper_weight`, the source-side ArcFace vector we are mixing *toward* (the painted doll) is not a single confident point — it has a wide vMF spread (low κ) because doll-images are OOD. The mean direction we read off `target_face.normed_embedding` is a *sample* from a high-variance distribution. That's part of the per-photo instability we observed in Phase 2.

### Intrinsic dimensionality: high (~300–400 of 512), not low-rank

Arc2Face [3] performs PCA on ArcFace embeddings of their generation corpus and reports that "at least 300–400 components" of the 512-d space are needed to maintain facial fidelity (Fig. 9). This is the single most important quantitative finding for our purposes: **ArcFace is not concentrated in a low-d identity subspace.** This rules out the idea that we could find a small handful of "identity-preserving" directions and walk along them.

It also implies that any *single* linear direction (gender, age, style) explains only a small fraction of total embedding variance. There is no analogue of the StyleGAN W-space "age direction" with clean disentanglement.

### Linearity: chord-path through the sphere works for in-distribution endpoints

Two independent results confirm linear interpolation produces useful intermediate identities:

- **Arc2Face Fig. 10** [3]: shows transitions between subject pairs by "linearly blending their ArcFace vectors"; intermediate faces are "plausible" and recognizable as smooth blends.
- **Arc2Morph** [4]: full morphing system built on linear interpolation in ArcFace space; reports successful identity-preserving morphs across standard FR benchmarks.

The lerp-vs-slerp question: published comparisons are limited. The one direct datapoint is Arc2Morph's finding that **slerp in CLIP space outperforms slerp in ArcFace space alone** for identity-preserving morphing [4]. This is *not* a result about lerp-vs-slerp inside ArcFace; it's a result about which embedding space carries the richer interpolation semantics. The inference for us: ArcFace alone is a thin description of identity, and pushing it around with simple operations will produce simple, sometimes brittle, results.

SlerpFace [8] uses slerp for face-template protection but does not benchmark slerp-vs-lerp identity preservation; it's a privacy-preserving construction, not a quality study.

**Practical conclusion for HyperSwap:** for our weight ranges (α ∈ [−0.35, +0.35], embedding angle change ≤ ~20°), lerp+renorm ≈ slerp to within a few percent. The choice is not load-bearing. **What matters more** is what the target endpoint *is*, and ours is OOD.

### Known operators / semantic directions: weaker than the GAN literature suggests

The InterFaceGAN [9] family of results (linear SVM boundaries for gender, age, expression, eyeglasses, pose) is famous but lives in **StyleGAN's W / W+ space, not in ArcFace.** It is a property of the *generator's* learned latent space, where attribute directions were curated to be linear and largely disentangled. ArcFace was trained for discrimination, not editability, and inherits no such guarantee.

The closest direct ArcFace-attribute analysis is Bortolato et al. 2025, "Attributes Shape the Embedding Space of Face Recognition Models" [10]:

- Confirms macroscale attribute organization exists: "real face recognition models exhibit strong global organization, such as the separation of male pictures from female pictures." So gender *does* have a recoverable direction (mean-of-male − mean-of-female), but it is not a clean low-d axis.
- Attribute-to-identity coupling is anisotropic: "attributes shaping the embedding space the most are the ones most deterministically linked to an identity" (intra-entropy vs KS Spearman ρ ∈ [−0.566, −0.656], p ≤ 10⁻³). Translation: the attributes most legible as directions are also the attributes ArcFace *uses* for identity, so editing them breaks identity.
- FaceNet has higher attribute-structural-dependency than ArcFace. ArcFace is *more* attribute-invariant by design — which means *less* clean attribute directions, not more.
- Intra-class / inter-class distance ratio ≈ 2–3× (Table 1). That's the geometric headroom for editing without identity collapse.

So: you *can* extract a male/female direction or an old/young direction by averaging labeled embeddings, the same way you can in any embedding space, but you should expect 5–10 dimensions of meaningful spread per attribute (not 1) and significant entanglement with identity.

The most reusable "operator" for our pipeline is the one we are already using and just hadn't recognized as such: **the source→target chord is itself a custom operator that says "move identity toward whatever this specific doll's ArcFace projection is."** That's the right framing for the Phase 3 sweep — not "weakening the swap," but "blending toward a specific OOD target point."

### Out-of-distribution: photo → portrait is a domain collapse, not an offset

StyleID [5] quantifies the damage:

| Setting | ArcFace TPR | StyleID TPR |
|---|---|---|
| StyleBench-H (in-distribution stylization, human-aligned) | 0.765 | 0.902 |
| Cross-method (unseen stylization) | 0.372 | 0.744 |
| Artist-drawn sketches (SKSF-A) | 0.619 | — |

The cross-method number is the relevant one: ArcFace loses **half its TPR** on stylization it was not exposed to in training [5]. The Stylized-Face dataset paper [12] (ICCV 2025) makes the same general claim — existing FR models are unsuitable for stylized recognition — but doesn't publish per-model ArcFace numbers.

Two failure modes, *both present, opposing each other* [5]:

- **Texture-sensitive identity drift:** changes in palette or material (lacquer, wood, paint vs. skin) read as identity changes that aren't there. ArcFace says "different person" when the human says "same person, stylized."
- **Geometry-insensitive false-match:** exaggerated geometry (matryoshka-stylized proportions, simplified features) does *not* drift identity enough. ArcFace says "same person" when the human says "obviously a different face."

Both biases apply to the matryoshka problem and they push in opposite directions for our blend term:

- Insofar as ArcFace under-reads the doll's stylized geometry, `target_face.normed_embedding` may still be close to the source's photoreal embedding — so `α·target` adds *little new information* (the chord is short), and the swap barely moves. This matches the "matryoshka doesn't actually defer" failure we saw at intermediate weights.
- Insofar as ArcFace over-reads the doll's wood/lacquer texture, `target_face.normed_embedding` is dominated by *appearance*, not the painted face's geometry. So `α·target` pulls the source toward a "wood-textured identity" — which is closer to what we want, but is being smuggled in via texture rather than as an explicit style axis.

There is no published *direction* in ArcFace that means "stylize toward matryoshka." The OOD region is a *collapse zone*, not a clean offset from photoreal.

### Implications for the Phase 3 sweep

1. **Lerp+renorm is fine as the math.** At our α range, the chord and arc agree. Don't switch to slerp expecting visible improvement.
2. **The signal is the target, not the algorithm.** Our outcomes are dominated by where ArcFace projects the specific doll image. Detector noise on stylized targets is high. Per-photo instability across Phase 2 is consistent with low-κ vMF on the target side.
3. **Stable target-direction trick (untested, easy):** instead of using the single `target_face.normed_embedding` from one render, **average target embeddings across multiple anchor doll renders** (matryoshka_bakeoff anchors, photo-anchor + nearby anchors). That reduces target-side variance by √N. Cheap to try; should reduce Phase 3 instability.
4. **The right way to "weaken identity transfer" is not a single knob.** We have three orthogonal levers, all currently fused into `face_swapper_weight`:
   - shorten the chord (the current weight knob),
   - choose a more stable target endpoint (average doll embeddings),
   - replace `target_face.normed_embedding` with a *curated* stylization direction (e.g. `mean(matryoshka_dolls) − mean(photoreal_faces)` from the bakeoff archive) — a hand-built operator.
5. **Hard ceiling is still HyperSwap's photoreal prior.** Even with the best target endpoint, the generator's training distribution is photoreal skin; pushing the conditioning won't change what the decoder paints. If `w ∈ [0, 0.5]` plateaus before producing acceptable matryoshka deference (as predicted in the [hyperswap-parameters survey](2026-05-20-hyperswap-parameters.md)), that ceiling — not the embedding geometry — is the wall we hit.

---

## Open questions

- **Lerp vs slerp at large α:** Arc2Morph's CLIP-vs-ArcFace finding [4] suggests interpolation quality drops in ArcFace alone, but we don't have a direct slerp-vs-lerp ablation inside ArcFace. If we push α past the clamped ±0.35 range (per the parameter survey's fallback plan), this becomes worth measuring.
- **Stylization direction as a constructed operator:** can we extract a usable "photo→matryoshka" mean-difference direction from the bakeoff corpus and use it *instead of* per-image target embeddings? Would give weight axis a fixed, calibrated meaning. Untested.
- **Per-identity κ as a confidence signal:** if we can estimate vMF concentration for each source identity (from face-augmentations of the same source), we could weight the source half of the blend by quality. Probably overkill at our scale.
- **No source addresses ArcFace embeddings of the same identity rendered as a matryoshka.** Our render → re-embed loop *is* the data that would answer this. Worth logging the embeddings of Phase 3 outputs alongside scores.

---

## Sources

[1] Deng, Guo, Xue, Zafeiriou. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition." CVPR 2019. <https://arxiv.org/pdf/1801.07698>
[2] Liu, Wen, Yu, Li, Raj, Song. "SphereFace: Deep Hypersphere Embedding for Face Recognition." CVPR 2017. <https://arxiv.org/abs/1704.08063>
[3] Papantoniou, Lattas, Moschoglou, Deng, Kainz, Zafeiriou. "Arc2Face: A Foundation Model for ID-Consistent Human Faces." ECCV 2024 (Oral). <https://arxiv.org/html/2403.11641v1> — Figs. 9 (PCA, 300–400 components) and 10 (linear ArcFace blending).
[4] "Arc2Morph: Identity-Preserving Facial Morphing with Arc2Face." Preprint. <https://arxiv.org/pdf/2602.16569> — slerp/lerp comparison, ArcFace-vs-CLIP space.
[5] "StyleID: A Perception-Aware Dataset and Metric for Stylization-Agnostic Facial Identity Recognition." Preprint 2026. <https://arxiv.org/html/2604.21689> — ArcFace TPR drop 0.765→0.372, bidirectional failure modes.
[6] Conti, Clémençon. "Mitigating Gender Bias in Face Recognition Using the von Mises-Fisher Mixture Model." ICML 2022. <https://proceedings.mlr.press/v162/conti22a/conti22a.pdf>
[7] "KappaFace: Adaptive Additive Angular Margin Loss for Deep Face Recognition." 2022. <https://arxiv.org/pdf/2201.07394> — vMF concentration κ as per-identity quality.
[8] "SlerpFace: Face Template Protection via Spherical Linear Interpolation." 2024. <https://arxiv.org/pdf/2407.03043>
[9] Shen, Yang, Tang, Zhou. "InterFaceGAN: Interpreting the Latent Space of GANs for Semantic Face Editing." CVPR 2020. <https://arxiv.org/pdf/1907.10786> — linear SVM boundaries in StyleGAN W, *not* ArcFace.
[10] Bortolato et al. "Attributes Shape the Embedding Space of Face Recognition Models." 2025. <https://arxiv.org/html/2507.11372v1> — ArcFace macroscale attribute organization, FR-vs-FaceNet comparison.
[11] Stylized-Face. ICCV 2025. <https://openaccess.thecvf.com/content/ICCV2025/papers/Peng_Stylized-Face_A_Million-level_Stylized_Face_Dataset_for_Face_Recognition_ICCV_2025_paper.pdf> — 4.6M images, 62k IDs, four style families.

## Cross-references

- [`2026-05-20-hyperswap-parameters.md`](2026-05-20-hyperswap-parameters.md) — the parameter knob that motivated this research; the survey assumed embedding-space mechanics that this doc justifies / qualifies.
- [`2026-05-20-photobooth-phase2-findings.md`](2026-05-20-photobooth-phase2-findings.md) — per-photo instability pattern matches low-κ target embeddings.
- `2026-05-18-face-swapper-landscape.md` — alternative swappers if the ArcFace OOD ceiling is the binding constraint.
- `[[reference-comfyui-shard-runbook]]`
