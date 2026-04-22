---
status: live
topic: manifold-geometry
summary: Lit read revises Phase 2 — sparse NMF, not PCA→ICA, is the right decomposition for our non-negative 52-ch blendshapes; k≈8–16 components at 95% VE, not the 25 originally planned.
---

# Three PCA/ICA/NMF-on-blendshapes papers — distilled, and how they change the plan

**Date:** 2026-04-22
**Papers read:**
`docs/papers/tripathi-2024-pca-au-keypoint.pdf` (arXiv 2406.09017),
`docs/papers/tripathi-2024-dfecs-nmf.pdf` (arXiv 2406.05434),
`docs/papers/li-2026-statistical-blendshape.pdf` (arXiv 2601.08234).
**Replaces Phase 2 of:** `2026-04-22-blendshape-bridge-plan.md`.

## TL;DR — plan updates

1. **Use sparse NMF, not PCA→ICA.** Our 52-channel MediaPipe output
   is already non-negative and semantically parted. Sparse NMF with
   non-negative encoding is purpose-built for this data shape; ICA
   rotations remain dense and sign-ambiguous for no reason.
2. **Component count is k ≈ 8–16, not 25.** 95% variance retention on
   keypoint/expression data of similar dimensionality sits at that
   range. Our earlier "25–30 effective dims" came from full FACS
   expression corpora; our Flux-restricted corpus is likely narrower.
3. **Interpretability is human-judged against a muscle atlas,** not a
   metric. Published numbers (50–87.5% interpretable) are
   majority-vote over 3 volunteers inspecting keypoint-displacement
   arrows. For us, the analogue is loading-weight inspection against
   ARKit channel names.
4. **Skip DFECS's 7-part split.** It exists because their 136-d
   keypoint input has spatial locality the algorithm doesn't know
   about. ARKit channel names *already* encode part membership.
5. **Li 2026 is not a decomposition paper.** Title misleading — it's a
   landmarks→ARKit regression system, not an analysis of blendshape
   coefficient space. Demote in papers index.

## Per-paper verdicts

### Tripathi & Garg 2024 (PCA, arXiv 2406.09017)

**Pipeline:** PCA on centered keypoint-displacement matrix X (136-d).
Retain components to 95% train VE → **k=8** on DISFA, BP4D, CK+
(Table 1, p.7). Same rule across all three datasets.

**Interpretability = binary muscle-plausibility judgment** (p.7–8,
Figs. 3–4): project a component's keypoint-displacement arrows onto a
neutral face, check against anatomical muscle directions
(UF1–UF4/MF1–MF2/LF1–LF2). A component is non-interpretable when the
implied muscle direction doesn't exist (cited failure examples:
component 3 shows lower lip sliding centrally along its own line;
components 7–8 show eyebrows pulling diagonally down toward ears).
**4 of 8 components pass → 50%.**

**Important caveat the authors are careful about:** they never label
components as "AU12" or "AU6." PCA components are not AUs. For actual
FACS-AU regression they build a separate **keypoint-based AU
dictionary** `U_AU ∈ ℝ^(136×26)` by photographing one subject
performing each AU's APEX frame (§2.4, p.5). That is a separate
supervised construction, not a property of PCA.

**For us:** 95% VE rule transfers directly. On 52-d already-compressed
data, expect k well below 25 — possibly 6–12. The muscle-plausibility
criterion becomes "does this component's high-loading channels form a
coherent anatomical group per ARKit naming" (e.g. `mouthSmileLeft` +
`mouthSmileRight` + `cheekSquintLeft` + `cheekSquintRight` is a
plausible Duchenne-smile atom; `browDownLeft` + `mouthSmileRight` is
not).

### Tripathi & Garg 2024 (DFECS, arXiv 2406.05434)

**Pipeline:** Two-level decomposition.

**Part Face Model (PFM):** 7 face parts (L/R brow, L/R eye, nose,
lips, jawline). Per-part dictionary learning with positive-LASSO
encoding (§2.3, eq. 1, p.10):

```
min ½‖X_f − U_f V_f‖²  +  α‖V_f‖₁   s.t. ‖u_j‖₂ ≤ 1, v_ij ≥ 0
```

Only the *encoding* V_f is constrained non-negative (keypoint
displacements are signed); atoms U_f are unconstrained. k_f grid
search 1..p; α grid 5→0.5 step 0.5; stop at per-part VE ≥ 95%.

**Hierarchical Face Model (HFM):** stack all U_f into U ∈ ℝ^(r×k),
then NMF on the concatenated encoding V with both factors non-negative
(eq. 2, p.11, since V ≥ 0 already):

```
min ‖V − AB‖²  +  α_A‖A‖₁  +  α_B‖B‖₁   s.t. A, B ≥ 0
```

**Result:** 16 DFECS atoms on DISFA at 95% train VE. Volunteer-voted
interpretability **62.5% (PCA, signed-split 16 atoms) → 87.5%
(DFECS)** under identical voting protocol — see §4.1.2, p.18 for the
fair comparison. The cross-paper 50→87.5 headline in the abstract mixes
two different voting procedures.

**Which constraint carries the gain?** The paper does not run a clean
ablation. The authors argue by design (§5, p.21) that non-negative
encoding + sparsity jointly prevent "one atom pulls keypoints in
biologically impossible directions on different samples." ICA-style
signed rotations inherit PCA's dual-direction ambiguity; NMF breaks
it. The 7-part split is there because global dictionary learning
would mix parts. **Sparsity and non-negativity are a joint package;
treat them as such.**

**Code:** https://github.com/Shivansh-ct/DFECS-AUs — reference impl.

**For us:** we have an even better setup than they do. Our 52-channel
input is itself non-negative, so standard NMF directly applies (X ≈
UV, both U, V ≥ 0). We don't need the PFM stage — ARKit channel
naming replaces it. Run **sparse NMF** on X ∈ ℝ^(52 × N=~1600) with:

```python
from sklearn.decomposition import NMF
model = NMF(n_components=k, init="nndsvd", beta_loss="frobenius",
            solver="cd", l1_ratio=0.5, alpha_W=0.1, alpha_H=0.1)
W = model.fit_transform(X)   # (52, k) atoms
H = model.components_         # (k, N) encodings
```

Sweep k ∈ {6, 8, 10, 12, 14, 16, 20}; plot VE vs k; pick knee. If
sparse NMF atoms are still messy, fall back to plain NMF or try
`MiniBatchDictionaryLearning` with `positive_code=True,
positive_dict=True`.

**ICA for comparison (not as primary method):** run FastICA on
PCA-whitened X restricted to 95% VE subspace, compare atom support
counts (how many channels per atom >0.05), atom-wise
muscle-plausibility, and reconstruction error. Expect NMF to dominate
on (support count) and (muscle plausibility); PCA→ICA to win on
(reconstruction at equal k).

### Li, Wang, Twombly 2026 (arXiv 2601.08234)

**Verdict:** not a decomposition paper, not useful for Phase 2.
Title's "statistical analysis" refers to per-blendshape regression
diagnostics (Durbin-Watson, Breusch-Pagan, Shapiro-Wilks) in their
landmarks→ARKit-coefficients regression pipeline. Training corpus is
131 images + MetaHuman-LiveLink synthetic frames; testing 18k frames
from 6 testers. They explicitly **do not decompose** the 52-ch
blendshape space; they take ARKit as the a-priori basis and build a
regression to produce its coefficients from MediaPipe landmarks.

**Update papers README**: retag this paper "landmarks→ARKit
coefficient regression pipeline, not blendshape-space decomposition."
Demote from "PCA-ICA reference" to "tangentially related; safe to
skip."

## Revised Phase 2 — concrete recipe

**Step 2.1 — assemble corpus.** Concatenate blendshape scores from
`bootstrap_v1` (288), `alpha_interp` (660), `smile_inphase` (330),
`jaw_inphase` (330), `intensity_full` (340). Total ≈ 1948 vectors.

**Step 2.2 — preprocess.**

- Stack all vectors into X ∈ ℝ^(52 × N).
- Drop channels with σ < 0.01 across corpus (neutral channel `_neutral`
  is typically zero-variance). Expect ~50 channels retained.
- Do **not** centre and do **not** whiten — NMF requires non-negative
  input.
- Do not halve via bilateral symmetrisation. Let NMF decide; a natural
  atom like "symmetric smile" should assemble both L and R channels
  together with similar weights.

**Step 2.3 — sparse NMF with k-sweep.**

```python
from sklearn.decomposition import NMF
ks = [6, 8, 10, 12, 14, 16, 20]
ve_by_k = {}
for k in ks:
    m = NMF(n_components=k, init="nndsvda", solver="cd",
            beta_loss="frobenius", l1_ratio=0.5,
            alpha_W=0.1, alpha_H=0.1, max_iter=500, random_state=0)
    m.fit(X)
    recon = m.transform(X) @ m.components_
    ve = 1 - ((X - recon)**2).sum() / ((X - X.mean())**2).sum()
    ve_by_k[k] = ve
```

Knee rule: pick smallest k where VE ≥ 0.95.

**Step 2.4 — interpret atoms (human-in-the-loop).** For each atom
(column of W, ℝ^52), list the top 5–8 channels by weight. Manual
check: do these channels share an anatomical region (brow / eye /
cheek / mouth-smile / mouth-stretch / jaw)? Label each atom with:

- `AU-plausible` name (e.g. "Duchenne-smile", "jawDrop", "browRaise")
  when loadings form a coherent group.
- `composite` if loadings mix across regions.
- `noise` if atom has diffuse loadings across ≥15 channels.

Target: ≥ 75% of atoms plausible or composite (Tripathi's bar was
87.5% on keypoints; we're on a cleaner basis, should do at least as
well).

**Step 2.5 — ICA comparison (one-shot).** FastICA on PCA-whitened X at
same k. Same atom inspection. Report: atom support count mean (lower
= sparser), fraction plausible, reconstruction VE at matched k. This
is a sanity check; if ICA wins on plausibility we switch; if not, NMF
is canonical.

**Step 2.6 — save the basis.** Save W (52 × k), atom names, atom
classification to `models/blendshape_nmf/` with a JSON manifest.

## What this does NOT change about the bridge plan

- Phase 3 (ridge-fit attention-cache features per atom) is unchanged.
  We just now have NMF atoms as targets instead of ICA axes.
- Phase 4 (per-axis linearity in `scale`) is unchanged, and the
  revised version from `2026-04-22-intensity-linearity.md` still
  applies: use weak-pair prompts, per-base baseline correction,
  low-scale regime.
- Phase 5 Riemann leg still gated on Phase 4 finding residual non-
  linearity after the engineering-curve corrections.

## Gotchas

- NMF is non-convex; different `random_state` produces different
  atoms. Report atom stability across 5–10 random inits; prune atoms
  that don't survive.
- `init="nndsvda"` is the most stable NMF initialisation for our
  data shape.
- Per Tripathi DFECS, expect some atoms to be "composite" (not
  cleanly one AU). Do not force 1-atom-per-AU naming.
- Sparsity regularisation `l1_ratio`/`alpha_*` trades VE for atom
  cleanliness. Sweep this before claiming NMF beats ICA.

## Checklist before running

- [ ] Score the last uncored corpus if any exists (currently all used
  datasets are scored).
- [ ] Confirm sklearn ≥ 1.3 (the `NMF` API parameters above need recent
  versions).
- [ ] Write `src/demographic_pc/blendshape_decomposition.py`
  implementing Steps 2.1–2.6.
- [ ] Run, produce NMF vs PCA→ICA comparison table.
- [ ] If Phase 2 succeeds, proceed to Phase 3 (ridge fits per atom).
