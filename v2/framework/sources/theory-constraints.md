# vamp-interface Theory Constraints — 2026-04-14 Late Session

**Status:** Two arguments saved verbatim for analysis after compaction. Both came out of a debugging conversation with the user after I'd over-committed to a pivot toward StyleGAN3. The first argument (HyperFace regime analysis) narrows a killed path. The second argument (continuity-vs-readability self-contradiction) is structural and reshapes what vamp-interface can realistically claim.

The rebuild plan and primitives memory have NOT been rewritten to reflect these arguments yet — that's the next pass, after compaction, which the user wants done with more rigor and explicit option analysis. This document is the input to that pass.

---

## Statement 1 — HyperFace is not a path killer, it's a mismatched tool under one regime

### My earlier claim

"HyperFace's objective is separation-maximizing → incompatible with the continuity hypothesis → kills Step 2 Option (d) as a path."

### The argument I actually had in mind (implicit)

1. vamp-interface needs `d(face_A, face_B) ≈ C · d(emb_A, emb_B)` — i.e., the composition `qwen → g → Arc2Face → image` is locally Lipschitz in qwen.
2. For that composition to be Lipschitz, `g: qwen → ArcFace` itself must be Lipschitz.
3. HyperFace packs its N targets to maximize the minimum pairwise angle on the hypersphere.
4. Therefore if I regress `g` to HyperFace targets, `g` will be non-smooth.

Step 4 is where the argument silently collapses. It only follows from 1–3 under an **unstated assumption** I didn't name.

### The hidden assumption

Step 4 assumes I'll use HyperFace targets as **exact, discrete, one-per-sample regression targets**. Under that assumption and *only* that assumption, the conclusion follows — because two nearby qwen embeddings would have to regress to two of the maximally-separated packed points, which are at least `min_pairwise_angle` apart no matter how close the inputs are. That forces `g` to be near-discontinuous (or: forces a massive Lipschitz constant).

But "exact discrete regression" is a choice I bolted onto HyperFace, not something HyperFace requires.

### What the argument actually shows, traced carefully

**Regime A — discrete exact regression (`g(e) → nearest_assigned_HyperFace_target(cluster(e))`).**
- Each qwen cluster maps to exactly one packed target.
- Within a cluster: all inputs collapse to the same target → distance = 0 regardless of qwen distance. *Perfectly Lipschitz within a cluster* (trivially, with constant 0).
- Between clusters: the jump is at least `min_pairwise_angle(HyperFace)`. The mapping is piecewise-constant with discrete jumps at cluster boundaries.
- This is **not "non-smooth" in the pathological sense**. It's a step function. The global Lipschitz constant is ∞ formally, but the local-within-cluster behavior is fine, and the between-cluster behavior is *exactly what the original vamp-interface's fixed-anchor design intended*: similar jobs get visually identical faces, different-cluster jobs get visually different faces.
- In this regime, HyperFace's max-separation is a *positive* — it makes the between-cluster jumps as visually distinguishable as possible.

**Regime B — interpolation among k-nearest packed targets (`g(e) = Σ w_k · target_k`, weights from qwen distances).**
- `g` is smooth by construction. Lipschitz constant is bounded by the interpolation weights' smoothness.
- The output lives in the convex hull of the packed targets, *not at the vertices*.
- HyperFace's max-separation is **irrelevant** — we're using interior points, so any well-spread set of N ArcFace vectors works equally well as anchors. Random samples from FFHQ-ArcFace, StyleGAN mapping-network samples, or uniform sphere sampling would all do the same job.
- So HyperFace-specifically buys us nothing in this regime, but it doesn't *hurt* either.

**Regime C — soft assignment with temperature τ.**
- Interpolates between A and B depending on τ.
- At τ → 0 (hard), we recover A. At τ → ∞, we recover B. Intermediate values give smooth-but-biased-toward-vertices.

### So what's actually true

1. HyperFace doesn't "kill" anything. It's a **mismatched tool** if you expected smooth exact-regression targets, and a **viable tool** in either of the two regimes above.
2. My earlier claim that "HyperFace is separation-maximizing, therefore incompatible with continuity" conflated "HyperFace's property" with "the regime in which we use HyperFace's output". The property only matters under Regime A.
3. **The correct role for HyperFace in the rebuild plan is as a codebook/anchor generator**, and Regime A is actually a reasonable fit for vamp-interface's original fixed-anchor thesis: one face per cluster, discrete jumps between clusters, consistent-within-cluster. The pivot to StyleGAN3 was motivated by wanting between-cluster smoothness, but vamp-interface may not actually need that — the original design had fixed anchors for exactly this reason.

### Broader lesson

I pattern-matched "separation-maximizing" to "incompatible with smooth" without stopping to ask *what function I was claiming was not smooth*. The function in question was `g`, but `g` is my design, not HyperFace's — HyperFace only provides targets. The smoothness of `g` is a property of how I use those targets, not of the targets themselves.

---

## Statement 2 — The continuity-vs-readability self-contradiction

### The user's challenge

"If we are projecting into face space AND we want faces to be different AND we want that difference to be associated with real world face features, then we are self-contradicting."

### The three requirements, restated

1. **Continuity / injectivity.** `g: qwen → face` preserves qwen-locality: close qwen → close face, distinct qwen → distinct face.
2. **Face differences exist.** The map is non-degenerate. Different inputs produce visually different outputs.
3. **Face differences are face-readable.** The *direction* in which two faces differ corresponds to a real-world face-feature axis — age, expression, bone structure, ethnicity, mood — something a human viewer could name.

### Why all three together is inconsistent

**qwen's axes and face-feature axes are different vector spaces that don't naturally share a basis.** qwen's principal directions encode things like "tech vs finance", "remote vs on-site", "fraud-y language patterns", "location cluster". Face-feature axes encode "age", "gender", "round vs long face", "skin tone", "expression". These are not the same axes and there is no empirical or theoretical reason they should be.

Now count dimensions:
- qwen has ~1024 effective dimensions (less after PCA, but still high).
- Face-feature perceptual space — the subspace of face variation that a human can *name* — is maybe 20-50 axes. Everything else is "these faces are different in some way I can't describe".
- ArcFace has 512 dimensions total, only a small fraction of which are perceptually named.

So any injective `g` into ArcFace must use most of ArcFace's 512 dimensions, but only ~50 of them are face-readable. Of the 1024 qwen directions we're asking the map to preserve, **at most ~50 can possibly land on readable face axes**. The other ~974 qwen directions must be packed into the "arbitrary pixel difference" subspace where viewers see "this face is different but I can't say how".

And even those 50 slots can only be filled with readable face-features if we *hand-choose which qwen direction maps to which face axis*. Since qwen axes aren't pre-labeled ("this is the dishonesty axis"), we can't align them without training a supervised probe per axis — which requires labels we don't have, for axes we haven't named.

**The theorem, informally:** any continuous injective map from qwen to face-pixels produces face differences that are mostly not face-readable. The fraction that *is* readable is bounded above by `dim(face-feature-vocabulary) / dim(qwen)`, and requires hand-chosen alignment for each readable axis.

This is exactly the Chernoff-faces problem. Chernoff (1973) picked 10 face features and *hand-assigned* 10 variables to them. The assignment was arbitrary and users had to memorize it. That's the price of requirement (3).

### Which requirement do we drop

vamp-interface can drop only one of the three without destroying itself:

- **Drop (1) continuity.** Go back to Chernoff's approach: hand-assign each salient variable to a specific face feature. No projection, no learned MLP, no diffusion — just a rule-based encoder. We lose the "face as continuous visualization of embedding space" thesis entirely. The face becomes a dashboard rendered with face parts instead of bars and gauges.
- **Drop (2) distinctness.** Make most faces identical. Useless for visualization.
- **Drop (3) readability.** Accept that the identity channel's face differences are *arbitrary distinguishers*, not *meaningful readouts*. Similar jobs → similar-looking faces (users can tell "these jobs cluster together") but the specific features that differ are unpredictable and not describable. Users treat each face as a visual identifier, not as a readout.

### What dropping (3) means concretely

Dropping (3) is the realistic path and it reshapes the product:

- **Identity channel becomes an identifier, not a readout.** A user who sees two jobs with similar faces can infer "these are semantically similar postings", but cannot say *what* makes them similar by looking at the face. It's like file icons: the icon of a Python file tells you it's Python, not *why* it's Python.
- **Drift channel stays readable — but only because it's hand-coded.** The `sus_level → uncanny` axis is the one place where we explicitly train a learned direction that corresponds to a named face-perception axis ("looks wrong"). This is a single hand-chosen variable, rendered on a single hand-chosen axis, exactly the way Chernoff would have done it — one slot in the dashboard, not a full projection.
- **The user study measures what's actually possible**: (a) can users distinguish clusters from face appearance? (yes, if the identity channel is injective and monotone), (b) does high-sus look "wrong"? (yes, if the drift axis training worked). Neither question requires readable projection of qwen axes.
- **The original vamp-interface product description was overreaching.** The earlier "d(face_A, face_B) ≈ C · d(embedding_A, embedding_B)" hypothesis is too strong — it conflates "similar-looking" (which needs monotonic similarity) with "readable difference" (which would need axis alignment we can't get). The achievable version is:
  - Identity channel: `sim(face_A, face_B)` is monotonic in `sim(qwen_A, qwen_B)` (direction-preserving, not metric-preserving)
  - Drift channel: `uncanny(face)` is monotonic in `sus_level`
  Two weaker, independent, achievable claims.

### How this reshapes the rebuild plan (to be worked out post-compact)

Once we only need **monotonic** identity-channel continuity (not metric-preserving, not readable):

- **Step 2 becomes much easier.** Any injective smooth-ish map works. StyleGAN3 W-space is fine, Arc2Face ArcFace-space is fine, HyperFace discrete cluster regression is fine (each cluster's jobs get the same face, clusters are distinct), even a hash-to-W fixed assignment is fine. The choice between them is a *product/quality* decision (photorealism, inference cost, license), not a *theoretical-correctness* decision. My pivot to StyleGAN3 was over-constrained.
- **HyperFace un-killed for the right reason.** Regime A (one-face-per-cluster discrete regression) satisfies monotonic identity-channel continuity perfectly — same cluster → same face, different cluster → maximally-distinct face. It's arguably *closer to what vamp-interface always was* than the smooth-projection approach: the original design had one anchor per cluster; HyperFace just generalizes one-anchor-fixed to N-anchors-deterministically-packed.
- **Step 4 is where the real engineering problem lives.** The drift axis is the only readable channel, and it's the only place where we need a learned perceptual direction. Everything else is bookkeeping. The mandatory α-sweep gate on Asyrp is still the critical experiment.
- **The Step 5 continuity pre-flight changes.** We don't need LPIPS monotonicity along interpolation sweeps — we only need LPIPS *monotonicity* (not slope-preserving). The test relaxes.

---

## Implications for the killed-paths list

The following are revisited against the two arguments above. The full blind-alleys doc at `docs/research/2026-04-14-rebuild-blind-alleys.md` is updated in parallel.

| Killed entry | Status after arguments | Reason |
|---|---|---|
| 1. Arc2Face "5-token CLIP conditioning space" | **Stays killed.** This was a misreading of the Arc2Face paper (only one 768-d slot carries identity via zero-padding). The theory arguments don't resurrect it — the paper architecture is what it is. |
| 2. Arc2Morph "proves ArcFace non-smooth" | **Stays killed.** This was a mischaracterization of Arc2Morph's results (they only studied the midpoint α=0.5, with no smoothness metric). Not affected by the theory pivot. |
| 3. Vox2Face drop-in clone | **Conditionally reopened.** Earlier objection: AM-Softmax + InfoNCE needs identity labels and positive pairs we don't have. Under identifier-not-readout framing, we don't need identity labels per-se — we can substitute qwen *cluster* indices for identity labels (AM-Softmax on cluster labels) and define positives as "same qwen cluster" for InfoNCE. This turns Vox2Face's Stage I into a cluster-alignment loss rather than identity-alignment loss. The residual problem: we lose Vox2Face's geometric guarantees about inter-class angular margins because cluster counts and cluster distributions are different from identity counts. Worth a closer look post-compact. |
| 4. Boundary Diffusion "class-mean drop-in" | **Stays killed as a drop-in.** The method is SVM hyperplane normals + symmetric one-shot shifts validated only on unconditional DDPM/iDDPM. The theory pivot doesn't change its architectural mismatch with CFG'd SD. Remains a Step 4C research bet if all other drift options fail. |
| 5. RigFace fallback | **Stays killed.** Per-layer FaceFusion features + no embedding port + no code. The theory pivot relaxes the Lipschitz requirement but RigFace's blockers are at the interface level (no way to supply a qwen vector), not the Lipschitz level. The input-interface problem is unaffected. |
| 6. NoiseCLR unsupervised alternative | **Stays killed.** Wrong slot (CLIP text-conditioning), wrong geometry (ε-output via CFG, not h-space), vanilla SD only. Theory pivot doesn't help — NoiseCLR still can't be run on Arc2Face's conditioning path. |
| 7. PhotoMaker fallback | **Stays killed.** Input is a token stack from real face images. Under identifier-not-readout, we still need to *supply* identity via PhotoMaker's expected interface, and that interface expects multi-image real references, not a single projected vector. Unaffected by the theory pivot. |
| 8. InstantID fallback | **Stays killed.** IdentityNet needs ArcFace + landmark spatial map + IP-Adapter image branch. Same interface-level blocker as PhotoMaker and RigFace. |
| HyperFace Step 2 Option (d) | **UN-KILLED.** Under Statement 1 (regime analysis) and Statement 2 (identifier-not-readout reframe), HyperFace-Regime-A (discrete cluster regression) is a principled match for the original vamp-interface fixed-anchor thesis. Residual problem to analyze post-compact: how to handle the within-cluster vs between-cluster information budget, and whether cluster granularity is tunable. |

Five entries stay killed because their blockers are at the *interface* level (what input the model accepts) rather than the *smoothness* level, and the theory pivot only relaxes the smoothness requirement. Two entries (Vox2Face, HyperFace) are re-opened with explicit residual problems that need post-compact analysis.

---

## Post-compact work list (for the next pass)

After compaction, the user wants a structured approach with more rigor and explicit option analysis. The inputs are:

1. This theory-constraints doc (two arguments verbatim)
2. The paper findings log at `docs/research/papers/2026-04-14-paper-findings.md`
3. The updated blind-alleys doc
4. The deeper-research queue
5. The current (pre-rewrite) rebuild plan, which will need a full rewrite against the new constraints

The structured approach should:

- Start from the two achievable claims (monotonic identity-channel continuity, readable drift-channel direction) and derive the requirements for each channel independently.
- Enumerate Step 2 options with a uniform evaluation grid: interface (what input it takes), smoothness regime (discrete / smooth / soft), readability (arbitrary distinguisher / coded axis / hybrid), training cost, inference cost, license, open questions.
- For each, say what's *proven* vs what's *unverified* against primary sources.
- Identify which residual reads from T1b.6-1b.10 are actually blocking.
- Make the Step 2 recommendation fall out of the grid rather than be asserted.
