---
status: archived
topic: archived-threads
summary: End-to-end detective-puzzle pipeline is buildable with differentiable reward through DRaFT-1 backprop; start with frozen-readout experiment on v3 faces.
---

# Full-Loop Quantification — Synthesis

**Date:** 2026-04-16
**Status:** Synthesis of four parallel research passes. Feeds the v0.12 framework pass and the Exp E/F experiment queue. Sources:

- [differentiable-flux-sampling.md](2026-04-16-differentiable-flux-sampling.md)
- [diffusion-rl-attribute-alignment.md](2026-04-16-diffusion-rl-attribute-alignment.md)
- [attribute-preserving-face-encoders.md](2026-04-16-attribute-preserving-face-encoders.md)
- [self-supervised-pattern-preservation.md](2026-04-16-self-supervised-pattern-preservation.md)

## TL;DR

**The task-anchored cycle is buildable end-to-end.** A concrete pipeline exists that is either fully differentiable (preferred) or policy-gradient-optimizable (fallback), and can run on a single 24 GB GPU for detective-scale problems. The recommended minimum viable form:

```
qwen(text)  →  P: ℝ¹⁰²⁴ → ℝ^{cond}  →  Flux.1-dev (frozen)
            →  image  →  Ψ = FaRL-B/16 (frozen)  →  R: linear probe
            →  D̂

Loss (labeled):    CE(D, D̂)  +  λ_align · InfoNCE(q, Ψ(I))  +  λ_collapse · VICReg(Ψ(I))
Loss (unlabeled):  InfoNCE  +  soft-rank Spearman  +  VICReg

Gradient through Flux:
  Tier 1 (cheap):   DRaFT-1 / DRaFT-LV  (last-step backprop, ~1× memory)
  Tier 2 (principled): Adjoint Matching (ICLR 2025, Microsoft repo for Flux)
  Tier 3 (fallback): Flow-GRPO or DanceGRPO (RL; only if differentiable path fails)
```

**Do this before any of the above:** a **readout-only** experiment on existing v3 faces — frozen Flux, frozen FaRL, train only `R`. If `R` already recovers planted axes, a huge chunk of the project just works without touching `P`. If it doesn't, we've localized the leak to either the projection or the generation step and know what to escalate to next.

## Cross-thread findings (what the four reports collectively say)

### 1. The reward is already differentiable — don't start with RL

Threads 1 and 2 agree strongly: the reward (`Ψ` frozen + `R` differentiable) is a differentiable function of pixels, and `P` is a differentiable neural net. The only non-differentiable link is Flux sampling, and the literature has cheap tricks (DRaFT-1) that make even that tractable. Starting with RL throws away this structure.

Thread 2 also flags a specific failure mode: **vanilla DDPO / DPOK diverge on rectified-flow samplers like Flux**, because Flux uses an ODE (not SDE) sampler and the likelihood ratio is ill-defined. This was reported in the Flow-GRPO and DanceGRPO papers and motivated the 2025 rectified-flow-specific methods. If we ever do need RL, use Flow-GRPO (arXiv:2505.05470, official FLUX.1-dev support) or DanceGRPO (arXiv:2505.07818). Do not reach for `trl.DDPOTrainer` out of the box.

### 2. Swap ArcFace for FaRL as the primary readout

Thread 3: ArcFace IR101 is trained with identity-invariant margin losses that **actively discard expression, attire, and gaze** — three of the five axis types the detective corpus plants. The closest published analysis (arXiv:2507.11372) confirms ArcFace compresses "smiling" and "mouth slightly open" while preserving "age" and "head angle." So ArcFace is adequate only for the pose/age axis family, and wrong for every other axis we care about.

**FaRL-B/16 (CVPR 2022, arXiv:2112.03109)** is a CLIP-ViT fine-tuned on 20M LAION-Face pairs. Published CelebA linear-probe mAcc: 89.66 / 90.99 / 91.39 at 1% / 10% / 100% training data — a consistent +0.5 mAcc over vanilla CLIP-B/16 across all budgets. The lift is small but structural: face-specific pretraining helps on face-specific attribute axes, and it remains strong on the general visual axes vanilla CLIP already covers.

**Ensemble with DINOv2-L/14** if compute allows — it captures complementary signal (pose geometry, fine-grained visual structure) that CLIP-lineage models underweight. Stacking gives a ~1024-d joint embedding and the readout `R` can learn its own weighting.

**Keep ArcFace as auxiliary** for identity-locked cluster detection (e.g., the signature-cluster planted axis in the detective spec). That's its actual strength.

Ψ is now a framework-level decision that should propagate to §2.6.1 and all future experiments.

### 3. Differentiable gradient path: DRaFT-1 first, Adjoint Matching second

Thread 1 ranks Flux-compatible gradient methods:

| Method | Memory | Compute | Gradient bias | Flux-ready? |
|---|---|---|---|---|
| **DRaFT-1** (arXiv:2309.17400) | ~1× | ~1× | biased (one-step) | yes, trivial |
| **DRaFT-LV / DRaFT-K** (K≈4–10) | ~K× or O(1) w/ checkpointing | ~K× | less biased | yes |
| **Adjoint Matching** (ICLR 2025, arXiv:2409.08861) | O(1) | ~2× | exact (memoryless noise) | **yes, Microsoft reference code** |
| Neural-ODE adjoint (vanilla) | O(1) | ~2× | exact | no Flux reference |
| Consistency distillation | — | — (ship-time cost) | zero once distilled | Schnell **dropped** (see note below) |
| RF-inversion / RF-Solver-Edit | — | — | **not a gradient method** | gives inversion only, not grad |

**Start with DRaFT-1.** Simple, cheap, proven on SD/Flux. If the one-step gradient bias doesn't kill training of `P`, we're done. If it does, escalate to DRaFT-LV with gradient checkpointing (K=4–10, keeps memory ~O(1)).

**Adjoint Matching is the principled option** — Microsoft's `github.com/microsoft/soc-fine-tuning-sd` repo ships working code for Flux/SD3 with the memoryless noise schedule that rectified flows provably require. Use this if DRaFT-K plateaus or the biased gradient produces bad attribute fidelity.

**Flux Schnell: dropped.** The research report identified it as a potential free consistency-distillation path (LADD-distilled 1–4-step sampling, Apache-2.0). **Dropped from active consideration (2026-04-16):** prior attempts in this project to use Schnell as a base for face generation did not produce usable outputs. If the project ever needs a one-step surrogate again, distill from Flux-dev directly (Stage 5+ escalation) rather than falling back to Schnell.

**Do NOT** try vanilla neural-ODE adjoint without a Flux-specific reference — no one has published gradient behavior through Flux-dev's distilled guidance-embedding path, and transformer-based velocity fields may have numerical quirks.

### 4. Objective function converges on a three-term recipe

Thread 4 argues for one loss form across both labeled and unlabeled regimes, with a task term toggled in/out:

```
L = λ_align · InfoNCE(q, f)                 # paired contrastive, shared-content identifiability
  + λ_struct · StructureLoss(D_Q, D_F)      # soft-rank Spearman or entropic GW
  + λ_collapse · VICReg(f)                  # variance floor + decorrelation on F
  [ + λ_task · CE(classifier(f), y) ]       # only when labels exist
```

**Why each term is needed:**

- InfoNCE alone preserves pair-identifiability (recovers the shared-content block per Daunhawer et al. ICLR 2023, arXiv:2303.09166) but doesn't constrain geometry.
- StructureLoss alone can be satisfied by degenerate embeddings (all points collapsed to one point has distance 0 to distance 0).
- VICReg alone prevents collapse but says nothing about which structure is preserved.
- CE when labels exist anchors the pipeline to the task directly.

**Daunhawer et al. 2023 is the strongest theoretical finding.** Paired contrastive on multimodal data *identifiably recovers the shared content block* under a content/style-independence assumption. This is the mathematical justification for InfoNCE being in our loss — not just "MI lower bound" but "provable recovery of the structure that's actually shared between qwen and face space."

**Practical recipe:** VICReg + soft-rank Spearman as the cheap baseline; upgrade the structure term to entropic Gromov-Wasserstein on subsampled batches for final validation. Gromov-Wasserstein as an unsupervised model-selection metric is called out as a promising open direction.

**What NOT to use:** IB degenerates to `min I(Q;F)` without labels (wrong direction). IRM needs environments; wrong shape for cross-modal alignment (though potentially useful as a cross-corpus generalization regularizer later). MINE is superseded by InfoNCE in practice.

## The staged buildout plan

Seven stages, each cheaper than the next and each either validating or killing the expensive stages that follow.

### Stage 0 — Exp E baseline (already specced in framework v0.11)

Run the §2.6.1 manifold-alignment metric sweep on the existing Flux v3 corpus with ArcFace as the readout (current production setup). Produces the baseline numbers that Stages 1–6 try to beat.

**Cost:** ~1 day. **Exit criterion:** report in `docs/research/2026-04-16-expE-manifold-metrics.md` with the agreement diagnostic. Handoff: decide Stage 1 priority based on which metrics pass/fail.

### Stage 1 — Readout-only on existing v3 faces

Swap Ψ from ArcFace → FaRL-B/16. Keep Flux v3 pipeline unchanged. Train only a linear probe `R` on FaRL embeddings of existing v3 faces, labeled by the 10 work-type archetypes (labels already exist on the 543-job corpus).

**Cost:** hours. **Exit criterion:** if R's accuracy on work-type classification is ≥80%, the face channel already carries archetype signal and we've localized the "does Flux preserve discoverable structure" question to "yes." If <50%, the problem is upstream of Ψ and we need to escalate.

**Cross-checks:** train the same probe on DINOv2-L/14, CLIP ViT-L/14, ArcFace IR101; report the ranking. This is the minimal empirical Ψ-selection experiment Thread 3 called out as not being in the published literature.

### Stage 2 — Detective corpus generation + readout measurement

Generate the 500-item detective corpus (Phase 1–3 of [hidden-pattern-experiment.md](../design/hidden-pattern-experiment.md)). Apply Stage 1 readouts to the rendered faces. Measure recovery of planted axes via linear probe on FaRL + DINOv2 ensemble.

**Cost:** ~2 days (1 day LLM generation + Flux render, 1 day readout + metrics). **Exit criterion:** per-axis recovery rate. If all 3 orthogonal axes + interaction cell + signature cluster recover at ≥70%, the pipeline works with the current v3 projection for the labeled detective task — and we can skip Stages 3–6 for this task. If recovery is <40% on any axis, escalate to Stage 3.

### Stage 3 — Hand-written conditioning sanity check

For the detective axes that failed Stage 2: bypass the learned projection and hand-write Flux conditioning prompts that encode each axis level directly. Re-render. Re-measure recovery.

**Cost:** ~1 day. **Exit criterion:** if hand-written conditioning recovers the axis, the problem is in the *projection* (P is not producing axis-aligned conditioning). If hand-written also fails, the problem is in *generation* or *readout* and no projection training will fix it — escalate to Ψ-ensemble or reconsider the axis definition.

If hand-written works and learned-P doesn't, **fit P by supervised regression to the hand-written conditioning.** Mean-squared-error loss in conditioning space. This bypasses both Flux gradients and policy-gradient RL entirely. Thread 2 flagged this as the biggest pre-RL win.

### Stage 4 — DRaFT-1 end-to-end on detective corpus

If Stage 3's regression-trained P is inadequate (hand-written conditioning too rigid, doesn't generalize across the 500 items, or we want to train P on many more qwen variants than we can hand-write):

Train `P + R` jointly with the labeled-regime loss (CE + InfoNCE + VICReg) using DRaFT-1 gradient through Flux. Last-step backprop only.

**Cost:** ~3–5 days including DRaFT-1 implementation on the Flux-dev flow sampler. **Exit criterion:** match or beat Stage 3 recovery rates. If DRaFT-1 plateaus below Stage 3, the one-step gradient bias is hurting — escalate to Stage 5.

### Stage 5 — Adjoint Matching / DRaFT-LV

Escalation path for Stage 4. Two forks:

- **DRaFT-LV with K=4–10** — multi-step backprop with gradient checkpointing. Factor of K compute cost vs DRaFT-1, but cheaper than Adjoint Matching and no reference-code dependency.
- **Adjoint Matching** (Microsoft `soc-fine-tuning-sd` repo) — theoretically exact, designed for rectified flows, known-working code. More engineering effort to port to our projection-only fine-tuning structure.

**Cost:** ~1 week each fork. **Exit criterion:** detective recovery ≥ 80% on all axes. If both fail, escalate to Stage 6.

### Stage 6 — Flow-GRPO / DanceGRPO (RL fallback)

If every differentiable path plateaus, or if we later add a non-differentiable reward (user-clicks-from-daily-puzzle-players as the reward signal — a live product telemetry loop), drop to policy gradient.

**Use Flow-GRPO** (arXiv:2505.05470, official Flux.1-dev support) as the primary. DanceGRPO (arXiv:2505.07818) is an alternative. **TexForce** (arXiv:2311.15657) is the architectural precedent — DDPO on a text-encoder LoRA with frozen UNet, structurally identical to our projection-only setup.

**Cost:** single 24 GB GPU budget per Thread 2: ~200–500 GPU-hours (1–3 weeks) for a 500-prompt corpus with LoRA on P. Realistically we'd rent H100s for this.

**Known gotcha:** published Flow-GRPO uses 8×80GB for full runs. Single-24GB LoRA-only runs are plausible but not benchmarked.

### Stage 7 — Unlabeled regime (scam corpus, arbitrary domain)

Once Stages 1–6 validate on the detective corpus, repeat on the real scam/labor corpus with the unlabeled-regime loss (InfoNCE + soft-rank + VICReg, no CE term). The pipeline stays the same; only the loss toggles.

Validation: measure recovery of any labels that do exist (work_type cluster membership, sus_level buckets) via linear probe trained post-hoc. Should retain most of the labeled-regime accuracy — otherwise the unsupervised objective is not matching the supervised one.

## Open questions we cannot answer from literature

Each of these requires in-house measurement before the staged plan can lock in:

1. **Does ArcFace v3 baseline's r(sus)=+0.914 reproduce with FaRL?** If FaRL gives weaker sus correlation but stronger axis recovery, we have a single-knob vs. multi-knob tradeoff to reconcile. The Stage 0 Exp E run should include FaRL as a second readout alongside ArcFace to answer this.
2. **Does learned `P` (via DRaFT-1 or AM) generalize off-corpus?** Train on 500-item detective corpus, test on 543-job scam corpus. If projection overfits to pattern-planting distribution, stages 4–6 are brittle.
3. **What's the right conditioning injection point in Flux.1-dev?** We've been writing "qwen → P → Flux conditioning" abstractly, but Flux has multiple injection surfaces (T5-XXL text embeddings, CLIP-L text embedding, pooled CLIP, guidance vector). Each is a different `P` target with different gradient paths. Thread 1 flags Flux's learned guidance-embedding path as a known-unknown for every gradient method. The Microsoft SOC repo chose specific injection surfaces — what did they choose, and is it right for our use?
4. **Conditioning-only vs. LoRA-on-Flux tradeoff.** The entire plan above freezes Flux. TexForce's result that "tuning upstream conditioning preserves attributes better than tuning the backbone" is our prior justification, but it's for SD, not Flux. We should measure this once we have Stage 4 running.
5. **Supervised-regression (Stage 3) ceiling.** If hand-written conditioning + regression-fit P matches DRaFT-1 performance, DRaFT-1 is over-engineered for detective tasks. We need to know the ceiling before committing to Stage 4+.

## Framework implications (v0.12 preview)

This synthesis suggests three framework-level updates for the next pass:

1. **§2.6.1 Ψ choice becomes a first-class decision.** Add a column in the metric table specifying "which encoder" and the reason. FaRL replaces ArcFace as the default Ψ for all attribute-axis metrics. ArcFace retained as auxiliary for identity-cluster metrics.
2. **§5 gains Stages 1–7 as an explicit experiment ladder.** Exp F (readout-only), Exp G (detective recovery), Exp H (hand-written + regression), Exp I (DRaFT-1), Exp J (Adjoint Matching), Exp K (Flow-GRPO), Exp L (unlabeled regime). Ordered by cost; each gates the next.
3. **New §2.8 "Optimization protocol" subsection.** Names the task-anchored cycle as the canonical training objective when labels exist, the InfoNCE+VICReg+soft-rank recipe when they don't, and the DRaFT-1 → AM → Flow-GRPO escalation ladder for gradient paths.

Not committing these yet — propose the v0.12 edits after we've either (a) run Stage 0 / Exp E and have baseline numbers, or (b) decided the framework update is useful on its own as a recorded design direction.

## Bottom line

**Yes, we can build this.** For the detective puzzle, cheaply — Stages 1–4 are a 2-week plan on a single 24 GB GPU. For arbitrary domain, with an additional week of unsupervised-objective validation (Stage 7). The full-loop quantification is a real optimization problem, end-to-end differentiable under the DRaFT-1 / Adjoint Matching path, and the major known risks are localized: Ψ-choice (empirical, tractable) and learned-projection generalization (requires cross-corpus validation).

The single biggest pre-commitment action is **Stage 1: swap in FaRL, train a linear probe on existing v3 faces.** Hours of work, tells us whether the rest of this plan is worth running.
