---
status: live
topic: neural-deformation-control
---

# Pivot decision — PersonaLive vs LivePortrait vs hybrid

Decision memo following today's stylized-anchor failure (5/5 photoreal
collapse to FFHQ-blonde-glasses on duck / demon / orc / zombie / anime)
and the subsequent question: do we pivot the realtime VTuber thread off
PersonaLive?

This document captures the option space, sunk-cost audit, quality
ceilings, and the gating empirical test that should run before any
commitment. Companion to
[`2026-05-06-rendering-stack-replacement-options.md`](2026-05-06-rendering-stack-replacement-options.md)
(broader option survey) and
[`2026-05-05-personalive-architecture-notes.md`](2026-05-05-personalive-architecture-notes.md)
(per-component architecture; *Inference-time control surface* section
documents what we already tried with no-retrain levers).

## Scope clarification (two threads, only one affected)

This pivot question concerns the **realtime VTuber-with-a-real-face
thread** (`_topics/neural-deformation-control.md`, default-path
decision dated 2026-05-04). Specifically: iPhone → Linux 5090 → OBS;
ARKit blendshapes drive a chosen reference portrait; output streamed
to OBS. PersonaLive is the current backbone here.

The **vamp-interface product** (telejobs job-posting faces rendered
offline as photoreal PNGs) is not affected. That thread runs on
slider/LoRA/Flux for static portraits and predates PersonaLive
adoption. No decision in this memo touches it.

The stylized-anchor batch we ran today (zombie / orc / duck /
demon / anime) tests the VTuber thread's robustness on non-photoreal
references — i.e., the "be an anime girl" / "be a duck" VTuber use
case. So today's evidence is on-thread, not a tangent.

## Reframing: two decisions, not one

The user's framing "either LivePortrait or FlashPortrait" conflates
two questions because **FlashPortrait is not an alternative to
LivePortrait**. FlashPortrait (CVPR 2026, arXiv 2512.16900) is a
step-skip accelerator that rides on top of a diffusion-portrait
pipeline. It accelerates PersonaLive; it doesn't replace it.

The genuine forks:

- **Speed question** — FlashPortrait integration vs continued
  TRT-only acceleration vs status quo.
- **Stylized-quality question** — pivot to LivePortrait, run hybrid,
  accept scope, or address root cause via Stage-2 retrain.

These should be decided independently. Bundling them obscures both.

## Option space

### A. Stay on PersonaLive (status quo)

Accept the stylized-ref failure as documented limitation. Photoreal
VTuber only. v2 loss redesign and TRT acceleration thread proceed as
planned.

### B. Wholesale pivot to LivePortrait

Drop PersonaLive entirely. Replace with LivePortrait (Kuaishou,
arXiv 2407.03168). Re-derive the ARKit bridge against LivePortrait's
existing MotionExtractor (no FAN-SA distill — LP doesn't have one).

### C. Hybrid (route by ref type)

PersonaLive for photoreal anchors, LivePortrait for stylized anchors.
Both pipelines maintained; reference-type classifier or manual flag
decides at job submission time.

### D. PersonaLive + FlashPortrait accelerator

Orthogonal speed lever. Stays on diffusion track. Doesn't address
stylized.

### E. PersonaLive Stage-2 discriminator swap

Address the root cause of stylized collapse without pivoting.
Replace StyleGAN2-FFHQ Stage-2 discriminator with a mixed panel
(FFHQ + Cartoon-StyleGAN + Anime-StyleGAN). Re-run Stage 2 only
(~3–5 days single H100). Keeps everything else.

## Sunk-cost audit

What transfers if we leave PersonaLive:

| Asset | Transfers to LivePortrait? |
|---|---|
| iPhone Live Link Face capture | ✓ driver-side, model-agnostic |
| ARKit-52 blendshape pipeline | ✓ |
| Closed-form pose (`EULER_SIGNS`, `F_KP_REF`) | ✓ already targets LP's MotionExtractor (we use LP's KP extractor in the bridge) |
| Distilled MotEncoder student (v1, v2 queued) | ✗ distilled FAN-SA, which is PersonaLive-specific |
| 4-step distill infra | ✗ N/A on LivePortrait |
| HKM long-clip stability mechanism | ✗ no equivalent in LivePortrait |
| Phase 2 photoreal grid validation (`2026-05-04-personalive-default-decision.md`) | ✗ would need redo |
| `personalive-acceleration` topic (TRT path, +77%) | ✗ different model |

Pivot cost estimate: ~5–7 days of work redone (the half of the ARKit
bridge that targets PersonaLive's motion encoder, plus Phase 2 grid
re-validation), plus loss of HKM-style long-clip stability that the
PersonaLive paper specifically calls out as a LivePortrait weakness.

## Quality ceiling audit

| Failure mode | PersonaLive | LivePortrait |
|---|---|---|
| Stylized identity collapse to FFHQ-blonde-glasses | ✗ catastrophic (today's evidence) | ? **unknown — gating uncertainty** |
| Long-clip identity drift | ✓ HKM mitigates | ✗ accumulates (paper explicitly criticizes) |
| Mouth open when ref mouth is closed | ✓ can hallucinate teeth | ✗ warp can't synthesize new pixels |
| Large head rotation, back-of-head not in ref | ✓ inpaints occluded regions | ✗ warp artifacts |
| Speed at 512² (RTX 4090, ours 5090 expected ~1.3×) | 15.8 FPS | ~78 FPS inner / 30–60 FPS E2E |
| VRAM | ~16 GB | ~6 GB |
| License | unclear / non-commercial-flavoured (paper, code TBD on release) | MIT |

The structural quality difference: warp can only **interpolate** the
reference; diffusion can **hallucinate**. For VTuber streams with
constant head motion and mouth opening, PersonaLive has a genuine
architectural advantage — *if* the stylized collapse can be addressed.

## The gating uncertainty

We do not actually know whether LivePortrait handles our specific
duck / demon / anime references. The literature says "cartoon works"
and ships an Animals checkpoint, but extreme stylization (beak,
demon teeth, non-humanoid topology) likely has its own failure modes
— probably different from PersonaLive's, not necessarily better.

If LivePortrait *also* fails on duck/demon, the entire pivot
question becomes moot — we'd be choosing between two failure modes
with PersonaLive having higher photoreal quality ceiling and HKM.

This makes pivoting before testing strictly dominated by testing
first. Cost of the test: ~1 hour. Cost of pivoting blind: 5–7 days
plus possible regret.

## Recommendation

**Decompose the decision and gate on cheap empirics.**

Speed question (separate, defer):

- Don't pivot for speed. PersonaLive 15.8 FPS + planned TRT path +
  FlashPortrait integration is the cheapest route to 60+ FPS
  without discarding the stack.
- Decision deferred until VTuber-thread bottleneck is actually
  speed (currently it is not — the bottleneck is stylized
  generalization).

Stylized question (decide soon, gate on test):

1. **Run the 1-hour LivePortrait sanity check** on the same 5
   stylized anchors against the same yaw clip used today. Document
   results in a new dated doc.
2. If LivePortrait handles them cleanly (identity preserved across
   the 5 anchors): **Option C (hybrid)** is the right answer. Both
   threads keep their stacks; route at the API boundary by
   ref-type flag.
3. If LivePortrait fails differently but not better: **Option A
   (accept scope)** for v1 of the VTuber thread, with **Option E
   (discriminator swap)** queued as a v2 R&D track on PersonaLive.
4. If LivePortrait is somehow both faster *and* better on
   photoreal: revisit Option B, but consider this unlikely —
   diffusion's quality ceiling is the architectural reason
   PersonaLive exists.

Decision NOT to take: don't pivot on the "1.5 years of progress"
intuition. LivePortrait is not an older PersonaLive; it is a
different cost/quality Pareto point in a different model family
(warp vs diffusion). Crossing families on speed reasoning alone
is paying for compute we already have headroom on, in exchange
for a quality ceiling we may regret.

## Validation experiment plan (the gating test)

Already have: 5 stylized anchors at `data/anchors/other/cropped/*.png`,
driver clip `data/llf-clips-auto/20260505_MySlate_5_yaw/`.

Steps:

1. Install LivePortrait if not already: `git clone
   https://github.com/KwaiVGI/LivePortrait`; checkpoints download.
2. Render each of the 5 anchors with the same yaw clip as the
   driver. Use both the human checkpoint and the Animals checkpoint
   on each (10 outputs total).
3. Visual judgment: did identity survive? Pose track? Mouth /
   eye animation plausible? Specific failure modes?
4. Quantitative: ArcFace cosine on first frame vs reference
   (only meaningful for the human-ish anchors); visual scorecard
   for non-human.
5. Side-by-side first/middle/last frame contact sheet against
   today's PersonaLive renders.
6. Save outputs to
   `exp_output/realtime/render/stylized_pool_liveportrait/` and
   results to `docs/research/2026-05-07-liveportrait-stylized-test.md`.

Expected wall-clock: ~1 hour including install and contact-sheet
build.

Decision criterion: a clear "yes / no / mixed" verdict on whether
LivePortrait preserves stylized identity. If mixed (e.g., orc OK,
duck fails), bias toward Option A + E (PersonaLive root-cause fix)
over Option C (hybrid), because stylized routing only pays off
when the stylized model has *broad* coverage, not just better
specific cases.

## Sharper question: can we replace just the UNet?

Asked after the initial pivot framing: keep the rest of PersonaLive's
infrastructure (ARKit bridge, conditioning stack, capture pipeline)
and surgically swap only the denoising UNet for a warp / deformation
core. Direct answer: **architecturally infeasible as a literal swap;
viable as UNet replacement *via distillation*.**

### Why a literal swap fails

PersonaLive's UNet is entangled with the rest of the stack via three
interfaces a warp method does not speak:

1. **Different intermediate space.** PersonaLive operates in VAE
   latent (B, 4, 64, 64). LivePortrait operates on pixel-adjacent
   appearance feature volumes (~B, 256, 64, 64) from its
   `AppearanceFeatureExtractor`. Different shapes, different units.
2. **Different conditioning interface.** PersonaLive's UNet consumes
   CLIP image embeds (cross-attn), motion features (32×16, cross-attn),
   pose-guider residual (320 ch additive at conv_in), and the
   ReferenceNet KV bank (`mutual_self_attention.py:171–217`).
   LivePortrait's renderer consumes a dense flow field computed from
   `kp_source × kp_driving` and applies a warp op. No shared
   interface.
3. **The "ReferenceNet" is a UNet itself.** It is not a separable
   component to keep — its purpose is to fill the bank the denoising
   UNet's self-attention reads from. Drop the denoising UNet and the
   ReferenceNet has nothing to talk to.

So "replace just the UNet" is functionally a synonym for "replace
the whole renderer and reference path." The bridge components (ARKit,
iPhone capture, OBS output) survive; everything from ReferenceNet
down does not. Notably, our closed-form pose code (`EULER_SIGNS`,
`F_KP_REF`) **already targets LivePortrait's MotionExtractor** because
we use LP's keypoint extractor inside the PersonaLive bridge — so if
we did move wholesale to LP, that piece transfers cleanly.

### Four UNet-replacement middle paths that are feasible

Broaden "replace the UNet" to include retraining and the option set
opens up:

**Path 1. Distill PersonaLive's 4-step UNet down to a 1-step student.**
Hyper-SD or DMD2-style distillation on the existing UNet. Same family,
same conditioning interface. ~5× speedup → ~75 FPS. Cost: 3–5 days
training + GAN-stability risk. **Does not fix stylized collapse** (same
UNet weights, same FFHQ-trained discriminator legacy if reused).
FlashPortrait achieves comparable speed via step-skipping with no
retrain, so the pragmatic version of Path 1 may be "use FlashPortrait
instead of distilling."

**Path 2. Warp-then-refine hybrid.** Run a warp method (LivePortrait-
class) to produce a structural draft, then a small 1–2-step diffusion
refiner to fix warping artifacts (teeth synthesis, occlusion fill, ID
sharpening). AniPortrait and parts of Hallo2 do internally similar
things. Combines warp's speed + stylized-friendliness with diffusion's
hallucination ability. Cost: substantial, no off-the-shelf recipe at
our scale, ~3–6 weeks research-grade work.

**Path 3. Distill PersonaLive into a single-pass feed-forward
generator.** Train a smaller CNN/GAN (warp-class architecture) where
the **teacher is PersonaLive's 4-step output** and the student
produces the same frame in one forward pass. DMD-style step distill
pushed to 1 step *and* compressed onto a smaller architecture. Result:
a custom warp-class model inheriting PersonaLive's photoreal quality
at warp-class speed. **Sidesteps the StyleGAN2-FFHQ discriminator
entirely** with the right supervision choice (LPIPS + ID + temporal
consistency, no FFHQ adversarial). Cost: ~1–2 weeks training, modest
GPU.

**Path 4. Warp-prior-into-latent.** Use a warp module to produce a
structural latent in PersonaLive's VAE space as the initialization
for the denoising loop instead of pure noise. Effectively shrinks 4
steps to 1–2 by giving the diffusion a head-start. Research-grade;
no published recipe; ~6+ weeks.

### How the gating LivePortrait test interacts with each path

The 1-hour empirical test does not just gate the wholesale pivot
(Option B in the prior section); it also gates which middle path is
viable:

- LivePortrait *fails* on our stylized refs → Paths 2 and 3 inherit
  the failure (they use warp). Path 1 (or FlashPortrait) becomes
  the only realistic speed lever; stylized remains R&D.
- LivePortrait *passes* → Paths 2 and 3 become much more attractive.
  Path 3 specifically gets a clear two-teacher recipe (PersonaLive
  on photoreal, LivePortrait on stylized → student learns to
  dispatch).
- LivePortrait *partially* passes → hybrid Path 2 is most appealing
  (warp where it works, refine elsewhere).

This raises the value of the gating test from "decides the pivot"
to "decides four downstream R&D options simultaneously."

## Pre-test research item: LivePortrait descendants

Before running the LP sanity check, we should map LivePortrait's
descendants — it is unlikely the community has stood still since
July 2024. Candidates to investigate as part of the deep-dive:

- LivePortrait v2 / official follow-ups from KwaiVGI.
- Forks with audio driving (Hallo3, EchoMimic-V2 use LP-class warp).
- ComfyUI-extended variants with extra conditioning slots.
- Stylized-trained variants (anyone retrained AppearanceFeature on
  cartoon/anime corpora?).
- 1024² variants (LP is 512²; some users have pushed it higher).

Output: short note appended here or filed as
`2026-05-07-liveportrait-descendants-survey.md` before the empirical
test runs.

## What this memo does not commit us to

- No code changes.
- No model installs.
- No restructuring of the realtime VTuber thread.
- No pause on the PersonaLive v2 loss redesign (different work,
  proceeds independently).

The only commitment: run the 1-hour test, then revisit this doc
with a verdict before any larger move.
