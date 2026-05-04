---
status: live
topic: neural-deformation-control
---

# LivePortrait + X-NeMo deep dive, and where our slider stack stands relative to them

**Date:** 2026-05-03
**Sources:** `docs/papers/liveportrait-2407.03168.pdf`, `docs/papers/x-nemo-2507.23143.pdf`, `docs/papers/personalive-2512.11253.pdf`.
**Companion:** `2026-05-03-talking-heads-2025-2026-survey.md` (broader landscape), `2026-05-03-slider-operational-handbook.md` (our slider stack).

This doc captures the architectural read on the two load-bearing methods in the deformation/diffusion-hybrid space and compares their loss zoo and design choices to ours, so a future session knows what we already share with the field and where we differ.

## LivePortrait — pure deformation, GAN-trained, 12.8 ms/frame

**Pure non-diffusion**, single forward pass, GAN-trained. The canonical "deformation only" answer at 2026 production quality.

### Stage 1 — base model (trained from scratch, 10 days × 8× A100)

Built on top of face vid2vid:
- **Appearance Feature Extractor F**: source image → 3D feature volume `f_s` (a learned voxelized representation).
- **Motion Extractor M** (single ConvNeXt-V2-Tiny backbone, unifying face vid2vid's 3 separate networks): from any image produces canonical 3D keypoints `x_c,s` + head pose R + expression deformation δ + scale s + translation t.
- **Cross-identity transformation** (eqn 2): `x_d = s_d · (x_c,s · R_d + δ_d) + t_d`. Note: driving frame contributes pose / expression / scale, but **canonical keypoints come from the source** — that's what holds identity put across reenactment.
- **Warping module W** generates a dense flow field from `(x_s, x_d)` and warps `f_s`.
- **SPADE decoder G** (heavier than face vid2vid's, with PixelShuffle to upsample 256² → 512²) renders the final image.
- Trained on 69M filtered frames (KVQ-filtered from 92M) + 60K static styled images, ~18.9K identities.

### Stage 2 — stitching + retargeting MLPs (everything else frozen)

This is where controllability lives:
- **Stitching MLP S(x_s, x_d) → Δ_st** corrects shoulder misalignment when pasting the warped face back into the original full image.
- **Eyes-retargeting MLP R_eyes(x_s, c_s,eyes, c_d,eyes) → Δ_eyes** takes a desired eyes-open scalar in [0, 0.8] and emits the keypoint offset that produces it. Crucially: trained with random driving eyes-open conditions decoupled from the driver — so at inference you can override eye-opening from a slider, independent of pose.
- **Lip-retargeting MLP R_lip** — same shape, lip-open scalar.
- Inference algorithm picks combinations via three indicator flags α_st, α_eyes, α_lip ∈ {0, 1}.

### Speed

12.8 ms/frame on RTX 4090 in plain PyTorch. Single forward pass, no iterative sampling.

The "12 sliders" in PowerHouseMan's AdvancedLivePortrait are hand-tuned linear deltas on these implicit keypoint indices, extending (not replacing) the official `c_eyes` / `c_lip` control surface.

## X-NeMo — diffusion-based, slow, but architecturally upstream of PersonaLive

UNet (Stable Diffusion LDM backbone) + ReferenceNet + AnimateDiff-style temporal modules. ~1.3 FPS / 15 s latency per PersonaLive's table. Important not for production speed but because its **motion-encoding recipe is what PersonaLive and HunyuanPortrait borrow**.

### The thesis

Prior diffusion animators (AniPortrait, X-Portrait) inject motion via spatially-aligned 2D control maps (ControlNet / PoseGuider). The UNet then takes a **shortcut**: instead of learning semantic correspondences between reference identity and driving motion, it just mimics the 2D layout of the control map. Result: identity drift in cross-identity reenactment because the 2D layout carries identity-specific structure.

### The fix — 1D motion bottleneck

Their Motion Encoder `E_mot`:
- Conv layers + self-attention + MLPs, output is a **1D global motion descriptor `f_mot`** (small dim, no spatial structure).
- "Low-pass filter" by construction — strips 2D geometry, keeps motion semantics.
- Plus a 4-dim `f_rts = (Δx/s_r, Δy/s_r, s_d/s_r)` for face-bbox translation/scale.
- Injected via **cross-attention** layers inserted after each spatial transformer block (NOT additively). CFG uses fully-masked appearance + motion-from-reference as negative prompt.

### Three reinforcing tricks

1. **Dual GAN head** — a StyleGAN-like decoder shares `f_app` (from a small conv encoder of the reference) with `f_mot`, trained jointly with reconstruction + adversarial + feature-matching + VGG losses. Why: pure diffusion loss is uniform-weight per latent pixel, biases the motion encoder toward low-frequency motion. The image-level GAN loss forces it to attend to subtle expressions (frowning, puckering). Without it, EMO-SIM 0.65 → 0.43.
2. **Color + spatial augmentations on driving** (color jitter, ±30% scale, piecewise affine + face-centered cropping). Forces `E_mot` to be identity-agnostic.
3. **Reference Feature Masking (RFM).** During training, 30% uniform random mask on the ref network's appearance feature maps before they flow into UNet self-attention. Prevents the UNet from shortcutting motion through high-dim appearance features when source and driving have similar expressions.

### Numbers

ID-SIM 0.787, AED 0.039, EMO-SIM 0.65 — beats LivePortrait (0.702 / 0.055 / 0.48), AniPortrait (0.713), X-Portrait (0.695), PD-FGC (0.604). Not real-time.

## Diffusion vs deformation — three regimes, two routes to real-time

Initial intuition was that real-time = diffusion eliminated. Half right.

| Regime | Examples | Diffusion? | Speed | Where it wins |
|---|---|---|---|---|
| **Pure deformation** | FOMM → face vid2vid → **LivePortrait** → MobilePortrait, FasterLivePortrait | None. GAN-trained warping + SPADE decode, single forward pass. | 12.8 ms/frame (4090); 60+ FPS on iPhone (MobilePortrait) | Photoreal sources, in-distribution faces, clean source image |
| **Pure diffusion** | AniPortrait, X-Portrait, HunyuanPortrait, **X-NeMo**, EMO, Sonic | Yes, 20–30 steps + ControlNet/PoseGuider | 1–2 FPS, 5–15 s latency | Stylized / cartoon / animal sources, extreme expressions, large pose deltas, novel identity at extreme angles |
| **Distilled + streaming diffusion** | **PersonaLive**, Teller, Hallo-Live, TalkingMachines | Yes, but compressed to 2–4 steps + AR micro-chunks | 15–25 FPS, 0.25–0.94 s latency | Best of both: diffusion's generative inpainting at near-deformation speed |

The field reached real-time **twice, by opposite routes**:
- The deformation branch (LivePortrait lineage) was real-time from the start because it never used diffusion. Production via better data scale (69M frames), better training (cascaded GAN losses), explicit retargeting MLPs.
- The diffusion branch caught up to real-time only in late 2025 / 2026 by aggressively distilling 20-step sampling down to 2–4 steps (PersonaLive uses StyleGAN2-adversarial distillation; TalkingMachines uses asymmetric KD; Teller skips diffusion entirely with AR motion generation).

### Why diffusion didn't get killed off entirely

1. **Occlusions and disocclusions** — when the warp exposes regions absent from source (back of head rotates into view, hand passes across face), pure deformation has nothing to inpaint with. Diffusion has a generative prior.
2. **Out-of-distribution sources** — LivePortrait's appearance encoder was trained on 69M *real* video frames. Stylized portraits (anime, cartoons, painterly Flux outputs) fall off-distribution and warp poorly. Diffusion methods inherit Stable Diffusion's broad prior and tolerate it better (though X-NeMo Fig 5 shows it still struggles on extreme cases).
3. **Identity at extreme expression** — mouth wide open / tongue out / dramatic frown. Warps lose detail because the *source pixels* for those regions never existed. Diffusion synthesizes them.
4. **Compositional motion across long sequences** — pure warps drift over minutes; diffusion + temporal modules + (PersonaLive's) historical keyframe bank stay stable.

### Practical implication for our pipeline

Flux Krea portraits are photorealistic and in-distribution for LivePortrait's training data. **For photo-real expression edits at near-neutral pose, LivePortrait alone is the right answer** — the diffusion stack is overkill, and the 12 ms inference budget lets us run hundreds of variations cheaply. The diffusion methods become necessary the moment we (a) stack a stylization LoRA on top of Flux, (b) want large head turns, or (c) need long sustained streaming with no drift. PersonaLive becomes the right escalation; X-NeMo is the upstream paper to read first because its 1D-motion + cross-attention recipe is what enables PersonaLive at all.

## How close is LivePortrait's loss zoo to ours?

Side-by-side on the *concepts* (not the specific weights):

| Concept | LivePortrait | Ours | Verdict |
|---|---|---|---|
| Identity preservation | `L_faceid` (ArcFace on decoded image) | `id_loss` (arc_distill ArcFace student in latent space) | **Same idea, different implementation.** Theirs runs full ArcFace per step on decoded RGB — accurate but expensive. Ours runs a 43.6M-param student on the latent — cheaper but more indirect. The "less direct" is exactly what enables our classifier-fooling problem; theirs is dense enough to be unfoolable in practice. |
| Face structural preservation | landmark-guided Wing on 10 eye+lip kp + cascaded face perceptual | bs_loss preserve mode (52-d MediaPipe blendshape critic) | **Different surface.** Theirs supervises the *geometry*, ours supervises the *blendshape vector*. Both penalise non-target perturbations; ours has finer semantic granularity (52 channels vs 10 landmarks); theirs has ground-truth supervision (Wing on actual annotated landmarks). |
| Region-localised supervision | cascaded global+face+lip perceptual + 3 separate GAN discriminators | `eye_mask_peak` geometric amplifier on z² | **Big gap in their favour.** They have *three full discriminators* trained from scratch (global, face, lip), each scrutinising its region. We have a Gaussian-mask gradient amplifier — much weaker. |
| Adversarial training | 3 GAN discriminators on the *generator* | None on the LoRA generator. PGD only on the *critic* (v4_pgd). | **Different target, opposite directions.** They use GAN to push image realism. We use PGD to make the critic robust to LoRA perturbation. Both reduce a "fooling" gradient, but at opposite ends of the loop. |
| Equivariance / kp consistency | `L_E` equivariance loss on extracted kp under affine | None | We don't predict keypoints, so it doesn't apply. Their analogue for us would be "your slider should commute with affine transforms of input" — could be tested. |
| Explicit bound on perturbation magnitude | L1 reg on Δ_eyes, Δ_lip, Δ_st in stage 2 | weight decay only on LoRA | **Their explicit L1 cap on the deformation offset is the trick we don't have.** Their controllable knobs can't blow up because the MLP output is L1-bounded. Our LoRA can drift arbitrarily — we discovered this the hard way, and the arc anchor is our workaround. A direct L1 cap on the LoRA output δv could be cheaper. |
| Region condition matching | `‖c_s − c_d‖_1` in retargeting MLPs | `bs_loss_engage_target` scalar match | **Conceptually identical.** Both: "MLP/LoRA, please make the readout match the target scalar." |
| Off-region preservation | implicit via global cascaded perceptual | geometric latent mask (Path 1a, planned, not built) | LivePortrait's choice — let the global perceptual force everything outside the active region to look like the source — is more elegant than our explicit mask. |

### Were we on the right track?

**Yes, broadly.** The two big alignments — ArcFace identity loss + scalar target matching for controllable parameters — are universal across both communities. Our latent-space reformulation (running distilled critics on Flux latents instead of decoded RGB) is a sensible compute optimization for diffusion-LoRA training where decode is the bottleneck.

**Where we diverged from best practice:**
- **No GAN on the generator.** This is the load-bearing trick LivePortrait uses to keep cross-id reenactment sharp without identity drift. Our LoRA has *no* discriminator anywhere. Combined with our reliance on a single classifier critic, this is structurally why we get classifier-fooling.
- **No explicit L1 bound on the perturbation.** LivePortrait's retargeting MLPs *cannot* fool ArcFace because their output is L1-bounded; ours can.
- **Single critic instead of cascaded supervision.** They get global + face + lip perceptual + 3 GANs + ArcFace + Wing landmarks all at once. Each loss term pulls in a slightly different direction; no single direction is cheap because some other loss penalises it. We get bs_loss + arc anchor — only 2 terms — which is exactly why we needed PGD-robust critics and arc anchor at w=5000 to avoid collapse.

### Original ideas worth contributing back

These are things our setup has that LivePortrait doesn't:

1. **PGD-trained robust critic (v4_pgd).** LivePortrait's identity loss is a frozen ArcFace; their dense loss zoo prevents fooling, but a *minimalist* setup would benefit from PGD-trained discriminators. The general technique ("train your loss head to be Lipschitz under the perturbations the generator can produce") is reusable.
2. **Latent-space distilled critics for fast training.** Running ArcFace on decoded RGB inside a diffusion training loop is expensive. arc_distill (43.6M params, takes Flux latent directly) gets ~95% of the signal at a fraction of the FLOPs. For any Flux-based animator this is a real speedup.
3. **Step-gating at inference.** Single-pass methods can't use this, but for any diffusion-based slider on Flux, restricting LoRA application to early-mid denoising percentages preserves detail. Direct wins for HunyuanPortrait / X-Portrait / X-NeMo if anyone trains LoRAs on top of them.
4. **NMF atom decomposition over MediaPipe blendshapes.** Gives us 8 perceptually-meaningful axes (DFECS-style) that aren't in ARKit. LivePortrait's "implicit blendshapes" are the K=21 keypoints with no semantic interpretation. Our NMF basis is interpretable and could be used as an editable basis on top of LivePortrait's keypoints if we ever wanted to.
5. **Solver C (inverse training-pair selection) on FFHQ.** They train on 69M unfiltered video pairs of the same person over time. We curate ~1000 high-J pairs per axis using the Mahalanobis-weighted scoring rule. For *small-data* situations or rare attributes (squint, brow asymmetry, micro-expressions), our approach is the better fit because it doesn't require temporal video pairs of the target attribute.

## Companion experiment plan

See `2026-05-03-personalive-experiment-plan.md` for the four-step verification of the claims in this doc.
