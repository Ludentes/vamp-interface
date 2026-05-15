---
status: live
topic: personalive-acceleration
---

# Decoupled-channel CLIP probe: stylization falsified, but stacks as a photoreal cleanup knob

Date: 2026-05-06

## Headline

PersonaLive feeds the reference image into two distinct conditioning channels: (1) `image_encoder` produces a (1, 1, 768) CLIP-image embed cross-attended at every diffusion step; (2) `reference_unet` reads VAE-encoded ref latents and writes spatial features into denoising_unet, plus those latents seed `init_latents`. Both currently take the same image.

We forked `apply_bridge_to_personalive.py` to add `--reference_clip <path>`: monkey-patches `pipe.image_encoder` so channel (1) is fed a different image than channel (2). Tested anime CLIP ref + photoreal grid spatial ref against the MySlate_5 yaw-stress driver; then stacked Ghibli LoRA on top.

## Findings

- **Decoupled CLIP alone (anime → channel 1, photoreal → channel 2)** produces a render that's a slightly cleaner photoreal portrait — not anime. CLIP-image surface carries real signal (mean|Δ| up to 6.78 vs same-ref baseline at peak yaw, comparable to a 1× LoRA full-merge), but RefNet's spatial copy of the photoreal ref dominates appearance and prevents categorical style flip.
- **Compound (decoupled CLIP + Ghibli LoRA `lora_targets=both` α=1.0)** stacks linearly with no interaction: deltas-vs-baseline go from 5.96 → 12.66 across yaw, deltas-vs-decoupled-only sit at 5.17 → 11.27 (≈ the LoRA-alone delta we measured prior). No nonlinear synergy; both perturbations contribute independently.
- **Visual outcome**: cleaner-than-baseline photoreal. Smoother shading, less skin texture noise, cleaner edges. **Not anime.** RefNet's spatial copy of the photoreal ref categorically anchors the output in photoreal space.

| metric | f=0 | f=50 | f=150 | f=280 |
|---|---|---|---|---|
| decoupled-only vs baseline | 2.19 | 4.77 | 6.78 | 4.63 |
| compound vs baseline | 5.96 | 10.33 | 12.66 | 10.79 |
| compound vs decoupled-only | 5.17 | 8.06 | 11.27 | 9.91 |

## Mechanistic reading

Both perturbations point in the **away-from-messy-real-photo direction**. Anime CLIP embeds emphasize smooth shading, simplified features, clean contours, idealized symmetry — the SD1.5 prior reads those tokens as "render an idealized face," not "render an anime face," because no spatial anime signal anchors that interpretation. Ghibli LoRA at α=1 in 4-step distilled inference integrates partially; the partial direction-of-shift away from photoreal is "smoother shading, less texture, cleaner edges."

Both stack as *cleanup* within the photoreal categorical floor. This is the two-channel image-space analogue of classic SD1.5 *negative prompts* (`"ugly, blemish, asymmetric, photo grain"`): perturbations toward the cleanest, most idealized point on the photoreal manifold without leaving it.

## Implications for product framing

For vamp-interface's uncanny-valley signal mechanism this is potentially a feature, not a curiosity. The conceit is that legit jobs (sus_level=0) anchor to a clean idealized face; fraud increases push toward valley-center wrongness. A cleaner-than-real portrait with both knobs maxed out lands at *idealized mannequin* — exactly where the bottom of the sus dial wants to live. Worth keeping the lever even though it doesn't deliver the originally hypothesized stylization.

For PersonaLive stylization more broadly: stylization needs to attack RefNet's write path (the per-pixel feature copy), not the cross-attention read path. Off-the-shelf SD1.5 LoRA additive merge at the 4-step distilled budget has already saturated. Next probe is a LoRA trained against PersonaLive's actual 4-step trajectory targeting RefNet specifically.

## Negative result on VAE swap for style

PersonaLive uses standard SD1.5 VAE (or TAESD `vae_tiny_path`). VAE is a near-identity decoder of latents to pixels — it doesn't make style decisions, it renders the latent the UNet produced. "Anime VAE" / "Orange Mix VAE" downloads share the SD1.5 latent space and only tweak chroma/contrast (effect: small, not structural). VAEs with distinct latents (Würstchen, SD3, FLUX) aren't swappable into PersonaLive at all.

Estimated effect of swapping in an anime-checkpoint-bundled VAE: mean|Δ| < 3, mostly saturation. Not pursued.

## Hypothesis tracker (per `_topics/personalive-acceleration.md`)

- **Falsified now**: stylized ref alone (prior thread); decoupled-CLIP alone; decoupled-CLIP + LoRA both; off-the-shelf SD1.5 LoRA at 4-step budget. All bounded by RefNet's photoreal spatial copy.
- **Confirmed real signal**: CLIP-image channel (≈ 1× LoRA worth of magnitude); LoRA additive merge surface (192/264 pairs match, linear in α to saturation).
- **Newly characterized**: compound is a clean *photoreal cleanup knob*, useful as the legit-end calibration anchor for vamp-interface's sus dial.
- **Next probe (open)**: monkey-patched 8-step schedule with off-the-shelf LoRA at higher α — Hail Mary on whether off-anchor budget alone unlocks style transfer despite distillation mismatch. If falsified, RefNet-targeted custom LoRA training is the only remaining lever short of Stage-1 retraining.

## Output artifacts

- `exp_output/personalive/decoupled_clip/anime_clip_x_photoreal_spatial.mp4` — decoupled CLIP alone
- `exp_output/personalive/decoupled_clip/anime_clip_x_photoreal_spatial__ghibli_both_a1.mp4` — compound
- `exp_output/personalive/decoupled_clip/_frames/sbs_f{0,150}.png` — side-by-side at frontal + peak yaw
- `runs/decoupled_clip_anime_x_photoreal.log`, `runs/decoupled_clip_compound.log` — full stdout

## Implementation

`scripts/apply_bridge_to_personalive.py` `--reference_clip <path>` flag. Monkey-patches `pipe.image_encoder` with an `nn.Module` shim that returns precomputed embeds; evicts the original from `pipe.components` to free GPU memory. Reviewer findings (2026-05-06): nn.Module subclass for `pipe.eval()` / `.to()` compatibility, explicit `.device` and `.dtype` exposure, `pipe.components.pop("image_encoder")` for full registry eviction.

## Companions

- `2026-05-06-stylized-vtuber-requires-personalive-lora.md` — sets up the LoRA-on-PersonaLive thesis; this doc falsifies the cheap CLIP-channel route to it.
- `_topics/personalive-acceleration.md` — topic index, update in same commit.
