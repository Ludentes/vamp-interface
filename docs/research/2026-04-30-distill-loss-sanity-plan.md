---
status: live
topic: arc-distill
---

# Distill-loss sanity plan — arc_distill + mediapipe_distill

Both adapters have shipped checkpoints; only `arc_distill` is wired into the
ai-toolkit ConceptSliderTrainer. This plan validates each as a usable
training-time loss term before they're used in squint v1.

## Context

- **arc_distill**: 43.6M-param adapter, ArcFace cos on noisy-latent `x0`
  estimate. Trainer integration done at
  `/home/newub/w/ai-toolkit/extensions_built_in/concept_slider/ConceptSliderTrainer.py`
  (`_id_loss_term`, `_get_id_adapter`, `id_loss_*` config knobs lines 46–59,
  used in pos/neg halves at lines 519 and 572). Sanity yaml exists at
  `/home/newub/w/ai-toolkit/config/glasses_slider_v9_idloss_sanity.yaml`,
  not yet run.
- **mediapipe_distill**: 5 variants (v1, v2b, v2c default, v2d, v2e). 11.47M
  params each. v2c is canonical 52-d bs predictor. **No trainer integration
  yet** — must mirror id_loss pattern.

## Step 1 — arc_distill sanity (shovel-ready, ~1h GPU)

Launch `glasses_slider_v9_idloss_sanity.yaml`. Differences from v8: id_loss
on (weight 1.0, t_max 0.4, latent_a2_full_native_shallow). Manual kill at
step ~600.

**Pass criteria (from yaml comments):**
1. `id_loss` finite, non-NaN, trending down
2. Glasses engagement at m=±1.5 matches v8 at step 600 (no regression on
   intended axis)
3. ArcFace cos to m=0 anchor at m=±1.5 **higher** than v8 baseline

**Fail modes:**
- Glasses don't engage by step 600 → id_loss too strong; drop weight 0.5/0.25
- id_loss NaN/Inf → x0 prediction misformulated for flowmatch
- Crash on adapter forward → import path / checkpoint path wrong

## Step 2 — wire `bs_loss` term into ConceptSliderTrainer (~1h coding)

Mirror id_loss pattern in `ConceptSliderTrainer.py`.

**Config knobs (`ConceptSliderTrainerConfig.__init__`):**
- `bs_loss_weight: float` (default 0.0 = off)
- `bs_loss_t_max: float` (default 1.0)
- `bs_loss_t_norm: float` (default 1000.0)
- `bs_loss_checkpoint: str` (path to v2c checkpoint.pt)
- `bs_loss_variant: str` (default `bs_a`)
- `bs_loss_repo: str` (default vamp-interface root)
- `bs_loss_mode: str` (`preserve` | `engage`, default `preserve`)
- `bs_loss_channels: List[str]` (channel names; resolved against the 52
  ARKit blendshape names in `BlendshapeStudent`)
- `bs_loss_engage_target: float` (only used in `engage` mode)

**Functions to add (mirror `_get_id_adapter` / `_id_loss_term`):**
- `_get_bs_adapter(repo, variant, checkpoint, device, dtype)`:
  - lazy-load `BlendshapeStudent("bs_a")` from `mediapipe_distill.student`
  - module global `_BS_ADAPTER`, `_BS_CHANNEL_INDEX`
  - load `ck["model"]`, eval, `requires_grad_(False)`
- `_bs_loss_term(class_pred, neutral_pred, noisy_latents, timesteps, cfg)`:
  - same t-gating, same x0 reconstruction (`x_t - t·v`)
  - `bs_anchor = adapter(x0_anchor).detach()` — (B, 52)
  - `bs_edited = adapter(x0_edited)` — (B, 52)
  - mask = bool tensor (52,) over `cfg.bs_loss_channels` (cache by id)
  - **preserve mode**: `weight * ((bs_anchor - bs_edited)**2)[:, mask].mean()`
  - **engage mode**: `weight * ((cfg.bs_loss_engage_target - bs_edited[:, mask])**2).mean()`

**Wiring:**
- Add `bs_pos = _bs_loss_term(...)` after `id_pos = ...` at line 519
- `total_pos_loss = base/3 + (id_pos or 0) + (bs_pos or 0)`
- Same for negative half (line 572)

**Channel-name resolution:** load the 52 ARKit names from
`BlendshapeStudent.CHANNELS` (or hardcode if not exposed; the canonical list
is in `models/mediapipe_distill/v2c/validation_report.json` keys).

## Step 3 — `glasses_slider_v10_bsloss_sanity.yaml` (~1h GPU)

Same recipe as v9 (glasses, known-good baseline) but swap id_loss → bs_loss:

```yaml
bs_loss_weight: 1.0
bs_loss_t_max: 0.4
bs_loss_t_norm: 1000.0
bs_loss_checkpoint: "/home/newub/w/vamp-interface/models/mediapipe_distill/v2c/checkpoint.pt"
bs_loss_variant: "bs_a"
bs_loss_repo: "/home/newub/w/vamp-interface"
bs_loss_mode: "preserve"
bs_loss_channels:
  - mouthSmileLeft
  - mouthSmileRight
  - jawOpen
  - jawForward
  - browDownLeft
  - browDownRight
```

Channels chosen because:
- they're in v2c's `confident_ship + ship` bucket (all R² ≥ 0.7)
- glasses-edit *shouldn't* touch them — they form a clean preservation gate

**Pass criteria:**
1. `bs_loss` finite, non-NaN, descending — wiring sanity
2. Glasses engagement at m=±1.5 matches v8 at step 600 — no regression
3. Δ(bs[selected channels]) between m=0 anchor and m=±1.5 is **smaller**
   than v8 baseline — the preservation effect we're buying

## Step 4 — squint v1 (after both sanities pass)

Stack both losses + the other v1 inputs already on file:

- `id_loss_weight: 0.5–1.0` (fixes id_pass collapsing by |m|=0.5)
- `bs_loss_mode: preserve` on `[mouthSmileLeft, mouthSmileRight,
  mouthDimpleLeft, mouthDimpleRight]` (fixes smile-creep that v0 step 1550
  exhibited at 0.271 vs v0 step 1800's 0.098)
- prompt sanitisation: drop "crow's feet", "compressed eyelids"
- `eye_mask_peak: 2.5–3.0` (was 5.0; over-concentrates gradient)
- `anchor_class: "a portrait photograph of a person"` (was null in v0)
- cosine lr, shorter schedule (~1500 steps; v0 saturated ~1500)

## Open questions

- `bs_loss_mode: engage` is more powerful than `preserve` (drives a channel
  to a target instead of just penalizing change). Keep that path for v2 once
  preserve mode is validated; don't try to test both modes in the same
  sanity run.
- Should `bs_loss` be applied to `class_pred` only (current id_loss design)
  or to both halves of `(class_pred, anchor_pred)` when anchor_class is on?
  Current id_loss does `class_pred` only; for preserve-mode bs_loss the
  same is correct (penalize change of edited vs neutral).
- v2c degenerate channels (`_neutral`, `cheekSquintL/R`, `noseSneerL/R`)
  must never appear in `bs_loss_channels`; add a sanity assert at adapter
  load time.
