# CFM Single-Pass Rewrite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rip out the twin-pass InfuseNet architecture, fix the silently-broken id_tokens precompute, and ship a single-pass CFM trainer that matches canonical InfiniteYou wiring. Gate every step on the existing sanity script (`scripts/cfm_sanity_iy_recipe.py`) before relaunching the pilot.

**Why this rewrite:** Two compounding bugs made identity transfer fail under the previous architecture:

1. **Precompute fed buffalo_l's `normed_embedding` to a resampler trained against facexlib `init_recognition_model('arcface')` (IR-SE-50 IR-SE-50 over 5-pt-aligned 112² norm_crop).** Different ArcFace heads → orthogonal 512-d output spaces → resampler returns junk tokens. Confirmed quantitatively: cos(iy_tokens, buf_tokens) = -0.12 to 0.00 across 4 picks.
2. **Forward wiring concatenated id_tokens onto T5 and passed the combined sequence to both InfuseNet and FLUX.** Canonical wiring (per `ComfyUI_InfiniteYou/infuse_net.py:62` and `pipeline_flux_infusenet.py:558,575`) replaces InfuseNet's cross-attention context with id_tokens alone, and feeds T5 only to the FLUX backbone.

The twin-pass architecture (frozen identity branch + LoRA expression branch + `disable_adapter()`/`no_grad` scaffolding) was a fabrication chasing symptoms of the two bugs above. With the precompute and wiring fixed, single-pass is the correct shape and is materially smaller.

**Reference docs:**
- Prior plan (superseded): `docs/superpowers/plans/2026-05-20-cfm-training-run.md`
- Asset report: `docs/research/2026-05-16-infiniteyou-asset-report.md`
- Architecture brief: `docs/research/2026-05-16-arkit-controlnet-infiniteyou.md`
- Upstream pipeline (canonical wiring source): `/tmp/InfiniteYou/pipelines/pipeline_flux_infusenet.py`
- Upstream ComfyUI node (id_embedding semantics): `/home/newub/w/ComfyUI/custom_nodes/ComfyUI_InfiniteYou/infuse_net.py`
- Working sanity script (the spec-by-example): `scripts/cfm_sanity_iy_recipe.py`

**File map:**

| File | Change |
|---|---|
| `src/arkit_controlnet/cfm/precompute.py` | swap ArcFace recognition path; force `id_ok=False` reset |
| `src/arkit_controlnet/cfm/model.py` | delete twin-pass; single-pass `velocity()` |
| `src/arkit_controlnet/cfm/train.py` | drop AdamW8bit; pass two encoder_hidden_states |
| `src/arkit_controlnet/cfm/eval.py` | 3-col collage; drop cs=0/cs=1 split |
| `scripts/cfm_sanity_iy_recipe.py` | promote to standing gate (no edits needed) |

**Tech stack unchanged:** nf4 FLUX via bitsandbytes, peft LoRA, gradient checkpointing, atomic per-sha cache. Bitsandbytes optimizer dropped in favor of `torch.optim.AdamW` (trainable params shrunk).

---

## Task 1: Re-precompute id_tokens with the InfiniteYou recipe

**Files:**
- Modify: `src/arkit_controlnet/cfm/precompute.py`
- Delete (after change lands): `output/cfm_precompute/id_tokens/`

The precompute already runs `face_crop_resize` and detects faces with buffalo_l for the 5-pt landmark. Replace only the **recognition** step: 5-pt `face_align.norm_crop(image_size=112)` → `facexlib init_recognition_model('arcface', device='cuda')` → 512-d.

- [ ] **Step 1: Replace `_load_arcface` with a recognition-only helper**

```python
# precompute.py — replace _load_arcface with two helpers
def _load_face_detector():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    return app


def _load_arcface_recognition():
    from facexlib.recognition import init_recognition_model
    return init_recognition_model("arcface", device="cuda")
```

- [ ] **Step 2: Rewrite the id-token block in `run()`**

```python
# inside run(), replace the buffalo_l-embedding block:
id_ok = it_path.exists()
if not id_ok:
    from insightface.utils import face_align
    bgr = crop[:, :, ::-1].copy()
    face_info = detector.get(bgr)
    if not face_info:
        id_ok = False
    else:
        face = sorted(
            face_info,
            key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]),
        )[-1]
        aligned = face_align.norm_crop(bgr, landmark=np.array(face.kps), image_size=112)
        x = (torch.from_numpy(aligned).unsqueeze(0).permute(0, 3, 1, 2).float() / 255.0)
        x = (2 * x - 1).to(device).contiguous()
        with torch.no_grad():
            emb = arcface(x)  # (1, 512)
            emb_t = emb.to(dtype=torch.bfloat16).view(1, 1, 512)
            tokens = resampler(emb_t)[0]  # (8, 4096)
        _atomic_torch_save(tokens.to("cpu", dtype=torch.bfloat16), it_path)
        id_ok = True
```

Drop the old `arcface = _load_arcface()` line above; build the new `detector = _load_face_detector()` and `arcface = _load_arcface_recognition()` in `run()`.

- [ ] **Step 3: Invalidate the existing cache**

```bash
rm -rf output/cfm_precompute/id_tokens
PYTHONPATH=src uv run python -c "
import pandas as pd
m = pd.read_parquet('output/cfm_precompute/meta.parquet')
m['id_ok'] = False
m.to_parquet('output/cfm_precompute/meta.parquet', index=False)
print('reset id_ok on', len(m), 'rows')
"
```

- [ ] **Step 4: Re-run precompute on the eval picks first (8 shas), confirm `id_ok=True` and cached file exists**

```bash
echo -e "$(jq -r '.[]' exp_output/cfm_train/pilot/eval_picks.json | head -8)" > /tmp/sanity_shas.txt
PYTHONPATH=src uv run python -m arkit_controlnet.cfm.precompute \
    --shas-file /tmp/sanity_shas.txt --out-dir output/cfm_precompute
```

- [ ] **Step 5: Sanity-gate the rebuild**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    uv run python scripts/cfm_sanity_iy_recipe.py --n 4
```

The sanity script reads the cached id_tokens (col 2/3) and recomputes IY-recipe id_tokens inline (col 2/3 sampling). After the rebuild, cos(iy_tokens, buf_tokens) should be replaced by cos(iy_tokens, new_cached_tokens) ≈ **1.0**. Edit the comparison in the script if needed, or accept the existing print is now self-comparison.

- [ ] **Step 6: Background-batch the rest of the corpus**

```bash
PYTHONPATH=src uv run python -m arkit_controlnet.cfm.precompute \
    --out-dir output/cfm_precompute
```

Wall: ~10 min for ~6000 rows on the 5090.

- [ ] **Step 7: Commit**

```bash
git add src/arkit_controlnet/cfm/precompute.py
git commit -m "fix(cfm): use facexlib ArcFace for id_tokens, not buffalo_l (resampler training-time match)"
```

---

## Task 2: Single-pass model.py

**Files:**
- Modify: `src/arkit_controlnet/cfm/model.py`

- [ ] **Step 1: Strip `CfmModel` to two fields**

```python
@dataclass
class CfmModel:
    flux: object
    infusenet: object
    trainable_params: list[Parameter]
```

Delete `frozen_x_embedder` and `frozen_controlnet_x_embedder`.

- [ ] **Step 2: Strip `build_model` of the twin-pass scaffolding**

Remove the `import copy`, the `frozen_x_emb`/`frozen_cn_x_emb` deep-copies, and the manual `requires_grad_(False)` → per-param `requires_grad=True` flip on `x_embedder`. Keep nf4 FLUX, peft LoRA on the same target list, `infusenet.requires_grad_(False)`, `flux.train()` for checkpointing, `enable_gradient_checkpointing()` on both.

Only `controlnet_x_embedder` becomes trainable now:

```python
mod = (infusenet.base_model.model.controlnet_x_embedder
       if hasattr(infusenet, "base_model")
       else infusenet.controlnet_x_embedder)
for p in mod.parameters():
    p.requires_grad = True
```

`x_embedder` stays frozen (z_t is the same distribution InfuseNet trained on).

- [ ] **Step 3: Replace `velocity()` with single-pass**

```python
def velocity(
    model: CfmModel,
    z_t_packed: torch.Tensor,
    sigma: torch.Tensor,
    id_tokens: torch.Tensor,
    t5_seq: torch.Tensor,
    pooled: torch.Tensor,
    id_txt_ids: torch.Tensor,
    t5_txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    control_packed: torch.Tensor,
    guidance: Optional[torch.Tensor] = None,
    conditioning_scale: float = 1.0,
) -> torch.Tensor:
    """Canonical InfiniteYou wiring: id_tokens → InfuseNet, t5 → FLUX."""
    cn_d, cn_s = model.infusenet(
        hidden_states=z_t_packed,
        controlnet_cond=control_packed,
        conditioning_scale=conditioning_scale,
        encoder_hidden_states=id_tokens,
        pooled_projections=pooled,
        timestep=sigma,
        img_ids=img_ids,
        txt_ids=id_txt_ids,
        guidance=guidance,
        return_dict=False,
    )
    v_packed = model.flux(
        hidden_states=z_t_packed,
        timestep=sigma,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=t5_seq,
        txt_ids=t5_txt_ids,
        img_ids=img_ids,
        controlnet_block_samples=cn_d,
        controlnet_single_block_samples=cn_s,
        return_dict=False,
    )[0]
    return v_packed
```

- [ ] **Step 4: Delete the helpers**

Remove `_infusenet_residuals` and `_sum_residuals`. They have no callers after Step 3.

- [ ] **Step 5: Smoke test**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    uv run python scripts/cfm_sanity_iy_recipe.py --n 4
```

Sanity collage should still produce identity-transferred faces (Task 1 already ensures the cached id_tokens are correct; Task 2 only changes the path that consumes them inside the library).

- [ ] **Step 6: Commit**

```bash
git add src/arkit_controlnet/cfm/model.py
git commit -m "refactor(cfm): single-pass InfuseNet with separate id/t5 cross-attention"
```

---

## Task 3: Single-pass train.py

**Files:**
- Modify: `src/arkit_controlnet/cfm/train.py`

- [ ] **Step 1: Replace optimizer**

```python
# Replace the bitsandbytes import + AdamW8bit construction with:
opt = torch.optim.AdamW(model.trainable_params, lr=lr, weight_decay=0.01)
```

Drop `import bitsandbytes as bnb`.

- [ ] **Step 2: Build two txt_ids one-time, outside the loop**

```python
id_txt_ids = prepare_text_ids(8, device, dtype)
t5_txt_ids = prepare_text_ids(t5_seq.shape[1], device, dtype)
```

Delete the existing `txt_ids = prepare_text_ids(8 + 512, ...)` line.

- [ ] **Step 3: Simplify the step body**

```python
photo_latent = batch["photo_latent"].to(device, dtype)
id_tokens = batch["id_tokens"].to(device, dtype)
ctrl_latent = batch["ctrl_latent"].to(device, dtype)

z0 = photo_latent
eps = torch.randn_like(z0)
sigma = _flux_sigma(torch.randn(1, device=device))
s = sigma.view(-1, 1, 1, 1).to(dtype)
z_t = (1 - s) * z0 + s * eps
target_packed = pack_latents(eps - z0)
z_t_packed = pack_latents(z_t)
control_packed = pack_latents(ctrl_latent)

v_pred = velocity(model, z_t_packed, sigma, id_tokens, t5_seq.expand(1, -1, -1),
                  pooled_one, id_txt_ids, t5_txt_ids, img_ids,
                  control_packed, guidance)
```

Delete:
- `eh = torch.cat([id_tokens, t5_seq.expand(1, -1, -1)], dim=1)` line
- `id_control_packed = pack_latents(photo_latent)` line
- `expr_control_packed = pack_latents(ctrl_latent)` line

- [ ] **Step 4: Update the eval-cadence call**

`dump_samples` signature changes in Task 4 — keep the call but pass new args:

```python
if step % eval_cadence == 0:
    from arkit_controlnet.cfm.eval import dump_samples
    dump_samples(model, step, out_dir=out_dir,
                 text_embeds=(t5_seq, pooled_one))
```

(No change needed if `dump_samples` continues to take `text_embeds`; the rewrite in Task 4 keeps that.)

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/cfm/train.py
git commit -m "refactor(cfm): single-pass training step + plain AdamW (trainable surface shrunk)"
```

---

## Task 4: Single-pass eval.py

**Files:**
- Modify: `src/arkit_controlnet/cfm/eval.py`

- [ ] **Step 1: Update `_sample` to the new `velocity` signature**

```python
@torch.no_grad()
def _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
            id_txt_ids, t5_txt_ids, img_ids, control_packed, guidance,
            conditioning_scale=1.0):
    from arkit_controlnet.cfm.model import velocity
    z = z_T
    for i in range(len(sigma_seq) - 1):
        s = sigma_seq[i]
        s_next = sigma_seq[i + 1]
        v = velocity(model, z, s.expand(z.shape[0]), id_tokens, t5_seq, pooled,
                     id_txt_ids, t5_txt_ids, img_ids, control_packed, guidance,
                     conditioning_scale=conditioning_scale)
        z = z + (s_next - s) * v
    return z
```

- [ ] **Step 2: Update `_setup` to return two txt_ids**

```python
def _setup(vae, text_embeds, device, dtype):
    if vae is None:
        vae = _vae_singleton(device, dtype)
    if text_embeds is None:
        t5_seq, pooled = _load_text_embeds(TEXT_EMBEDS_PATH, device, dtype)
    else:
        t5_seq, pooled = text_embeds
    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    id_txt_ids = prepare_text_ids(8, device, dtype)
    t5_txt_ids = prepare_text_ids(t5_seq.shape[1], device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)
    return vae, t5_seq, pooled, img_ids, id_txt_ids, t5_txt_ids, guidance
```

- [ ] **Step 3: Rewrite `dump_samples` columns**

```python
@torch.no_grad()
def dump_samples(model, step, out_dir, n=8, sample_steps=25,
                 vae=None, text_embeds=None):
    """Render n eval picks: [target | control | generated]."""
    out = Path(out_dir)
    samples_dir = out / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device, dtype = "cuda", torch.bfloat16
    vae, t5_seq, pooled, img_ids, id_txt_ids, t5_txt_ids, guidance = _setup(
        vae, text_embeds, device, dtype)

    shas = _picks(out, n)
    ds = CfmPairDataset(split="eval")
    sub = ds.df[ds.df.image_sha256.isin(shas)].copy()
    sub["__order"] = sub.image_sha256.map({s: i for i, s in enumerate(shas)})
    sub = sub.sort_values("__order").reset_index(drop=True)
    ds.df = sub.drop(columns="__order")

    ffhq_idx, shards = _ffhq_index_and_shards()
    sigma_seq = _build_sigma_seq(sample_steps, device, dtype)

    rows = []
    for i in range(len(ds)):
        item = ds[i]
        sha = item["sha"]
        row = ds.df.iloc[i]
        photo_latent = item["photo_latent"].to(device, dtype).unsqueeze(0)
        id_tokens = item["id_tokens"].to(device, dtype).unsqueeze(0)
        ctrl_latent = item["ctrl_latent"].to(device, dtype).unsqueeze(0)
        control_packed = pack_latents(ctrl_latent)

        bs_target_dict = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
        for n in ARKIT_BLENDSHAPE_NAMES:
            if n != "tongueOut":
                bs_target_dict[n] = float(row.get(f"bs_{n}", 0.0))
        ctrl_uint8 = _render_control_from_bs(bs_target_dict, row)

        gen = torch.Generator(device=device).manual_seed(
            int(hashlib.md5(sha.encode()).hexdigest()[:8], 16))
        z_T_unpacked = torch.randn(photo_latent.shape, generator=gen,
                                   device=device, dtype=dtype)
        z_T = pack_latents(z_T_unpacked)

        z = _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled,
                    id_txt_ids, t5_txt_ids, img_ids, control_packed, guidance)
        gen_img = _decode(vae, z)[0].permute(1, 2, 0).numpy()

        target = _load_photo_crop_uint8(sha, row, ffhq_idx, shards)
        rows.append(np.concatenate([target, ctrl_uint8, gen_img], axis=1))

    collage = np.concatenate(rows, axis=0)
    Image.fromarray(collage).save(samples_dir / f"step_{step:06d}.png")
```

- [ ] **Step 4: Rewrite `metrics`**

Same pattern — drop the `cs=0`/`cs=1` split. Keep the neutral-control baseline pass (`bs_cos_neutral_mean`) — it's the expression floor and still useful. Sample twice: once with real ctrl_latent, once with VAE-encoded zero-bs neutral render. Both use `conditioning_scale=1.0`. Same id_tokens path for both.

- [ ] **Step 5: Smoke test**

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    uv run python scripts/cfm_sanity_iy_recipe.py --n 4
```

Then run training for 1 step against a small filter (validate the loop runs end-to-end):

```bash
rm -rf exp_output/cfm_train/smoke
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    uv run python -m arkit_controlnet.cfm.train \
    --out-dir exp_output/cfm_train/smoke --max-steps 4 \
    --eval-every 4 --save-every 4 --dataset-filter-n 100
```

Verify: one collage at step_000004.png, three columns, identity-transferred targets.

- [ ] **Step 6: Commit**

```bash
git add src/arkit_controlnet/cfm/eval.py
git commit -m "refactor(cfm): single-pass eval — [target|control|generated] collage"
```

---

## Task 5: Pilot relaunch + Gate 2

**Files:** none (operational)

- [ ] **Step 1: Run the pilot at the original gating settings**

```bash
rm -rf exp_output/cfm_train/pilot
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src \
    uv run python -m arkit_controlnet.cfm.train \
    --out-dir exp_output/cfm_train/pilot --max-steps 500 \
    --eval-every 50 --save-every 100 --dataset-filter-n 5000 \
    2>&1 | tee /tmp/cfm_pilot2.log
```

- [ ] **Step 2: Step-50 visual gate**

Expect: col 3 (generated) shows identity transfer + some spatial bias toward the FLAME-normals control image. If col 3 still has no identity, stop — re-run sanity gate; do not grind to 500.

- [ ] **Step 3: Step-500 visual gate**

Expect: col 3 follows the FLAME pose visibly. Mouth open / brow position / head yaw should track the control image at least crudely. Cleanliness is not required.

- [ ] **Step 4: Decide on full 20k**

If both gates pass, launch the full run at the same settings minus the row filter. If step-500 looks marginal but learning, schedule a 2000-step probe before committing.

---

## Task 6 (deferred): X-mirror in FLAME render

**Files:**
- Modify: `src/arkit_controlnet/flame_render.py`

Investigation, not implementation, until Task 5 confirms identity is solid. Render one FLAME row, overlay against the matching photo crop, check whether eye/nostril/mouth-corner landmarks align only after horizontal flip. Most likely a sign flip in the rotation matrix or projection convention.

---

## Self-review

Spec coverage: every change listed in the diagnosis has a task. The twin-pass strip (model.py), the precompute fix (precompute.py), the train loop simplification (train.py), the eval collage simplification (eval.py), the pilot relaunch (operational), the mirror (deferred). No spec gaps.

Placeholders: none. Code blocks contain exact replacements.

Type consistency: `velocity()` signature is the same in model.py, train.py, eval.py.

Cleanup remaining: the bloated commit `756eac7` (out of scope; rebase later if needed). The bnb optimizer drop is intentional in Task 3.
