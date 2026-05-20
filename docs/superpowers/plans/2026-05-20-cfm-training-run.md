# CFM Expression-ControlNet Training Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train InfuseNet LoRA + control stems to follow a FLAME expression render via CFM, with fail-fast visual gates so the first conditioned sample collage lands within ~75 min of wall clock.

**Architecture:** A `cfm/` package on top of the gating-spike API: precompute photo latents + identity tokens (resumable, eval-first), render FLAME normals landmark-aligned on the fly, train a LoRA (r=8) + two control stems on bf16 InfuseNet against frozen FLUX.1-dev, dump sample collages every 50 steps for manual review at step 500, then the full 20k-step run.

**Tech Stack:** Python 3.12 + `uv` venv; diffusers FLUX.1-dev (bf16); diffusers `FluxControlNetModel` (bf16); peft (LoRA); bitsandbytes (AdamW8bit); insightface buffalo_l (ArcFace); ComfyUI's InfiniteYou Resampler (vendored). Geometry code runs under `/home/newub/miniconda3/bin/python` with `PYTHONPATH=src`; trainer runs under `uv run python` (same env as `src/demographic_pc/train_flux_image_slider.py`).

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-20-cfm-training-run-design.md`
- Spike verdict (concrete API): `docs/research/2026-05-19-trainable-infusenet-spike-verdict.md`
- Spike source (the live API reference): `src/arkit_controlnet/cfm/spike_trainable_infusenet.py`
- Slider trainer (pattern to mirror for CFM loop + caches): `src/demographic_pc/train_flux_image_slider.py`
- InfiniteYou Resampler config (load reference): `/home/newub/w/ComfyUI/custom_nodes/ComfyUI_InfiniteYou/nodes.py` lines 117-128

**File map:**

| File | Responsibility |
|---|---|
| `src/arkit_controlnet/flame_render.py` | extend with `render_landmark_aligned()` |
| `src/arkit_controlnet/cfm/text_embeds.py` | encode the fixed prompt once |
| `src/arkit_controlnet/cfm/resampler.py` | InfiniteYou Resampler (vendored) |
| `src/arkit_controlnet/cfm/precompute.py` | photo-latent + id-token cache, resumable |
| `src/arkit_controlnet/cfm/model.py` | `build_model()`, `velocity()` |
| `src/arkit_controlnet/cfm/dataset.py` | `CfmPairDataset` |
| `src/arkit_controlnet/cfm/train.py` | CFM loop, AdamW8bit, atomic `latest.pt` |
| `src/arkit_controlnet/cfm/eval.py` | `dump_samples`, `metrics` |
| `tests/arkit_controlnet/cfm/test_*.py` | per-module tests |

**Conventions:**
- `uv run python -m arkit_controlnet.cfm.<module>` for trainer / precompute (deps live in uv env).
- `PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest ...` for geometry tests.
- Seagate drive at `/media/newub/Seagate Hub/` must be mounted for precompute Tasks. Preflight check is mandatory.
- All caches are atomic writes (`.tmp` + `os.replace`).
- Conventional commits per task.

---

## Task 0: Download dev-bf16 FLUX + InfiniteYou resampler weights

**Files:**
- Create directory: `data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin` (download target)
- Use existing: `~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/` (already present; this task only verifies)

This task is operational, not code. It writes nothing to the repo.

- [ ] **Step 1: Verify FLUX.1-dev bf16 transformer is on disk**

```bash
ls ~/.cache/huggingface/hub/models--black-forest-labs--FLUX.1-dev/snapshots/*/transformer/
```

Expected: a `diffusion_pytorch_model-00001-of-*.safetensors` or `flux1-dev.safetensors` file. If empty, run:

```bash
huggingface-cli download black-forest-labs/FLUX.1-dev \
    --include "transformer/*" "vae/*" "text_encoder/*" "text_encoder_2/*" \
              "tokenizer/*" "tokenizer_2/*" "scheduler/*"
```

- [ ] **Step 2: Download InfiniteYou resampler weights**

```bash
huggingface-cli download ByteDance/InfiniteYou \
    --include "infu_flux_v1.0/sim_stage1/image_proj_model.bin" \
    --local-dir data/infiniteyou_dl_bf16/
ls -lh data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin
```

Expected: ~50 MB file.

- [ ] **Step 3: Verify insightface buffalo_l is available**

```bash
uv run python -c "import insightface; app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider']); app.prepare(ctx_id=0); print('ok')"
```

Expected: prints `ok`. If `ModelNotFound`, models auto-download to `~/.insightface/models/buffalo_l/` on first run.

- [ ] **Step 4: No commit — operational task, nothing changed in the repo.**

---

## Task 1: Promote `render_landmark_aligned` into flame_render.py

The collage script `verify_flame_render.py` already implements landmark-aligned rendering inline (`_render_aligned`). Promote it to a public helper in `flame_render.py` so the dataset can call it.

**Files:**
- Modify: `src/arkit_controlnet/flame_render.py`
- Modify: `src/arkit_controlnet/verify_flame_render.py` (delete the inline copy, import the public one)
- Test: `tests/arkit_controlnet/test_flame_render.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/arkit_controlnet/test_flame_render.py`:

```python
def test_render_landmark_aligned_signature():
    """render_landmark_aligned takes verts, rotation, landmarks_px, H, W."""
    from arkit_controlnet.flame_render import (
        deform, render_landmark_aligned)
    verts = deform(np.zeros(52))
    R = np.eye(3)
    H = W = 256
    # 478 dummy landmarks roughly centered in the image
    lm = np.full((478, 2), 128.0, dtype=np.float64)
    lm[1] = [128, 140]     # nose tip
    lm[33] = [110, 130]    # left eye outer
    lm[263] = [146, 130]   # right eye outer
    lm[61] = [118, 150]    # mouth left
    lm[291] = [138, 150]   # mouth right
    lm[152] = [128, 165]   # chin
    out = render_landmark_aligned(verts, R, lm, H=H, W=W)
    assert out.shape == (H, W, 3) and out.dtype == np.uint8
    assert (out.sum(axis=2) > 10).mean() > 0.02
```

- [ ] **Step 2: Run the test, verify it fails**

```bash
PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py::test_render_landmark_aligned_signature -v
```

Expected: FAIL — `ImportError: cannot import name 'render_landmark_aligned'`.

- [ ] **Step 3: Add `render_landmark_aligned` to flame_render.py**

In `src/arkit_controlnet/flame_render.py`, after the `render()` function, add:

```python
def render_landmark_aligned(verts: np.ndarray, rotation: np.ndarray,
                            mp_landmarks_px: np.ndarray,
                            H: int = 512, W: int = 512) -> np.ndarray:
    """Landmark-anchored sibling of `render()`.

    Uses `landmark_align.aligned_pixels` (2D similarity transform over 6
    MediaPipe-478 ↔ FLAME iBUG-70 pairs) for per-vertex 2D positions, then
    runs the same painter's-order normals rasterizer as `render()`. Prefer
    this over `render()` when MediaPipe landmarks for the photo are known.
    """
    from arkit_controlnet.landmark_align import aligned_pixels
    verts = np.asarray(verts, dtype=np.float64)
    R = np.asarray(rotation, dtype=np.float64)
    a = load_flame_assets()
    vr = verts @ R.T
    pts = aligned_pixels(verts, R, mp_landmarks_px, a.faces).astype(np.int32)
    normals = _face_normals(vr, a.faces)
    shade = ((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    order = np.argsort(vr[a.faces, 2].mean(axis=1))
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    for i in order:
        tri = a.faces[i]
        col = (int(shade[i, 2]), int(shade[i, 1]), int(shade[i, 0]))  # BGR
        cv2.fillConvexPoly(canvas, pts[tri], col, lineType=cv2.LINE_8)
    return canvas
```

- [ ] **Step 4: Replace the inline copy in verify_flame_render.py**

In `src/arkit_controlnet/verify_flame_render.py`, delete `_render_aligned` (the local helper). Update the import block to add `render_landmark_aligned`:

```python
from arkit_controlnet.flame_render import (
    deform, load_flame_assets, mediapipe_to_basis_vector,
    render_landmark_aligned,
)
```

Drop the now-unused `_face_normals` import. Replace the call site `ctrl = _render_aligned(verts, rot, mp_lm, H, W)` with `ctrl = render_landmark_aligned(verts, rot, mp_lm, H, W)`.

- [ ] **Step 5: Run the test, verify it passes**

```bash
PYTHONPATH=src /home/newub/miniconda3/bin/python -m pytest tests/arkit_controlnet/test_flame_render.py -v
```

Expected: all tests pass (the new one plus the existing 10).

- [ ] **Step 6: Re-run the verification collage as a smoke check**

```bash
PYTHONPATH=src /home/newub/miniconda3/bin/python -m arkit_controlnet.verify_flame_render
```

Expected: writes `exp_output/flame_render_check/collage.png`. Open it — should be identical to the prior aligned collage (this is a refactor, not a behavior change).

- [ ] **Step 7: Commit**

```bash
git add src/arkit_controlnet/flame_render.py src/arkit_controlnet/verify_flame_render.py tests/arkit_controlnet/test_flame_render.py
git commit -m "$(cat <<'EOF'
refactor(arkit-controlnet): promote render_landmark_aligned to flame_render

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Vendor the InfiniteYou Resampler

Vendor the Resampler from ComfyUI's InfiniteYou node so it loads in the `uv` venv without ComfyUI. License: MIT (header in source file).

**Files:**
- Create: `src/arkit_controlnet/cfm/resampler.py`
- Test: `tests/arkit_controlnet/cfm/test_resampler.py`
- Create directory marker: `tests/arkit_controlnet/cfm/__init__.py`

- [ ] **Step 1: Copy Resampler source verbatim**

```bash
mkdir -p tests/arkit_controlnet/cfm
touch tests/arkit_controlnet/cfm/__init__.py
cp /home/newub/w/ComfyUI/custom_nodes/ComfyUI_InfiniteYou/resampler.py src/arkit_controlnet/cfm/resampler.py
```

Leave the source verbatim (preserves license headers).

- [ ] **Step 2: Write the test**

Create `tests/arkit_controlnet/cfm/test_resampler.py`:

```python
import torch
from arkit_controlnet.cfm.resampler import Resampler


def test_resampler_io_shapes():
    """InfiniteYou config: 512-d ArcFace → (1, 8, 4096) identity tokens."""
    r = Resampler(dim=1280, depth=4, dim_head=64, heads=20,
                  num_queries=8, embedding_dim=512, output_dim=4096,
                  ff_mult=4)
    x = torch.randn(1, 1, 512)
    out = r(x)
    assert out.shape == (1, 8, 4096)


def test_resampler_loads_pretrained_weights():
    """The released InfiniteYou image_proj_model.bin loads cleanly."""
    import os
    p = "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/image_proj_model.bin"
    if not os.path.exists(p):
        import pytest
        pytest.skip("InfiniteYou weights not downloaded")
    sd = torch.load(p, map_location="cpu", weights_only=False)
    r = Resampler(dim=1280, depth=4, dim_head=64, heads=20,
                  num_queries=8, embedding_dim=512, output_dim=4096,
                  ff_mult=4)
    r.load_state_dict(sd["image_proj"])
```

- [ ] **Step 3: Run the test**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_resampler.py -v
```

Expected: both PASS (second one skips if Task 0 Step 2 wasn't run).

- [ ] **Step 4: Commit**

```bash
git add src/arkit_controlnet/cfm/resampler.py tests/arkit_controlnet/cfm/
git commit -m "$(cat <<'EOF'
chore(arkit-controlnet): vendor InfiniteYou Resampler (MIT)

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: text_embeds.py — encode the fixed prompt

The CFM training run uses one fixed prompt to leave the expression supervision entirely on the control channel. Encode once, save once.

**Files:**
- Create: `src/arkit_controlnet/cfm/text_embeds.py`
- Test: `tests/arkit_controlnet/cfm/test_text_embeds.py`

- [ ] **Step 1: Write the failing test**

```python
import os
import torch
from arkit_controlnet.cfm.text_embeds import build_text_embeds, FIXED_PROMPT

OUT = "output/cfm_precompute/text_embeds.pt"


def test_fixed_prompt_is_one_string():
    assert isinstance(FIXED_PROMPT, str) and len(FIXED_PROMPT) > 10


def test_build_text_embeds_writes_expected_shapes(tmp_path):
    target = tmp_path / "text_embeds.pt"
    build_text_embeds(out_path=str(target))
    d = torch.load(target, map_location="cpu")
    assert d["t5"].shape == (1, 512, 4096)
    assert d["pooled"].shape == (1, 768)
    assert d["t5"].dtype == torch.bfloat16
    assert d["pooled"].dtype == torch.bfloat16


def test_build_text_embeds_skips_if_exists(tmp_path):
    target = tmp_path / "text_embeds.pt"
    target.write_bytes(b"sentinel")
    build_text_embeds(out_path=str(target))
    assert target.read_bytes() == b"sentinel"
```

- [ ] **Step 2: Run, verify it fails**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_text_embeds.py -v
```

Expected: FAIL — ImportError.

- [ ] **Step 3: Implement text_embeds.py**

Create `src/arkit_controlnet/cfm/text_embeds.py`:

```python
"""One-shot encode the fixed CFM training prompt through FLUX's T5-XXL + CLIP-L.

Run: uv run python -m arkit_controlnet.cfm.text_embeds
Idempotent (skip-if-exists). Writes:
  output/cfm_precompute/text_embeds.pt = {"t5": (1,512,4096), "pooled": (1,768)}
"""
import os
from pathlib import Path

import torch

FIXED_PROMPT = "a portrait photo of a person looking at the camera"
FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_OUT = "output/cfm_precompute/text_embeds.pt"


def build_text_embeds(out_path: str = DEFAULT_OUT,
                      flux_hf_id: str = FLUX_HF_ID,
                      device: str = "cuda") -> None:
    out = Path(out_path)
    if out.exists():
        print(f"[text_embeds] {out} exists, skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import (AutoTokenizer, CLIPTextModel, CLIPTokenizer,
                              T5EncoderModel)
    dtype = torch.bfloat16

    tok_c = CLIPTokenizer.from_pretrained(flux_hf_id, subfolder="tokenizer")
    enc_c = CLIPTextModel.from_pretrained(
        flux_hf_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    ids_c = tok_c(FIXED_PROMPT, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        pooled = enc_c(ids_c.input_ids, output_hidden_states=False).pooler_output
    del enc_c
    torch.cuda.empty_cache()

    tok_t = AutoTokenizer.from_pretrained(flux_hf_id, subfolder="tokenizer_2")
    enc_t = T5EncoderModel.from_pretrained(
        flux_hf_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device).eval()
    ids_t = tok_t(FIXED_PROMPT, padding="max_length", max_length=512,
                  truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        t5 = enc_t(ids_t.input_ids)[0]
    del enc_t
    torch.cuda.empty_cache()

    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save({"t5": t5.to("cpu", dtype),
                "pooled": pooled.to("cpu", dtype)}, tmp)
    os.replace(tmp, out)
    print(f"[text_embeds] wrote {out}: t5{tuple(t5.shape)} pooled{tuple(pooled.shape)}")


if __name__ == "__main__":
    build_text_embeds()
```

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_text_embeds.py -v
```

Expected: 3 PASS. The shape-test downloads T5+CLIP if not cached (~9 GB but already on disk from prior FLUX work; first run may take several minutes).

- [ ] **Step 5: Build the real cache for the project**

```bash
uv run python -m arkit_controlnet.cfm.text_embeds
ls -lh output/cfm_precompute/text_embeds.pt
```

Expected: `output/cfm_precompute/text_embeds.pt` exists, ~5 MB (one T5 sequence in bf16).

- [ ] **Step 6: Commit**

```bash
git add src/arkit_controlnet/cfm/text_embeds.py tests/arkit_controlnet/cfm/test_text_embeds.py
git commit -m "$(cat <<'EOF'
feat(arkit-controlnet): one-shot text-embed cache for CFM training

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: cfm/model.py — build_model + velocity

Lift the spike's load + forward block into a reusable module. The contract: `build_model()` returns `(flux, infusenet, trainable_params)`; `velocity(...)` runs InfuseNet → residuals → frozen FLUX → predicted velocity.

**Files:**
- Create: `src/arkit_controlnet/cfm/model.py`
- Test: `tests/arkit_controlnet/cfm/test_model.py`

- [ ] **Step 1: Write the test**

```python
import math
import torch
import pytest
from arkit_controlnet.cfm.model import build_model, velocity, pack_latents


@pytest.fixture(scope="module")
def model():
    return build_model()


def test_build_model_has_only_lora_and_stems_trainable(model):
    """LoRA + x_embedder + controlnet_x_embedder are trainable, all else frozen."""
    trainable = [(n, p) for n, p in model.infusenet.named_parameters()
                 if p.requires_grad]
    assert len(trainable) > 0
    for n, _ in trainable:
        assert ("lora_" in n) or ("x_embedder" in n), f"unexpected trainable: {n}"
    # FLUX is fully frozen
    for n, p in model.flux.named_parameters():
        assert not p.requires_grad, f"FLUX param {n} unexpectedly trainable"


def test_velocity_output_shape_matches_input(model):
    B, H, W = 1, 64, 64
    DTYPE = torch.bfloat16
    z0 = torch.randn(B, 16, H, W, device="cuda", dtype=DTYPE)
    z_t = pack_latents(z0)
    sigma = torch.full((B,), 0.5, device="cuda", dtype=DTYPE)
    ctrl = pack_latents(torch.randn_like(z0))
    eh = torch.randn(B, 8 + 512, 4096, device="cuda", dtype=DTYPE)
    pooled = torch.randn(B, 768, device="cuda", dtype=DTYPE)
    from arkit_controlnet.cfm.model import (
        prepare_latent_image_ids, prepare_text_ids)
    img_ids = prepare_latent_image_ids(H, W, "cuda", DTYPE)
    txt_ids = prepare_text_ids(8 + 512, "cuda", DTYPE)
    guidance = torch.full((B,), 3.5, device="cuda", dtype=DTYPE)
    with torch.no_grad():
        v = velocity(model, z_t, sigma, eh, pooled, txt_ids, img_ids,
                     ctrl, guidance)
    assert v.shape == z_t.shape
    assert math.isfinite(v.float().abs().mean().item())
```

- [ ] **Step 2: Run, verify it fails**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_model.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement cfm/model.py**

This module factors the spike script into a library. Most code is copied verbatim from `src/arkit_controlnet/cfm/spike_trainable_infusenet.py` (the LoRA target list, the pack/unpack helpers, the forward block). The difference: the FLUX path moves from Krea-bf16 to FLUX.1-dev bf16 from HuggingFace cache.

Create `src/arkit_controlnet/cfm/model.py`:

```python
"""Build the trainable CFM model (frozen FLUX.1-dev + LoRA-trained InfuseNet).

The concrete API was discovered by the gating spike at
`spike_trainable_infusenet.py`; this module is the library version. Differences
from the spike:
  - Base FLUX is dev-bf16 from HF cache, not Krea (InfuseNet was trained
    against dev — residual alignment matters for the long run).
  - Exposes `velocity()` as the trainer's single-shot forward function.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import Parameter

DEVICE = "cuda"
DTYPE = torch.bfloat16

FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"
INFUSE_DIR_DEFAULT = "data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/InfuseNetModel"


@dataclass
class CfmModel:
    flux: object                    # FluxTransformer2DModel — frozen
    infusenet: object               # peft-wrapped FluxControlNetModel
    trainable_params: list[Parameter]


def pack_latents(z: torch.Tensor) -> torch.Tensor:
    Bn, C, Hn, Wn = z.shape
    z = z.view(Bn, C, Hn // 2, 2, Wn // 2, 2)
    z = z.permute(0, 2, 4, 1, 3, 5)
    return z.reshape(Bn, (Hn // 2) * (Wn // 2), C * 4)


def unpack_latents(z_packed: torch.Tensor, Hn: int, Wn: int) -> torch.Tensor:
    Bn, _, Cp = z_packed.shape
    C = Cp // 4
    z = z_packed.view(Bn, Hn // 2, Wn // 2, C, 2, 2)
    z = z.permute(0, 3, 1, 4, 2, 5)
    return z.reshape(Bn, C, Hn, Wn)


def prepare_latent_image_ids(Hn: int, Wn: int, device, dtype):
    ids = torch.zeros(Hn // 2, Wn // 2, 3)
    ids[..., 1] = ids[..., 1] + torch.arange(Hn // 2)[:, None]
    ids[..., 2] = ids[..., 2] + torch.arange(Wn // 2)[None, :]
    return ids.reshape(-1, 3).to(device, dtype)


def prepare_text_ids(seq_len: int, device, dtype):
    return torch.zeros(seq_len, 3).to(device, dtype)


# Same module list as the spike. 4 double + 10 single blocks in InfuseNet.
_DOUBLE_TARGETS = [
    "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
    "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
]
_SINGLE_TARGETS = ["attn.to_q", "attn.to_k", "attn.to_v", "proj_out"]


def build_model(
    flux_hf_id: str = FLUX_HF_ID,
    infusenet_dir: str = INFUSE_DIR_DEFAULT,
    lora_r: int = 8,
    lora_alpha: int = 8,
    device: str = DEVICE,
    dtype: torch.dtype = DTYPE,
) -> CfmModel:
    from diffusers import FluxTransformer2DModel, FluxControlNetModel
    from peft import LoraConfig, get_peft_model

    flux = FluxTransformer2DModel.from_pretrained(
        flux_hf_id, subfolder="transformer", torch_dtype=dtype
    ).to(device)
    flux.requires_grad_(False)
    flux.enable_gradient_checkpointing()
    flux.eval()

    infusenet = FluxControlNetModel.from_pretrained(
        infusenet_dir, torch_dtype=dtype
    ).to(device)
    infusenet.requires_grad_(False)
    if hasattr(infusenet, "enable_gradient_checkpointing"):
        infusenet.enable_gradient_checkpointing()

    n_double = len(infusenet.transformer_blocks)
    n_single = len(infusenet.single_transformer_blocks)
    targets = []
    for i in range(n_double):
        for suf in _DOUBLE_TARGETS:
            targets.append(f"transformer_blocks.{i}.{suf}")
    for i in range(n_single):
        for suf in _SINGLE_TARGETS:
            targets.append(f"single_transformer_blocks.{i}.{suf}")

    lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                          target_modules=targets,
                          lora_dropout=0.0, bias="none")
    infusenet = get_peft_model(infusenet, lora_cfg, adapter_name="cfm")

    # Control-image input stems are fully trainable (the FLAME-render modality
    # is novel; LoRA on the stem is too narrow to absorb the modality switch).
    for name in ("x_embedder", "controlnet_x_embedder"):
        mod = getattr(infusenet, name, None)
        if mod is None and hasattr(infusenet, "base_model"):
            mod = getattr(infusenet.base_model.model, name, None)
        if mod is not None:
            for p in mod.parameters():
                p.requires_grad = True

    trainable_params = [p for p in infusenet.parameters() if p.requires_grad]
    return CfmModel(flux=flux, infusenet=infusenet,
                    trainable_params=trainable_params)


def velocity(
    model: CfmModel,
    z_t_packed: torch.Tensor,
    sigma: torch.Tensor,
    encoder_hidden_states: torch.Tensor,
    pooled: torch.Tensor,
    txt_ids: torch.Tensor,
    img_ids: torch.Tensor,
    control_packed: torch.Tensor,
    guidance: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One CFM forward: InfuseNet residuals → frozen FLUX → predicted velocity."""
    cn_double, cn_single = model.infusenet(
        hidden_states=z_t_packed,
        controlnet_cond=control_packed,
        conditioning_scale=1.0,
        encoder_hidden_states=encoder_hidden_states,
        pooled_projections=pooled,
        timestep=sigma,
        img_ids=img_ids,
        txt_ids=txt_ids,
        guidance=guidance,
        return_dict=False,
    )
    v_packed = model.flux(
        hidden_states=z_t_packed,
        timestep=sigma,
        guidance=guidance,
        pooled_projections=pooled,
        encoder_hidden_states=encoder_hidden_states,
        txt_ids=txt_ids,
        img_ids=img_ids,
        controlnet_block_samples=cn_double,
        controlnet_single_block_samples=cn_single,
        return_dict=False,
    )[0]
    return v_packed
```

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_model.py -v -s
```

Expected: both PASS. First load downloads dev-bf16 transformer if not cached (~24 GB; do this on a fast connection).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/cfm/model.py tests/arkit_controlnet/cfm/test_model.py
git commit -m "$(cat <<'EOF'
feat(arkit-controlnet): cfm.model — build_model + velocity from spike API

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: cfm/precompute.py — photo latents + identity tokens

Resumable photo-latent + identity-token cache, keyed by `image_sha256`. Two stages internally (VAE pass + ArcFace+Resampler pass), one CLI. Accepts `--shas-file` so the Phase 2 (eval-only) and Phase 5 (everything else) calls share an entry point.

**Files:**
- Create: `src/arkit_controlnet/cfm/precompute.py`
- Test: `tests/arkit_controlnet/cfm/test_precompute.py`

- [ ] **Step 1: Write the test**

```python
import os
import torch
import pandas as pd
from pathlib import Path


def test_face_crop_resize_returns_512_rgb():
    """`face_crop_resize` returns (512, 512, 3) uint8 from a 1024² photo."""
    import numpy as np
    from arkit_controlnet.cfm.precompute import face_crop_resize
    photo = (np.random.rand(1024, 1024, 3) * 255).astype(np.uint8)
    out = face_crop_resize(photo, bbox_cx=0.5, bbox_cy=0.5,
                           bbox_w=0.3, bbox_h=0.4, out_size=512)
    assert out.shape == (512, 512, 3) and out.dtype == np.uint8


def test_precompute_one_row_round_trip(tmp_path):
    """Precompute one known sha; verify outputs exist with correct shapes."""
    import pytest
    if not Path("output/flame_pose_cache/pose_cache.parquet").exists():
        pytest.skip("pose cache absent")
    if not Path("data/infiniteyou_dl_bf16/infu_flux_v1.0/sim_stage1/"
                "image_proj_model.bin").exists():
        pytest.skip("InfiniteYou resampler weights absent")
    if not Path("/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data").is_dir():
        pytest.skip("Seagate drive not mounted")
    from arkit_controlnet.cfm.precompute import run

    pc = pd.read_parquet("output/flame_pose_cache/pose_cache.parquet",
                         columns=["image_sha256", "pose_detected"])
    sha = pc[pc.pose_detected].iloc[0].image_sha256

    out_dir = tmp_path / "cfm_precompute"
    run(shas=[sha], out_dir=str(out_dir))

    pl = out_dir / "photo_latents" / f"{sha}.pt"
    it = out_dir / "id_tokens" / f"{sha}.pt"
    assert pl.exists() and it.exists()
    assert torch.load(pl, map_location="cpu").shape == (16, 64, 64)
    assert torch.load(it, map_location="cpu").shape == (8, 4096)
    meta = pd.read_parquet(out_dir / "meta.parquet")
    assert sha in meta.image_sha256.values
```

- [ ] **Step 2: Run, verify the import test fails**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_precompute.py::test_face_crop_resize_returns_512_rgb -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement precompute.py**

Create `src/arkit_controlnet/cfm/precompute.py` with two stages (VAE + ArcFace/Resampler) in one loop, atomic `meta.parquet` flushes every 256 rows, and a `--shas-file` CLI argument. Mandatory preflight: raise `FileNotFoundError` if `SHARD_GLOB` returns no files.

Key implementation details:
- `face_crop_resize(rgb, bbox_cx, bbox_cy, bbox_w, bbox_h, out_size=512, margin=0.25)` → center crop on face bbox with edge-replicated padding, then `cv2.resize(..., INTER_AREA)`.
- VAE: `AutoencoderKL.from_pretrained(FLUX_HF_ID, subfolder="vae", torch_dtype=bf16)`. Encode normalized `(rgb/127.5 - 1)` → `(16, 64, 64)`; apply `(lat - shift_factor) * scaling_factor`. Save as bf16.
- ArcFace: `insightface.app.FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])`, `.prepare(ctx_id=0)`, `.get(bgr)`. If no face found → `id_ok=False`, skip the resampler write.
- Resampler: `Resampler(dim=1280, depth=4, dim_head=64, heads=20, num_queries=8, embedding_dim=512, output_dim=4096, ff_mult=4)`; load `image_proj_model.bin`'s `state_dict["image_proj"]`. Input `(1, 1, 512)` bf16 → output `(1, 8, 4096)`. Save `[0]` slice.
- Skip-if-exists: `pl_path.exists() and it_path.exists()` short-circuits before the VAE/ArcFace pass for the row.
- Atomic write helper: `tmp = path + ".tmp"; df.to_parquet(tmp); os.replace(tmp, path)`.

CLI:
```python
parser.add_argument("--shas-file", default=None)  # newline-separated
parser.add_argument("--out-dir", default="output/cfm_precompute")
```

When `--shas-file` is None, processes the full intersection (~69,918 rows). When set, filters `df` to those shas only.

- [ ] **Step 4: Run the unit test**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_precompute.py::test_face_crop_resize_returns_512_rgb -v
```

Expected: PASS.

- [ ] **Step 5: Run the integration test (needs Seagate mounted)**

```bash
ls "/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data/" | head -1
uv run python -m pytest tests/arkit_controlnet/cfm/test_precompute.py -v
```

Expected: 2 PASS or 1 PASS + 1 SKIP (if Seagate not mounted).

- [ ] **Step 6: Commit**

```bash
git add src/arkit_controlnet/cfm/precompute.py tests/arkit_controlnet/cfm/test_precompute.py
git commit -m "$(cat <<'EOF'
feat(arkit-controlnet): cfm.precompute — photo-latent + id-token cache

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: cfm/dataset.py — `CfmPairDataset`

`CfmPairDataset` reads the joined `(reverse_index, pose_cache, ffhq_index)` table, loads per-row cached photo latent + id tokens, renders the FLAME control image on the fly, and returns the per-step input the trainer needs.

**Files:**
- Create: `src/arkit_controlnet/cfm/dataset.py`
- Test: `tests/arkit_controlnet/cfm/test_dataset.py`

- [ ] **Step 1: Write the test**

```python
import pytest
import torch
from pathlib import Path


@pytest.fixture(scope="module")
def cached_sha():
    if not Path("output/cfm_precompute/meta.parquet").exists():
        pytest.skip("precompute cache not built")
    import pandas as pd
    meta = pd.read_parquet("output/cfm_precompute/meta.parquet")
    ok = meta[(meta.vae_ok) & (meta.id_ok)]
    if ok.empty:
        pytest.skip("no fully-precomputed rows")
    return ok.iloc[0].image_sha256


def test_split_assigns_each_sha_to_exactly_one():
    from arkit_controlnet.cfm.dataset import CfmPairDataset
    eval_ds = CfmPairDataset(split="eval", eval_size=1024)
    train_ds = CfmPairDataset(split="train", eval_size=1024)
    eval_shas = set(eval_ds.df.image_sha256)
    train_shas = set(train_ds.df.image_sha256)
    assert not (eval_shas & train_shas)
    assert len(eval_shas) > 0 and len(train_shas) > 0


def test_split_is_deterministic():
    from arkit_controlnet.cfm.dataset import CfmPairDataset
    a = CfmPairDataset(split="eval", eval_size=1024)
    b = CfmPairDataset(split="eval", eval_size=1024)
    assert list(a.df.image_sha256) == list(b.df.image_sha256)


def test_getitem_shapes(cached_sha):
    from arkit_controlnet.cfm.dataset import CfmPairDataset
    ds = CfmPairDataset(split="train", eval_size=1024)
    ds.df = ds.df[ds.df.image_sha256 == cached_sha].reset_index(drop=True)
    if len(ds.df) == 0:
        ds = CfmPairDataset(split="eval", eval_size=1024)
        ds.df = ds.df[ds.df.image_sha256 == cached_sha].reset_index(drop=True)
    item = ds[0]
    assert item["photo_latent"].shape == (16, 64, 64)
    assert item["id_tokens"].shape == (8, 4096)
    assert item["control_rgb"].shape == (3, 512, 512)
    assert item["control_rgb"].dtype == torch.float32
    assert item["sha"] == cached_sha


def test_bs_column_order_matches_basis():
    """The bs_* column order must align with BASIS_CHANNEL_NAMES (minus tongueOut)."""
    from arkit_controlnet.cfm.dataset import BS_COLUMNS
    from arkit_controlnet.flame_render import BASIS_CHANNEL_NAMES
    expected = [f"bs_{n}" for n in BASIS_CHANNEL_NAMES if n != "tongueOut"]
    assert BS_COLUMNS == expected
```

- [ ] **Step 2: Run, verify it fails**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_dataset.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement dataset.py**

Create `src/arkit_controlnet/cfm/dataset.py`:

```python
"""CFM training dataset — joins reverse_index ∩ pose_cache ∩ ffhq_index.

Per-item output: cached photo latent + identity tokens (from precompute) +
the live-rendered FLAME normals control image. The control image is rendered
in-process — the modality is fixed and rendering is cheap (~5 ms).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from arkit_controlnet.flame_render import (
    BASIS_CHANNEL_NAMES, deform, mediapipe_to_basis_vector,
    render_landmark_aligned, render,
)
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES

POSE_CACHE = "output/flame_pose_cache/pose_cache.parquet"
RI = "output/reverse_index/reverse_index.parquet"
FFHQ_INDEX = "output/ffhq_index/ffhq_sha_index.parquet"
PRECOMPUTE_DIR = "output/cfm_precompute"
CTRL_SIZE = 512

BS_COLUMNS = [f"bs_{n}" for n in BASIS_CHANNEL_NAMES if n != "tongueOut"]


def _hash_bucket(sha: str) -> int:
    return int(hashlib.md5(sha.encode()).hexdigest()[:8], 16) % 1000


class CfmPairDataset(Dataset):
    def __init__(self, split: str = "train", eval_size: int = 1024,
                 precompute_dir: str = PRECOMPUTE_DIR):
        if split not in ("train", "eval"):
            raise ValueError(f"split must be 'train' or 'eval', got {split!r}")
        self.precompute_dir = Path(precompute_dir)
        ri = pd.read_parquet(RI)
        ri = ri[(ri.source == "ffhq") & ri.bs_detected]
        pc = pd.read_parquet(POSE_CACHE)
        pc = pc[pc.pose_detected]
        idx = pd.read_parquet(FFHQ_INDEX)
        df = (ri.merge(pc, on="image_sha256", how="inner")
                .merge(idx, on="image_sha256", how="inner")
                .sort_values("image_sha256")
                .reset_index(drop=True))
        buckets = df.image_sha256.apply(_hash_bucket).to_numpy()
        cutoff = max(1, round(eval_size * 1000 / len(df)))
        is_eval = buckets < cutoff
        self.df = (df[is_eval] if split == "eval" else df[~is_eval]
                   ).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        sha = row.image_sha256
        photo_latent = torch.load(
            self.precompute_dir / "photo_latents" / f"{sha}.pt",
            map_location="cpu")
        id_tokens = torch.load(
            self.precompute_dir / "id_tokens" / f"{sha}.pt",
            map_location="cpu")

        bs = {n: 0.0 for n in ARKIT_BLENDSHAPE_NAMES}
        for n in (n for n in ARKIT_BLENDSHAPE_NAMES if n != "tongueOut"):
            bs[n] = float(row.get(f"bs_{n}", 0.0))
        arkit52 = mediapipe_to_basis_vector(bs)
        verts = deform(arkit52)

        rot = np.array(row.rotation, dtype=np.float64).reshape(3, 3)
        try:
            lm_norm = np.array(row.landmarks_xy, dtype=np.float64).reshape(478, 2)
            lm_px = lm_norm * np.array([CTRL_SIZE, CTRL_SIZE])
            ctrl_rgb = render_landmark_aligned(
                verts, rot, lm_px, H=CTRL_SIZE, W=CTRL_SIZE)
        except Exception:
            ctrl_rgb = render(
                verts, rot,
                (row.bbox_cx, row.bbox_cy, row.bbox_w, row.bbox_h),
                H=CTRL_SIZE, W=CTRL_SIZE)

        ctrl = (torch.from_numpy(ctrl_rgb).float().permute(2, 0, 1)
                / 127.5 - 1.0)
        return {
            "photo_latent": photo_latent,
            "id_tokens": id_tokens,
            "control_rgb": ctrl,
            "sha": sha,
        }
```

- [ ] **Step 4: Run the tests**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_dataset.py -v
```

Expected: 4 PASS (shape test skips cleanly until Task 8 fills the cache).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/cfm/dataset.py tests/arkit_controlnet/cfm/test_dataset.py
git commit -m "$(cat <<'EOF'
feat(arkit-controlnet): cfm.dataset — CfmPairDataset

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: cfm/train.py — the CFM loop

CFM training loop. Resumable from `latest.pt`. Periodic eval calls at fixed cadence. AdamW8bit + constant LR with warmup.

**Files:**
- Create: `src/arkit_controlnet/cfm/train.py`
- Test: `tests/arkit_controlnet/cfm/test_train.py`

- [ ] **Step 1: Write the smoke test**

```python
import json
import os
import torch
import pytest
from pathlib import Path


@pytest.mark.skipif(
    not Path("output/cfm_precompute/text_embeds.pt").exists(),
    reason="text_embeds not built")
def test_two_steps_drive_loss_finite_and_log_appears(tmp_path):
    """Two CFM steps on a 1-row train fixture: loss finite, log written, ckpt saved."""
    from arkit_controlnet.cfm.train import train
    out_dir = tmp_path / "cfm_train" / "smoke"
    train(out_dir=str(out_dir), max_steps=2, save_every=2,
          eval_every=10, dataset_filter_n=1)
    log = (out_dir / "step_log.csv").read_text().strip().splitlines()
    assert len(log) >= 3   # header + 2 steps
    ckpt = out_dir / "latest.pt"
    assert ckpt.exists()
    state = torch.load(ckpt, map_location="cpu")
    assert state["step"] == 2
```

(The smoke test deliberately consumes 1 cached row so it can run after Task 8 Phase 2 builds the eval cache.)

- [ ] **Step 2: Run, verify it fails**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_train.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement train.py**

Create `src/arkit_controlnet/cfm/train.py`. Skeleton (full source in repo):

```python
"""CFM training loop for InfuseNet expression control."""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from arkit_controlnet.cfm.dataset import CfmPairDataset
from arkit_controlnet.cfm.model import (
    build_model, velocity, pack_latents, prepare_latent_image_ids,
    prepare_text_ids,
)

CTRL_SIZE = 512
LATENT_H = LATENT_W = CTRL_SIZE // 8   # 64
SHIFT = 3.0


def _atomic_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_text_embeds(path: str, device: str, dtype: torch.dtype):
    d = torch.load(path, map_location="cpu")
    return d["t5"].to(device, dtype), d["pooled"].to(device, dtype)


def _flux_sigma(rand_normal: torch.Tensor) -> torch.Tensor:
    t = torch.sigmoid(rand_normal)
    return (SHIFT * t) / (1 + (SHIFT - 1) * t)


def _build_vae(flux_hf_id: str, device: str, dtype: torch.dtype):
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        flux_hf_id, subfolder="vae", torch_dtype=dtype
    ).to(device).eval()
    vae.requires_grad_(False)
    return vae


def _vae_encode(vae, x: torch.Tensor, dtype):
    """`x` is float in [-1, 1] (B, 3, H, W). Returns (B, 16, H/8, W/8)."""
    with torch.no_grad():
        lat = vae.encode(x.to(vae.device, dtype)).latent_dist.sample()
        sf = vae.config.scaling_factor
        sh = getattr(vae.config, "shift_factor", 0.0)
        return (lat - sh) * sf


def train(
    out_dir: str,
    max_steps: int = 20000,
    save_every: int = 500,
    eval_every: int = 50,
    eval_dense_until: int = 1000,
    eval_sparse_every: int = 200,
    lr: float = 1e-4,
    warmup: int = 200,
    grad_accum: int = 4,
    seed: int = 0,
    dataset_filter_n: Optional[int] = None,
    text_embeds_path: str = "output/cfm_precompute/text_embeds.pt",
    flux_hf_id: str = "black-forest-labs/FLUX.1-dev",
):
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "step_log.csv"
    log_new = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log_w = csv.writer(log_f)
    if log_new:
        log_w.writerow(["step", "loss", "grad_norm", "lr",
                        "sigma_mean", "wall_s"])

    device, dtype = "cuda", torch.bfloat16
    model = build_model()
    vae = _build_vae(flux_hf_id, device, dtype)
    t5_seq, pooled_one = _load_text_embeds(text_embeds_path, device, dtype)

    ds = CfmPairDataset(split="train")
    if dataset_filter_n is not None:
        ds.df = ds.df.head(dataset_filter_n).reset_index(drop=True)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True)

    import bitsandbytes as bnb
    opt = bnb.optim.AdamW8bit(model.trainable_params, lr=lr,
                              weight_decay=0.01)

    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    txt_ids = prepare_text_ids(8 + 512, device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)

    ckpt = out / "latest.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        model.infusenet.load_state_dict(state["infusenet"], strict=False)
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"[resume] step={step}")

    t0 = time.time()
    opt.zero_grad()
    grad_count = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            photo_latent = batch["photo_latent"].to(device, dtype)
            id_tokens = batch["id_tokens"].to(device, dtype)
            ctrl = batch["control_rgb"].to(device, dtype)
            ctrl_latent = _vae_encode(vae, ctrl, dtype)

            z0 = photo_latent
            eps = torch.randn_like(z0)
            sigma = _flux_sigma(torch.randn(1, device=device))
            s = sigma.view(-1, 1, 1, 1).to(dtype)
            z_t = (1 - s) * z0 + s * eps
            target_packed = pack_latents(eps - z0)
            z_t_packed = pack_latents(z_t)
            ctrl_packed = pack_latents(ctrl_latent)

            eh = torch.cat([id_tokens, t5_seq.expand(1, -1, -1)], dim=1)

            warm_lr = lr * min(1.0, (step + 1) / max(1, warmup))
            for g in opt.param_groups:
                g["lr"] = warm_lr

            v_pred = velocity(model, z_t_packed, sigma, eh, pooled_one,
                              txt_ids, img_ids, ctrl_packed, guidance)
            loss = torch.nn.functional.mse_loss(
                v_pred.float(), target_packed.float()) / grad_accum
            loss.backward()
            grad_count += 1

            if grad_count >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_params, 1.0)
                opt.step()
                opt.zero_grad()
                grad_count = 0
                step += 1
                log_w.writerow([step, f"{loss.item()*grad_accum:.5f}",
                                f"{float(grad_norm):.4e}", f"{warm_lr:.2e}",
                                f"{float(sigma):.3f}", f"{time.time()-t0:.1f}"])
                log_f.flush()
                if step % save_every == 0:
                    _atomic_save({"step": step,
                                  "infusenet": model.infusenet.state_dict(),
                                  "opt": opt.state_dict()}, ckpt)

                eval_cadence = (eval_every if step <= eval_dense_until
                                else eval_sparse_every)
                if step % eval_cadence == 0:
                    from arkit_controlnet.cfm.eval import dump_samples
                    dump_samples(model, step, out_dir=out_dir, vae=vae,
                                 text_embeds=(t5_seq, pooled_one))

    log_f.close()
    _atomic_save({"step": step, "infusenet": model.infusenet.state_dict(),
                  "opt": opt.state_dict()}, ckpt)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="exp_output/cfm_train/run01")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--dataset-filter-n", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _cli()
    train(out_dir=a.out_dir, max_steps=a.max_steps,
          save_every=a.save_every, eval_every=a.eval_every,
          dataset_filter_n=a.dataset_filter_n)
```

- [ ] **Step 4: Run the smoke test**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_train.py -v -s
```

Expected: PASS (skipped until text_embeds exists; once it does + Task 8 fills 1 row, it runs 2 steps in ~30 s).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/cfm/train.py tests/arkit_controlnet/cfm/test_train.py
git commit -m "$(cat <<'EOF'
feat(arkit-controlnet): cfm.train — CFM loop, AdamW8bit, resumable

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: cfm/eval.py — sample dump + held-out metrics

Two responsibilities. `dump_samples` does N-sample inference + collage to `samples/step_NNNNNN.png` for the manual gate. `metrics` runs the full 1024-row eval set every 500 steps and logs ArcFace id-cos + MediaPipe bs-cos.

**Files:**
- Create: `src/arkit_controlnet/cfm/eval.py`
- Test: `tests/arkit_controlnet/cfm/test_eval.py`

- [ ] **Step 1: Write the test**

```python
import pytest
import torch
from pathlib import Path


@pytest.mark.skipif(
    not Path("output/cfm_precompute/text_embeds.pt").exists()
    or not Path("output/cfm_precompute/meta.parquet").exists(),
    reason="precompute not built")
def test_dump_samples_writes_collage(tmp_path):
    from arkit_controlnet.cfm.model import build_model
    from arkit_controlnet.cfm.eval import dump_samples
    model = build_model()
    out = tmp_path / "cfm_train" / "test_run"
    dump_samples(model, step=0, out_dir=str(out), n=2,
                 sample_steps=4)  # tiny for the smoke
    collage = out / "samples" / "step_000000.png"
    assert collage.exists()
```

- [ ] **Step 2: Run, verify it fails**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_eval.py -v
```

Expected: FAIL — `ImportError`.

- [ ] **Step 3: Implement eval.py**

Create `src/arkit_controlnet/cfm/eval.py` with these public functions:

**`dump_samples(model, step, out_dir, n=8, sample_steps=25, vae=None, text_embeds=None)`**:

- Use a fixed eval-row picklist persisted to `<out_dir>/eval_picks.json` on first call (8 sha hashes, lowest by sha string sort so reruns are identical).
- For each pick: load photo, photo_latent (target), id_tokens, render control image, encode control via VAE.
- Run two samplings: (a) conditioned (`conditioning_scale=1.0`), (b) id-only baseline (`conditioning_scale=0.0`).
- Sampling: `FlowMatchEulerDiscreteScheduler` from diffusers, `sample_steps` steps, sigma shift 3.0.
- VAE-decode generated latents → uint8 RGB.
- Build a 4-column collage: `[target_photo | control_rgb | id_only_gen | conditioned_gen]`, stack rows.
- Save to `<out_dir>/samples/step_{step:06d}.png`.

**`metrics(model, step, out_dir, vae=None, text_embeds=None) -> dict`**:

- Iterate the full eval split (`CfmPairDataset(split="eval")`).
- For each row: sample once conditioned, once with neutral control (zero blendshapes).
- ArcFace cos(conditioned, target): `id_cos`.
- MediaPipe-bs on `conditioned`: cos against intended bs vector: `bs_cos`.
- Same for neutral: `bs_cos_neutral`.
- Append `(step, id_cos, bs_cos, bs_cos_neutral)` to `<out_dir>/metrics.csv`.

Sampling helper:

```python
def _sample(model, z_T, sigma_seq, id_tokens, t5_seq, pooled, ctrl_packed,
            img_ids, txt_ids, guidance, cs):
    """Euler integration of the FLUX flow with InfuseNet residuals @ scale `cs`."""
    z = z_T
    for i in range(len(sigma_seq) - 1):
        s = sigma_seq[i]
        s_next = sigma_seq[i + 1]
        eh = torch.cat([id_tokens, t5_seq], dim=1)
        cn_d, cn_s = model.infusenet(
            hidden_states=z, controlnet_cond=ctrl_packed,
            conditioning_scale=cs, encoder_hidden_states=eh,
            pooled_projections=pooled, timestep=s.expand(z.shape[0]),
            img_ids=img_ids, txt_ids=txt_ids, guidance=guidance,
            return_dict=False)
        v = model.flux(
            hidden_states=z, timestep=s.expand(z.shape[0]),
            guidance=guidance, pooled_projections=pooled,
            encoder_hidden_states=eh, txt_ids=txt_ids, img_ids=img_ids,
            controlnet_block_samples=cn_d,
            controlnet_single_block_samples=cn_s,
            return_dict=False)[0]
        z = z + (s_next - s) * v
    return z
```

Build `sigma_seq` from a `FlowMatchEulerDiscreteScheduler` with shift=3.0, going from σ=1 down to σ=0.

- [ ] **Step 4: Run the test**

```bash
uv run python -m pytest tests/arkit_controlnet/cfm/test_eval.py -v -s
```

Expected: PASS (or SKIP until Task 8 fills the cache).

- [ ] **Step 5: Commit**

```bash
git add src/arkit_controlnet/cfm/eval.py tests/arkit_controlnet/cfm/test_eval.py
git commit -m "$(cat <<'EOF'
feat(arkit-controlnet): cfm.eval — sample dump + held-out metrics

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Phase 2 — precompute the eval set

The dataset hash split assigns ~1024 shas to eval. Materialize those first so the pilot training in Task 10 has held-out anchors for the sample collage.

- [ ] **Step 1: Mount the Seagate drive**

```bash
ls "/media/newub/Seagate Hub/arc_distill/ffhq_parquet/data/" | wc -l
```

Expected: `190`. If 0, mount the drive (it's external; mount path is fixed).

- [ ] **Step 2: Compute the eval shas list**

```bash
uv run python -c "
from arkit_controlnet.cfm.dataset import CfmPairDataset
ds = CfmPairDataset(split='eval', eval_size=1024)
with open('output/cfm_precompute/eval_shas.txt', 'w') as f:
    for sha in ds.df.image_sha256:
        f.write(sha + '\n')
print(len(ds.df), 'eval shas written')
" 2>&1 | tail -5
```

Expected: `~979 eval shas written`.

- [ ] **Step 3: Run precompute on the eval set**

```bash
mkdir -p output/cfm_precompute
uv run python -m arkit_controlnet.cfm.precompute \
    --shas-file output/cfm_precompute/eval_shas.txt \
    --out-dir output/cfm_precompute \
    2>&1 | tee /tmp/cfm_precompute_eval.log
```

Expected: ~15 min wall clock; final line reports `XXX rows processed, XXX vae_ok, XXX id_ok`.

- [ ] **Step 4: Verify the cache shape**

```bash
ls output/cfm_precompute/photo_latents | wc -l
ls output/cfm_precompute/id_tokens | wc -l
du -sh output/cfm_precompute/
```

Expected: both directory counts within 1% of the eval set size (~979); total size ~140 MB for the eval slice.

- [ ] **Step 5: No commit — caches are gitignored data.**

(Verify `output/` is in `.gitignore`; if not, add it before continuing.)

---

## Task 10: Phase 3 — pilot training to step 500 (Gate 1)

Train on a 5k-row subset for 500 steps with sample dumps every 50 steps. Manual eyeball gate at the end.

- [ ] **Step 1: Build the 5k pilot shas list**

```bash
uv run python -c "
from arkit_controlnet.cfm.dataset import CfmPairDataset
ds = CfmPairDataset(split='train', eval_size=1024)
with open('output/cfm_precompute/pilot_shas.txt', 'w') as f:
    for sha in ds.df.image_sha256.head(5000):
        f.write(sha + '\n')
print(5000, 'pilot shas written')
"
```

- [ ] **Step 2: Precompute the pilot subset**

```bash
uv run python -m arkit_controlnet.cfm.precompute \
    --shas-file output/cfm_precompute/pilot_shas.txt \
    --out-dir output/cfm_precompute \
    2>&1 | tee /tmp/cfm_precompute_pilot.log
```

Expected: ~75 min wall clock (5× the eval set). The eval-set rows already on disk are skipped.

- [ ] **Step 3: Launch the pilot training run**

```bash
uv run python -m arkit_controlnet.cfm.train \
    --out-dir exp_output/cfm_train/pilot \
    --max-steps 500 \
    --eval-every 50 \
    --save-every 100 \
    --dataset-filter-n 5000 \
    2>&1 | tee -a /tmp/cfm_pilot.log &
```

Expected: ~30 min wall clock. Sample collages at `exp_output/cfm_train/pilot/samples/step_{000050,000100,...,000500}.png`.

- [ ] **Step 4: Open the step_000500 collage**

```bash
ls -lh exp_output/cfm_train/pilot/samples/
```

Open `exp_output/cfm_train/pilot/samples/step_000500.png` and inspect the 8 rows. Each row should show 4 panels: target photo, FLAME render, identity-only generation, conditioned generation.

**Gate 1 criterion (manual):** at least 3 of the 8 conditioned generations visibly differ from their identity-only baseline in a direction consistent with the FLAME render (e.g. open mouth when the FLAME render shows open mouth, raised eyebrow when it shows raised eyebrow). If yes → proceed to Task 11. If no → STOP, write a failure-mode note (which axis of the pipeline is broken: VAE crop? text embed? control packing? LoRA target?), escalate.

- [ ] **Step 5: No commit yet — Gate 1 is a manual decision.**

---

## Task 11: Phase 5 — full-corpus precompute (background)

If Gate 1 passes, kick off the remaining ~64k precompute in the background while training continues.

- [ ] **Step 1: Build the remaining-shas list**

```bash
uv run python -c "
import pandas as pd
from arkit_controlnet.cfm.dataset import CfmPairDataset
ds = CfmPairDataset(split='train', eval_size=1024)
done = set()
import os
for f in os.listdir('output/cfm_precompute/photo_latents'):
    done.add(f.removesuffix('.pt'))
remaining = [s for s in ds.df.image_sha256 if s not in done]
with open('output/cfm_precompute/remaining_shas.txt', 'w') as f:
    for s in remaining:
        f.write(s + '\n')
print(len(remaining), 'remaining shas')
"
```

- [ ] **Step 2: Run the remaining-corpus precompute in the background**

```bash
nohup uv run python -m arkit_controlnet.cfm.precompute \
    --shas-file output/cfm_precompute/remaining_shas.txt \
    --out-dir output/cfm_precompute \
    > /tmp/cfm_precompute_full.log 2>&1 &
echo $! > /tmp/cfm_precompute_full.pid
```

Expected: ~2 h wall clock. Resumable per row.

- [ ] **Step 3: No commit.**

---

## Task 12: Phase 6+7 — full training + Gate 2 + final eval

Continue from the pilot checkpoint to 20k total steps.

- [ ] **Step 1: Copy the pilot checkpoint as the seed for the full run**

```bash
mkdir -p exp_output/cfm_train/run01
cp exp_output/cfm_train/pilot/latest.pt exp_output/cfm_train/run01/latest.pt
```

- [ ] **Step 2: Launch the full training run**

```bash
uv run python -m arkit_controlnet.cfm.train \
    --out-dir exp_output/cfm_train/run01 \
    --max-steps 20000 \
    --eval-every 50 \
    --save-every 500 \
    2>&1 | tee -a /tmp/cfm_run01.log &
```

Expected: ~12 h wall clock. As the corpus precompute completes in parallel, the dataset transparently sees more rows on each epoch boundary.

- [ ] **Step 3: Gate 2 — automatic check at step 2000**

```bash
tail -5 exp_output/cfm_train/run01/metrics.csv
```

Expected by step 2000: `bs_cos - bs_cos_neutral > 0.05` AND `id_cos > 0.50`. If not, halt with `kill $(cat /tmp/cfm_precompute_full.pid)` (precompute) and `pkill -f cfm.train` (training), write a failure note, escalate.

- [ ] **Step 4: Final eval at step 20000**

```bash
uv run python -c "
from arkit_controlnet.cfm.eval import dump_samples, metrics
from arkit_controlnet.cfm.model import build_model
import torch
model = build_model()
state = torch.load('exp_output/cfm_train/run01/latest.pt', map_location='cpu')
model.infusenet.load_state_dict(state['infusenet'], strict=False)
dump_samples(model, step=20000, out_dir='exp_output/cfm_train/run01', n=16)
m = metrics(model, step=20000, out_dir='exp_output/cfm_train/run01')
print(m)
"
```

- [ ] **Step 5: Write the run summary**

Create `docs/research/2026-05-21-cfm-training-run-results.md` with: final `id_cos`, `bs_cos`, `bs_cos_neutral`, link to the step_020000 collage, and a one-paragraph verdict.

- [ ] **Step 6: Commit results doc**

```bash
git add docs/research/2026-05-21-cfm-training-run-results.md docs/research/_topics/arkit-controlnet.md
git commit -m "$(cat <<'EOF'
docs(arkit-controlnet): CFM training run results

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Implementing task |
|---|---|
| Constraints already fixed (modality A, landmark alignment, dataset, dev-bf16, resampler) | Tasks 0, 1, 4, 5, 6 |
| `flame_render` extension (`render_landmark_aligned`) | Task 1 |
| `cfm/text_embeds.py` | Task 3 |
| `cfm/model.py` | Task 4 |
| `cfm/precompute.py` | Task 5 |
| `cfm/dataset.py` | Task 6 |
| `cfm/train.py` | Task 7 |
| `cfm/eval.py` | Task 8 |
| Phase 0 — weight downloads | Task 0 |
| Phase 2 — eval precompute | Task 9 |
| Phase 3 — pilot + Gate 1 | Task 10 |
| Phase 5 — background precompute | Task 11 |
| Phase 6+7 — full run, Gate 2, final eval | Task 12 |
| Resampler vendoring | Task 2 |

No gaps.

**Placeholder scan:** Task 5 and Task 8 omit the full implementation source in favor of structured implementation notes — these modules are ~150 lines each and reproducing them verbatim would bloat the plan without adding instruction; the bullet specifications give concrete contracts (function signatures, side effects, output shapes) that any implementer can flesh out without ambiguity. Task 7's `train.py` is shown in full because its structure is load-bearing. No `TBD`, no `implement later`.

**Type consistency:** 
- `CfmModel` is `(flux, infusenet, trainable_params)` everywhere.
- `velocity()` signature matches the spike script and the test in Task 4.
- `photo_latent` is `(16, 64, 64)` bf16 throughout (Tasks 5/6/7).
- `id_tokens` is `(8, 4096)` bf16 throughout.
- `control_rgb` from dataset is `(3, 512, 512) float32 in [-1, 1]`; trainer VAE-encodes it to `(16, 64, 64) bf16` then packs to `(1, 1024, 64)`.
- `BS_COLUMNS` order matches `BASIS_CHANNEL_NAMES` (sans tongueOut) — tested in Task 6.
- LoRA target list is identical between spike, Task 4, and the LoraConfig regex.
- `sigma` is a `(B,)` tensor at every callsite (Task 4 test, Task 7 train loop, Task 8 sampling).

