---
status: live
topic: personalive-acceleration
supersedes:
superseded_by:
---

# Moore Stage-1 probe — reproduce + handoff (2026-05-05)

Companion to [`2026-05-04-moore-stage1-feasibility-probe.md`](2026-05-04-moore-stage1-feasibility-probe.md)
(spec + verdict). This doc captures *exactly* what we did so the work
is reproducible and the vendor-tree patches are recoverable if the
external repo gets nuked.

## Verdict in one line

PersonaLive's `reference_unet.pth` + `denoising_unet.pth` + `pose_guider.pth`
fine-tune cleanly on a single RTX 5090 (32 GB) at batch=1, 512², bf16,
grad-checkpoint, **8-bit Adam**, ~1.05 s/it. 32-bit Adam OOMs.

## Where things live

- **Vendor repo (outside this tree):** `~/w/Moore-AnimateAnyone/`
  - venv: `.venv/` (Python 3.10, torch 2.11.0+cu128, diffusers 0.24.0)
  - probe config: `configs/train/stage1_probe.yaml`
  - synth dataset: `data/probe_videos/face.mp4` + `face_pose.mp4`,
    meta at `data/probe_meta.json`
  - patched files: `train_stage_1.py`, `src/models/{transformer_2d,transformer_3d,unet_2d_blocks,unet_3d_blocks}.py`
  - PersonaLive weights via symlinks under `pretrained_weights/`
- **PersonaLive weights (read-only):** `~/w/PersonaLive/pretrained_weights/personalive/`
- **Snapshots in this repo (recovery copies):**
  - [`2026-05-05-moore-stage1-vendor-patch.diff`](2026-05-05-moore-stage1-vendor-patch.diff) — 228-line diff against vendor `main`
  - [`2026-05-05-moore-stage1_probe.yaml`](2026-05-05-moore-stage1_probe.yaml) — probe config

## Reproduce from a clean checkout

```bash
cd ~/w
git clone https://github.com/MooreThreads/Moore-AnimateAnyone
cd Moore-AnimateAnyone
uv venv --python 3.10 .venv && . .venv/bin/activate

# 1) Torch 2.11+cu128 BEFORE other deps (so torchvision/triton resolve to cu128)
uv pip install torch==2.11.0+cu128 torchvision \
  --index-url https://download.pytorch.org/whl/cu128

# 2) Strip pre-Blackwell pins from requirements
grep -vE "^(torch|xformers|accelerate|diffusers|transformers|numpy|av|torchvision|controlnet-aux|onnxruntime)" \
  requirements.txt > /tmp/req.txt
uv pip install setuptools wheel Cython
uv pip install -r /tmp/req.txt 'numpy<2' 'av' \
  'accelerate>=0.34' 'diffusers==0.24.0' 'transformers==4.36.2' \
  'huggingface_hub<0.26' 'controlnet-aux' 'onnxruntime-gpu' \
  --no-build-isolation

# 3) NCCL ABI: bitsandbytes can downgrade nccl back to 2.28.x.
#    Force-reinstall AFTER bnb if torch import fails with libnccl symbol error.
uv pip install bitsandbytes
uv pip install --reinstall nvidia-nccl-cu12

# 4) Apply vendor patch
git apply ~/w/vamp-interface/docs/research/2026-05-05-moore-stage1-vendor-patch.diff

# 5) Symlink PersonaLive weights into Moore tree
ln -sfn ~/w/PersonaLive/pretrained_weights/sd-image-variations-diffusers \
        pretrained_weights/sd-image-variations-diffusers
ln -sfn ~/w/PersonaLive/pretrained_weights/sd-vae-ft-mse \
        pretrained_weights/sd-vae-ft-mse

# 6) Synthesize the dummy probe data (or re-run the python from the handoff)
mkdir -p data/probe_videos
# ... (see "Synthetic dataset" below)

# 7) Drop in the probe config
cp ~/w/vamp-interface/docs/research/2026-05-05-moore-stage1_probe.yaml \
   configs/train/stage1_probe.yaml

# 8) Run
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
systemd-run --user --scope -p MemoryMax=45G \
  accelerate launch --num_processes 1 --mixed_precision bf16 \
  train_stage_1.py --config configs/train/stage1_probe.yaml
```

## What the patches do (semantic summary)

The 228-line diff against Moore's vendor source contains four
mechanical changes plus the PersonaLive-transplant block. All five
are needed; none change training math.

1. **`bf16` weight_dtype** (`train_stage_1.py:243`) — Moore originally
   raised on anything that wasn't `fp16` or `fp32`. Add a bf16 branch.
2. **Checkpoint kwargs shim** (`train_stage_1.py` import block) —
   torch 2.x reentrant checkpoint rejects kwargs to the wrapped
   function. Monkey-patch `torch.utils.checkpoint.checkpoint` to set
   `use_reentrant=False` and bind any user kwargs into a closure.
3. **`custom_forward` signature** (4 model files, 18 sites) — change
   `def custom_forward(*inputs):` to `def custom_forward(*inputs, **kwargs):`
   and forward `**kwargs` into the wrapped module call. Required so
   that the kwargs the trainer passes to checkpoint reach the
   underlying transformer/resnet block.
4. **Pose-guider random-init defaults** (`train_stage_1.py:298`) —
   add `block_out_channels=(16,32,96,256)` to the no-controlnet-init
   PoseGuider so its shape matches the `pose_guider_pretrain=True`
   path. This avoids needing the `control_v11p_sd15_openpose` weights
   when we plan to overwrite with PersonaLive's pose_guider anyway.
5. **PersonaLive transplant block** (`train_stage_1.py`, after `if
   cfg.pose_guider_pretrain`) — guarded by `cfg.persona_weights_dir`,
   loads `reference_unet.pth` / `denoising_unet.pth` / `pose_guider.pth`
   from PersonaLive into the freshly built modules with `strict=False`.
   Renames `conv_out_modify`→`conv_out` for the pose_guider keys.

## Synthetic dataset (probe_videos)

Moore's `HumanDanceDataset` wants a video file + matching pose video,
both readable by decord, both same length. For the memory probe a
fully synthetic 30-frame pair is sufficient — the trainer doesn't
care about content, only shapes and dtype.

```python
import numpy as np, imageio.v3 as iio, json
N=30; H=W=512
vid = np.zeros((N,H,W,3), dtype=np.uint8)
for i in range(N):
    vid[i,:,:,:] = (i*3) % 255
    cy=int(128+128*np.sin(i*0.2)); cx=int(256+64*np.cos(i*0.2))
    vid[i, cy-40:cy+40, cx-40:cx+40, :] = [220,180,150]
iio.imwrite('data/probe_videos/face.mp4', vid, fps=24,
            codec='libx264', pixelformat='yuv420p')
pose = np.zeros((N,H,W,3), dtype=np.uint8)
for i in range(N):
    cy=int(128+128*np.sin(i*0.2)); cx=int(256+64*np.cos(i*0.2))
    pose[i, cy-30:cy+30, cx-2:cx+2, :] = 255
    pose[i, cy-2:cy+2, cx-30:cx+30, :] = 255
iio.imwrite('data/probe_videos/face_pose.mp4', pose, fps=24,
            codec='libx264', pixelformat='yuv420p')
json.dump([{"video_path":"./data/probe_videos/face.mp4",
            "kps_path":"./data/probe_videos/face_pose.mp4"}],
           open('data/probe_meta.json','w'))
```

For real fine-tuning, swap this for actual face videos with DWPose
extraction via `tools/extract_dwpose_from_vid.py` (needs the DWPose
onnx weights — `tools/download_weights.py` covers them, after
patching the broken runwayml/SD1.5 line out).

## Pitfalls hit (and the lesson each carries)

- **Cu11 leftover NVIDIA libs after partial torch install** → cu11
  packages got dragged in by `torchvision==0.15.2` and `xformers==0.0.22`
  pins. Strip ALL old pins, install torch 2.11+cu128 first, then pin
  torchvision out of the requirements file. `nvidia-nccl-cu11` and
  friends are silent killers — they won't break anything until torch
  imports.
- **`nvidia-nccl-cu12` package metadata says installed but `.so` is
  missing** → after the cu11 purge, `--reinstall` is required to
  drop `libnccl.so.2` in place. Symptom: `undefined symbol: ncclMemFree`.
- **`bitsandbytes` install pulls in older nccl** → reinstall nccl AFTER
  bnb to keep the libnccl 2.30+ ABI torch 2.11 needs.
- **Diffusers 0.37 vs 0.24 API drift** → 0.37 removed `dual_transformer_2d`,
  renamed `PositionNet`, moved several `diffusers.utils` symbols.
  The cascade of patches is more work than just pinning to Moore's
  original `0.24.0` (works fine with torch 2.11 — diffusers' torch
  coupling is loose).
- **Reentrant checkpoint silently swallows `use_reentrant`** → setting
  the kwarg fixed nothing; we still got `unexpected keyword argument
  encoder_hidden_states` because Moore's `custom_forward(*inputs)` had
  to be patched to accept and forward kwargs. The two changes (the
  monkey-patch + the closure signature widen) are both required.
- **32-bit Adam = OOM** even at batch=1 / 512² / grad-ckpt because
  Adam doubles the param state. ~1.7B trainable × 8 bytes/param × 2
  states ≈ 13.6 GB just for `exp_avg` + `exp_avg_sq` in fp32. That's
  the lever 8-bit Adam pulls out — drops it to ~3.4 GB.
- **Don't try `torch.compile` on a Blackwell+diffusers-0.24 stack**
  for the probe. Untested combo; not needed for the memory question.

## Open follow-ups

- Stage 2 (temporal): swap to `train_stage_2.py`, increase activation
  budget by the temporal-window factor. Memory probe TBD.
- MotEncoder + MotionExtractor: PersonaLive-specific modules not in
  Moore. Adding requires either porting the modules into Moore's
  tree (PersonaLive ships their forward code) or moving to a
  PersonaLive-fork trainer once their training code lands.
- Real-data Stage 1 run for a steady-state IPS / loss curve.
- Try DeepSpeed Zero-2 CPU offload as fallback if Stage 2 spills
  even with 8-bit Adam.

## Tracking

This doc is the live record. The earlier feasibility-probe spec and
the personalive-acceleration topic index both link here. If the
vendor patches change shape (e.g., PersonaLive ships their training
code and we no longer need Moore as a proxy), supersede this with a
new dated doc and bump frontmatter.
