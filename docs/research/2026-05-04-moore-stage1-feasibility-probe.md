---
status: live
topic: personalive-acceleration
---

# Moore-AnimateAnyone Stage-1 feasibility probe on RTX 5090

## Why

PersonaLive training code is deferred indefinitely (issue #17, maintainer
explicitly redirects to Moore-AnimateAnyone). We want to know whether
PersonaLive training is *possible* on a single RTX 5090 (32 GB) before
the upstream code lands, so we can decide whether to invest in a
reconstruction or wait.

Moore-AnimateAnyone is the closest releasable proxy: same UNet2D /
UNet3D / ReferenceNet / PoseGuider skeleton, same state-dict layout
for those three modules. The proxy does **not** exercise PersonaLive's
`MotEncoder` (FAN face-crop motion) or `MotionExtractor` (LivePortrait
3D keypoints), so the result is a *floor* on memory and throughput,
not a ceiling.

Decision rule:

- Moore Stage 1 OOMs at batch=1 on the 5090 → PersonaLive training
  here is not feasible without architectural surgery (offload, CPU
  optim, LoRA-only). Stop and wait for upstream.
- Moore Stage 1 fits at batch=1 → marginal, MotEncoder addition could
  tip it over. Plausible but tight.
- Moore Stage 1 fits at batch≥2 with grad ckpt + bf16 → headroom for
  PersonaLive's extra modules. Green light to start a reconstruction.

## Scope

Stage 1 only. Image-level reference + pose, no temporal module.
1-2 epochs on a tiny VFHQ subset (target: ≤50 clips, ≤10 GB on disk
given the **49 GB free** on `/` — current disk pressure is the
binding constraint, not GPU memory).

Stage 2 (temporal) is out of scope for the probe. If Stage 1 passes,
follow-up probe sizes Stage 2 separately.

## Setup

Clone alongside PersonaLive, do not install into the same env:

```
cd /home/newub/w
git clone https://github.com/MooreThreads/Moore-AnimateAnyone
cd Moore-AnimateAnyone
uv venv --python 3.10 .venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

Blackwell caveats (carried over from our PersonaLive acceleration
thread, see `_topics/personalive-acceleration.md`):

- xformers source build for sm_120 falsified — use plain SDPA, set
  `enable_xformers_memory_efficient_attention=False` in any config
  flag if Moore exposes one.
- torch ≥ 2.7 + CUDA ≥ 12.8 wheel needed for sm_120 kernels. If
  Moore pins old torch, override after install:
  `uv pip install torch==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128`.
- diffusers/transformers versions must be compatible with the
  pinned torch — expect to bump.

If accelerate is the trainer driver (Explore agent reported it is),
generate a single-GPU bf16 config:

```
accelerate config default --mixed_precision bf16
```

## Dataset

Moore's Stage-1 dataloader wants paired (reference, target) frames
with pose conditioning extracted offline. Adapting to a face-video
setting:

- Source: existing VFHQ test split, or 20–50 clips from any face-
  video corpus we already have on disk. Avoid a full VFHQ pull —
  100+ GB doesn't fit our 49 GB headroom.
- Per clip: extract every Nth frame (target ~30 frames/clip,
  ≈1500 frames total).
- Pose extraction: Moore expects DWPose / OpenPose-style 2D
  keypoints. For a face-only probe, hand-rolling a stub PoseGuider
  input from MediaPipe FaceMesh edges is acceptable — we are not
  optimizing for output quality, we want a forward+backward pass
  that does not OOM.
- Target resolution: 512×512 (Moore default 768 is wasteful for the
  feasibility question and triples activation memory).

If clip extraction itself blows past 10 GB, stop and downsample
further. The probe is not a research run.

## Run

Stage 1, smallest possible config, gradient checkpointing on,
bf16, batch=1 with grad accumulation = 4 (effective batch 4):

```
accelerate launch \
  --num_processes 1 \
  --mixed_precision bf16 \
  train_stage_1.py \
  --config configs/train/stage1.yaml \
  data.resolution=512 \
  data.batch_size=1 \
  train.gradient_accumulation_steps=4 \
  train.gradient_checkpointing=true \
  train.max_train_epochs=2 \
  train.checkpointing_steps=500 \
  data.train_data_root=/home/newub/w/face-probe-data
```

(Exact CLI overrides depend on Moore's actual config schema —
discover during execution; the above is the shape.)

Wrap with the standard memory cap our `feedback_eval_memory_cap`
note prescribes:

```
systemd-run --user --scope -p MemoryMax=45G \
  accelerate launch ...
```

Expected wall time: with ~1500 frames × 2 epochs at single-sample
forward+backward on 512², 5090 should land in the 30–90 minute
range. If it stretches past 3 hours, kill — something is wrong
(likely no grad ckpt or fp32 fallback on a Blackwell-incompatible
kernel).

## Memory ceiling estimate

Back-of-envelope, bf16, grad ckpt on, batch=1, 512²:

- ReferenceUNet (~860 M params): 1.7 GB weights + 3.4 GB Adam
  optimizer state ≈ 5.1 GB
- DenoisingUNet3D (~890 M): 1.8 + 3.6 ≈ 5.4 GB
- PoseGuider (~10 M): negligible
- VAE (frozen, fp16): ~330 MB
- Image encoder (CLIP, frozen): ~600 MB
- Activations w/ grad ckpt at 512² batch=1, 16 frames: ~6–8 GB
- Misc fragmentation, cache: ~2 GB

→ Working estimate **~20–22 GB**. Should fit on the 5090 with ~10 GB
headroom for kernel scratch / fragmentation. If we see >28 GB
sustained, something's off.

## What to log

Per training step, append to a CSV under
`docs/research/_data/moore-probe/run-<ts>.csv`:

- step
- wall-clock dt
- `nvidia-smi --query-gpu=memory.used,utilization.gpu` snapshot
- loss (if exposed)
- effective images/sec

Plus a single end-of-run summary:

- did Stage 1 complete 2 epochs without OOM (yes/no)
- peak GPU memory observed
- mean step time at steady state (after warm-up step)
- effective images/sec
- any crashes or kernel fallbacks (search dmesg + accelerate log
  for `at::native::` CPU fallbacks)

Spawn `nvidia-smi dmon` in a side terminal logging to file for the
peak-memory trace.

## Success criteria

The probe answers a single question: does Stage 1 fit? The verdict
is binary, with a tier:

- **Green** — completes 2 epochs, peak ≤ 24 GB, no crashes. Implies
  ~8 GB headroom for PersonaLive's MotEncoder + MotionExtractor.
  Recommend starting a PersonaLive Stage 1 reconstruction.
- **Yellow** — completes at batch=1 only, peak 24–30 GB. PersonaLive
  reconstruction would likely require offload (CPU Adam, paged
  optimizer) or LoRA on the UNets.
- **Red** — OOMs at batch=1 even with grad ckpt + bf16. Wait for
  upstream training code; do not invest in reconstruction.

## Out of scope (explicitly)

- Quality of generated samples. Output frames will look bad — we
  ran 2 epochs on 50 clips with hacked pose input. Do not bench
  quality.
- Stage 2 temporal. Separate probe if Stage 1 passes Green.
- MotEncoder integration. Separate probe if Stage 1 passes Green.
- Distillation (PersonaLive Stage 2 paper sense). Out of scope
  entirely until Moore Stage 1 + PersonaLive reconstruction Stage 1
  both work.

## Follow-ups

- If Green: spec a PersonaLive Stage 1 reconstruction probe that
  adds MotEncoder + MotionExtractor on top of Moore's Stage 1
  trainer. Re-run, compare memory and throughput delta.
- If Yellow: research CPU offload (`bitsandbytes` 8-bit Adam, FSDP
  CPU offload, DeepSpeed Zero-2). Each adds complexity but unblocks.
- If Red: monitor PersonaLive issue #17. Revisit after upstream
  releases or after a 64 GB / 96 GB GPU becomes available.

## Pre-flight checklist

Before launching the actual training run:

- [ ] `df -h /home/newub/w` shows ≥ 15 GB free after dataset extract.
      Currently 49 GB free; clip extraction must stay under 10 GB.
- [ ] Moore-AnimateAnyone clones, `uv pip install` succeeds without
      manual torch override → if torch override needed, log it.
- [ ] `python -c "import torch; print(torch.cuda.get_device_name())"`
      shows RTX 5090.
- [ ] Single forward pass on a dummy batch completes without
      `at::native::` CPU fallback warnings.
- [ ] `accelerate launch` with `--num_processes 1` works on a
      10-step dry run before committing to 2 epochs.
