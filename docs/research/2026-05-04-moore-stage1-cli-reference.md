---
status: live
topic: personalive-acceleration
---

# Moore-AnimateAnyone Stage-1 CLI reference (single 5090, 32 GB)

Source-of-truth: [MooreThreads/Moore-AnimateAnyone @ master](https://github.com/MooreThreads/Moore-AnimateAnyone/tree/master). Default branch is `master` (not `main`).

## Entry-point script

- File: `train_stage_1.py` at repo root ([source](https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/train_stage_1.py)).
- Argparse defines exactly one CLI flag: `--config`, default `./configs/training/stage1.yaml`. (Note the README refers to `configs/train/stage1.yaml`; the shipped YAML actually lives at [`configs/train/stage1.yaml`](https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/configs/train/stage1.yaml). The argparse default path is stale — pass `--config configs/train/stage1.yaml` explicitly.)
- Config loader: OmegaConf for `.yaml`, `import_filename(...).cfg` for `.py`. There are **no other CLI overrides** — all tuning happens by editing the YAML.
- Stage 2: `train_stage_2.py`, default `./configs/training/stage2.yaml`, same loader contract.

## Config schema (Stage 1, override-relevant keys)

From [`configs/train/stage1.yaml`](https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/configs/train/stage1.yaml):

- `data.train_bs: 4` — per-device batch size. Lower for 32 GB.
- `data.train_width: 768` / `data.train_height: 768` — try 512 for the probe.
- `data.meta_paths: ["./data/fashion_meta.json"]` — list of metadata JSONs (multi-dataset).
- `solver.gradient_checkpointing: False` — flip to `True` for VRAM headroom.
- `solver.mixed_precision: 'fp16'` (also `weight_dtype: 'fp16'`).
- `solver.enable_xformers_memory_efficient_attention: True`.
- `solver.gradient_accumulation_steps: 1`, `solver.max_train_steps: 30000`, `solver.learning_rate: 1.0e-5`, `solver.use_8bit_adam: False`.
- `checkpointing_steps: 2000`, `val.validation_steps: 200`, `save_model_epoch_interval: 5`.
- Model paths: `base_model_path`, `vae_model_path`, `image_encoder_path`, `controlnet_openpose_path` — populated via `python tools/download_weights.py`.
- `seed: 12580`, `resume_from_checkpoint: ''`, `exp_name: 'stage1'`, `output_dir: './exp_output'`.

## Dataset format

Layout expected by `tools/extract_meta_info.py`:

```
<root_path>/**/*.mp4              # source videos (any subdir nesting)
<root_path>_dwpose/**/*.mp4       # parallel tree of pose-rendered videos
```

Pipeline:

```
python tools/extract_dwpose_from_vid.py --video_root /path/to/videos
python tools/extract_meta_info.py --root_path /path/to/videos --dataset_name anyone
# -> ./data/anyone_meta.json with [{"video_path": ..., "kps_path": ...}, ...]
```

Pose extractor: DWPose (`src.dwpose.DWposeDetector`); the script auto-shards across visible GPUs. README does not list UBC fashion / TikTok URLs — bring your own video corpus.

## Single-GPU launch

There's only one supported invocation:

```
accelerate launch train_stage_1.py --config configs/train/stage1.yaml
```

For single-GPU, run `accelerate config` once and select 1 process / no DeepSpeed / fp16 — or override inline:

```
accelerate launch --num_processes=1 --mixed_precision=fp16 \
  train_stage_1.py --config configs/train/stage1.yaml
```

All other knobs (batch, resolution, ckpt) are config-file edits — there is no Hydra/CLI override layer.

## Known issues for consumer GPUs

- [#113](https://github.com/MooreThreads/Moore-AnimateAnyone/issues/113): default Stage 1 OOMs on 4090 (24 GB) and even V100 32 GB. A100 screenshot shown as the working baseline — implies stock config is sized for ~40 GB. 32 GB on 5090 will need `train_width/height: 512`, `gradient_checkpointing: True`, `train_bs: 1`, possibly `use_8bit_adam: True`.
- [#99](https://github.com/MooreThreads/Moore-AnimateAnyone/issues/99): single-4090 + DeepSpeed offload crashes at validation step 200 with `mat1/mat2 dtype mismatch (Float vs Half)` inside `pipeline_pose2img.py`. Workaround not posted; likely needs disabling validation or forcing fp32 cast at the `time_embedding` call. Unresolved upstream.
- [#146](https://github.com/MooreThreads/Moore-AnimateAnyone/issues/146): ZeRO-3 vs ZeRO-2 memory discussion — no clean resolution.

## Pinned versions ([`requirements.txt`](https://github.com/MooreThreads/Moore-AnimateAnyone/blob/master/requirements.txt))

`torch==2.0.1`, `torchvision==0.15.2`, `xformers==0.0.22`, `diffusers==0.24.0`, `transformers==4.30.2`, `accelerate==0.21.0`, `omegaconf==2.2.3`, `onnxruntime-gpu==1.16.3`, `numpy==1.23.5`, `controlnet-aux==0.0.7`. README pins CUDA 11.7 + Python ≥3.10.

**sm_120 / Blackwell conflict:** all of `torch==2.0.1`, `xformers==0.0.22`, `onnxruntime-gpu==1.16.3` predate Blackwell and will not have sm_120 kernels. RTX 5090 needs torch ≥2.7 + cu128 (and a matching xformers, or fall back to PyTorch SDPA). Plan to bump torch/xformers and re-pin diffusers/transformers to versions compatible with the new torch — verify Stage 1 still loads (`UNet2DConditionModel` import path may have shifted in diffusers ≥0.30). Compatibility of `accelerate==0.21.0` with torch 2.7+ is not verifiable from repo, would need to discover at run time.
