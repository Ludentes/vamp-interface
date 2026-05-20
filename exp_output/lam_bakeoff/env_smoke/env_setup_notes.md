# LAM env setup on RTX 5090 (sm_120) — patches applied

Upstream LAM commit: `abbf38a Release report of MeshLAM, CVPR 2026`
Conda env: `/home/newub/miniconda3/envs/lam/` (python 3.10)

## Stack that actually runs

- torch **2.11.0+cu128** (cu121 + cu124 channels both lack sm_120 kernels — confirmed via `torch.cuda.get_arch_list()`)
- xformers 0.0.35 (kept for the import, but disabled at runtime — no sm_120 attention kernels yet)
- CUDA extensions, all rebuilt with `TORCH_CUDA_ARCH_LIST=12.0` against torch 2.11.0+cu128:
  - `simple_knn` (camenduru fork, `--no-build-isolation`)
  - `diff_gaussian_rasterization` (ashawkey, with glm submodule, `--no-build-isolation`)
  - `nvdiffrast` 0.3.3 (ShenhanQian @ backface-culling)
  - `pytorch3d` 0.7.9
- FaceBoxesV2 cpu_nms: rebuilt after `np.int → np.int_` (numpy 1.20+ alias removed)
- helper deps: `fdlite` (= `face-detection-tflite==0.6.0`, pulls tensorflow 2.21), `chumpy`, `ninja`, `tensorboard`

## Source patches

| File | Reason | Patch |
|---|---|---|
| `external/landmark_detection/lib/metric/fr_and_auc.py` | scipy 1.14+ removed `simps` | try/except → `simpson as simps` |
| `external/vgghead_detector/VGGDetector.py:24` | torch 2.6+ default `weights_only=True` breaks TorchScript load | add `weights_only=False` |
| `external/landmark_detection/FaceBoxesV2/utils/nms/cpu_nms.pyx:29` | `np.int` removed | `np.int_` then rebuild via `make.sh` |
| site-packages `chumpy/__init__.py:11` | numpy alias removal | `from numpy import nan, inf` only |
| site-packages `sitecustomize.py` | (1) global `torch.load(..., weights_only=False)` shim covers the many other `torch.load` callsites; (2) prepend torch-cpp-extension JIT cache dir to `sys.path` so `nvdiffrast_plugin.so` is importable after build | new file |

## Required env at run-time

```bash
PATH=/home/newub/miniconda3/envs/lam/bin:$PATH       # nvdiffrast JIT needs ninja on PATH
PYTHONPATH=.                                         # from LAM repo root
XFORMERS_DISABLED=1                                  # no sm_120 attention kernels in xformers 0.0.35 yet → falls back to torch matmul attention (DINOv2 inference, not training, so fine)
```

## Bundled smoke (Task 1.5) — PASS

- Input: `assets/sample_input/status.png` + `assets/sample_motion/export/Look_In_My_Eyes/` (519 frames)
- Output: `exps/videos/lam/lam_20k/status_audio.mp4` (copied to `01_status_bundled.mp4`, 902 KB)
- **Render time**: 1.75 s for 519 frames → ~297 fps GS render on 5090 (excludes one-time VHAP tracking + DINOv2 + transformer pass)
- Total wall: ~2 min including VHAP single-image FLAME tracking and DINOv2 forward (one-shot, per-anchor)

## Open

- xformers fp16 path likely works (cutlass etc. compute_cap≤9.0 only, but fa2F supports fp16/bf16) — would need to cast in DINOv2 attention. Not pursued; torch native matmul is fast enough at 1 image × 1301 tokens.
- VHAP step is the bottleneck on first-time setup (~10 s/anchor); cached via `tracking_output/export/<uid>/`. The 1.75 s render number is the steady-state path.

## Anchor sweep (Task 6) — results

| Anchor | Class | Result | Time |
|---|---|---|---|
| status.png | photoreal portrait | ✅ | 1.75 s |
| pushkin.jpg | oil painting (Kiprensky 1827) | ✅ | 1.73 s |
| tikhonov.jpg | B&W photo (1948) | ✅ | 1.57 s |
| other_a.jpg | photoreal | ✅ (only after FaceBoxes thresh 0.8 → 0.2) | 1.63 s |
| anime.png | anime/cel (Haruhi-style) | ❌ FaceBoxes zero detect at thresh 0.2 |
| other_b.jpg | non-human cartoon (duck) | ❌ out of FLAME morphology by construction |

**Patch:** `tools/flame_tracking_single_image.py:205` — FaceBoxes confidence threshold 0.8 → 0.2 to admit non-FFHQ photoreal anchors (rescued `other_a`; did not rescue anime/duck).

## Headline numbers

- 4/4 human anchors pass (incl. painting + B&W photo).
- ~1.65 s / 519 frames @ 30 fps render = ~310 fps GS render on RTX 5090.
- One-shot setup per anchor: ~10 s (FaceBoxes + STAR landmark + matting + VHAP single-image tracking + DINOv2 + LAM identity encode).
- Anime: out-of-domain for both FaceBoxes and STAR; would need anime-trained detector + landmark + FLAME-equivalent for ducks/cels.
- Render quality (subjective, from artifacts in `env_smoke/0[1-4]_*.mp4`): pending your eyeball.
