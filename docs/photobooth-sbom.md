# Photobooth dependency manifest (SBOM)

What must be installed and where, for the single-face photobooth + group photobooth to function. Read this before standing up the pipeline on a new ComfyUI host.

**Last verified:** 2026-05-24 against live ComfyUI at `http://127.0.0.1:8188`.

---

## ComfyUI host

- **Install path:** `/home/newub/w/ComfyUI/`
- **Server:** `http://127.0.0.1:8188`
- **GPU:** RTX 5090 (24 GB)
- **Custom-nodes dir:** `/home/newub/w/ComfyUI/custom_nodes/`

Restart ComfyUI after installing any new custom node so it gets registered.

---

## Required custom nodes

| Node pack | Class types used | Install URL | License | Used by |
|---|---|---|---|---|
| `kijai/ComfyUI-segment-anything-2` | `DownloadAndLoadSAM2Model`, `Sam2Segmentation`, `MaskToImage` | https://github.com/kijai/ComfyUI-segment-anything-2 | Apache-2.0 | Group photobooth `detect.py` (body mask) |
| `1038lab/ComfyUI-RMBG` | `RMBG` (INSPYRENET mode) | https://github.com/1038lab/ComfyUI-RMBG | AGPL-3.0 (model: INSPYRENET MIT) | Group photobooth `silhouette.py` (doll cutout) |
| `group_photobooth_helpers` (local) | `BBoxFromJSON` | (not on github) source: `/home/newub/w/ComfyUI/custom_nodes/group_photobooth_helpers/__init__.py` | Project-internal | Bridges JSON-string bbox → BBOX-typed Sam2Segmentation input |

Built-in ComfyUI nodes the workflows also rely on: `LoadImage`, `SaveImage`, `KSampler`, `VAEDecode`, `VAEEncode`, `EmptyLatentImage`, `CLIPTextEncode`, `ControlNetApplyAdvanced`, `LoadImageMask`, `SetLatentNoiseMask` — all stock.

Stack required by the single-face photobooth (not group-specific):

| Node pack | Class types | Install URL | Used by |
|---|---|---|---|
| Z-Image (built-in workflow) | model loader for `z_image_turbo` | shipped with ComfyUI ≥ Nov 2025 | `photobooth_zimage_cn.api.json` |
| `Z-Image-Turbo-Fun-CN-Union` ControlNet | CN model + loader | https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-CN-Union | `cn_workflow` ControlNet branch |
| `HyperSwap-1c` ONNX | invoked via `scripts/swap_core.py`, not as a ComfyUI node | https://huggingface.co/your-source-here (see `scripts/swap_core.py:load_swapper`) | post-render identity swap |

---

## Required model weights

All paths are under `/home/newub/w/ComfyUI/models/` (or noted otherwise). Weights auto-download on first workflow execution unless flagged "manual".

### Diffusion + ControlNet

| Weight | Path | Source | Size | Notes |
|---|---|---|---|---|
| Z-Image Turbo (6-step) | `models/checkpoints/z_image_turbo.safetensors` | https://huggingface.co/alibaba-pai/Z-Image-Turbo | ~6 GB | Manual download |
| Z-Image Turbo CN-Union | `models/controlnet/zimage_turbo_cn_union.safetensors` | https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-CN-Union | ~1.5 GB | Manual download |

### Identity swap (HyperSwap)

| Weight | Path | Source | Notes |
|---|---|---|---|
| `hyperswap_1c.onnx` | resolved by `scripts/swap_core.py:load_swapper()` | (see `swap_core.py` constants) | Auto-loaded by `swap_core` |
| `buffalo_l` (face detection + ArcFace embedding) | `~/.insightface/models/buffalo_l/` | insightface model zoo | Auto-downloaded by `insightface.app.FaceAnalysis(name="buffalo_l")` on first use |

### SAM2 (group photobooth)

| Weight | Path | Source | Size |
|---|---|---|---|
| `sam2_hiera_base_plus.safetensors` | `models/sam2/sam2_hiera_base_plus.safetensors` | `DownloadAndLoadSAM2Model` auto-downloads from kijai's HF mirror | ~325 MB |

### Background removal (group photobooth)

| Weight | Path | Source | Size |
|---|---|---|---|
| INSPYRENET | `models/RMBG/INSPYRENET/` | `RMBG` node auto-downloads from `1038lab/INSPYRENET` HF | ~350 MB |

### Person detection (group photobooth — runs in Python, not ComfyUI)

| Weight | Path | Source | Size |
|---|---|---|---|
| `yolov8n.pt` | `~/.cache/ultralytics/yolov8n.pt` | `ultralytics` auto-downloads on first `YOLO("yolov8n.pt")` call | ~6 MB |

---

## Python deps

Group photobooth additions beyond what the single-face photobooth already requires:

| Package | Why | Install |
|---|---|---|
| `ultralytics` | YOLOv8 person detection | `uv pip install ultralytics` |
| `requests` | ComfyUI HTTP API | already a transitive dep |
| `opencv-python` | image I/O + Lab conversion | already installed |
| `numpy`, `Pillow` | array ops, fallback image I/O | already installed |

Single-face photobooth python deps (carry over): `insightface`, `onnxruntime-gpu`, `pyarrow` (parquet score table), `cv2`, `numpy`.

---

## First-time setup checklist

```bash
# 1. Custom nodes
cd /home/newub/w/ComfyUI/custom_nodes/
git clone https://github.com/kijai/ComfyUI-segment-anything-2
git clone https://github.com/1038lab/ComfyUI-RMBG
# group_photobooth_helpers/ already exists if cloned from a vamp-interface dev's
# machine; otherwise copy from another host or recreate from the source at
# /home/newub/w/ComfyUI/custom_nodes/group_photobooth_helpers/__init__.py.

# 2. Restart ComfyUI (user-owned lifecycle on this machine)

# 3. Python deps (in vamp-interface venv)
cd /home/newub/w/vamp-interface
uv pip install ultralytics
uv run --no-project python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# 4. Verify
COMFY_URL=http://127.0.0.1:8188 uv run --no-project \
  pytest tests/group_photobooth/test_comfy_io_e2e.py -v
# Expect: 2 passed (first run ~30-60s for model auto-downloads, ~5-10s after)
```

---

## What's NOT in this manifest (intentional)

- Other workflows in `comfyui/workflows/` (matryoshka_*, arkit_*, flux_*) that aren't part of the photobooth pipeline — see their own runbooks.
- LoRAs / IP-Adapters used by adjacent threads (`personalive`, `cfm_train`, etc.).
- The host OS, CUDA toolkit, Python interpreter version — assumed already provisioned.

---

## Cross-references

- `docs/research/_topics/photobooth-sweep.md` — current photobooth recipe + Phase 2/3 findings
- `docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md` — group photobooth design
- `docs/superpowers/plans/2026-05-21-group-photobooth.md` — implementation plan
- `CLAUDE.md` "External paths" — quick reference to repo-external paths
