---
status: live
topic: photobooth-sweep
---

# Group photobooth architecture + runbook

Turn a photo of 1–5 people into a matryoshka group portrait composited onto a chosen pre-rendered background. Approach A from the spec: per-face render via the existing single-face photobooth (Phase 3 heavy mix, `swap_weight=0.10`), then INSPYRENET cutout + painter's-order alpha composite onto a library background.

Spec: `docs/superpowers/specs/2026-05-21-group-photobooth-pipeline-design.md`
Plan: `docs/superpowers/plans/2026-05-21-group-photobooth.md`
SBOM: `docs/photobooth-sbom.md`

## Pipeline

```
photo ──► detect.detect_people ──► [Person(body_bbox, body_mask, face_bbox), ...]
                │                       │
                │                       └──► for each Person:
                │                              face_renderer.render_doll(face_crop) ──► doll BGR
                │                              silhouette.cutout(doll)               ──► doll BGRA
                │
                └──► layout.solve_placements(persons, photo_hw, bg_hw)
                                                │
                                                ▼
                            composite.composite(bg, dolls_rgba, placements)
                                                │  (optional polish: lab_match + drop_shadow)
                                                ▼
                                          result.png
```

`detect.detect_people` runs YOLOv8-person (CPU) + buffalo_l face detection in-process; SAM2 body-mask prediction is one HTTP call per person to ComfyUI using `comfyui/workflows/group_sam2_mask.api.json`. `silhouette.cutout` posts `group_cutout.api.json` (INSPYRENET via `1038lab/ComfyUI-RMBG`). `face_renderer.render_doll` reuses `scripts/photobooth_sweep/{driver,preprocess}.py` and `scripts/swap_core.py` verbatim with `HEAVY_MIX_CFG` locked at module level.

Layout letterboxes the photo aspect into the background aspect (uniform scale `min(bh/ph, bw/pw)`, centered). Painter's order = ascending `y_bottom` so lower-in-photo persons paint last (closer to camera). Doll height tracks person bbox height in projected coords.

## CLI

```bash
COMFY_URL=http://127.0.0.1:8188 uv run --no-project \
    python -m group_photobooth.driver \
    --photo data/group_photobooth_inputs/test_two_people.jpg \
    --background blank_studio \
    --out exp_output/group_photobooth/test_two_people/
```

Add `--polish` for Lab color match to background palette + elliptical drop shadow. Add `--demo-json '{"gender":"F","age_bin":"30s","race":"european"}'` to inject demographic prompt tokens (default `{gender:person, age_bin:30s, race:""}` works but skips the demographic conditioning).

Per-person intermediates cache to `<out_dir>/people/NN/{face,doll,doll_rgba}.png`. Reruns skip every step whose output exists, so partial failures are cheap to recover.

## Failure modes

- **No face detected on a person** — `detect_people` skips and logs `[detect] skipping bbox (...): no face`. Lying-down persons typically trip this (YOLO finds the body, buffalo_l rejects the rotated face). Workaround: rotate input photo manually.
- **No persons at all** — driver writes the bare background as `result.png` and exits 0.
- **Doll clipped at canvas edge** — composite clips to canvas; if a placement falls partially outside the bg, the visible portion still renders. Use a larger background or pre-crop the input photo so all persons fit after the letterbox scale.
- **SAM2 cold start** — first request after ComfyUI start auto-downloads `sam2_hiera_base_plus.safetensors` (~325 MB). Subsequent calls are sub-second.
- **INSPYRENET cold start** — same pattern, model auto-downloads to `models/RMBG/INSPYRENET/`.

## Per-person cache layout

```
<out_dir>/
├── result.png
└── people/
    ├── 00/
    │   ├── face.png        # cropped face from input photo (margin_frac=0.35)
    │   ├── doll.png        # post-swap doll portrait (Phase 3 heavy mix output)
    │   └── doll_rgba.png   # INSPYRENET cutout (BGRA)
    ├── 01/
    │   └── ...
    └── ...
```

Seed is deterministic per (photo_id, face_index) — `_seed_for_face` uses md5(`<stem>__<i>`). Same photo at the same path always renders the same dolls.

## Background library

`assets/backgrounds/manifest.json` is an array of `{id, description, dims, palette_lab_mean, lighting_hint}` records. Each background's PNG lives at `assets/backgrounds/<id>/bg.png`. Currently shipped: `blank_studio` (1024×1024 vertical gray gradient 220→180). Add a background by dropping `assets/backgrounds/<new_id>/bg.png` and appending a record to the manifest; `--background <new_id>` picks it up.

`palette_lab_mean` drives the `--polish` Lab match step. Pick a value by eyeballing the bg's mean in Lab, or compute via `cv2.cvtColor(bg, cv2.COLOR_BGR2LAB).mean(axis=(0,1))`.

## Smoke tests

```bash
COMFY_URL=http://127.0.0.1:8188 uv run --no-project \
    pytest tests/group_photobooth/test_smoke_e2e.py -v -s
```

Expect 2 passed; first cold run ~60 s for N=1 (face render + cutout + SAM2 + INSPYRENET model loads), ~2× that for N=2. Subsequent warm runs ~30 s/face.

The two ControlNet weights referenced by `photobooth_zimage_cn.api.json` (`models/vae/z_image_ae.safetensors`, `models/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors`) live on the Seagate Hub archive at `/media/newub/Seagate Hub/comfyui-models/` and need to be present (real file or symlink) in those ComfyUI subdirs. ComfyUI's `ModelPatchLoader` reads `models/model_patches/` — placing the CN file in `models/controlnet/` is not sufficient.

## Cross-references

- `docs/research/_topics/photobooth-sweep.md` — current photobooth recipe + Phase 2/3 findings.
- `docs/research/2026-05-21-comfyui-compositor-toolkit.md` — node-pack survey that informed the Python/ComfyUI split (Python-side YOLO + Pillow composite, ComfyUI for SAM2 + INSPYRENET).
- `docs/photobooth-sbom.md` — full dependency manifest (custom nodes, model weights, Python deps).
