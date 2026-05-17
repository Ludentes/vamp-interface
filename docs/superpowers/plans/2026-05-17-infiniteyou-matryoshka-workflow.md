# InfiniteYou Matryoshka Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Render identity-preserving single matryoshka dolls from photos using InfiniteYou (InfuseNet) for identity, stacked with a fixed Canny doll-silhouette ControlNet for structure, on FLUX.1-dev.

**Architecture:** A ComfyUI API-format workflow JSON wires `flux1-dev-fp8` → DualCLIP → Canny `ControlNetApplyAdvanced` → `InfuseNetApply` → KSampler → SaveImage. InfuseNet chains after the Canny ControlNet via prev-controlnet, so structure and identity compose without a custom node. A sweep script cloned from `matryoshka_sweep.py` drives a deterministic grid over identity / InfuseNet strength / InfuseNet end-percent.

**Tech Stack:** Python 3.12, ComfyUI REST API, FLUX.1-dev fp8, InfiniteYou `ComfyUI_InfiniteYou` node, InstantX FLUX Union ControlNet, insightface antelopev2.

Spec: `docs/superpowers/specs/2026-05-17-infiniteyou-matryoshka-workflow-design.md`

---

## File Structure

- Create: `comfyui/workflows/flux_infiniteyou_canny.api.json` — the API-format node graph (placeholders `$$...` substituted per cell).
- Create: `scripts/matryoshka_infiniteyou_sweep.py` — the sweep harness (cloned from `scripts/matryoshka_sweep.py`).
- Output: `data/importer/refs_matryoshka_infu/chibi/*.png`, manifest `data/importer/manifest_matryoshka_infu.parquet`.

---

## Task 1: Author the InfiniteYou API workflow JSON

**Files:**
- Create: `comfyui/workflows/flux_infiniteyou_canny.api.json`

The graph in ComfyUI API format (a flat dict of node-id → `{class_type, inputs}`).
Node-id assignment is fixed below so the sweep's injection assertion is stable.
`$$...` tokens are substituted by the sweep script.

- [ ] **Step 1: Write the workflow JSON**

```json
{
  "1":  {"class_type": "UNETLoader", "inputs": {"unet_name": "FLUX1/flux1-dev-fp8.safetensors", "weight_dtype": "fp8_e4m3fn"}},
  "2":  {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": "t5/t5xxl_fp8_e4m3fn.safetensors", "clip_name2": "clip_l.safetensors", "type": "flux", "device": "default"}},
  "3":  {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1/ae.safetensors"}},
  "4":  {"class_type": "CLIPTextEncodeFlux", "inputs": {"clip": ["2", 0], "clip_l": "$$POSITIVE_PROMPT", "t5xxl": "$$POSITIVE_PROMPT", "guidance": 3.5}},
  "5":  {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "$$NEGATIVE_PROMPT"}},
  "6":  {"class_type": "LoadImage", "inputs": {"image": "$$CANNY_FILENAME"}},
  "7":  {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "FLUX.1/instantx-union/diffusion_pytorch_model.safetensors"}},
  "8":  {"class_type": "ControlNetApplyAdvanced", "inputs": {"positive": ["4", 0], "negative": ["5", 0], "control_net": ["7", 0], "image": ["6", 0], "strength": "$$CN_STRENGTH", "start_percent": "$$CN_START", "end_percent": "$$CN_END", "vae": ["3", 0]}},
  "9":  {"class_type": "LoadImage", "inputs": {"image": "$$IDENTITY_FILENAME"}},
  "10": {"class_type": "IDEmbeddingModelLoader", "inputs": {"image_proj_model_name": "sim_stage1/image_proj_model.bin", "image_proj_num_tokens": 8, "face_analysis_provider": "CUDA", "face_analysis_det_size": "AUTO"}},
  "11": {"class_type": "ExtractIDEmbedding", "inputs": {"face_detector": ["10", 0], "arcface_model": ["10", 1], "image_proj_model": ["10", 2], "image": ["9", 0]}},
  "12": {"class_type": "InfuseNetLoader", "inputs": {"controlnet_name": "sim_stage1/infusenet_sim_fp8e4m3fn.safetensors"}},
  "13": {"class_type": "EmptyImage", "inputs": {"width": 864, "height": 1152, "batch_size": 1, "color": 0}},
  "14": {"class_type": "InfuseNetApply", "inputs": {"positive": ["8", 0], "negative": ["8", 1], "id_embedding": ["11", 0], "control_net": ["12", 0], "image": ["13", 0], "vae": ["3", 0], "strength": "$$INFU_STRENGTH", "start_percent": "$$INFU_START", "end_percent": "$$INFU_END"}},
  "15": {"class_type": "EmptyLatentImage", "inputs": {"width": 864, "height": 1152, "batch_size": 1}},
  "16": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["14", 0], "negative": ["14", 1], "latent_image": ["15", 0], "seed": "$$SEED", "steps": 20, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta", "denoise": 1.0}},
  "17": {"class_type": "VAEDecode", "inputs": {"samples": ["16", 0], "vae": ["3", 0]}},
  "18": {"class_type": "SaveImage", "inputs": {"images": ["17", 0], "filename_prefix": "$$OUTPUT_PREFIX"}}
}
```

- [ ] **Step 2: Validate it is well-formed JSON**

Run: `python3 -c "import json; json.load(open('comfyui/workflows/flux_infiniteyou_canny.api.json'))" && echo OK`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add comfyui/workflows/flux_infiniteyou_canny.api.json
git commit -m "feat(matryoshka): InfiniteYou + Canny API workflow"
```

---

## Task 2: Verify ControlNet and InfuseNet weight filenames resolve

Before writing the sweep, confirm the exact filenames the workflow references
exist in the ComfyUI tree. The InstantX union filename in particular is a guess.

- [ ] **Step 1: List the actual files**

Run:
```bash
ls /home/newub/w/ComfyUI/models/controlnet/FLUX.1/instantx-union/
ls /home/newub/w/ComfyUI/models/infinite_you/sim_stage1/
ls /home/newub/w/ComfyUI/models/diffusion_models/FLUX1/
```
Expected: a `.safetensors` under `instantx-union/`; `infusenet_sim_fp8e4m3fn.safetensors` + `image_proj_model.bin` under `sim_stage1/`; `flux1-dev-fp8.safetensors` under `FLUX1/`.

- [ ] **Step 2: Fix any mismatched filename in the workflow JSON**

If the `instantx-union/` file is not named `diffusion_pytorch_model.safetensors`,
edit node `"7"`'s `control_net_name` in `comfyui/workflows/flux_infiniteyou_canny.api.json`
to the real path (relative to `models/controlnet/`). Same check for nodes `"1"`, `"12"`, `"10"`.

- [ ] **Step 3: Commit if changed**

```bash
git add comfyui/workflows/flux_infiniteyou_canny.api.json
git commit -m "fix(matryoshka): correct InfiniteYou workflow weight paths" || echo "no change"
```

---

## Task 3: Write the sweep script

**Files:**
- Create: `scripts/matryoshka_infiniteyou_sweep.py`

Clone `scripts/matryoshka_sweep.py`. Reuse verbatim: `_retry`, `queue`, `wait`,
`download`. Replace the grid, the substitution map, and the node assertion.

- [ ] **Step 1: Write the script**

```python
"""Matryoshka InfiniteYou sweep: identity-preserving single nesting doll.

InfiniteYou (InfuseNet) replaces PuLID for identity. InfuseNet chains after a
fixed Canny doll-silhouette ControlNet, so doll structure and face identity
compose. Base = FLUX.1-dev fp8, sim_stage1 variant.

Grid: identity x infusenet_strength x infusenet_end x 2 seeds = 48 cells.
Canny is the SAME doll-silhouette template for every cell (structure tool).
Manifest written up front, resumable (skip-if-exists), atomic PNG writes.

Usage (cwd = data/importer):
    python scripts/matryoshka_infiniteyou_sweep.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow ../../comfyui/workflows/flux_infiniteyou_canny.api.json \\
        --id-dir identities_flux --template refs/matryoshka/template_canny.png \\
        --out refs_matryoshka_infu --manifest manifest_matryoshka_infu.parquet \\
        --comfy-input-dir /home/newub/w/ComfyUI/input
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import requests

WORKFLOW_VERSION = "matryoshka_infu_2026-05-17"
CANNY_TEMPLATE = "matryoshka_template_canny.png"
NEGATIVE_PROMPT = (
    "hands, body, full body, multiple people, watermark, text, low quality, "
    "blurry, deformed, distorted, signature"
)
BASE_PROMPT = ("a Russian matryoshka nesting doll, single doll, painted "
               "wooden figure, glossy lacquer finish, floral folk-art shawl "
               "and apron, flat painted face, plain background, centered")
STYLE_SUFFIX = (", traditional khokhloma painting, rosy painted cheeks, "
                "hand-painted detail")
PROMPT = BASE_PROMPT + STYLE_SUFFIX

CN_STRENGTH = 0.5
CN_START, CN_END = 0.0, 0.5
INFU_START = 0.0

INFU_STRENGTHS = [0.6, 0.8, 1.0]
INFU_ENDS = [0.5, 0.8]
IDENTITIES = [3, 8, 14, 15]
SEEDS_PER_CELL = 2

# Nodes whose literals the sweep overwrites -- asserted at startup so a
# re-exported (renumbered) workflow fails loud.
SCHEDULE_NODES = {"8": "ControlNetApplyAdvanced", "14": "InfuseNetApply",
                  "16": "KSampler"}


def build_grid() -> list[dict]:
    rows: list[dict] = []
    cell = 0
    for idx in IDENTITIES:
        for st in INFU_STRENGTHS:
            for en in INFU_ENDS:
                for _ in range(SEEDS_PER_CELL):
                    seed = 71_000_000 + cell * 7919
                    st_tag = f"st{int(round(st * 100)):03d}"
                    en_tag = f"en{int(round(en * 100)):03d}"
                    identity = f"id_{idx:02d}"
                    stem = f"{identity}_{st_tag}_{en_tag}_seed{seed}"
                    rows.append({
                        "render": f"{stem}.png", "stem": stem, "id_idx": idx,
                        "identity": identity, "infu_strength": st,
                        "infu_end": en, "infu_start": INFU_START,
                        "cn_strength": CN_STRENGTH, "cn_start": CN_START,
                        "cn_end": CN_END, "seed": seed, "prompt": PROMPT,
                        "workflow_version": WORKFLOW_VERSION,
                    })
                    cell += 1
    return rows


def build_workflow(template: dict, cell: dict, output_prefix: str) -> dict:
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": f"{cell['identity']}.png",
        "$$CANNY_FILENAME": CANNY_TEMPLATE,
        "$$POSITIVE_PROMPT": cell["prompt"],
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$SEED": int(cell["seed"]),
        "$$CN_STRENGTH": float(cell["cn_strength"]),
        "$$CN_START": float(cell["cn_start"]),
        "$$CN_END": float(cell["cn_end"]),
        "$$INFU_STRENGTH": float(cell["infu_strength"]),
        "$$INFU_START": float(cell["infu_start"]),
        "$$INFU_END": float(cell["infu_end"]),
        "$$OUTPUT_PREFIX": output_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items() if not k.startswith("_")}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    return _sub(copy.deepcopy(template))


def _retry(fn, *, tries: int = 3, what: str = ""):
    for attempt in range(tries):
        try:
            return fn()
        except (requests.RequestException, ConnectionError) as e:
            if attempt == tries - 1:
                raise
            print(f"  [retry {attempt+1}/{tries}] {what}: {e}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def queue(sess: requests.Session, url: str, wf: dict, client_id: str) -> str:
    def _do():
        r = sess.post(f"{url}/prompt", json={"prompt": wf, "client_id": client_id},
                      timeout=30)
        r.raise_for_status()
        return r.json()["prompt_id"]
    return _retry(_do, what="queue")


def wait(sess: requests.Session, url: str, pid: str, timeout: float = 420) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = _retry(lambda: sess.get(f"{url}/history/{pid}", timeout=10), what="history")
        if r is not None and r.status_code == 200 and pid in r.json():
            return r.json()[pid]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {pid} did not complete within {timeout}s")


def download(sess: requests.Session, url: str, entry: dict, out_path: Path) -> bool:
    status = entry.get("status", {})
    if status.get("status_str") not in (None, "success"):
        print(f"  [fail] {out_path.stem}: ComfyUI status={status.get('status_str')}")
        return False
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            r = _retry(lambda: sess.get(f"{url}/view", params={
                "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output")}, timeout=30), what="view")
            if r is None or r.status_code != 200 or r.content[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".png.tmp")
            tmp.write_bytes(r.content)
            os.replace(tmp, out_path)
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--id-dir", type=Path, default=Path("identities_flux"))
    ap.add_argument("--template", type=Path,
                    default=Path("refs/matryoshka/template_canny.png"))
    ap.add_argument("--out", type=Path, default=Path("refs_matryoshka_infu"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("manifest_matryoshka_infu.parquet"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="render at most N cells (0 = all); for smoke tests")
    args = ap.parse_args()

    template = json.loads(args.workflow.read_text())
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(f"workflow node {nid} is {got!r}, expected {cls!r}")

    chibi_dir = args.out / "chibi"
    chibi_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    pd.DataFrame([{k: v for k, v in c.items() if k != "stem"} for c in grid]
                 ).to_parquet(args.manifest, index=False)
    print(f"[sweep] {len(grid)} cells -> manifest {args.manifest}")

    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        if args.template.exists():
            shutil.copy2(args.template, args.comfy_input_dir / CANNY_TEMPLATE)
        else:
            raise SystemExit(f"doll template not found: {args.template}")
        for idx in IDENTITIES:
            src = args.id_dir / f"id_{idx:02d}.png"
            if src.exists():
                shutil.copy2(src, args.comfy_input_dir / src.name)
        print(f"[sweep] staged template + ID PNGs into {args.comfy_input_dir}")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for cell in grid:
        if args.limit and done >= args.limit:
            break
        out_png = chibi_dir / cell["render"]
        if out_png.exists():
            skipped += 1
            continue

        wf = build_workflow(template, cell,
                            output_prefix=f"matryoshka_infu/{cell['stem']}")
        t_start = time.time()
        try:
            pid = queue(sess, args.comfy_url, wf, client_id)
            entry = wait(sess, args.comfy_url, pid)
            if not download(sess, args.comfy_url, entry, out_png):
                failed += 1
                continue
        except Exception as e:
            print(f"  [fail] {cell['stem']}: {e}")
            failed += 1
            continue

        done += 1
        rate = done / max(time.time() - t0, 1) * 60
        print(f"  [ok] {cell['stem']} ({time.time()-t_start:.1f}s) "
              f"- {done} done, {skipped} skipped, {failed} failed, {rate:.1f}/min")

    print(f"[sweep] complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the script parses and the grid is 48 cells**

Run: `python3 -c "import sys; sys.path.insert(0,'scripts'); import matryoshka_infiniteyou_sweep as m; g=m.build_grid(); print(len(g), g[0]['stem'])"`
Expected: `48 id_03_st060_en050_seed71000000`

- [ ] **Step 3: Commit**

```bash
git add scripts/matryoshka_infiniteyou_sweep.py
git commit -m "feat(matryoshka): InfiniteYou identity sweep pipeline"
```

---

## Task 4: Pre-flight the identity PNGs

`ExtractIDEmbedding` raises if antelopev2 detects no face in the identity PNG.
Confirm all 4 identities are usable before the sweep.

- [ ] **Step 1: Run the face-detection pre-flight**

Run:
```bash
cd /home/newub/w/vamp-interface/data/importer && python3 -c "
from insightface.app import FaceAnalysis
app = FaceAnalysis(name='antelopev2', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640))
import cv2
for i in (3,8,14,15):
    img = cv2.imread(f'identities_flux/id_{i:02d}.png')
    n = len(app.get(img)) if img is not None else -1
    print(f'id_{i:02d}: {n} face(s)')
"
```
Expected: each line `id_NN: 1 face(s)` (or more). Any `0` or `-1` line means
that identity must be dropped from `IDENTITIES` in the sweep script before
running. If one fails, edit `IDENTITIES` in `scripts/matryoshka_infiniteyou_sweep.py`
to remove it and commit `fix(matryoshka): drop no-face identity from InfiniteYou sweep`.

---

## Task 5: Smoke test — one render

- [ ] **Step 1: Confirm ComfyUI is up**

Run: `curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8188/system_stats`
Expected: `200`. If not, start ComfyUI: `cd /home/newub/w/ComfyUI && nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &` and wait ~20 s.

- [ ] **Step 2: Run a single-cell smoke test**

Run:
```bash
cd /home/newub/w/vamp-interface/data/importer && python3 ../../scripts/matryoshka_infiniteyou_sweep.py \
  --workflow ../../comfyui/workflows/flux_infiniteyou_canny.api.json \
  --comfy-input-dir /home/newub/w/ComfyUI/input \
  --limit 1
```
Expected: `[ok] id_03_st060_en050_seed71000000 (...)` then `complete: 1 done`.
The first render is slow (arcface weight auto-downloads via facexlib). If it
fails, read `/tmp/comfyui.log` for the node error and fix the workflow JSON.

- [ ] **Step 3: Inspect the render**

Read the PNG at `data/importer/refs_matryoshka_infu/chibi/id_03_st060_en050_seed71000000.png`.
Confirm it is a matryoshka doll (not a portrait, not noise) with a painted face.

---

## Task 6: Run the full sweep

- [ ] **Step 1: Run all 48 cells**

Run:
```bash
cd /home/newub/w/vamp-interface/data/importer && python3 ../../scripts/matryoshka_infiniteyou_sweep.py \
  --workflow ../../comfyui/workflows/flux_infiniteyou_canny.api.json \
  --comfy-input-dir /home/newub/w/ComfyUI/input
```
Expected: `complete: 47 done, 1 skipped, 0 failed` (the smoke-test cell is skipped). ~12 min.

- [ ] **Step 2: Confirm output count**

Run: `ls data/importer/refs_matryoshka_infu/chibi/*.png | wc -l`
Expected: `48`

---

## Task 7: Build the comparison montage

**Files:**
- Create: `scripts/matryoshka_infu_montage.py`

A per-identity contact sheet: rows = identity, columns = (strength × end)
cells, so identity transfer through the flat-paint style is visible at a glance.

- [ ] **Step 1: Write the montage script**

```python
"""Contact sheet for the InfiniteYou matryoshka sweep.

One row per identity, columns ordered by (infu_strength, infu_end). Each tile
is labelled with its stem. Output a single PNG per identity plus a combined
grid, written to refs_matryoshka_infu/montage/.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHIBI = ROOT / "data" / "importer" / "refs_matryoshka_infu" / "chibi"
OUT = ROOT / "data" / "importer" / "refs_matryoshka_infu" / "montage"
IDENTITIES = [3, 8, 14, 15]
STRENGTHS = [60, 80, 100]
ENDS = [50, 80]
TILE = 320


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx in IDENTITIES:
        cols = []
        for st in STRENGTHS:
            for en in ENDS:
                # first seed of the cell
                hits = sorted(CHIBI.glob(
                    f"id_{idx:02d}_st{st:03d}_en{en:03d}_seed*.png"))
                if hits:
                    tile = cv2.resize(cv2.imread(str(hits[0])), (TILE, TILE))
                else:
                    tile = np.full((TILE, TILE, 3), 64, np.uint8)
                cols.append(label(tile, f"id{idx:02d} st{st} en{en}"))
        rows.append(np.hstack(cols))
    grid = np.vstack(rows)
    cv2.imwrite(str(OUT / "infu_sweep_grid.png"), grid)
    print(f"wrote {OUT / 'infu_sweep_grid.png'} ({grid.shape[1]}x{grid.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `cd /home/newub/w/vamp-interface && python3 scripts/matryoshka_infu_montage.py`
Expected: `wrote .../infu_sweep_grid.png (...)`

- [ ] **Step 3: Inspect the grid**

Read `data/importer/refs_matryoshka_infu/montage/infu_sweep_grid.png`. Judge:
does identity transfer through the flat-painted style (the thing PuLID failed)?
Does higher `infu_strength` increase likeness without breaking the doll form?

- [ ] **Step 4: Commit**

```bash
git add scripts/matryoshka_infu_montage.py
git commit -m "feat(matryoshka): InfiniteYou sweep comparison montage"
```

---

## Task 8: Record the verdict

**Files:**
- Create: `docs/research/2026-05-17-infiniteyou-matryoshka-results.md`

- [ ] **Step 1: Write the results doc**

Frontmatter `status: live`, `topic: arkit-controlnet`. Record: which
`(infu_strength, infu_end)` cell gave the best identity-through-flat-paint
result, whether InfiniteYou beat PuLID and the inswapper face-swap, the montage
path, and the recommended settings for matryoshka v2. If InfiniteYou also
failed, say so plainly and note what the failure looked like.

- [ ] **Step 2: Commit**

```bash
git add docs/research/2026-05-17-infiniteyou-matryoshka-results.md
git commit -m "docs: InfiniteYou matryoshka sweep results"
```

---

## Self-Review

**Spec coverage:** node graph → Task 1; base-model/variant/control-image
decisions → Task 1 JSON literals; sweep axes (identity/strength/end) → Task 3
grid; identity-image precondition → Task 4; smoke test → Task 5; sweep → Task 6;
comparison montage → Task 7; verdict → Task 8. All spec sections covered.

**Placeholder scan:** no TBD/TODO; all code blocks complete; Task 2 handles the
one genuine unknown (the InstantX union filename) by listing then fixing.

**Type consistency:** node IDs `8`/`14`/`16` in the JSON match `SCHEDULE_NODES`
in the sweep script; `$$` tokens in the JSON match the `subs` dict keys in
`build_workflow`; output dir `refs_matryoshka_infu` consistent across Tasks 3,
6, 7; stem format `id_NN_stXXX_enXXX_seedN` consistent between `build_grid` and
the montage glob.
