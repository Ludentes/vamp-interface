# Matryoshka Identity-Swap Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a 252-cell sweep on the Windows ComfyUI box that generates a generic matryoshka doll with Flux-Krea + PuLID and swaps a real identity onto its painted face with `inswapper_128`.

**Architecture:** Two new scripts. `scripts/swap_core.py` is a sweep-agnostic detect-and-swap module (insightface SCRFD with a MediaPipe fallback, CPU-only). `scripts/matryoshka_swap_sweep.py` is the runner — it reuses `matryoshka_sweep.py`'s ComfyUI helpers for doll generation, then calls `swap_core` per cell. The runner ships to the Windows box and runs as a Scheduled Task.

**Tech Stack:** Python 3.12 (local `.venv`) / 3.10 (Windows ComfyUI venv), insightface `buffalo_l` + `inswapper_128.onnx`, mediapipe, ComfyUI REST, pandas/pyarrow, pytest.

**Spec:** `docs/superpowers/specs/2026-05-16-matryoshka-swap-sweep-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `scripts/swap_core.py` (create) | Reusable detect + swap. No sweep knowledge. |
| `scripts/matryoshka_swap_sweep.py` (create) | 252-cell grid, ComfyUI loop, per-cell swap, manifest. |
| `scripts/matryoshka_swap_montage.py` (create) | Post-sweep per-anchor contact sheets. |
| `tests/test_swap_core.py` (create) | Unit tests for swap_core logic (fakes, no ML models). |
| `tests/test_matryoshka_swap_grid.py` (create) | Unit tests for the 252-cell grid builder. |
| `scripts/matryoshka_sweep.py` (reuse, unchanged) | Source of `queue`, `wait`, `download`, `build_workflow`, `_retry`, `NEGATIVE_PROMPT`, `SCHEDULE_NODES`. |

All tests import scripts by inserting `scripts/` on `sys.path` (the repo's scripts are not a package). Run tests with `.venv/bin/python -m pytest`.

---

### Task 1: swap_core.py — detect and swap

**Files:**
- Create: `scripts/swap_core.py`
- Test: `tests/test_swap_core.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_swap_core.py`:

```python
"""Unit tests for swap_core -- logic only, with fakes. No ML models."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import swap_core  # noqa: E402


class FakeFace:
    def __init__(self, det_score):
        self.det_score = det_score


class FakeApp:
    """Stand-in for insightface FaceAnalysis."""
    def __init__(self, faces):
        self._faces = faces

    def get(self, img):
        return list(self._faces)


class FakeSwapper:
    """Stand-in for inswapper -- echoes the doll, records the call."""
    def __init__(self):
        self.calls = []

    def get(self, img, target, source, paste_back=True):
        self.calls.append((target, source))
        return img


def _doll():
    return np.zeros((1024, 1024, 3), dtype=np.uint8)


def test_detect_source_picks_highest_score():
    app = FakeApp([FakeFace(0.4), FakeFace(0.9), FakeFace(0.7)])
    face = swap_core.detect_source(app, _doll())
    assert face.det_score == 0.9


def test_detect_source_none_when_no_faces():
    assert swap_core.detect_source(FakeApp([]), _doll()) is None


def test_swap_identity_default_mode_when_scrfd_hits():
    app = FakeApp([FakeFace(0.83)])
    swapper = FakeSwapper()
    result, mode, score = swap_core.swap_identity(
        app, swapper, _doll(), source_face="SRC")
    assert mode == "default"
    assert score == pytest.approx(0.83)
    assert len(swapper.calls) == 1
    assert swapper.calls[0][1] == "SRC"


def test_swap_identity_forced_when_scrfd_misses(monkeypatch):
    kps = np.array([[1, 1], [2, 1], [1.5, 2], [1, 3], [2, 3]],
                   dtype=np.float32)
    bbox = np.array([0, 0, 3, 4], dtype=np.float32)
    monkeypatch.setattr(swap_core, "mediapipe_kps_bbox",
                        lambda img: (kps, bbox))
    app = FakeApp([])
    swapper = FakeSwapper()
    result, mode, score = swap_core.swap_identity(
        app, swapper, _doll(), source_face="SRC")
    assert mode == "forced"
    assert score == 0.0
    assert len(swapper.calls) == 1


def test_swap_identity_failed_when_both_miss(monkeypatch):
    monkeypatch.setattr(swap_core, "mediapipe_kps_bbox",
                        lambda img: (None, None))
    doll = _doll()
    swapper = FakeSwapper()
    result, mode, score = swap_core.swap_identity(
        FakeApp([]), swapper, doll, source_face="SRC")
    assert mode == "failed"
    assert score == 0.0
    assert result is doll
    assert swapper.calls == []


def test_mediapipe_kps_bbox_unavailable_returns_none(monkeypatch):
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    kps, bbox = swap_core.mediapipe_kps_bbox(_doll())
    assert kps is None and bbox is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_swap_core.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'swap_core'`.

- [ ] **Step 3: Implement scripts/swap_core.py**

```python
"""Detect-and-swap core for the matryoshka identity pipeline.

No sweep knowledge. PuLID generates a generic doll; this module detects the
doll's painted face (insightface SCRFD, MediaPipe fallback) and swaps a real
identity onto it with inswapper_128. CPU-only -- the swap never contends
with ComfyUI for GPU VRAM.
"""
from __future__ import annotations

import cv2
import numpy as np

_CPU = ["CPUExecutionProvider"]


def make_face_app():
    """insightface buffalo_l, detection + recognition, CPU only."""
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l",
                       allowed_modules=["detection", "recognition"],
                       providers=_CPU)
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def load_swapper(model_path):
    """inswapper_128 swapper model, CPU only."""
    from insightface.model_zoo import get_model
    return get_model(model_path, download=False, providers=_CPU)


def detect_source(app, img_bgr):
    """Highest-det_score face in img_bgr, or None if no face."""
    faces = app.get(img_bgr)
    if not faces:
        return None
    return max(faces, key=lambda f: f.det_score)


def mediapipe_kps_bbox(img_bgr):
    """(kps[5,2], bbox[4]) from MediaPipe FaceMesh, else (None, None).

    Fallback landmarker for doll faces SCRFD cannot see. kps order is
    [eyeL, eyeR, nose, mouthL, mouthR] (arcface convention, image-left
    first). Returns (None, None) when mediapipe.solutions is unavailable --
    the caller treats that as 'fallback unavailable', it never crashes.
    """
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh
    except (ImportError, AttributeError):
        return None, None

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_bgr.shape[:2]
    with face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                            refine_landmarks=True,
                            min_detection_confidence=0.1) as fm:
        res = fm.process(rgb)
    if not res.multi_face_landmarks:
        return None, None
    lm = res.multi_face_landmarks[0].landmark

    def pt(i):
        return np.array([lm[i].x * w, lm[i].y * h])

    eye_a = np.mean([pt(i) for i in range(468, 473)], axis=0)  # iris rings
    eye_b = np.mean([pt(i) for i in range(473, 478)], axis=0)
    nose, m_a, m_b = pt(1), pt(61), pt(291)
    eyes = sorted([eye_a, eye_b], key=lambda p: p[0])
    mouth = sorted([m_a, m_b], key=lambda p: p[0])
    kps = np.array([eyes[0], eyes[1], nose, mouth[0], mouth[1]],
                   dtype=np.float32)
    xs = [lm[i].x * w for i in range(468)]   # 0..467 = face mesh, no iris
    ys = [lm[i].y * h for i in range(468)]
    bbox = np.array([min(xs), min(ys), max(xs), max(ys)], dtype=np.float32)
    return kps, bbox


def swap_identity(app, swapper, doll_bgr, source_face):
    """Swap source_face's identity onto the doll.

    Returns (result_bgr, mode, det_score). mode is 'default' (SCRFD found
    the doll face), 'forced' (MediaPipe synthetic-kps fallback), or 'failed'
    (neither -- the un-swapped doll is returned unchanged).
    """
    det = app.get(doll_bgr)
    if det:
        target = max(det, key=lambda f: f.det_score)
        result = swapper.get(doll_bgr.copy(), target, source_face,
                             paste_back=True)
        return result, "default", float(target.det_score)

    kps, bbox = mediapipe_kps_bbox(doll_bgr)
    if kps is not None:
        from insightface.app.common import Face
        target = Face(bbox=bbox, kps=kps, det_score=1.0)
        result = swapper.get(doll_bgr.copy(), target, source_face,
                             paste_back=True)
        return result, "forced", 0.0

    return doll_bgr, "failed", 0.0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_swap_core.py -v`
Expected: PASS — 6 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/swap_core.py tests/test_swap_core.py
git commit -m "feat(matryoshka): swap_core detect-and-swap module

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: matryoshka_swap_sweep.py — the 252-cell grid builder

**Files:**
- Create: `scripts/matryoshka_swap_sweep.py` (grid builder + constants only in this task)
- Test: `tests/test_matryoshka_swap_grid.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matryoshka_swap_grid.py`:

```python
"""Unit tests for the matryoshka swap-sweep grid builder."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matryoshka_swap_sweep as mss  # noqa: E402


def test_grid_has_252_cells():
    assert len(mss.build_grid()) == 252


def test_grid_covers_all_21_anchors():
    anchors = {c["anchor"] for c in mss.build_grid()}
    assert anchors == {"id_user"} | {f"id_{i:02d}" for i in range(20)}


def test_grid_covers_all_axes():
    grid = mss.build_grid()
    assert {c["finish"] for c in grid} == {"glossy", "satin", "matte"}
    assert {c["cn_strength"] for c in grid} == {0.0, 0.5}
    assert {c["pulid_start"] for c in grid} == {0.0, 0.1}


def test_pulid_weight_and_end_are_fixed():
    for c in mss.build_grid():
        assert c["pulid_weight"] == 1.5
        assert c["pulid_end"] == 1.0


def test_seeds_are_unique_and_deterministic():
    grid = mss.build_grid()
    seeds = [c["seed"] for c in grid]
    assert len(set(seeds)) == 252
    assert grid[0]["seed"] == 70_000_000
    assert grid[1]["seed"] == 70_000_000 + 7919


def test_stem_format_and_uniqueness():
    grid = mss.build_grid()
    stems = {c["stem"] for c in grid}
    assert len(stems) == 252
    c0 = grid[0]
    assert c0["stem"] == (f"{c0['anchor']}_{c0['finish']}"
                          f"_cn{int(round(c0['cn_strength']*100)):03d}"
                          f"_ps{int(round(c0['pulid_start']*100)):03d}"
                          f"_seed{c0['seed']}")


def test_prompt_contains_finish_token():
    by_finish = {c["finish"]: c["prompt"] for c in mss.build_grid()}
    assert "glossy lacquer finish" in by_finish["glossy"]
    assert "satin lacquer finish" in by_finish["satin"]
    assert "matte painted finish" in by_finish["matte"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_matryoshka_swap_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matryoshka_swap_sweep'`.

- [ ] **Step 3: Implement the grid builder in scripts/matryoshka_swap_sweep.py**

Create `scripts/matryoshka_swap_sweep.py` with this content (the runner `main()` is added in Task 3):

```python
"""Matryoshka identity-swap sweep: PuLID doll + inswapper identity.

The PuLID-weight ladder proved PuLID cannot transfer identity through the
flat-painted matryoshka style -- but a PuLID doll IS a coherent doll with a
detectable face, and inswapper swaps a real identity onto it cleanly. Roles
split: PuLID generates the doll, inswapper is the identity vehicle.

This sweep is 252 cells: 21 anchors x 3 prompt finishes x 2 CN strengths x
2 PuLID starts, one deterministic seed each. Recipe fixed: PuLID weight 1.5,
SCRFD-default swap, no repaint. Run ON the ComfyUI box, cwd = data/importer:

    python ../../scripts/matryoshka_swap_sweep.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow workflows/flux_pulid_canny_lora.api.json \\
        --id-dir identities_flux --template refs/matryoshka/template_canny.png \\
        --swapper C:/comfy/ComfyUI/models/insightface/inswapper_128.onnx \\
        --out refs_matryoshka --comfy-input-dir C:/comfy/ComfyUI/input
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import pandas as pd
import requests

from matryoshka_sweep import (
    CANNY_TEMPLATE,
    NEGATIVE_PROMPT,
    SCHEDULE_NODES,
    build_workflow,
    download,
    queue,
    wait,
)
from swap_core import detect_source, load_swapper, make_face_app, swap_identity

WORKFLOW_VERSION = "matryoshka_swap_2026-05-16"

# anchors: the user photo first, then id_00..id_19
ANCHORS = ["id_user"] + [f"id_{i:02d}" for i in range(20)]

# prompt finish: substitutes the lacquer token (the "too glossy" axis)
BASE_PROMPT = ("a Russian matryoshka nesting doll, single doll, painted "
               "wooden figure, {finish}, floral folk-art shawl and apron, "
               "flat painted face, plain background, centered")
STYLE_SUFFIX = (", traditional khokhloma painting, rosy painted cheeks, "
                "hand-painted detail")
FINISHES = {
    "glossy": "glossy lacquer finish",
    "satin": "satin lacquer finish",
    "matte": "matte painted finish",
}

# fixed recipe (NOT swept)
PULID_WEIGHT = 1.5
PULID_END = 1.0
CN_START, CN_END = 0.0, 0.5

# swept axes
FINISH_KEYS = ["glossy", "satin", "matte"]
CN_STRENGTHS = [0.0, 0.5]
PULID_STARTS = [0.0, 0.1]

SEED_BASE = 70_000_000


def make_prompt(finish_key: str) -> str:
    return BASE_PROMPT.format(finish=FINISHES[finish_key]) + STYLE_SUFFIX


def build_grid() -> list[dict]:
    """Deterministic, ordered 252-cell grid. cell index drives the seed."""
    rows: list[dict] = []
    cell = 0
    for anchor in ANCHORS:
        for finish in FINISH_KEYS:
            for cn in CN_STRENGTHS:
                for ps in PULID_STARTS:
                    seed = SEED_BASE + cell * 7919
                    stem = (f"{anchor}_{finish}"
                            f"_cn{int(round(cn * 100)):03d}"
                            f"_ps{int(round(ps * 100)):03d}_seed{seed}")
                    rows.append({
                        "cell": cell, "anchor": anchor, "identity": anchor,
                        "finish": finish, "prompt": make_prompt(finish),
                        "cn_strength": cn, "pulid_start": ps,
                        "pulid_weight": PULID_WEIGHT, "pulid_end": PULID_END,
                        "cn_start": CN_START, "cn_end": CN_END,
                        "seed": seed, "stem": stem,
                        "workflow_version": WORKFLOW_VERSION,
                    })
                    cell += 1
    return rows
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_matryoshka_swap_grid.py -v`
Expected: PASS — 7 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/matryoshka_swap_sweep.py tests/test_matryoshka_swap_grid.py
git commit -m "feat(matryoshka): swap-sweep 252-cell grid builder

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: matryoshka_swap_sweep.py — the runner

**Files:**
- Modify: `scripts/matryoshka_swap_sweep.py` (append `main()` and the `__main__` guard)

No unit test — the runner needs a live ComfyUI. It is exercised by the
Task 5 smoke test (`--limit 1`) and the full Task 6 run. Correctness is
guarded by pre-flight asserts that fail before any compute is spent.

- [ ] **Step 1: Append the runner to scripts/matryoshka_swap_sweep.py**

Append this to the file created in Task 2:

```python
def _preflight(args, grid) -> dict:
    """Fail loud before any compute. Returns the parsed workflow template."""
    template = json.loads(args.workflow.read_text())
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(f"workflow node {nid} is {got!r}, "
                             f"expected {cls!r}")
    for anchor in {c["anchor"] for c in grid}:
        if not (args.id_dir / f"{anchor}.png").exists():
            raise SystemExit(f"missing anchor PNG: "
                             f"{args.id_dir / (anchor + '.png')}")
    if not Path(args.swapper).exists():
        raise SystemExit(f"inswapper model not found: {args.swapper}")
    if not args.template.exists():
        raise SystemExit(f"doll template not found: {args.template}")
    try:
        requests.get(f"{args.comfy_url}/system_stats", timeout=10
                     ).raise_for_status()
    except requests.RequestException as e:
        raise SystemExit(f"ComfyUI unreachable at {args.comfy_url}: {e}")
    return template


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--id-dir", type=Path, default=Path("identities_flux"))
    ap.add_argument("--template", type=Path,
                    default=Path("refs/matryoshka/template_canny.png"))
    ap.add_argument("--swapper", required=True,
                    help="path to inswapper_128.onnx")
    ap.add_argument("--out", type=Path, default=Path("refs_matryoshka"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="process at most N cells (0 = all) -- smoke test")
    args = ap.parse_args()

    grid = build_grid()
    if args.limit > 0:
        grid = grid[:args.limit]
    template = _preflight(args, grid)
    print(f"[swap-sweep] {len(grid)} cells, preflight ok")

    sweep_dir = args.out / "swap_sweep"
    dolls_dir = sweep_dir / "dolls"
    swapped_dir = sweep_dir / "swapped"
    dolls_dir.mkdir(parents=True, exist_ok=True)
    swapped_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest_swap_sweep.parquet"

    # stage the fixed Canny template + every anchor PNG into ComfyUI input
    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.template, args.comfy_input_dir / CANNY_TEMPLATE)
        for anchor in {c["anchor"] for c in grid}:
            shutil.copy2(args.id_dir / f"{anchor}.png",
                         args.comfy_input_dir / f"{anchor}.png")
        print(f"[swap-sweep] staged template + anchors into "
              f"{args.comfy_input_dir}")

    # resumable manifest: keep prior rows so a resumed run stays complete
    manifest_rows: list[dict] = []
    if manifest_path.exists():
        manifest_rows = pd.read_parquet(manifest_path).to_dict("records")
    done_stems = {r["stem"] for r in manifest_rows}

    app = make_face_app()
    swapper = load_swapper(args.swapper)
    source_cache: dict[str, object] = {}    # anchor -> source Face

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for cell in grid:
        stem = cell["stem"]
        swapped_png = swapped_dir / f"{stem}.png"
        if stem in done_stems and swapped_png.exists():
            skipped += 1
            continue

        t_start = time.time()
        doll_png = dolls_dir / f"{stem}.png"
        try:
            if not doll_png.exists():
                wf = build_workflow(
                    template, cell,
                    output_prefix=f"matryoshka/swap_sweep/{stem}")
                pid = queue(sess, args.comfy_url, wf, client_id)
                entry = wait(sess, args.comfy_url, pid)
                if not download(sess, args.comfy_url, entry, doll_png):
                    print(f"  [fail] {stem}: ComfyUI produced no image")
                    failed += 1
                    continue
        except Exception as e:                       # noqa: BLE001
            print(f"  [fail] {stem}: generation: {e}")
            failed += 1
            continue

        # swap stage
        import cv2
        anchor = cell["anchor"]
        if anchor not in source_cache:
            src_img = cv2.imread(str(args.id_dir / f"{anchor}.png"))
            source_cache[anchor] = detect_source(app, src_img)
        source_face = source_cache[anchor]
        if source_face is None:
            print(f"  [fail] {stem}: no face in anchor {anchor}")
            failed += 1
            continue

        doll = cv2.imread(str(doll_png))
        result, mode, score = swap_identity(app, swapper, doll, source_face)
        tmp = swapped_png.with_suffix(".png.tmp")
        cv2.imwrite(str(tmp), result)
        os.replace(tmp, swapped_png)

        manifest_rows.append({
            "cell": cell["cell"], "anchor": anchor,
            "finish": cell["finish"], "cn_strength": cell["cn_strength"],
            "pulid_start": cell["pulid_start"],
            "pulid_weight": cell["pulid_weight"], "seed": cell["seed"],
            "doll_png": str(doll_png), "swapped_png": str(swapped_png),
            "swap_mode": mode, "swap_det_score": score,
            "workflow_version": WORKFLOW_VERSION,
        })
        pd.DataFrame(manifest_rows).to_parquet(manifest_path, index=False)
        done += 1
        rate = done / max(time.time() - t0, 1) * 60
        print(f"  [ok] {stem} ({time.time()-t_start:.1f}s) swap={mode} "
              f"score={score:.2f} - {done} done, {skipped} skipped, "
              f"{failed} failed, {rate:.1f}/min")

    print(f"[swap-sweep] complete: {done} done, {skipped} skipped, "
          f"{failed} failed in {(time.time()-t0)/60:.1f} min")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the module still imports and the grid tests still pass**

Run: `.venv/bin/python -m pytest tests/test_matryoshka_swap_grid.py -v`
Expected: PASS — 7 passed (appending `main()` must not break the grid).

- [ ] **Step 3: Verify the CLI parses**

Run: `.venv/bin/python scripts/matryoshka_swap_sweep.py --help`
Expected: argparse usage text listing `--comfy-url`, `--workflow`, `--swapper`, `--limit`, etc. (No network call — `--help` exits before `main()` body.)

- [ ] **Step 4: Commit**

```bash
git add scripts/matryoshka_swap_sweep.py
git commit -m "feat(matryoshka): swap-sweep runner -- ComfyUI loop + per-cell swap

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: matryoshka_swap_montage.py — post-sweep contact sheets

**Files:**
- Create: `scripts/matryoshka_swap_montage.py`

No unit test — pure I/O glue over the sweep outputs, verified by eyeballing
one produced sheet.

- [ ] **Step 1: Implement scripts/matryoshka_swap_montage.py**

```python
"""Per-anchor contact sheets for the matryoshka swap sweep.

One PNG per anchor: rows = prompt finish (glossy/satin/matte), columns =
the 4 (cn_strength, pulid_start) combinations, cells = the swapped doll.
Run after matryoshka_swap_sweep.py completes.

    python scripts/matryoshka_swap_montage.py \\
        --sweep-dir refs_matryoshka/swap_sweep
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

FINISHES = ["glossy", "satin", "matte"]
COLS = [(0.0, 0.0), (0.0, 0.1), (0.5, 0.0), (0.5, 0.1)]  # (cn, pulid_start)
CELL = 360
LABEL = 28


def _find(swapped_dir: Path, anchor: str, finish: str,
          cn: float, ps: float):
    cn_tag = f"cn{int(round(cn * 100)):03d}"
    ps_tag = f"ps{int(round(ps * 100)):03d}"
    prefix = f"{anchor}_{finish}_{cn_tag}_{ps_tag}_seed"
    hits = sorted(swapped_dir.glob(f"{prefix}*.png"))
    return hits[0] if hits else None


def build_sheet(swapped_dir: Path, anchor: str, out_path: Path) -> bool:
    rows, cols = len(FINISHES), len(COLS)
    canvas = np.full((rows * (CELL + LABEL), cols * CELL, 3), 240,
                     dtype=np.uint8)
    any_cell = False
    for r, finish in enumerate(FINISHES):
        for c, (cn, ps) in enumerate(COLS):
            x, y = c * CELL, r * (CELL + LABEL)
            cv2.rectangle(canvas, (x, y), (x + CELL, y + LABEL),
                          (20, 20, 20), -1)
            cv2.putText(canvas, f"{finish} cn{cn} ps{ps}", (x + 6, y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            path = _find(swapped_dir, anchor, finish, cn, ps)
            if path is None:
                continue
            im = cv2.resize(cv2.imread(str(path)), (CELL, CELL))
            canvas[y + LABEL:y + LABEL + CELL, x:x + CELL] = im
            any_cell = True
    if any_cell:
        cv2.imwrite(str(out_path), canvas)
    return any_cell


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path,
                    default=Path("refs_matryoshka/swap_sweep"))
    args = ap.parse_args()

    swapped_dir = args.sweep_dir / "swapped"
    montage_dir = args.sweep_dir / "montage"
    montage_dir.mkdir(parents=True, exist_ok=True)

    anchors = sorted({p.name.split("_")[0] + "_" + p.name.split("_")[1]
                      for p in swapped_dir.glob("*.png")})
    written = 0
    for anchor in anchors:
        out = montage_dir / f"{anchor}.png"
        if build_sheet(swapped_dir, anchor, out):
            written += 1
            print(f"  wrote {out}")
    print(f"[montage] {written} contact sheet(s) -> {montage_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Note: anchor stems are `id_user` and `id_NN` — both are two underscore-joined
tokens, so `split("_")[0] + "_" + split("_")[1]` recovers the anchor name.

- [ ] **Step 2: Verify the CLI parses**

Run: `.venv/bin/python scripts/matryoshka_swap_montage.py --help`
Expected: argparse usage text with `--sweep-dir`.

- [ ] **Step 3: Commit**

```bash
git add scripts/matryoshka_swap_montage.py
git commit -m "feat(matryoshka): swap-sweep per-anchor contact sheets

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Stage assets to the Windows box and smoke-test one cell

**Files:** none in-repo. SSH/scp to `videocard@192.168.87.25`.

The Windows ComfyUI venv is `C:\comfy\ComfyUI\venv`; repo scripts live at
`C:\comfy\importer\scripts\`; the sweep runs cwd `C:\comfy\importer\data\importer`.

- [ ] **Step 1: Copy the three scripts to the Windows scripts dir**

```bash
scp scripts/swap_core.py scripts/matryoshka_swap_sweep.py \
    scripts/matryoshka_swap_montage.py \
    videocard@192.168.87.25:C:/comfy/importer/scripts/
```
Expected: three files copied, no error.

- [ ] **Step 2: Copy the user anchor PNG into the Windows identities dir**

```bash
scp /tmp/matpeek/SOURCE_user.png \
    videocard@192.168.87.25:C:/comfy/importer/data/importer/identities_flux/id_user.png
```
Expected: one file copied. (The 20 `id_NN.png` are already on the box from
the earlier sweep.)

- [ ] **Step 3: Install mediapipe into the Windows ComfyUI venv**

The MediaPipe fallback in `swap_core` needs `mediapipe.solutions`. Install it:

```bash
ssh videocard@192.168.87.25 'C:\comfy\ComfyUI\venv\Scripts\python.exe -m pip install -q mediapipe'
```
Expected: install completes (or "already satisfied"). Non-fatal if it fails —
`swap_core.mediapipe_kps_bbox` degrades gracefully — but the fallback is then
unavailable, so prefer it to succeed.

- [ ] **Step 4: Verify all 21 anchors are present on the box**

```bash
ssh videocard@192.168.87.25 'powershell -NoProfile -Command "(Get-ChildItem C:\comfy\importer\data\importer\identities_flux\id_*.png).Count"'
```
Expected: `21`.

- [ ] **Step 5: Smoke-test one cell**

```bash
ssh videocard@192.168.87.25 'cd /d C:\comfy\importer\data\importer && C:\comfy\ComfyUI\venv\Scripts\python.exe -u ..\..\scripts\matryoshka_swap_sweep.py --comfy-url http://127.0.0.1:8188 --workflow workflows\flux_pulid_canny_lora.api.json --id-dir identities_flux --template refs\matryoshka\template_canny.png --swapper C:\comfy\ComfyUI\models\insightface\inswapper_128.onnx --out refs_matryoshka --comfy-input-dir C:\comfy\ComfyUI\input --limit 1'
```
Expected: `[swap-sweep] 1 cells, preflight ok`, then `[ok] id_user_glossy_cn000_ps000_seed70000000 (...s) swap=default score=0.xx - 1 done, ...`. Confirms generation, swap, manifest write, and the `id_user` anchor all work end to end.

- [ ] **Step 6: Pull the smoke-test output and eyeball it**

```bash
scp videocard@192.168.87.25:C:/comfy/importer/data/importer/refs_matryoshka/swap_sweep/swapped/id_user_glossy_cn000_ps000_seed70000000.png /tmp/matpeek/smoketest_swap.png
```
Read `/tmp/matpeek/smoketest_swap.png`: it should be a matryoshka doll whose
face is the user's, swapped in. If `swap_mode` was `failed`, stop and
diagnose before the full run.

- [ ] **Step 7: Commit (none — staging only)**

No repo change in this task. Proceed to Task 6.

---

### Task 6: Launch the full sweep as a Windows Scheduled Task

**Files:** none in-repo. SSH to `videocard@192.168.87.25`.

The task must survive logout and SSH disconnect — same pattern as the
InfiniteYou download. The wrapper `.cmd` is written via base64 so the `>`
redirect is not eaten by the remote shell.

- [ ] **Step 1: Write the wrapper .cmd to the Windows box (base64)**

```bash
printf '@echo off\r\ncd /d C:\\comfy\\importer\\data\\importer\r\nC:\\comfy\\ComfyUI\\venv\\Scripts\\python.exe -u ..\\..\\scripts\\matryoshka_swap_sweep.py --comfy-url http://127.0.0.1:8188 --workflow workflows\\flux_pulid_canny_lora.api.json --id-dir identities_flux --template refs\\matryoshka\\template_canny.png --swapper C:\\comfy\\ComfyUI\\models\\insightface\\inswapper_128.onnx --out refs_matryoshka --comfy-input-dir C:\\comfy\\ComfyUI\\input > C:\\comfy\\importer\\runner_swap_sweep.log 2>&1\r\n' > /tmp/swap_sweep.cmd
B64=$(base64 -w0 /tmp/swap_sweep.cmd)
ssh videocard@192.168.87.25 "powershell -NoProfile -Command \"[IO.File]::WriteAllBytes('C:\\comfy\\importer\\swap_sweep.cmd', [Convert]::FromBase64String('$B64')); Get-Content C:\\comfy\\importer\\swap_sweep.cmd\""
```
Expected: the printed `.cmd` shows all three lines including the `>` log
redirect intact.

- [ ] **Step 2: Create and run the Scheduled Task**

```bash
ssh videocard@192.168.87.25 'schtasks /create /tn MatryoshkaSwapSweep /tr "C:\comfy\importer\swap_sweep.cmd" /sc once /st 00:00 /sd 01/01/2030 /ru SYSTEM /rl HIGHEST /f && schtasks /run /tn MatryoshkaSwapSweep'
```
Expected: "SUCCESS: The scheduled task ... was successfully created" and
"... attempted to run".

- [ ] **Step 3: Confirm the run started**

Wait ~90 s, then:

```bash
ssh videocard@192.168.87.25 'powershell -NoProfile -Command "Get-Content C:\comfy\importer\runner_swap_sweep.log -Tail 15"'
```
Expected: `[swap-sweep] 252 cells, preflight ok` followed by `[ok] ...` lines
accumulating. `251 cells` (not 252) is also fine — the smoke-test cell is
already done and will show as skipped.

- [ ] **Step 4: Record progress-check commands for the user**

Print, for the user, the two commands to monitor the unattended run:

```bash
# tail the log
ssh videocard@192.168.87.25 'powershell -NoProfile -Command "Get-Content C:\comfy\importer\runner_swap_sweep.log -Tail 20"'
# count finished swapped PNGs (done when this reaches 252)
ssh videocard@192.168.87.25 'powershell -NoProfile -Command "(Get-ChildItem C:\comfy\importer\data\importer\refs_matryoshka\swap_sweep\swapped\*.png).Count"'
```

- [ ] **Step 5: Commit (none — launch only)**

No repo change. The sweep runs ~3.3 h unattended. When it completes, pull
the `swapped/` dir and `manifest_swap_sweep.parquet`, run
`matryoshka_swap_montage.py` locally, and review the contact sheets.

---

## Self-Review

**Spec coverage:**
- Recipe (pw1.5, SCRFD-default, no repaint) → Task 2 constants + Task 1 `swap_identity`. ✓
- 252-cell grid (21×3×2×2) → Task 2 `build_grid` + tests. ✓
- `swap_core.py` (make_face_app/load_swapper/detect_source/mediapipe_kps_bbox/swap_identity, CPU-only) → Task 1. ✓
- Runner reusing `matryoshka_sweep` helpers → Task 3. ✓
- Outputs `dolls/`, `swapped/`, manifest with all columns → Task 3. ✓
- Error handling: retry, SCRFD→forced→failed, resumable skip-if-exists, atomic writes → Task 1 + Task 3. ✓
- Scheduled Task launch → Task 6. ✓
- Testing: swap_core logic tests, grid tests, pre-flight asserts → Tasks 1–3. ✓
- Post-sweep montage → Task 4. ✓
- mediapipe fallback availability on Windows → Task 5 Step 3. ✓

**Placeholder scan:** No TBD/TODO. All code blocks complete. ✓

**Type consistency:** `swap_identity` returns `(result, mode, score)` — used consistently in Task 3. `build_grid` cell keys (`identity`, `pulid_weight`, `pulid_end`, `cn_start`, `cn_end`, `pulid_start`, `prompt`) match what `matryoshka_sweep.build_workflow` consumes. `stem` format identical in Task 2 builder, Task 2 test, and Task 4 `_find`. ✓
