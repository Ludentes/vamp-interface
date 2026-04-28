# FFHQ Reverse-Index Extension Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run all five reverse-index metric families on the 70k FFHQ corpus (resized to 512² for Flux parity, executing on the remote Windows GPU box), then merge into a unified `reverse_index.parquet` keyed by `image_sha256` covering both `flux_corpus_v3` and `ffhq` sources.

**Architecture:** A single-pass extractor processes one parquet shard at a time, holds all five model wrappers (MiVOLO + FairFace + InsightFace + SigLIP-2 + MediaPipe FaceLandmarker) resident in GPU memory, and emits one `.pt` per FFHQ shard with a flat schema. NMF atom projection is done CPU-side via NNLS against the saved `au_library.npz` H matrix. Resumability via skip-if-exists per shard. Final merge step concatenates per-shard `.pt` outputs and joins with the existing Flux-corpus reverse-index parquets through a shared `image_sha256` column (which has to be back-filled on the Flux side first).

**Tech Stack:** torch + diffusers + insightface + open-clip + transformers (siglip2) + mediapipe + dlib + timm + scipy.optimize.nnls + pyarrow. Existing extractors live under `src/demographic_pc/`. Remote target: Windows 10, RTX 3090, Python 3.10 venv at `C:\comfy\ComfyUI\venv\`. The Flux VAE encode is already running there — this work is decoupled from it and can be developed in parallel.

---

## File Structure

**New files:**
- `tests/test_ffhq_extractor.py` — pytest fixtures + smoke tests (run locally against a 6-image FFHQ slice before pushing to Windows)
- `tests/fixtures/ffhq_smoke.parquet` — pre-built fixture (6 FFHQ images sampled from shard 0)
- `tests/fixtures/ffhq_smoke_expected.json` — golden values from the local run
- `src/demographic_pc/extract_ffhq_metrics.py` — main extractor; CLI: `--shards-dir / --out-dir / --log / --resolution 512 / --limit-shards N`
- `src/demographic_pc/build_unified_reverse_index.py` — merger; CLI: `--ffhq-dir / --flux-corpus-out / --out`
- `src/demographic_pc/backfill_flux_corpus_sha.py` — recomputes `image_sha256` for existing Flux PNGs, joins into existing reverse-index parquets
- `scripts/sync_extractor_assets_to_windows.sh` — rsync vendor weights + mediapipe model + au_library.npz to Windows
- `scripts/run_ffhq_metrics.bat` — Windows batch wrapper (mirrors existing `run_encode.bat`)
- `docs/research/2026-04-28-ffhq-reverse-index-extension.md` — research note (status=live, topic=demographic-pc-pipeline)

**Modified files:**
- `src/demographic_pc/classifiers.py` — no changes expected; reused as-is. If we discover MiVOLO loading too aggressively occupies VRAM alongside the other four models we may need a `device_map` parameter, but that's a fix-when-we-see-it concern, not a planned modification.

**Read-only references (already exist; the plan consumes them):**
- `src/demographic_pc/classifiers.py` — `MiVOLOClassifier`, `FairFaceClassifier`, `InsightFaceClassifier` wrappers
- `src/demographic_pc/score_clip_probes.py` — `Siglip2Backend` + `PROBES` list (12 attributes)
- `src/demographic_pc/score_blendshapes.py` — MediaPipe FaceLandmarker pattern (52-d ARKit blendshapes)
- `models/blendshape_nmf/au_library.npz` — keys: `H (8, 52)`, `names (52,)`, `tags (8,)` — used for NMF projection
- `vendor/MiVOLO/` — required as `sys.path` entry for MiVOLO model registration
- `vendor/weights/mivolo_volo_d1_face_age_gender_imdb.pth.tar`, `vendor/weights/fairface/res34_fair_align_multi_7_20190809.pt`, `vendor/weights/mmod_human_face_detector.dat`, `vendor/weights/shape_predictor_5_face_landmarks.dat`
- `models/mediapipe/face_landmarker.task`

**Per-shard `.pt` schema (FFHQ side):**
```
{
    "image_sha256":      list[str]               (length N)
    "shard_name":        str
    "format_version":    int (= 1)
    "resolution":        int (= 512)
    # demographic classifiers
    "mv_age":            np.float32 (N,)         (NaN where mivolo did not return)
    "mv_gender":         np.array of "M"|"F"|"" (N,)
    "mv_gender_conf":    np.float32 (N,)
    "ff_detected":       np.bool_ (N,)
    "ff_age_bin":        np.array (N,)           ("" where not detected)
    "ff_gender":         np.array (N,)
    "ff_race":           np.array (N,)
    "ff_age_probs":      np.float32 (N, 9)       (zeros where not detected)
    "ff_gender_probs":   np.float32 (N, 2)
    "ff_race_probs":     np.float32 (N, 7)
    "ins_detected":      np.bool_ (N,)
    "ins_age":           np.float32 (N,)         (NaN where not detected)
    "ins_gender":        np.array (N,)
    # siglip-2 attribute probes (12 attrs; see PROBES list)
    "sg_<probe>_margin": np.float32 (N,)         for each of 12 probe names
    # mediapipe blendshapes
    "bs_detected":       np.bool_ (N,)
    "bs_<name>":         np.float32 (N,)         for each of 52 ARKit names
                                                 (zeros where not detected)
    # NMF atom loadings (k=8 from au_library.npz)
    "atom_<i>":          np.float32 (N,)         for i in 0..7
                                                 (NaN where bs_detected=False)
}
```
ArcFace 512-d embedding is *not* re-stored here — it's already in the encode_ffhq.py outputs at `C:\arc_distill\encoded\<shard>.pt`. The merger reads both.

---

## Task 1: Build the local fixture (6-image FFHQ smoke set)

**Files:**
- Create: `tests/fixtures/build_ffhq_smoke.py`
- Create: `tests/fixtures/ffhq_smoke.parquet`

We need a tiny, version-controlled FFHQ slice we can run smoke tests against locally without copying 95 GB of parquet around. Six rows is enough to exercise: detected-face path, undetected-face path (FFHQ has ~40% miss rate on SCRFD@0.5), and at least one row where MediaPipe FaceLandmarker returns blendshapes.

- [ ] **Step 1: Write the fixture builder**

```python
# tests/fixtures/build_ffhq_smoke.py
"""One-shot builder. Reads the first FFHQ parquet shard from the Windows box
(or local copy if you have one) and saves the first 6 rows to a small
parquet under tests/fixtures/. Run once, commit the parquet."""
from __future__ import annotations
import argparse
from pathlib import Path
import pyarrow.parquet as pq

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="path to a FFHQ train-*.parquet shard")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).parent / "ffhq_smoke.parquet")
    args = ap.parse_args()
    table = pq.read_table(args.src)
    sliced = table.slice(0, args.n)
    pq.write_table(sliced, args.out)
    print(f"wrote {args.out} ({sliced.num_rows} rows, {args.out.stat().st_size/1e6:.1f} MB)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the builder against the Windows shard**

```bash
# pull one shard locally (or scp it down):
scp videocard@192.168.87.25:C:/arc_distill/ffhq_parquet/data/train-00000-of-00190.parquet /tmp/ffhq_shard0.parquet
uv run python tests/fixtures/build_ffhq_smoke.py --src /tmp/ffhq_shard0.parquet --n 6
rm /tmp/ffhq_shard0.parquet
```

Expected output: `wrote tests/fixtures/ffhq_smoke.parquet (6 rows, ~3-5 MB)`

- [ ] **Step 3: Commit the fixture**

```bash
git add tests/fixtures/build_ffhq_smoke.py tests/fixtures/ffhq_smoke.parquet
git commit -m "test(ffhq): smoke fixture (6 rows from FFHQ shard 0)"
```

---

## Task 2: Write the unit-level test for the extractor's pure functions

**Files:**
- Create: `tests/test_ffhq_extractor.py`

Pure functions that can be unit-tested without GPUs: PNG-bytes-to-RGB conversion, sha256 hashing, NNLS atom projection from a 52-d blendshape vector against the saved H matrix. These let us catch logic bugs locally before the slow Windows runs.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ffhq_extractor.py
"""Unit tests for the FFHQ extractor's pure logic.

Heavy GPU-dependent paths (mivolo/fairface/insightface/siglip2/mediapipe) are
covered by the Windows-side smoke run, not here. These tests hit only the
deterministic plumbing: hashing, image decode + resize, NMF atom projection.
"""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from demographic_pc.extract_ffhq_metrics import (
    decode_and_resize,
    compute_image_sha256,
    project_blendshapes_to_atoms,
)


FIXTURE = Path(__file__).parent / "fixtures" / "ffhq_smoke.parquet"
AU_LIBRARY = Path(__file__).resolve().parents[1] / "models" / "blendshape_nmf" / "au_library.npz"


def test_compute_image_sha256_matches_hashlib():
    payload = b"some png bytes"
    expected = hashlib.sha256(payload).hexdigest()
    assert compute_image_sha256(payload) == expected


def test_decode_and_resize_produces_expected_shape():
    table = pq.read_table(FIXTURE, columns=["image"])
    row = table.column("image").to_pylist()[0]
    rgb = decode_and_resize(row["bytes"], 512)
    assert rgb.shape == (512, 512, 3)
    assert rgb.dtype == np.uint8


def test_project_blendshapes_to_atoms_zero_input_gives_zero_atoms():
    H = np.load(AU_LIBRARY)["H"].astype(np.float32)  # (8, 52)
    y = np.zeros(52, dtype=np.float32)
    atoms = project_blendshapes_to_atoms(y, H)
    assert atoms.shape == (8,)
    assert np.allclose(atoms, 0.0)


def test_project_blendshapes_reconstruction_roughly_recovers_atom_basis():
    """If we feed in one row of H (an atom pattern), the projection should put
    most of the loading on that atom."""
    H = np.load(AU_LIBRARY)["H"].astype(np.float32)  # (8, 52)
    for k in range(H.shape[0]):
        y = H[k].copy()
        atoms = project_blendshapes_to_atoms(y, H)
        # The k-th atom should dominate.
        assert atoms[k] > 0.5 * atoms.sum(), \
            f"atom {k} not dominant: atoms={atoms}"
```

- [ ] **Step 2: Run tests to confirm they fail with import errors**

Run: `uv run pytest tests/test_ffhq_extractor.py -v`

Expected: 4 failures, all `ImportError: cannot import name '...' from 'demographic_pc.extract_ffhq_metrics'` or similar (file doesn't exist yet).

---

## Task 3: Implement the pure helper functions in the extractor module

**Files:**
- Create: `src/demographic_pc/extract_ffhq_metrics.py`

Stand up the file with just the three pure functions — heavy classes (the model wrappers) come in Task 4. This keeps Task 2's tests passing as soon as possible.

- [ ] **Step 1: Create the module skeleton with the three pure helpers**

```python
# src/demographic_pc/extract_ffhq_metrics.py
"""Run all reverse-index metric families on a FFHQ parquet corpus.

Single-pass per shard: image bytes -> RGB@512 -> {mivolo, fairface,
insightface attrs, siglip2 probes, mediapipe blendshapes -> NMF atoms}.

The five model wrappers are imported lazily inside `main()` so that the
pure helpers below remain testable on a CPU-only laptop.
"""
from __future__ import annotations

import hashlib
import io

import numpy as np
from PIL import Image
from scipy.optimize import nnls


def compute_image_sha256(png_bytes: bytes) -> str:
    """Hex-digest of the raw PNG bytes. Stable across decoders."""
    return hashlib.sha256(png_bytes).hexdigest()


def decode_and_resize(png_bytes: bytes, resolution: int) -> np.ndarray:
    """Decode PNG, convert to RGB, resize to (resolution, resolution) using
    LANCZOS (matches the encode_ffhq.py + Flux corpus convention).
    Returns uint8 (H, W, 3)."""
    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.uint8)


def project_blendshapes_to_atoms(y: np.ndarray, H: np.ndarray) -> np.ndarray:
    """NNLS projection: find non-negative w (k,) minimising ||y - w @ H||^2.

    Args:
        y: (52,) blendshape activations in [0, 1].
        H: (k, 52) NMF atom-pattern matrix from au_library.npz.

    Returns:
        (k,) atom loadings (>= 0).
    """
    w, _ = nnls(H.T, y)  # solves ||H.T @ w - y||
    return w.astype(np.float32)
```

- [ ] **Step 2: Run the tests**

Run: `uv run pytest tests/test_ffhq_extractor.py -v`

Expected: all 4 PASS.

- [ ] **Step 3: Commit**

```bash
git add src/demographic_pc/extract_ffhq_metrics.py tests/test_ffhq_extractor.py
git commit -m "feat(ffhq): extractor pure helpers + unit tests"
```

---

## Task 4: Add the model-wrapper classes and the per-shard pipeline

**Files:**
- Modify: `src/demographic_pc/extract_ffhq_metrics.py` (extend)

Adds the heavy lifting — instantiates all five extractors, loops over a parquet shard, builds the per-shard `.pt` payload. This is the Windows-side workhorse. We can't unit-test this slice meaningfully on a CPU laptop, so we'll exercise it via a `--smoke` mode pointed at the fixture parquet, in Task 5.

- [ ] **Step 1: Extend the module with the extractor pipeline**

Append to `src/demographic_pc/extract_ffhq_metrics.py`:

```python
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import pyarrow.parquet as pq
import torch

from demographic_pc.classifiers import (
    MiVOLOClassifier, FairFaceClassifier, InsightFaceClassifier,
)
from demographic_pc.score_clip_probes import Siglip2Backend, PROBES


# ARKit blendshape names in canonical order (52 channels, matches our existing
# sample_index columns). Hardcoded to avoid a model-load dependency at import.
ARKIT_BLENDSHAPE_NAMES: list[str] = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp",
    "browOuterUpLeft", "browOuterUpRight", "cheekPuff",
    "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight",
    "mouthRollLower", "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "noseSneerLeft", "noseSneerRight",
]

FORMAT_VERSION = 1


def build_landmarker(model_path: Path):
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    opts = mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return mp_vision.FaceLandmarker.create_from_options(opts), mp


def score_blendshapes(landmarker, mp, rgb: np.ndarray) -> dict[str, float] | None:
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res = landmarker.detect(img)
    if not res.face_blendshapes:
        return None
    return {c.category_name: float(c.score) for c in res.face_blendshapes[0]}


def encode_siglip_probes(siglip: Siglip2Backend) -> dict:
    """Pre-encode the 12 probe text pairs once."""
    return siglip.encode_probes()


def score_siglip_image(siglip: Siglip2Backend, probe_feats, rgb: np.ndarray) -> dict[str, float]:
    """Score one image against pre-encoded probes. Mirrors score_image() from
    score_clip_probes.py but takes an in-memory RGB array, not a path."""
    img = Image.fromarray(rgb)
    inputs = siglip.processor(images=img, return_tensors="pt").to(siglip.device)
    with torch.no_grad():
        feat = siglip._as_tensor(siglip.model.get_image_features(**inputs))
    feat = feat / feat.norm(dim=-1, keepdim=True)
    out: dict[str, float] = {}
    for name, pf in probe_feats.items():
        sims = (feat @ pf.T).squeeze(0)
        out[f"sg_{name}_margin"] = float(sims[0] - sims[1])
    return out


def process_shard(
    shard_path: Path,
    out_path: Path,
    *,
    mv: MiVOLOClassifier,
    ff: FairFaceClassifier,
    ins: InsightFaceClassifier,
    siglip: Siglip2Backend,
    probe_feats,
    landmarker,
    mp,
    H: np.ndarray,
    resolution: int,
    log: logging.Logger,
) -> dict:
    t0 = time.time()
    table = pq.read_table(shard_path, columns=["image"])
    rows = table.column("image").to_pylist()
    n = len(rows)
    log.info(f"  {shard_path.name}: {n} rows")

    # Allocate output buffers.
    payload: dict = {
        "image_sha256": [],
        "mv_age":          np.full(n, np.nan, dtype=np.float32),
        "mv_gender":       np.full(n, "", dtype=object),
        "mv_gender_conf":  np.full(n, np.nan, dtype=np.float32),
        "ff_detected":     np.zeros(n, dtype=bool),
        "ff_age_bin":      np.full(n, "", dtype=object),
        "ff_gender":       np.full(n, "", dtype=object),
        "ff_race":         np.full(n, "", dtype=object),
        "ff_age_probs":    np.zeros((n, 9), dtype=np.float32),
        "ff_gender_probs": np.zeros((n, 2), dtype=np.float32),
        "ff_race_probs":   np.zeros((n, 7), dtype=np.float32),
        "ins_detected":    np.zeros(n, dtype=bool),
        "ins_age":         np.full(n, np.nan, dtype=np.float32),
        "ins_gender":      np.full(n, "", dtype=object),
        "bs_detected":     np.zeros(n, dtype=bool),
    }
    for name in ARKIT_BLENDSHAPE_NAMES:
        payload[f"bs_{name}"] = np.zeros(n, dtype=np.float32)
    for k in range(H.shape[0]):
        payload[f"atom_{k:02d}"] = np.full(n, np.nan, dtype=np.float32)
    for probe_name, _, _ in PROBES:
        payload[f"sg_{probe_name}_margin"] = np.zeros(n, dtype=np.float32)

    for i, row in enumerate(rows):
        png_bytes = row["bytes"]
        payload["image_sha256"].append(compute_image_sha256(png_bytes))
        rgb = decode_and_resize(png_bytes, resolution)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # mivolo (no detection step; predicts on whole image)
        mv_out = mv.predict(bgr)
        payload["mv_age"][i] = mv_out["age"]
        payload["mv_gender"][i] = mv_out["gender"]
        payload["mv_gender_conf"][i] = mv_out["gender_conf"]

        # fairface (dlib detect + 5-landmark align)
        ff_out = ff.predict(bgr)
        payload["ff_detected"][i] = ff_out["detected"]
        if ff_out["detected"]:
            payload["ff_age_bin"][i] = ff_out["age_bin"]
            payload["ff_gender"][i] = ff_out["gender"]
            payload["ff_race"][i] = ff_out["race"]
            payload["ff_age_probs"][i] = ff_out["age_probs"]
            payload["ff_gender_probs"][i] = ff_out["gender_probs"]
            payload["ff_race_probs"][i] = ff_out["race_probs"]

        # insightface (genderage only; recognition handled by encode_ffhq.py)
        ins_out = ins.predict(bgr)
        payload["ins_detected"][i] = ins_out["detected"]
        if ins_out["detected"]:
            payload["ins_age"][i] = ins_out["age"]
            payload["ins_gender"][i] = ins_out["gender"]

        # siglip-2 probes
        sg_out = score_siglip_image(siglip, probe_feats, rgb)
        for k, v in sg_out.items():
            payload[k][i] = v

        # mediapipe blendshapes
        bs = score_blendshapes(landmarker, mp, rgb)
        if bs is not None:
            payload["bs_detected"][i] = True
            y = np.zeros(52, dtype=np.float32)
            for j, name in enumerate(ARKIT_BLENDSHAPE_NAMES):
                v = float(bs.get(name, 0.0))
                payload[f"bs_{name}"][i] = v
                y[j] = v
            atoms = project_blendshapes_to_atoms(y, H)
            for k_idx in range(H.shape[0]):
                payload[f"atom_{k_idx:02d}"][i] = atoms[k_idx]

        if (i + 1) % 25 == 0:
            elapsed = time.time() - t0
            log.info(
                f"    {i+1}/{n} ({(i+1)/elapsed:.1f} img/s, "
                f"ff_det={payload['ff_detected'][:i+1].mean():.2f}, "
                f"bs_det={payload['bs_detected'][:i+1].mean():.2f})"
            )

    payload["shard_name"] = shard_path.name
    payload["format_version"] = FORMAT_VERSION
    payload["resolution"] = resolution
    elapsed = time.time() - t0
    log.info(
        f"  {shard_path.name} done: {n} imgs in {elapsed:.1f}s "
        f"({n/elapsed:.1f} img/s); "
        f"ff_det={payload['ff_detected'].mean():.3f} "
        f"bs_det={payload['bs_detected'].mean():.3f}"
    )

    tmp_path = out_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, out_path)
    log.info(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    return {
        "shard": shard_path.name, "n": n, "elapsed_s": elapsed,
        "ff_det": float(payload["ff_detected"].mean()),
        "bs_det": float(payload["bs_detected"].mean()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--mediapipe-model", type=Path, required=True)
    p.add_argument("--au-library", type=Path, required=True)
    p.add_argument("--resolution", type=int, default=512)
    p.add_argument("--limit-shards", type=int, default=0, help="0 = all")
    p.add_argument("--smoke", action="store_true",
                   help="if set, --shards-dir is ignored and we run on a single fixture parquet")
    p.add_argument("--smoke-fixture", type=Path,
                   default=Path("tests/fixtures/ffhq_smoke.parquet"))
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.FileHandler(args.log, mode="a"), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("extract_ffhq")
    log.info("=" * 60)
    log.info(f"start extract_ffhq pid={os.getpid()}")

    H = np.load(args.au_library)["H"].astype(np.float32)
    log.info(f"loaded au_library H={H.shape}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"device={device}")

    log.info("loading MiVOLO...");        mv = MiVOLOClassifier(device=device)
    log.info("loading FairFace...");      ff = FairFaceClassifier(device=device)
    log.info("loading InsightFace...");   ins = InsightFaceClassifier(ctx_id=0, with_embedding=False)
    log.info("loading SigLIP-2...");      siglip = Siglip2Backend(device=device)
    log.info("loading FaceLandmarker..."); landmarker, mp = build_landmarker(args.mediapipe_model)

    log.info("encoding SigLIP-2 probes...")
    probe_feats = encode_siglip_probes(siglip)

    if args.smoke:
        shards = [args.smoke_fixture]
    else:
        shards = sorted(args.shards_dir.glob("*.parquet"))
        if args.limit_shards:
            shards = shards[: args.limit_shards]
    log.info(f"processing {len(shards)} shard(s)")

    summary = []
    for idx, shard in enumerate(shards):
        out_path = args.out_dir / f"{shard.stem}.pt"
        if out_path.exists():
            log.info(f"[{idx+1}/{len(shards)}] {shard.name} -> SKIP (exists)")
            continue
        log.info(f"[{idx+1}/{len(shards)}] processing {shard.name}")
        try:
            stats = process_shard(
                shard, out_path,
                mv=mv, ff=ff, ins=ins,
                siglip=siglip, probe_feats=probe_feats,
                landmarker=landmarker, mp=mp,
                H=H, resolution=args.resolution, log=log,
            )
            summary.append(stats)
        except Exception:
            log.exception(f"FAILED on {shard.name}")
            raise

    summary_path = args.out_dir / "extract_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"all done, wrote {summary_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Confirm the module still imports cleanly (the heavy classes are inside `main()` so import-time is light)**

Run: `uv run python -c "from demographic_pc.extract_ffhq_metrics import process_shard, main; print('ok')"`

Expected: `ok`

- [ ] **Step 3: Run the unit tests again to make sure we haven't broken them**

Run: `uv run pytest tests/test_ffhq_extractor.py -v`

Expected: all 4 PASS.

- [ ] **Step 4: Commit**

```bash
git add src/demographic_pc/extract_ffhq_metrics.py
git commit -m "feat(ffhq): per-shard extractor pipeline (mivolo+fairface+insightface+siglip2+blendshapes+atoms)"
```

---

## Task 5: Local smoke test against the 6-image fixture

**Files:**
- Modify: `src/demographic_pc/extract_ffhq_metrics.py` (no code change; just exercising it)

Run end-to-end on 6 images locally to catch model-loading / wiring bugs before pushing to Windows. We expect the local GPU to be busy with experiments — this only needs ~30 seconds, so even queuing on it is fine. If the local GPU is genuinely unavailable, fall back to CPU (slower but still tractable for 6 images).

- [ ] **Step 1: Run the smoke pipeline**

```bash
uv run python -m demographic_pc.extract_ffhq_metrics \
    --smoke \
    --out-dir /tmp/ffhq_smoke_out \
    --log /tmp/ffhq_smoke.log \
    --mediapipe-model models/mediapipe/face_landmarker.task \
    --au-library models/blendshape_nmf/au_library.npz \
    --shards-dir tests/fixtures
```

Expected output (last lines):
```
... ffhq_smoke.parquet done: 6 imgs in <60>s (~0.2 img/s); ff_det=<rate> bs_det=<rate>
... wrote /tmp/ffhq_smoke_out/ffhq_smoke.pt (~0.2 MB)
... all done, wrote /tmp/ffhq_smoke_out/extract_summary.json
```

If it crashes: read the traceback, fix the bug, commit the fix, re-run.

- [ ] **Step 2: Inspect the output and lock in golden values**

```bash
uv run python - <<'EOF'
import torch, json, numpy as np
p = torch.load("/tmp/ffhq_smoke_out/ffhq_smoke.pt", weights_only=False)
golden = {
    "image_sha256": p["image_sha256"],
    "ff_detected": p["ff_detected"].tolist(),
    "ins_detected": p["ins_detected"].tolist(),
    "bs_detected": p["bs_detected"].tolist(),
    "mv_age_first": float(p["mv_age"][0]) if not np.isnan(p["mv_age"][0]) else None,
    "format_version": p["format_version"],
}
print(json.dumps(golden, indent=2))
EOF
```

Save this output to `tests/fixtures/ffhq_smoke_expected.json`. (Hand-edit if numerics are non-deterministic across runs — pin only the structural fields like `image_sha256` and `format_version`.)

- [ ] **Step 3: Add a regression test that loads the smoke output and checks it against the golden file**

Append to `tests/test_ffhq_extractor.py`:

```python
import json

EXPECTED = Path(__file__).parent / "fixtures" / "ffhq_smoke_expected.json"


@pytest.mark.skipif(
    not Path("/tmp/ffhq_smoke_out/ffhq_smoke.pt").exists(),
    reason="run the smoke extractor first (Task 5 step 1)",
)
def test_smoke_output_matches_golden():
    import torch
    p = torch.load("/tmp/ffhq_smoke_out/ffhq_smoke.pt", weights_only=False)
    expected = json.loads(EXPECTED.read_text())
    assert p["image_sha256"] == expected["image_sha256"]
    assert p["format_version"] == expected["format_version"]
```

Run: `uv run pytest tests/test_ffhq_extractor.py -v`

Expected: 5 tests, all PASS (or 4 PASS + 1 SKIP if `/tmp` was cleared).

- [ ] **Step 4: Commit**

```bash
git add tests/fixtures/ffhq_smoke_expected.json tests/test_ffhq_extractor.py
git commit -m "test(ffhq): smoke regression against golden output"
```

---

## Task 6: Sync extractor assets to the Windows box

**Files:**
- Create: `scripts/sync_extractor_assets_to_windows.sh`

Vendor weights and the MediaPipe model and `au_library.npz` need to live on Windows for the extractor to load them. The encoder pipeline already pushed `vendor/MiVOLO/` indirectly (it isn't actually on Windows yet — we'll push it now). One rsync, ~1 GB total.

Approximate sizes:
- MiVOLO weights .pth.tar: ~390 MB
- FairFace res34 .pt: ~85 MB
- dlib mmod CNN detector: ~700 KB
- dlib 5-landmark predictor: ~9 MB
- mediapipe `face_landmarker.task`: ~3 MB
- au_library.npz: ~1 MB
- vendor/MiVOLO Python source tree: ~1 MB (no large binaries)
- vendor/FairFace dlib model dir: covered above

- [ ] **Step 1: Write the sync script**

```bash
#!/usr/bin/env bash
# scripts/sync_extractor_assets_to_windows.sh
# One-shot: copy extractor weights + source dependencies to the Windows GPU box.
# Idempotent (rsync skips unchanged files).

set -euo pipefail

REMOTE="videocard@192.168.87.25"
DST_ROOT="C:/arc_distill/repo_assets"

# scp doesn't do trees natively on stock Windows OpenSSH the way rsync would.
# We use scp -r (slow but works without rsync on the remote).

ssh "$REMOTE" "mkdir C:\\arc_distill\\repo_assets 2>nul; mkdir C:\\arc_distill\\repo_assets\\vendor 2>nul; mkdir C:\\arc_distill\\repo_assets\\models 2>nul; exit 0"

echo "[sync] vendor/weights -> $DST_ROOT/vendor/weights"
scp -r vendor/weights                 "$REMOTE:$DST_ROOT/vendor/weights"

echo "[sync] vendor/MiVOLO -> $DST_ROOT/vendor/MiVOLO"
scp -r vendor/MiVOLO                  "$REMOTE:$DST_ROOT/vendor/MiVOLO"

echo "[sync] vendor/FairFace/dlib_models -> $DST_ROOT/vendor/FairFace/dlib_models"
ssh  "$REMOTE" "mkdir C:\\arc_distill\\repo_assets\\vendor\\FairFace 2>nul; exit 0"
scp -r vendor/FairFace/dlib_models    "$REMOTE:$DST_ROOT/vendor/FairFace/dlib_models"

echo "[sync] models/mediapipe -> $DST_ROOT/models/mediapipe"
scp -r models/mediapipe               "$REMOTE:$DST_ROOT/models/mediapipe"

echo "[sync] models/blendshape_nmf/au_library.npz -> $DST_ROOT/models/blendshape_nmf/"
ssh  "$REMOTE" "mkdir C:\\arc_distill\\repo_assets\\models\\blendshape_nmf 2>nul; exit 0"
scp models/blendshape_nmf/au_library.npz "$REMOTE:$DST_ROOT/models/blendshape_nmf/au_library.npz"

echo "[sync] src/demographic_pc -> $DST_ROOT/src/demographic_pc"
ssh  "$REMOTE" "mkdir C:\\arc_distill\\repo_assets\\src 2>nul; exit 0"
scp -r src/demographic_pc             "$REMOTE:$DST_ROOT/src/demographic_pc"

echo "[sync] done"
```

- [ ] **Step 2: Make executable and run it**

```bash
chmod +x scripts/sync_extractor_assets_to_windows.sh
bash scripts/sync_extractor_assets_to_windows.sh
```

Expected: `[sync] done` after ~5-10 minutes (depending on VPN bandwidth).

- [ ] **Step 3: Verify the assets landed on Windows**

```bash
ssh videocard@192.168.87.25 "dir C:\arc_distill\repo_assets\vendor\weights & dir C:\arc_distill\repo_assets\models\mediapipe & dir C:\arc_distill\repo_assets\models\blendshape_nmf"
```

Expected: `mivolo_volo_d1_face_age_gender_imdb.pth.tar`, `face_landmarker.task`, `au_library.npz` all present and non-zero size.

- [ ] **Step 4: Commit the sync script**

```bash
git add scripts/sync_extractor_assets_to_windows.sh
git commit -m "ops(ffhq): rsync extractor weights + source to Windows GPU"
```

---

## Task 7: Install the extra Python deps on the Windows box

**Files:** none in the repo. This is one ssh invocation that mutates the remote venv.

Required new packages (everything else was installed during the encode_ffhq path): `timm`, `dlib`, `mediapipe`, `open_clip_torch`, `transformers`, `scikit-learn` (already there from insightface), `scipy` (already there).

`dlib` on Windows requires Visual Studio Build Tools or a prebuilt wheel. We'll try the prebuilt wheel first and only fall back to building from source if pip can't find one.

- [ ] **Step 1: Install the deps**

```bash
ssh videocard@192.168.87.25 "C:\comfy\ComfyUI\venv\Scripts\python.exe -m pip install timm dlib mediapipe open_clip_torch transformers"
```

Expected: `Successfully installed ...` lines for each.

- [ ] **Step 2: Verify each import works**

```bash
ssh videocard@192.168.87.25 'C:\comfy\ComfyUI\venv\Scripts\python.exe -c "import timm, dlib, mediapipe, open_clip, transformers; print(\"ok\")"'
```

Expected: `ok`

- [ ] **Step 3: If `dlib` install failed, install the prebuilt wheel from https://github.com/sachadee/Dlib (Python 3.10 wheel for Windows)**

```bash
# Only if needed:
ssh videocard@192.168.87.25 "C:\comfy\ComfyUI\venv\Scripts\python.exe -m pip install https://github.com/sachadee/Dlib/raw/main/dlib-19.22.99-cp310-cp310-win_amd64.whl"
```

(Verify this URL is still alive at run time — third-party wheels rot. If 404, build from source after installing Visual Studio Build Tools.)

---

## Task 8: Smoke-run the extractor on 1 FFHQ shard on Windows

**Files:**
- Create: `scripts/run_ffhq_metrics.bat`

Mirror the existing `run_encode.bat` pattern. Passes the right paths for the assets we synced in Task 6.

- [ ] **Step 1: Write the batch wrapper**

```batch
@echo off
REM scripts/run_ffhq_metrics.bat
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
C:\comfy\ComfyUI\venv\Scripts\python.exe -m demographic_pc.extract_ffhq_metrics ^
    --shards-dir C:\arc_distill\ffhq_parquet\data ^
    --out-dir C:\arc_distill\metrics ^
    --log C:\arc_distill\metrics.log ^
    --mediapipe-model %REPO%\models\mediapipe\face_landmarker.task ^
    --au-library      %REPO%\models\blendshape_nmf\au_library.npz
echo metrics_done > C:\arc_distill\metrics.done
```

Note: this uses `PYTHONPATH=%REPO%\src` so that `demographic_pc.classifiers` and `demographic_pc.score_clip_probes` resolve. We also need to ensure `MiVOLO` import works — `classifiers.py` does `sys.path.insert(0, str(ROOT / "vendor" / "MiVOLO"))` based on `Path(__file__).resolve().parents[2]`. That walks up 2 levels from `repo_assets/src/demographic_pc/classifiers.py` to `repo_assets/`, so `repo_assets/vendor/MiVOLO/` is the right place — which is where the sync script put it. ✓

- [ ] **Step 2: scp the batch file to Windows**

```bash
scp scripts/run_ffhq_metrics.bat videocard@192.168.87.25:C:/arc_distill/run_ffhq_metrics.bat
```

- [ ] **Step 3: Run on 1 shard with `--limit-shards 1` to smoke**

```bash
ssh videocard@192.168.87.25 "set PYTHONPATH=C:\arc_distill\repo_assets\src && C:\comfy\ComfyUI\venv\Scripts\python.exe -m demographic_pc.extract_ffhq_metrics --shards-dir C:\arc_distill\ffhq_parquet\data --out-dir C:\arc_distill\metrics --log C:\arc_distill\metrics.log --mediapipe-model C:\arc_distill\repo_assets\models\mediapipe\face_landmarker.task --au-library C:\arc_distill\repo_assets\models\blendshape_nmf\au_library.npz --limit-shards 1"
```

Expected:
- All 5 model loads succeed (one INFO line each)
- ~369 rows in shard 0
- Per-image throughput in the log: roughly 3-6 img/s (~1-2 min total for 1 shard)
- Final line `wrote C:\arc_distill\metrics\train-00000-of-00190.pt (~0.5 MB)`

If anything fails: read the traceback, fix in the local `extract_ffhq_metrics.py`, re-sync via `scripts/sync_extractor_assets_to_windows.sh`, retry.

- [ ] **Step 4: Sanity-check the output schema**

```bash
ssh videocard@192.168.87.25 "C:\comfy\ComfyUI\venv\Scripts\python.exe -c \"import torch; p = torch.load(r'C:\arc_distill\metrics\train-00000-of-00190.pt', weights_only=False); print('keys:', sorted(p.keys())); print('n:', len(p['image_sha256'])); print('ff_det:', float(p['ff_detected'].mean())); print('bs_det:', float(p['bs_detected'].mean()))\""
```

Expected:
- `keys:` lists all the fields from the schema in this plan's File Structure section
- `n: 369`
- `ff_det:` and `bs_det:` are non-zero (sanity that detection ran)

- [ ] **Step 5: Commit the batch wrapper**

```bash
git add scripts/run_ffhq_metrics.bat
git commit -m "ops(ffhq): batch wrapper to launch the metrics extractor on Windows"
```

---

## Task 9: Launch the full FFHQ metrics extraction (autonomous, schtasks)

**Files:** none new. This step launches a long-running task.

Same pattern as the FFHQ encode launch: `schtasks /create` so it survives ssh disconnect, then `schtasks /run`.

Throughput projection: if smoke shows ~5 img/s, 70 000 images / 5 = ~3.9 hours. Plan for ~4-6 hours wall time.

- [ ] **Step 1: Confirm `C:\arc_distill\encoded\` (the VAE encode output dir) is *not* the same as `C:\arc_distill\metrics\` — they must not collide**

```bash
ssh videocard@192.168.87.25 "dir C:\arc_distill\metrics 2>&1 | findstr /i \"\"  & dir C:\arc_distill\encoded 2>&1 | findstr /i \"\""
```

Expected: distinct dirs, both exist (one with smoke output, the other with VAE encodes from the parallel job).

- [ ] **Step 2: Create + launch the scheduled task**

```bash
ssh videocard@192.168.87.25 "schtasks /delete /tn ffhq_metrics /f 2>nul & schtasks /create /tn ffhq_metrics /tr \"C:\arc_distill\run_ffhq_metrics.bat\" /sc once /st 23:59 /f & schtasks /run /tn ffhq_metrics"
```

Expected: `Успешно: запланированное задание "ffhq_metrics" создано` and `Успешно: попытка выполнения задания "ffhq_metrics"` (or English equivalents).

- [ ] **Step 3: Verify it's running**

```bash
ssh videocard@192.168.87.25 "schtasks /query /tn ffhq_metrics /fo list"
```

Expected: `Состояние / Status: Выполняется / Running`

- [ ] **Step 4: Spot-check the log after ~5 min**

```bash
ssh videocard@192.168.87.25 "powershell -Command \"Get-Content C:\arc_distill\metrics.log | Select-Object -Last 20\""
```

Expected: progress lines like `[3/190] processing train-00002-of-00190.parquet` and per-50-row throughput updates.

- [ ] **Step 5: Once `C:\arc_distill\metrics.done` exists, pull the per-shard outputs back locally**

```bash
mkdir -p output/ffhq_metrics
scp -r videocard@192.168.87.25:C:/arc_distill/metrics output/ffhq_metrics/
```

(190 .pt files × ~0.5 MB ≈ ~100 MB — quick.)

- [ ] **Step 6: Commit nothing yet — this step only produces external artifacts. The reverse-index merge in Task 10 is what produces a committed parquet.**

---

## Task 10: Backfill `image_sha256` for the existing Flux corpus

**Files:**
- Create: `src/demographic_pc/backfill_flux_corpus_sha.py`
- Modify: `output/demographic_pc/labels.parquet` (in-place add column; back up first)
- Modify: `output/demographic_pc/classifier_scores.parquet` (same)
- Modify: `output/demographic_pc/clip_probes_siglip2.parquet` (same)
- Modify: `models/blendshape_nmf/sample_index.parquet` (same)

Each Flux-corpus row knows its source PNG via an `img_path` or `rel_from_overnight` column. We compute `sha256(open(p, "rb").read())` for each, then write the column back in place. This is the join key that lets us merge with FFHQ later.

- [ ] **Step 1: Write the backfill script**

```python
# src/demographic_pc/backfill_flux_corpus_sha.py
"""Add image_sha256 to the existing Flux-corpus reverse-index parquets.

For each parquet that has a path column pointing at an existing PNG, compute
the sha256 of the raw bytes and add a column. Writes back in place after a
backup. Idempotent: skips parquets that already have image_sha256.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# (parquet_path, path_column, path_root_for_resolution)
TARGETS = [
    (ROOT / "models/blendshape_nmf/sample_index.parquet",
     "img_path", ROOT),
    (ROOT / "output/demographic_pc/classifier_scores.parquet",
     "img_path", ROOT),
    (ROOT / "output/demographic_pc/clip_probes_siglip2.parquet",
     "rel_from_overnight",
     ROOT / "output/demographic_pc/overnight_drift"),
    # labels.parquet has a sample_id, not a path, so it's joined indirectly later.
]


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def backfill_one(parquet_path: Path, path_col: str, path_root: Path) -> None:
    if not parquet_path.exists():
        print(f"  SKIP (missing): {parquet_path}")
        return
    df = pd.read_parquet(parquet_path)
    if "image_sha256" in df.columns:
        print(f"  SKIP (already has image_sha256): {parquet_path}")
        return
    if path_col not in df.columns:
        print(f"  SKIP (no '{path_col}' column): {parquet_path}")
        return

    backup = parquet_path.with_suffix(parquet_path.suffix + ".bak.before_sha")
    if not backup.exists():
        shutil.copy2(parquet_path, backup)
        print(f"  backed up to {backup.name}")

    sha_col: list[str | None] = []
    missing = 0
    for rel in df[path_col]:
        if rel is None or (isinstance(rel, float)):  # NaN
            sha_col.append(None); missing += 1; continue
        p = (path_root / rel).resolve()
        if not p.exists():
            sha_col.append(None); missing += 1; continue
        sha_col.append(sha256_file(p))
    df["image_sha256"] = sha_col
    df.to_parquet(parquet_path, index=False)
    print(f"  wrote {parquet_path} ({len(df)} rows, {missing} unresolved paths)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for parquet_path, path_col, path_root in TARGETS:
        print(f"backfill {parquet_path.relative_to(ROOT)}")
        if args.dry_run:
            print("  (dry-run; skipping write)")
            continue
        backfill_one(parquet_path, path_col, path_root)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Dry-run first**

```bash
uv run python -m demographic_pc.backfill_flux_corpus_sha --dry-run
```

Expected: lists three parquets, all "(dry-run; skipping write)".

- [ ] **Step 3: Real run (creates `.bak.before_sha` backups)**

```bash
uv run python -m demographic_pc.backfill_flux_corpus_sha
```

Expected output, e.g.:
```
backfill models/blendshape_nmf/sample_index.parquet
  backed up to sample_index.parquet.bak.before_sha
  wrote models/blendshape_nmf/sample_index.parquet (7772 rows, 0 unresolved paths)
backfill output/demographic_pc/classifier_scores.parquet
  ...
```

If `unresolved paths` is non-zero, that's expected (some Flux PNGs were deleted between rendering and now); they get NULL `image_sha256` and won't join into the unified index. Investigate only if >5% are unresolved.

- [ ] **Step 4: Verify**

```bash
uv run python -c "
import pandas as pd
for p in ['models/blendshape_nmf/sample_index.parquet',
          'output/demographic_pc/classifier_scores.parquet',
          'output/demographic_pc/clip_probes_siglip2.parquet']:
    df = pd.read_parquet(p)
    assert 'image_sha256' in df.columns
    n_null = df['image_sha256'].isna().sum()
    print(f'{p}: rows={len(df)}, null_sha={n_null}')
"
```

Expected: each parquet now has the column; null counts are small.

- [ ] **Step 5: Commit (parquet binaries go in too — the .bak files are gitignored already; if not, add them to .gitignore)**

```bash
# verify backups are gitignored
git check-ignore models/blendshape_nmf/sample_index.parquet.bak.before_sha
# if not: echo "*.bak.before_sha" >> .gitignore
git add src/demographic_pc/backfill_flux_corpus_sha.py
git add models/blendshape_nmf/sample_index.parquet \
        output/demographic_pc/classifier_scores.parquet \
        output/demographic_pc/clip_probes_siglip2.parquet
git commit -m "feat(reverse-index): backfill image_sha256 on Flux-corpus parquets"
```

---

## Task 11: Build the unified reverse-index parquet

**Files:**
- Create: `src/demographic_pc/build_unified_reverse_index.py`
- Create: `output/reverse_index/reverse_index.parquet` (artefact; gitignore via existing `output/` rules — verify)
- Create: `output/reverse_index/build_log.json` (small artefact; commit)

Concatenates per-shard FFHQ `.pt` files + the existing Flux-corpus parquets into one long parquet keyed by `image_sha256` with a `source` column. Schema is the union of the FFHQ schema (Task 4) and the Flux-corpus schemas, with NaN/empty fill for rows that don't have a given column on their side.

- [ ] **Step 1: Write the merger**

```python
# src/demographic_pc/build_unified_reverse_index.py
"""Concatenate per-shard FFHQ extractor outputs + Flux-corpus reverse-index
parquets into one long parquet keyed by image_sha256.

source column values:
    "flux_corpus_v3"  - synthetic renders (existing reverse-index parquets)
    "ffhq"            - real-world FFHQ images (per-shard .pt outputs)

Columns missing in one side are filled with NaN/empty.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def load_ffhq_shard(pt_path: Path) -> pd.DataFrame:
    p = torch.load(pt_path, weights_only=False)
    n = len(p["image_sha256"])
    out = {"source": ["ffhq"] * n, "image_sha256": list(p["image_sha256"])}
    for k, v in p.items():
        if k in {"image_sha256", "shard_name", "format_version", "resolution"}:
            continue
        if isinstance(v, np.ndarray) and v.ndim == 1:
            out[k] = v.tolist()
        elif isinstance(v, np.ndarray) and v.ndim == 2:
            # split into per-row lists (e.g. ff_age_probs (N,9))
            out[k] = [row.tolist() for row in v]
        elif isinstance(v, list):
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.tolist()
    return pd.DataFrame(out)


def load_flux_arcface(pt_dir: Path) -> pd.DataFrame:
    """Pull the ArcFace 512-d embeddings produced by encode_ffhq.py."""
    rows = []
    for pt_path in sorted(pt_dir.glob("*.pt")):
        if pt_path.name == "encode_summary.json":
            continue
        p = torch.load(pt_path, weights_only=False)
        for sha, emb, det in zip(p["image_sha256"],
                                 p["arcface_fp32"].numpy(),
                                 p["detected"].numpy()):
            rows.append({
                "image_sha256": sha,
                "arcface_fp32": emb.tolist() if det else None,
                "ffhq_arcface_detected": bool(det),
            })
    return pd.DataFrame(rows)


def load_flux_corpus(root: Path) -> pd.DataFrame:
    """Merge the three backfilled Flux-corpus parquets on image_sha256."""
    paths = {
        "sample_index":  root / "models/blendshape_nmf/sample_index.parquet",
        "classifier":    root / "output/demographic_pc/classifier_scores.parquet",
        "siglip":        root / "output/demographic_pc/clip_probes_siglip2.parquet",
    }
    dfs = {k: pd.read_parquet(p) for k, p in paths.items() if p.exists()}
    base = dfs["classifier"][["image_sha256",
                              "mv_age", "mv_gender", "mv_gender_conf",
                              "ff_age_bin", "ff_gender", "ff_race",
                              "ff_age_probs", "ff_gender_probs", "ff_race_probs",
                              "ff_detected",
                              "ins_age", "ins_gender", "ins_detected",
                              "ins_embedding"]].copy()
    base = base.rename(columns={"ins_embedding": "arcface_fp32"})
    base["source"] = "flux_corpus_v3"

    # blendshapes + atoms from sample_index
    bs_cols = [c for c in dfs["sample_index"].columns if c.startswith(("bs_", "atom_"))]
    base = base.merge(
        dfs["sample_index"][["image_sha256"] + bs_cols],
        on="image_sha256", how="left",
    )

    # siglip probes (rename to sg_*_margin to match FFHQ schema)
    sg = dfs["siglip"].copy()
    sg = sg.rename(columns={c: f"sg_{c}" for c in sg.columns if c.endswith("_margin")})
    sg_cols = [c for c in sg.columns if c.startswith("sg_") and c.endswith("_margin")]
    base = base.merge(sg[["image_sha256"] + sg_cols], on="image_sha256", how="left")

    # drop rows with null sha256 (unresolved in backfill)
    base = base[base["image_sha256"].notna()].reset_index(drop=True)
    return base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ffhq-metrics-dir", type=Path,
                    default=ROOT / "output/ffhq_metrics/metrics")
    ap.add_argument("--ffhq-encoded-dir", type=Path,
                    default=ROOT / "output/ffhq_metrics/encoded")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "output/reverse_index/reverse_index.parquet")
    ap.add_argument("--log", type=Path,
                    default=ROOT / "output/reverse_index/build_log.json")
    args = ap.parse_args()

    print("[reverse-index] loading FFHQ metrics shards")
    ffhq_metrics = pd.concat(
        [load_ffhq_shard(p) for p in sorted(args.ffhq_metrics_dir.glob("*.pt"))],
        ignore_index=True,
    )
    print(f"  ffhq_metrics: {len(ffhq_metrics)} rows")

    print("[reverse-index] loading FFHQ ArcFace shards")
    ffhq_arc = load_flux_arcface(args.ffhq_encoded_dir)
    print(f"  ffhq_arcface: {len(ffhq_arc)} rows")

    print("[reverse-index] joining FFHQ metrics + ArcFace on image_sha256")
    ffhq = ffhq_metrics.merge(ffhq_arc, on="image_sha256", how="left")
    print(f"  ffhq joined: {len(ffhq)} rows")

    print("[reverse-index] loading Flux corpus")
    flux = load_flux_corpus(ROOT)
    print(f"  flux_corpus_v3: {len(flux)} rows")

    print("[reverse-index] concatenating")
    unified = pd.concat([flux, ffhq], ignore_index=True, sort=False)
    print(f"  unified: {len(unified)} rows, {len(unified.columns)} cols")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    unified.to_parquet(args.out, index=False)
    print(f"[reverse-index] wrote {args.out} ({args.out.stat().st_size/1e6:.1f} MB)")

    log = {
        "n_total": len(unified),
        "n_ffhq": int((unified["source"] == "ffhq").sum()),
        "n_flux": int((unified["source"] == "flux_corpus_v3").sum()),
        "columns": sorted(unified.columns.tolist()),
        "ffhq_metrics_dir": str(args.ffhq_metrics_dir),
        "ffhq_encoded_dir": str(args.ffhq_encoded_dir),
    }
    with args.log.open("w") as f:
        json.dump(log, f, indent=2)
    print(f"[reverse-index] log -> {args.log}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the merger**

```bash
uv run python -m demographic_pc.build_unified_reverse_index \
    --ffhq-metrics-dir output/ffhq_metrics/metrics \
    --ffhq-encoded-dir output/ffhq_metrics/encoded
```

Expected:
- ffhq_metrics: ~70000 rows
- ffhq_arcface: ~70000 rows
- ffhq joined: ~70000 rows
- flux_corpus_v3: ~7400-7800 rows (depends on backfill resolution rate)
- unified: ~77000-78000 rows
- output ~50-200 MB

- [ ] **Step 3: Sanity-check distributions**

```bash
uv run python - <<'EOF'
import pandas as pd
df = pd.read_parquet("output/reverse_index/reverse_index.parquet")
print("=== overall counts ===")
print(df.groupby("source").size())
print()
print("=== detection rates by source ===")
for col in ["ff_detected", "ins_detected", "bs_detected"]:
    if col in df.columns:
        print(df.groupby("source")[col].mean())
print()
print("=== gender split per source (ff_gender) ===")
print(df.groupby(["source", "ff_gender"]).size().unstack(fill_value=0))
EOF
```

Expected (approximate):
- FFHQ: ~50/50 gender, ~60% bs_detected (MediaPipe is more permissive than SCRFD)
- Flux corpus: skewed by prompt-pair distribution (probably more M than F by a few percent)

- [ ] **Step 4: Commit the merger + small log; the parquet is large and goes under output/ which is already gitignored — verify**

```bash
git check-ignore output/reverse_index/reverse_index.parquet
# Expected: prints the path = it IS ignored. If not, add 'output/reverse_index/*.parquet' to .gitignore.
git add src/demographic_pc/build_unified_reverse_index.py output/reverse_index/build_log.json
git commit -m "feat(reverse-index): merge FFHQ + Flux corpus into one parquet"
```

---

## Task 12: Write the research note

**Files:**
- Create: `docs/research/2026-04-28-ffhq-reverse-index-extension.md`
- Modify: `docs/research/_topics/demographic-pc-pipeline.md` (append a 2026-04-28 entry)

The skill rule: load-bearing decisions go in a dated research doc; the topic file is the mutable interpretation layer.

- [ ] **Step 1: Write the dated research doc**

```markdown
---
status: live
topic: demographic-pc-pipeline
---

# FFHQ reverse-index extension (2026-04-28)

## Decision

Run the same five reverse-index metric families that we have on the
Flux-corpus side (mivolo + fairface + insightface attrs + siglip-2
attribute probes + ARKit blendshapes + NMF atom projection from
`au_library.npz`) on all 70k FFHQ images, at 512² resolution to match
Flux corpus dimensions, on the remote Windows GPU box.

## Why

  - We already have the FFHQ corpus on the Windows box (95 GB, 190
    parquet shards) for the arc_latent distillation work.
  - The Flux corpus is small (~7800 rows) and synthetically biased.
    A real-world reference at 10× the size lets us:
      * sanity-check the classifiers and probes on natural-image
        distributions (gender ~50/50, age curve broader, etc.);
      * compare manifold geometry of synthetic vs real on shared
        metrics;
      * give the arc_latent student a real-world test set even though
        its training set is also FFHQ.

## How

Single-pass extractor (`src/demographic_pc/extract_ffhq_metrics.py`)
loads all five wrappers once, holds them resident on the RTX 3090,
and processes one parquet shard at a time. Per-shard `.pt` outputs
under `C:\arc_distill\metrics\` are merged locally with the
encode_ffhq.py ArcFace outputs (`C:\arc_distill\encoded\`) into a
unified `output/reverse_index/reverse_index.parquet` keyed by
`image_sha256`.

Detection-rate caveat carried over from
`docs/research/2026-04-27-arcface-detection-threshold.md`: at the
canonical SCRFD det_thresh=0.5, FFHQ has ~40% miss rate from the
ArcFace path. MiVOLO does not detect (predicts on the whole image),
so it returns a number for every row — this is fine for "what does
the model think" but should not be interpreted as "every row is a
face." MediaPipe FaceLandmarker has its own internal detector with
different priors and recovers a different (typically wider) subset.

## Anti-goals

  - Not training anything new — purely an extraction pass over
    existing models.
  - Not changing the Flux corpus reverse-index schema — only
    backfilling `image_sha256` so the join key exists.
  - Not extending the FluxSpace per-render measurements
    (`axis/scale/attn_*`) to FFHQ — those are render-time and don't
    apply to real photos.

## Storage

  - Per-shard FFHQ metrics `.pt`: ~0.5 MB × 190 shards ≈ 100 MB
  - Per-shard FFHQ encoded `.pt` (already produced by
    encode_ffhq.py): ~50 MB × 190 ≈ 9.4 GB
  - Unified `reverse_index.parquet`: estimated 100-200 MB
```

- [ ] **Step 2: Append a topic entry**

Append to `docs/research/_topics/demographic-pc-pipeline.md`:

```markdown
- **2026-04-28 — FFHQ reverse-index extension.** All five reverse-index metric families (mivolo + fairface + insightface attrs + siglip-2 attribute probes + ARKit blendshapes + NMF atom projection from `au_library.npz`) now extracted over the 70k FFHQ corpus at 512² to match Flux dimensions. Single-pass extractor at `src/demographic_pc/extract_ffhq_metrics.py`; merger at `src/demographic_pc/build_unified_reverse_index.py`. Output: `output/reverse_index/reverse_index.parquet` (sources: `flux_corpus_v3`, `ffhq`; join key: `image_sha256`). FFHQ detection rates: ArcFace/SCRFD ~60% (det_thresh=0.5; per `2026-04-27-arcface-detection-threshold.md`), MediaPipe ~higher (different detector priors). MiVOLO predicts on every row — interpret accordingly. See `2026-04-28-ffhq-reverse-index-extension.md`.
```

- [ ] **Step 3: Commit**

```bash
git add docs/research/2026-04-28-ffhq-reverse-index-extension.md \
        docs/research/_topics/demographic-pc-pipeline.md
git commit -m "docs(ffhq): reverse-index extension research note + topic entry"
```

---

## Self-Review

**Spec coverage** — every requirement from the trigger message:
  - ✅ All five metric families covered: demographic classifiers (Tasks 4, 8, 9); SigLIP-2 probes (Task 4); ARKit blendshapes + NMF atom projection (Task 4 + helpers in Task 3).
  - ✅ FFHQ at 512² for Flux parity: locked in `decode_and_resize` (Task 3) and in the `--resolution 512` default everywhere downstream (Tasks 4, 8, 9).
  - ✅ Remote Windows execution: assets sync (Task 6), deps install (Task 7), batch wrapper + smoke (Task 8), full launch via schtasks (Task 9).
  - ✅ Unified reverse-index parquet keyed by `image_sha256`: backfill on Flux side (Task 10), merger (Task 11).
  - ✅ ArcFace not re-run: extractor's `InsightFaceClassifier` is built with `with_embedding=False` (default in classifiers.py); ArcFace embeddings come from `encode_ffhq.py` outputs and are joined in Task 11.
  - ✅ Per-shard, resumable: skip-if-exists in Task 4's `main()`, mirrors encode_ffhq.py.

**Placeholder scan** — searched for "TODO", "TBD", "implement later", "fill in details", "appropriate error handling", "similar to". Only false positives are inside the 'Anti-goals' bullets in Task 12 (the literal word "training" appears, but in negation). No actual placeholder steps.

**Type / signature consistency** — `decode_and_resize`, `compute_image_sha256`, and `project_blendshapes_to_atoms` are referenced by exactly the names defined in Task 3. The `--mediapipe-model` and `--au-library` CLI flags are consistent across Tasks 4, 5, 8. The output schema fields (`mv_age`, `ff_detected`, `bs_<name>`, `atom_NN`, `sg_<probe>_margin`) match across Task 4 (writer) and Task 11 (reader). The merger's `load_flux_corpus` consumes `bs_*` and `atom_*` from `sample_index.parquet`, which matches the existing schema we observed.

**Risks the engineer should know about** —
  1. **MiVOLO + FairFace + InsightFace + SigLIP-2 + MediaPipe co-resident on a 24 GB 3090.** Estimate is ~14 GB peak; if it OOMs, easy fix is to run SigLIP-2 in fp16 (`.to(torch.float16)`) — current behaviour is fp32. Defer until Task 8 smoke shows whether it's a problem.
  2. **`dlib` install on Windows can be painful.** Task 7 step 3 has a fallback wheel URL; if both fail, the FairFace path is the casualty (it's the only consumer of dlib). MiVOLO + InsightFace + SigLIP-2 + blendshapes still work — extractor would emit zeros for the `ff_*` columns. Worth doing FairFace second-priority.
  3. **Flux corpus PNGs may have been deleted since the parquets were generated.** Task 10 step 3 prints `unresolved paths`; if it's >5%, investigate before merging — those rows would silently disappear from the unified parquet.

---

## Execution Handoff

After all 12 tasks are complete, the next concrete user-facing question is:

> *Does the unified reverse-index reproduce known patterns when sliced by source?*

That's a one-evening exploratory analysis (`uv run jupyter ...`), not part of this plan. Suggest spinning it up as a follow-up note (`docs/research/2026-04-29-reverse-index-cross-corpus-eda.md`) once the parquet is built.
