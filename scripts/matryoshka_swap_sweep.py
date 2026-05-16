"""Matryoshka identity-swap sweep: PuLID doll + inswapper identity.

The PuLID-weight ladder proved PuLID cannot transfer identity through the
flat-painted matryoshka style -- but a PuLID doll IS a coherent doll with a
detectable face, and inswapper swaps a real identity onto it cleanly. Roles
split: PuLID generates the doll, inswapper is the identity vehicle.

This sweep is 252 cells: 21 anchors x 3 prompt finishes x 2 CN strengths x
2 PuLID starts, one deterministic seed each. Recipe fixed: PuLID weight 1.5,
SCRFD-default swap, no repaint. Run ON the ComfyUI box, cwd = data/importer:

    python ../../scripts/matryoshka_swap_sweep.py \
        --comfy-url http://127.0.0.1:8188 \
        --workflow workflows/flux_pulid_canny_lora.api.json \
        --id-dir identities_flux --template refs/matryoshka/template_canny.png \
        --swapper C:/comfy/ComfyUI/models/insightface/inswapper_128.onnx \
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
