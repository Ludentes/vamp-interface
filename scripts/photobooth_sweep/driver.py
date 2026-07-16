"""Photobooth Phase-1 sweep driver.

Reads `data/importer/identities/<pid>.png` for the 4 phase-1 photos, samples
LHS over the 6 axes, and for each cell runs:

    src → frame_face → ctrl(canny|depth) → Z-Image CN render → HyperSwap →
    refine (low denoise) → score → save intermediates + append row

Resumes by re-reading scores.parquet; cells already present are skipped.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import swap_core  # noqa: E402
from scripts.photobooth_sweep import axes as ax_mod  # noqa: E402
from scripts.photobooth_sweep import preprocess as pp  # noqa: E402
from scripts.photobooth_sweep import refine as rf  # noqa: E402
from scripts.photobooth_sweep import scorer as sc  # noqa: E402

PHOTO_IDS = ["id_00", "id_11", "id_16", "id_01"]
PHASE2_PHOTO_IDS = [f"id_{i:02d}" for i in range(20)]
ANCHOR = ROOT / "exp_output/matryoshka_bakeoff/renders/zimage_turbo_st06_euler_simple_seed74029470.png"
BASE_PROMPT = (
    "a vibrant traditional Russian matryoshka nesting doll, glossy red and "
    "gold lacquer, ornate floral painting, with a realistic photographic "
    "human face, soft three-dimensional shading, correct facial proportions, "
    "defined nose and lips, natural-sized eyes, centered frontal face, "
    "wooden doll, plain background")
DEMO_LOOKUP_PATH = ROOT / "data/importer/identities/manifest.csv"


def load_demo_lookup() -> dict[str, dict[str, str]]:
    import csv
    out: dict[str, dict[str, str]] = {}
    with open(DEMO_LOOKUP_PATH, newline="") as f:
        for row in csv.DictReader(f):
            pid = row["filename"].replace(".png", "")
            out[pid] = {"gender": row["gender"], "age_bin": row["age_bin"],
                        "race": row["race"]}
    return out


def make_prompt(pid: str, demo: dict[str, str], inject: str) -> str:
    if inject == "off":
        return BASE_PROMPT
    g = {"M": "man's", "F": "woman's"}.get(demo["gender"], "person's")
    return (f"a {demo['age_bin']}-year-old {demo['race']} {g} face, "
            + BASE_PROMPT)


def cn_workflow(comfy_url: str, *, ctrl_name: str, prompt: str,
                render_hw: tuple[int, int], cn_strength: float, seed: int,
                prefix: str) -> dict:
    wf = json.load(
        open(ROOT / "comfyui/workflows/photobooth_zimage_cn.api.json"))
    subs = {"$$POSITIVE_PROMPT": prompt, "$$WIDTH": render_hw[1],
            "$$HEIGHT": render_hw[0], "$$CTRL_IMAGE": ctrl_name,
            "$$CN_STRENGTH": float(cn_strength), "$$SEED": int(seed),
            "$$STEPS": 6, "$$OUTPUT_PREFIX": prefix}
    for n in wf.values():
        for k, v in n.get("inputs", {}).items():
            if isinstance(v, str) and v in subs:
                n["inputs"][k] = subs[v]
    return wf


def comfy_submit(comfy_url: str, wf: dict) -> np.ndarray:
    cid = str(uuid.uuid4())
    r = requests.post(f"{comfy_url}/prompt",
                      json={"prompt": wf, "client_id": cid}, timeout=30)
    r.raise_for_status()
    pid = r.json()["prompt_id"]
    for _ in range(600):
        h = requests.get(f"{comfy_url}/history/{pid}", timeout=10).json()
        if pid in h:
            st = h[pid]["status"]
            if st.get("status_str") != "success":
                errs = [m for m in st.get("messages", [])
                        if m[0] == "execution_error"]
                raise RuntimeError(f"cn render failed: {errs[:1]}")
            for o in h[pid].get("outputs", {}).values():
                for im in o.get("images", []):
                    rr = requests.get(f"{comfy_url}/view", params={
                        "filename": im["filename"],
                        "subfolder": im.get("subfolder", ""),
                        "type": "output"}, timeout=60)
                    return cv2.imdecode(np.frombuffer(rr.content, np.uint8),
                                        cv2.IMREAD_COLOR)
            raise RuntimeError("cn render: no image")
        time.sleep(0.5)
    raise RuntimeError("cn render: timeout")


def existing_cells(scores_path: Path) -> set[str]:
    if not scores_path.exists():
        return set()
    t = pq.read_table(scores_path, columns=["cell_id"])
    return set(t.column("cell_id").to_pylist())


def append_row(scores_path: Path, row: dict) -> None:
    new = pa.table({k: [v] for k, v in row.items()})
    if scores_path.exists():
        old = pq.read_table(scores_path)
        # align schemas — promote types
        combined = pa.concat_tables([old, new], promote_options="default")
    else:
        combined = new
    pq.write_table(combined, scores_path)


def run_cell(app, swapper, comfy_url: str, *, pid: str, cfg_idx: int,
             cfg: dict, src_bgr: np.ndarray, src_face,
             demo: dict[str, str], anchor_emb: np.ndarray,
             cells_root: Path, seed_iter: int = 0,
             reuse_render_from: Path | None = None) -> dict:
    cell = ax_mod.cell_id(pid, cfg_idx, seed_iter)
    cdir = cells_root / cell
    cdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    prompt = make_prompt(pid, demo, cfg["demo_inject"])
    seed = 100000 + cfg_idx + 1000000 * seed_iter

    # 1+2. control image + CN render -- reused from another phase if the
    # render is invariant under this phase's swept axes.
    reuse_src = None
    if reuse_render_from is not None:
        # Phase 3's renders are cfg000..cfg005 per photo, all with the locked
        # generation defaults but different seeds. cfg000 is the canonical
        # source -- copy its render.png into this Phase-4 cell.
        candidate = reuse_render_from / f"{pid}__cfg000" / "render.png"
        if candidate.exists():
            reuse_src = candidate

    if reuse_src is not None:
        ctrl_src = reuse_render_from / f"{pid}__cfg000" / "ctrl.png"
        if ctrl_src.exists():
            ctrl_img = cv2.imread(str(ctrl_src))
            if ctrl_img is not None:
                cv2.imwrite(str(cdir / "ctrl.png"), ctrl_img)
        render = cv2.imread(str(reuse_src))
        if render is None:
            raise RuntimeError(f"reuse: failed to read {reuse_src}")
        cv2.imwrite(str(cdir / "render.png"), render)
        render_hw = (render.shape[0], render.shape[1])
    else:
        ctrl, render_hw = pp.build_control(
            app, comfy_url, src_bgr, cfg["cn_condition"], cfg["canny_preset"],
            cfg["face_pixel_budget"])
        cv2.imwrite(str(cdir / "ctrl.png"), ctrl)
        ctrl_name = pp.upload_control(comfy_url, ctrl, cell)
        wf = cn_workflow(comfy_url, ctrl_name=ctrl_name, prompt=prompt,
                         render_hw=render_hw, cn_strength=cfg["cn_strength"],
                         seed=seed, prefix=f"phb_cn_{cell}")
        render = comfy_submit(comfy_url, wf)
        cv2.imwrite(str(cdir / "render.png"), render)

    # 3. swap (with optional mask shaping)
    swap_w = float(cfg.get("swap_weight", 0.5))
    mask_mode = cfg.get("mask_mode")  # None | "erode" | "feather"
    mask_radius = int(cfg.get("mask_radius", 0))
    erode_px = mask_radius if mask_mode == "erode" else 0
    feather_px = mask_radius if mask_mode == "feather" else 0
    swap, det_mode, det_pre = swap_core.swap_identity(
        app, swapper, render, src_face, collapse=True, restore=False,
        swap_weight=swap_w, mask_erode_px=erode_px,
        mask_feather_px=feather_px)
    cv2.imwrite(str(cdir / "swap.png"), swap)

    # 4. refine
    refined = rf.refine(comfy_url, swap, prompt=prompt,
                        denoise=cfg["refine_denoise"], seed=seed, tag=cell)
    cv2.imwrite(str(cdir / "refined.png"), refined)

    # 5. score
    src_emb = src_face.normed_embedding
    s = sc.score_cell(app, refined_bgr=refined, src_emb=src_emb,
                      anchor_emb=anchor_emb, det_mode=det_mode,
                      wall_clock=time.time() - t0)
    row = {
        "cell_id": cell, "photo_id": pid, "cfg_idx": cfg_idx,
        "seed_iter": seed_iter, "seed": seed,
        "face_pixel_budget": cfg["face_pixel_budget"],
        "cn_condition": cfg["cn_condition"],
        "cn_strength": float(cfg["cn_strength"]),
        "canny_preset": cfg["canny_preset"] or "",
        "refine_denoise": float(cfg["refine_denoise"]),
        "demo_inject": cfg["demo_inject"],
        "swap_weight": swap_w,
        "mask_mode": mask_mode or "",
        "mask_radius": mask_radius,
        **{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
           for k, v in s.items()},
        "ok": True, "err": "",
    }
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="exp_output/photobooth_phase1",
                   help="output root (cells go in `cells/<cell_id>/`)")
    p.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    p.add_argument("--n-cells", type=int, default=40)
    p.add_argument("--photos", default=None,
                   help="comma list; default: phase1=4 / phase2=20")
    p.add_argument("--seed", type=int, default=42, help="LHS seed (phase1)")
    p.add_argument("--mode", choices=("phase1", "phase2", "phase3", "phase4"),
                   default="phase1")
    p.add_argument("--seed-iter", type=int, default=0,
                   help="phase2: 0/1/2 — adds 1e6 to base seed per iter")
    p.add_argument("--reuse-render-from", default=None,
                   help="phase root to copy render.png from (e.g. "
                        "exp_output/photobooth_phase3); only the cfg000 cell "
                        "per photo is reused. Skips CN render call entirely.")
    args = p.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cells_root = out / "cells"
    cells_root.mkdir(exist_ok=True)
    scores_path = out / "scores.parquet"
    manifest_path = out / "manifest.json"

    if args.mode == "phase2":
        cfgs = ax_mod.phase2_configs()
    elif args.mode == "phase3":
        cfgs = ax_mod.phase3_configs()
    elif args.mode == "phase4":
        cfgs = ax_mod.phase4_configs()
    else:
        cfgs = ax_mod.lhs_sample(args.n_cells, seed=args.seed)
    json.dump(cfgs, open(manifest_path, "w"), indent=2)
    print(f"manifest: {manifest_path}  mode={args.mode}  "
          f"({len(cfgs)} configs, seed_iter={args.seed_iter})")

    app = swap_core.make_face_app()
    swapper = swap_core.load_swapper()
    demo = load_demo_lookup()

    anchor_emb = sc.clip_embed(cv2.imread(str(ANCHOR)))

    done = existing_cells(scores_path)
    print(f"resume: {len(done)} cells already in scores.parquet")

    if args.photos:
        photos = [p for p in args.photos.split(",") if p]
    else:
        photos = (PHASE2_PHOTO_IDS
                  if args.mode in ("phase2", "phase3", "phase4")
                  else PHOTO_IDS)

    reuse_root = ROOT / args.reuse_render_from / "cells" if args.reuse_render_from else None
    if reuse_root is not None:
        print(f"reuse renders from: {reuse_root}")

    for pid in photos:
        src = cv2.imread(str(ROOT / f"data/importer/identities/{pid}.png"))
        if src is None:
            print(f"SKIP {pid}: not found"); continue
        src_face = swap_core.detect_source(app, src)
        if src_face is None:
            print(f"SKIP {pid}: no face on source"); continue
        for i, cfg in enumerate(cfgs):
            cell = ax_mod.cell_id(pid, i, args.seed_iter)
            if cell in done:
                continue
            try:
                row = run_cell(app, swapper, args.comfy_url, pid=pid,
                               cfg_idx=i, cfg=cfg, src_bgr=src,
                               src_face=src_face, demo=demo[pid],
                               anchor_emb=anchor_emb,
                               cells_root=cells_root,
                               seed_iter=args.seed_iter,
                               reuse_render_from=reuse_root)
                append_row(scores_path, row)
                print(f"{cell}  id={row['id_cos']:.3f}  "
                      f"det={row['det_mode']:>7}  "
                      f"ff={row['face_frac']:.3f}  "
                      f"cs={row['clip_style']:.3f}  "
                      f"wall={row['wall_clock']:.1f}s")
            except Exception as e:
                err_row = {
                    "cell_id": cell, "photo_id": pid, "cfg_idx": i,
                    "seed_iter": args.seed_iter,
                    "seed": 100000 + i + 1000000 * args.seed_iter,
                    "face_pixel_budget": cfg["face_pixel_budget"],
                    "cn_condition": cfg["cn_condition"],
                    "cn_strength": float(cfg["cn_strength"]),
                    "canny_preset": cfg["canny_preset"] or "",
                    "refine_denoise": float(cfg["refine_denoise"]),
                    "demo_inject": cfg["demo_inject"],
                    "swap_weight": float(cfg.get("swap_weight", 0.5)),
                    "mask_mode": cfg.get("mask_mode") or "",
                    "mask_radius": int(cfg.get("mask_radius", 0)),
                    "id_cos": float("nan"), "det_score": float("nan"),
                    "det_mode": "exception", "face_frac": float("nan"),
                    "clip_style": float("nan"), "wall_clock": 0.0,
                    "ok": False, "err": str(e)[:300],
                }
                append_row(scores_path, err_row)
                print(f"{cell}  FAIL  {err_row['err']}")


if __name__ == "__main__":
    main()
