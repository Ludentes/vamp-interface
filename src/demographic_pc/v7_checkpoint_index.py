"""Cheap numerical index over v7 training-sample collages.

Walks `output/ai_toolkit_runs/glasses_slider_v7/samples/` for the 9-cell
grids (3 demographics × 3 strengths) saved every `save_every` step by
ai-toolkit. For each sample image computes:

  - SigLIP-2 probe margins (glasses + 5 bundle channels)
  - ArcFace 512-d identity embedding (insightface buffalo_l)

Then aggregates per checkpoint:

  - engagement_pos        mean siglip_glasses_margin at +1.5 across demos
  - engagement_per_demo   the 3 individual values
  - engagement_consistency  min of those three (worst-demo glasses)
  - bundle_pos            sum_p mean |Δ_demo siglip_p_margin(s=+1.5) - (s=0)|
                          across the 5 non-glasses probes
  - identity_pos          mean ArcFace cos vs s=0-same-demo at +1.5
  - identity_neg          same at -1.5

Output: `models/sliders/glasses_v7/checkpoint_index.parquet`

The point of this index is to answer:
  "which checkpoint should we branch from for the lower-LR refinement?"
numerically, instead of by eye.

CPU-only by default so it can run alongside the trainer. Re-run any time
to pick up new checkpoints — existing rows are skipped via `--force`
gating.

Usage:
    PYTHONPATH=src uv run python -m demographic_pc.v7_checkpoint_index
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/newub/w/vamp-interface")
SAMPLES = ROOT / "output/ai_toolkit_runs/glasses_slider_v7/samples"
OUT = ROOT / "models/sliders/glasses_v7/checkpoint_index.parquet"

# v7 yaml prompt order — index 0..8 maps to (demo, strength)
GRID = [
    ("east_asian_m", -1.5),
    ("east_asian_m",  0.0),
    ("east_asian_m", +1.5),
    ("black_f",      -1.5),
    ("black_f",       0.0),
    ("black_f",      +1.5),
    ("european_m",   -1.5),
    ("european_m",    0.0),
    ("european_m",   +1.5),
]

DEMOS = ["east_asian_m", "black_f", "european_m"]
NAME_RE = re.compile(r".*__(\d{9})_(\d)\.jpg$")


@dataclass
class Cell:
    step: int
    demo: str
    strength: float
    path: Path


def discover_cells(samples: Path) -> list[Cell]:
    out = []
    for p in sorted(samples.iterdir()):
        m = NAME_RE.match(p.name)
        if not m:
            continue
        step = int(m.group(1))
        idx = int(m.group(2))
        demo, strength = GRID[idx]
        out.append(Cell(step, demo, strength, p))
    return out


def score_all(cells: list[Cell], device: str) -> pd.DataFrame:
    """Run SigLIP + ArcFace once per image. Returns per-cell long-form df."""
    from src.demographic_pc.score_clip_probes import Siglip2Backend
    from src.demographic_pc.classifiers import InsightFaceClassifier
    from PIL import Image
    import torch

    print(f"[score] {len(cells)} cells on device={device}")
    print("[score] loading SigLIP-2-so400m")
    siglip = Siglip2Backend(device)

    # Pre-encode probe text features
    from demographic_pc.measure_slider import SIGLIP_PROBES
    probe_text = {}
    for name, pos, neg in SIGLIP_PROBES:
        with torch.no_grad():
            inp = siglip.processor(text=[pos, neg], return_tensors="pt",
                                   padding="max_length", truncation=True).to(device)
            tf = siglip._as_tensor(siglip.model.get_text_features(**inp))
            tf = tf / tf.norm(dim=-1, keepdim=True)
        probe_text[name] = tf  # [2, D]

    print("[score] loading InsightFace (ArcFace embedding)")
    import cv2
    ctx = -1 if device == "cpu" else 0
    ins = InsightFaceClassifier(ctx_id=ctx, with_embedding=True)

    rows = []
    for i, c in enumerate(cells):
        if i % 25 == 0:
            print(f"  [{i+1}/{len(cells)}] step={c.step} {c.demo} s={c.strength:+.1f}")
        row: dict = {"step": c.step, "demo": c.demo, "strength": c.strength,
                     "rel": c.path.name}
        # SigLIP margins
        feat = siglip.encode_image(c.path)  # [1, D] L2-normed
        for name, tf in probe_text.items():
            sims = (feat @ tf.T).squeeze(0).tolist()  # [2]
            row[f"siglip_{name}_margin"] = float(sims[0] - sims[1])
        # ArcFace embedding via InsightFace.predict(bgr)
        bgr = cv2.imread(str(c.path), cv2.IMREAD_COLOR)
        ins_out = ins.predict(bgr) if bgr is not None else {"embedding": None,
                                                            "detected": False}
        emb = ins_out.get("embedding")
        if emb is not None:
            row["arc_emb"] = np.asarray(emb, dtype=np.float32).tobytes()
            row["face_detected"] = True
        else:
            row["arc_emb"] = None
            row["face_detected"] = False
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Per-checkpoint summary."""
    # ArcFace identity cosine vs same-step-same-demo s=0
    base_df = per_cell[per_cell["strength"] == 0.0]
    base: dict[tuple[int, str], bytes | None] = {}
    for _, br in base_df.iterrows():
        base[(int(br["step"]), str(br["demo"]))] = br["arc_emb"]
    cosines: list[float] = []
    for _, r in per_cell.iterrows():
        if not bool(r.get("face_detected", False)) or r["arc_emb"] is None:
            cosines.append(float("nan")); continue
        key = (int(r["step"]), str(r["demo"]))
        b_buf = base.get(key)
        if b_buf is None:
            cosines.append(float("nan")); continue
        a = np.frombuffer(bytes(r["arc_emb"]), dtype=np.float32)
        b = np.frombuffer(bytes(b_buf), dtype=np.float32)
        cosines.append(float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
    per_cell = per_cell.copy()
    per_cell["identity_cos_to_base"] = cosines

    bundle_probes = ["earrings", "hair_long", "hair_curly", "formal_clothing", "beard"]

    summary = []
    for step, sub in per_cell.groupby("step"):
        row: dict = {"step": int(step)}
        # engagement
        pos = sub[sub["strength"] == 1.5]
        zero = sub[sub["strength"] == 0.0]
        neg = sub[sub["strength"] == -1.5]

        per_demo_pos = {d: float(pos[pos["demo"] == d]["siglip_glasses_margin"].mean())
                        for d in DEMOS}
        per_demo_zero = {d: float(zero[zero["demo"] == d]["siglip_glasses_margin"].mean())
                         for d in DEMOS}
        per_demo_engage = {d: per_demo_pos[d] - per_demo_zero[d] for d in DEMOS}

        for d in DEMOS:
            row[f"glasses_pos_{d}"] = per_demo_pos[d]
            row[f"glasses_engage_{d}"] = per_demo_engage[d]
        row["engagement_pos_mean"] = float(np.mean(list(per_demo_pos.values())))
        row["engagement_delta_mean"] = float(np.mean(list(per_demo_engage.values())))
        row["engagement_consistency"] = float(min(per_demo_engage.values()))
        row["glasses_neg_mean"] = float(neg["siglip_glasses_margin"].mean())

        # bundle: per-probe Δ from s=0 to s=+1.5, averaged abs across demos, summed
        bundle_total = 0.0
        for p in bundle_probes:
            col = f"siglip_{p}_margin"
            d_per = []
            for d in DEMOS:
                p15 = pos[pos["demo"] == d][col].mean()
                p00 = zero[zero["demo"] == d][col].mean()
                d_per.append(abs(float(p15 - p00)))
            row[f"bundle_{p}"] = float(np.mean(d_per))
            bundle_total += row[f"bundle_{p}"]
        row["bundle_pos"] = bundle_total

        # identity
        row["identity_pos_mean"] = float(pos["identity_cos_to_base"].mean())
        row["identity_pos_min"] = float(pos["identity_cos_to_base"].min())
        row["identity_neg_mean"] = float(neg["identity_cos_to_base"].mean())
        row["identity_neg_min"] = float(neg["identity_cos_to_base"].min())

        # composite — high engagement, high identity, low bundle, all demos engaged
        row["score_balanced"] = (
            row["engagement_consistency"]                         # worst-demo glasses delta
            - 0.5 * row["bundle_pos"]                             # bundle penalty
            + 0.3 * max(row["identity_pos_mean"] - 0.4, 0.0)      # identity bonus over passing line
        )
        summary.append(row)

    return pd.DataFrame(summary).sort_values("step").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--force", action="store_true",
                    help="Re-score all cells even if a parquet already exists")
    args = ap.parse_args()

    cells = discover_cells(SAMPLES)
    print(f"[discover] {len(cells)} sample cells across "
          f"{len({c.step for c in cells})} checkpoints")

    cells_pq = OUT.with_name(OUT.stem + "_cells.parquet")
    if cells_pq.exists() and not args.force:
        prior = pd.read_parquet(cells_pq)
        seen = set(prior["rel"])
        new = [c for c in cells if c.path.name not in seen]
        print(f"[discover] {len(new)} new cells (skipping {len(prior)} existing)")
        if new:
            new_df = score_all(new, args.device)
            per_cell = pd.concat([prior, new_df], ignore_index=True)
        else:
            per_cell = prior
    else:
        per_cell = score_all(cells, args.device)

    cells_pq.parent.mkdir(parents=True, exist_ok=True)
    per_cell.to_parquet(cells_pq, index=False)
    print(f"[write] {cells_pq} ({len(per_cell)} rows)")

    summary = aggregate(per_cell)
    summary.to_parquet(OUT, index=False)
    print(f"[write] {OUT} ({len(summary)} checkpoints)")

    # Pretty print
    cols = ["step", "engagement_pos_mean", "engagement_consistency",
            "bundle_pos", "identity_pos_mean", "identity_neg_mean", "score_balanced"]
    print("\n== per-checkpoint summary ==")
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

    best = summary.loc[summary["score_balanced"].idxmax()]
    print(f"\n[recommendation] highest score_balanced: step {int(best['step'])} "
          f"(score={best['score_balanced']:+.3f}, "
          f"eng_consist={best['engagement_consistency']:+.3f}, "
          f"bundle={best['bundle_pos']:+.3f}, "
          f"id={best['identity_pos_mean']:+.3f})")


if __name__ == "__main__":
    main()
