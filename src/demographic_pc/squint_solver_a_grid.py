"""Solver A squint composition — full demographic grid (7 races × 2 genders × 3 ages × 4 seeds).

Renders the squint primary pair at scale=1.0 (no iterative counters)
across the full grid, scores each cell, builds a collage. Findings on
european_m (2 seeds, both clean) and young_european_f (scale-collapse
beyond ~1.0) suggest the bundle isn't the issue at scale=1.0; what we
want from this grid is which *bases* engage cleanly.

Outputs:
  output/demographic_pc/solver_a_squint_grid/<race>_<gender>_<age>/seed<N>.png
  output/demographic_pc/solver_a_squint_grid/scores.parquet
  output/demographic_pc/solver_a_squint_grid/grid_collage.png
  output/demographic_pc/solver_a_squint_grid/report.md
"""

from __future__ import annotations

import asyncio
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output/demographic_pc/solver_a_squint_grid"

RACES = [
    ("white", "European"),
    ("black", "Black"),
    ("east_asian", "East Asian"),
    ("south_asian", "South Asian"),
    ("middle_eastern", "Middle Eastern"),
    ("southeast_asian", "Southeast Asian"),
    ("latino", "Latin American"),
]
GENDERS = [("m", "man"), ("f", "woman")]
AGES = [
    ("young", "young"),
    ("adult", "adult"),
    ("elderly", "elderly"),
]
SEEDS = [
    2026, 4242, 1337, 8675,
    1010, 2929, 3838, 4747,
    5151, 6262, 7373, 8484,
    9595, 1212, 2323, 3434,
]
SCALE = 1.0
START_PCT = 0.15
END_PCT = 1.0


def base_prompt(age: str, race: str, gender: str) -> str:
    return (
        f"A photorealistic portrait photograph of {'an' if age[0] in 'aeiou' else 'a'} "
        f"{age} {race} {gender}, neutral expression, plain grey background, "
        f"studio lighting, sharp focus."
    )


def squint_pos(age: str, race: str, gender: str) -> str:
    return (
        f"{'An' if age[0] in 'aeiou' else 'A'} {age} {race} {gender} with eyes "
        f"squinted tightly together, mouth completely neutral and relaxed, "
        f"plain grey background, studio lighting, sharp focus."
    )


def squint_neg(age: str, race: str, gender: str) -> str:
    return (
        f"{'An' if age[0] in 'aeiou' else 'A'} {age} {race} {gender} with eyes "
        f"wide open and relaxed, mouth completely neutral and relaxed, "
        f"plain grey background, studio lighting, sharp focus."
    )


async def render_grid() -> list[dict]:
    from src.demographic_pc.comfy_flux import ComfyClient
    from src.demographic_pc.fluxspace_metrics import pair_measure_workflow

    rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async with ComfyClient() as client:
        # anchors per (race, gender, age) — scale=0 short-circuits the edit
        for r_key, r_name in RACES:
            for g_key, g_word in GENDERS:
                for a_key, a_word in AGES:
                    cell = f"{r_key}_{g_key}_{a_key}"
                    cell_dir = OUT_DIR / cell
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    base = base_prompt(a_word, r_name, g_word)
                    edit_a = squint_pos(a_word, r_name, g_word)
                    edit_b = squint_neg(a_word, r_name, g_word)
                    for seed in SEEDS:
                        # anchor (scale=0) — used for identity reference
                        anchor_png = cell_dir / f"seed{seed}_anchor.png"
                        edit_png = cell_dir / f"seed{seed}_squint.png"
                        if not anchor_png.exists():
                            wf = pair_measure_workflow(
                                seed, None, f"sa_{cell}_anchor_s{seed}",
                                base_prompt=base, edit_a=base, edit_b=base,
                                scale=0.0, start_percent=START_PCT, end_percent=END_PCT)
                            await client.generate(wf, anchor_png)
                        if not edit_png.exists():
                            wf = pair_measure_workflow(
                                seed, None, f"sa_{cell}_squint_s{seed}",
                                base_prompt=base, edit_a=edit_a, edit_b=edit_b,
                                scale=SCALE, start_percent=START_PCT, end_percent=END_PCT)
                            await client.generate(wf, edit_png)
                        rows.append({
                            "race": r_key, "gender": g_key, "age": a_key,
                            "seed": seed,
                            "anchor_png": str(anchor_png.relative_to(ROOT)),
                            "edit_png": str(edit_png.relative_to(ROOT)),
                        })
    return rows


def score_grid(rows: list[dict]) -> pd.DataFrame:
    """Score each anchor + edit; report Δsquint, smile drift, cheek drift, identity cosine."""
    import cv2
    from src.demographic_pc.score_blendshapes import make_landmarker, score_png
    from src.demographic_pc.classifiers import InsightFaceClassifier
    from src.demographic_pc.score_clip_probes import Siglip2Backend
    import torch

    print("[load] InsightFace + SigLIP-2…")
    ins = InsightFaceClassifier(with_embedding=True)
    sig = Siglip2Backend("cuda" if torch.cuda.is_available() else "cpu")
    probe_feats = sig.encode_probes()

    out = []
    with make_landmarker() as lm:
        for r in rows:
            row = dict(r)
            for tag, key in (("anchor", "anchor_png"), ("edit", "edit_png")):
                p = ROOT / r[key]
                bgr = cv2.imread(str(p))
                bs = score_png(lm, p) if bgr is not None else None
                if bs is not None:
                    row[f"{tag}_squint"] = float(
                        bs.get("eyeSquintLeft", 0.0) + bs.get("eyeSquintRight", 0.0)
                    )
                    row[f"{tag}_smile_bs"] = float(
                        bs.get("mouthSmileLeft", 0.0) + bs.get("mouthSmileRight", 0.0)
                    )
                    row[f"{tag}_cheek"] = float(
                        bs.get("cheekSquintLeft", 0.0) + bs.get("cheekSquintRight", 0.0)
                    )
                    row[f"{tag}_brow"] = float(
                        bs.get("browDownLeft", 0.0) + bs.get("browDownRight", 0.0)
                    )
                else:
                    for k in ("squint", "smile_bs", "cheek", "brow"):
                        row[f"{tag}_{k}"] = float("nan")
                if bgr is not None:
                    feat = sig.encode_image(p)
                    for name, pf in probe_feats.items():
                        sims = (feat @ pf.T).squeeze(0)
                        row[f"{tag}_siglip_{name}"] = float(sims[0] - sims[1])
                    i = ins.predict(bgr)
                    row[f"{tag}_emb"] = i["embedding"] if i["embedding"] is not None else None
                    row[f"{tag}_face_detected"] = bool(i["detected"])
            # identity cos
            ae = row.get("anchor_emb"); ee = row.get("edit_emb")
            if ae is not None and ee is not None:
                a = np.asarray(ae); b = np.asarray(ee)
                row["id_cos"] = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            else:
                row["id_cos"] = float("nan")
            for k in ("anchor_emb", "edit_emb"):
                row.pop(k, None)
            row["d_squint"] = row["edit_squint"] - row["anchor_squint"]
            row["d_smile_bs"] = row["edit_smile_bs"] - row["anchor_smile_bs"]
            row["d_cheek"] = row["edit_cheek"] - row["anchor_cheek"]
            row["d_smile_siglip"] = row["edit_siglip_smiling"] - row["anchor_siglip_smiling"]
            out.append(row)
    return pd.DataFrame(out)


def make_collage(df: pd.DataFrame, out_path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    cells = sorted({(r["race"], r["gender"], r["age"]) for _, r in df.iterrows()})
    seeds = sorted(df["seed"].unique().tolist())
    thumb = 96
    pad = 2
    label_h = 28
    cell_w = (thumb * 2 + pad) * len(seeds) + pad
    cell_h = thumb + label_h + pad
    rows = math.ceil(len(cells) / 3)
    W = 3 * cell_w + 24
    H = rows * cell_h + 32
    img = Image.new("RGB", (W, H), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()

    for k, (r, g, a) in enumerate(cells):
        col = k % 3
        rr = k // 3
        x0 = 12 + col * cell_w
        y0 = 16 + rr * cell_h
        cell_rows = df[(df["race"] == r) & (df["gender"] == g) & (df["age"] == a)].sort_values("seed")
        cell_rows = cell_rows.reset_index(drop=True)
        for s, (_, row) in enumerate(cell_rows.iterrows()):
            try:
                a_im = Image.open(ROOT / row["anchor_png"]).resize((thumb, thumb))
                e_im = Image.open(ROOT / row["edit_png"]).resize((thumb, thumb))
                xs = x0 + s * (thumb * 2 + pad)
                img.paste(a_im, (xs, y0))
                img.paste(e_im, (xs + thumb, y0))
                # mark per-seed |Δsquint|
                ds = row.get("d_squint", float("nan"))
                ds_text = f"{ds:+.2f}" if ds == ds else "nan"
                draw.text((xs, y0 + thumb + 1), ds_text, fill=(220, 220, 220), font=font)
            except Exception:
                pass
        label = f"{r}/{g}/{a}"
        draw.text((x0, y0 - 14), label, fill=(255, 200, 90), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def write_report(df: pd.DataFrame, out_path: Path) -> None:
    grouped = df.groupby(["race", "gender", "age"]).agg(
        d_squint_mean=("d_squint", "mean"),
        d_squint_std=("d_squint", "std"),
        d_smile_bs_mean=("d_smile_bs", "mean"),
        d_smile_siglip_mean=("d_smile_siglip", "mean"),
        d_cheek_mean=("d_cheek", "mean"),
        id_cos_mean=("id_cos", "mean"),
        face_det_rate=("edit_face_detected", "mean"),
    ).reset_index()

    n_cells = len(grouped)
    pass_engage = (grouped["d_squint_mean"] > 0.10).sum()
    pass_clean = ((grouped["d_squint_mean"] > 0.10) & (grouped["d_smile_siglip_mean"].abs() < 0.05)).sum()
    pass_intact = (grouped["face_det_rate"] >= 0.75).sum()

    lines = [
        "# Solver A squint grid result",
        "",
        f"- cells (race × gender × age): {n_cells}",
        f"- seeds per cell: {len(SEEDS)}",
        f"- scale: {SCALE} (primary only, no counters)",
        "",
        "## Pass card",
        "",
        f"- cells with mean Δsquint > 0.10: {pass_engage} / {n_cells}",
        f"- cells with engagement *and* |Δsmile_siglip| < 0.05: {pass_clean} / {n_cells}",
        f"- cells with face detection rate ≥ 0.75 (no scale-collapse): {pass_intact} / {n_cells}",
        "",
        "## Per-cell summary (mean over 4 seeds)",
        "",
        "| race | gender | age | Δsquint | Δsmile_bs | Δsmile_siglip | Δcheek | id_cos | face_det |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in grouped.iterrows():
        lines.append(
            f"| {r['race']} | {r['gender']} | {r['age']} | "
            f"{r['d_squint_mean']:+.3f} | {r['d_smile_bs_mean']:+.3f} | "
            f"{r['d_smile_siglip_mean']:+.3f} | {r['d_cheek_mean']:+.3f} | "
            f"{r['id_cos_mean']:.3f} | {r['face_det_rate']:.2f} |"
        )
    out_path.write_text("\n".join(lines))
    print("\n".join(lines[:30]))


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[render] {len(RACES) * len(GENDERS) * len(AGES) * len(SEEDS)} edit renders + same anchors")
    rows = await render_grid()
    print("[score] all renders…")
    df = score_grid(rows)
    df.to_parquet(OUT_DIR / "scores.parquet")
    make_collage(df, OUT_DIR / "grid_collage.png")
    write_report(df, OUT_DIR / "report.md")


if __name__ == "__main__":
    asyncio.run(main())
