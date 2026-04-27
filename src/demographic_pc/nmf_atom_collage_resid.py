"""Top-activating-samples collage for residualised NMF atoms, annotated with
(base, gender, age) so demographic spread is visible per row.

Loads: models/blendshape_nmf/{W_nmf_resid.npy, H_nmf_resid.npy, manifest_resid.json}
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
NMF_DIR = ROOT / "models/blendshape_nmf"
OUT_PNG = NMF_DIR / "atom_extremes_collage_resid.png"

N_COLS = 5
THUMB = 180
LABEL_W = 280

# Mirrors blendshape_decomposition tag logic.
SOURCE_DIRS = {
    "bootstrap_v1":                METRICS / "bootstrap_v1",
    "alpha_interp":                METRICS / "crossdemo/smile/alpha_interp",
    "smile_inphase":               METRICS / "crossdemo/smile/smile_inphase",
    "jaw_inphase":                 METRICS / "crossdemo/smile/jaw_inphase",
    "intensity_full":              METRICS / "crossdemo/smile/intensity_full",
    "anger/rebalance":             METRICS / "crossdemo/anger/rebalance",
    "surprise/rebalance":          METRICS / "crossdemo/surprise/rebalance",
    "disgust/rebalance":           METRICS / "crossdemo/disgust/rebalance",
    "pucker/rebalance":            METRICS / "crossdemo/pucker/rebalance",
    "lip_press/rebalance":         METRICS / "crossdemo/lip_press/rebalance",
}


def resolve_path(sid: str) -> Path | None:
    # sid format: "<tag>/<rel>" where tag may contain a slash (anger/rebalance).
    for tag, dir_ in sorted(SOURCE_DIRS.items(), key=lambda x: -len(x[0])):
        prefix = tag + "/"
        if sid.startswith(prefix):
            return dir_ / sid[len(prefix):]
    return None


def top_channel_str(W_k: np.ndarray, channels_stack: list[str], top_n: int = 3) -> str:
    idx = np.argsort(-W_k)[:top_n]
    keep = [i for i in idx if W_k[i] > 1e-3]
    return ", ".join(f"{channels_stack[i]}({W_k[i]:.2f})" for i in keep) or "(dead)"


def main() -> None:
    manifest = json.loads((NMF_DIR / "manifest_resid.json").read_text())
    W = np.load(NMF_DIR / "W_nmf_resid.npy")
    H = np.load(NMF_DIR / "H_nmf_resid.npy")
    sample_ids = manifest["sample_ids"]
    bases = manifest["bases"]
    base_meta = manifest["base_meta"]
    channels_stack = manifest["channels_stack"]
    k = W.shape[0]
    print(f"[collage-resid] k={k}, N={H.shape[0]}")

    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 12)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)

    W_img = LABEL_W + N_COLS * THUMB
    ROW_H = THUMB + 12
    H_img = k * ROW_H + 30
    canvas = Image.new("RGB", (W_img, H_img), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 4), f"residualised NMF (k={k}) — top-{N_COLS} activators per atom "
              f"with (base | gender | age)", fill="black", font=font)
    y0 = 24

    atom_report: list[dict] = []
    for a in range(k):
        col = H[:, a]
        top_idx = np.argsort(-col)[:N_COLS]
        y = y0 + a * ROW_H
        head = f"#{a:02d}\n" + top_channel_str(W[a], channels_stack, top_n=3)
        draw.multiline_text((8, y + 4), head, fill="black", font=small, spacing=2)
        picks = []
        for ci, i in enumerate(top_idx):
            sid = sample_ids[int(i)]
            path = resolve_path(sid)
            base = bases[int(i)]
            eth, gen, age = base_meta.get(base, ("?", "?", "?"))
            x = LABEL_W + ci * THUMB
            if path and path.exists():
                img = Image.open(path).convert("RGB")
                img.thumbnail((THUMB, THUMB))
                cell = Image.new("RGB", (THUMB, THUMB), (240, 240, 240))
                cell.paste(img, ((THUMB - img.width) // 2,
                                 (THUMB - img.height) // 2))
                canvas.paste(cell, (x, y))
            else:
                draw.rectangle([x, y, x + THUMB, y + THUMB], outline="red")
                draw.text((x + 8, y + THUMB // 2), "missing", fill="red", font=small)
            tag = f"{base}\n{gen}|{age}  c={col[int(i)]:.2f}"
            draw.multiline_text((x + 4, y + THUMB - 30), tag, fill=(200, 40, 40),
                                font=small, spacing=1)
            picks.append({"sid": sid, "base": base, "gender": gen,
                          "age": age, "coeff": float(col[int(i)])})
        # Per-atom demographic-spread metric
        top_bases_set = {p["base"] for p in picks}
        top_genders = {p["gender"] for p in picks}
        atom_report.append({
            "atom": a,
            "n_distinct_bases_topK": len(top_bases_set),
            "n_distinct_genders_topK": len(top_genders),
            "picks": picks,
        })

    canvas.save(OUT_PNG)
    print(f"[save] {OUT_PNG}  ({canvas.width}×{canvas.height})")
    (NMF_DIR / "atom_extremes_resid_report.json").write_text(
        json.dumps(atom_report, indent=2))
    # Summary
    n_base_hist = [0] * (N_COLS + 1)
    for r in atom_report:
        n_base_hist[r["n_distinct_bases_topK"]] += 1
    print(f"[summary] distribution of distinct bases in top-{N_COLS} per atom:")
    for nb, count in enumerate(n_base_hist):
        if count:
            print(f"  {nb} distinct bases: {count} atoms")
    mean_spread = float(np.mean([r["n_distinct_bases_topK"] for r in atom_report]))
    print(f"[summary] mean distinct bases in top-K = {mean_spread:.2f}  "
          f"(perfect spread would be {min(N_COLS, 6)})")


if __name__ == "__main__":
    main()
