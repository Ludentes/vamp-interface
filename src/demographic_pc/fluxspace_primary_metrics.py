"""Primary-metric analysis for FluxSpace cross-demo glasses axis.

Three passes, all over existing renders and measurement pkls:
  1. ArcFace drift — cosine distance from s=0 baseline per base, per scale.
  2. CLIP glasses-presence — P(wearing glasses) per render.
  3. Geometry cos(δ_mix, attn_base) per measurement pkl — aggregated stats.

Outputs:
  - output/.../crossdemo/primary_metrics.json
  - output/.../crossdemo/primary_metrics.png (twin-axis per base)
  - output/.../crossdemo/geometry.json

Usage:
    uv run python -m src.demographic_pc.fluxspace_primary_metrics
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from src.demographic_pc.fluxspace_metrics import AXES, _axis_dir

# Overridden in main() based on --axis
AXIS = "glasses"
CROSS_BASES = AXES[AXIS]["bases"]
ROOT_DIR = _axis_dir(AXIS)
ANAL_DIR = ROOT_DIR / "primary"
CLIP_POS = AXES[AXIS]["clip_pos"]
CLIP_NEG = AXES[AXIS]["clip_neg"]


def _arcface_model():
    from src.demographic_pc.stage4_5_evaluate import load_arcface
    return load_arcface()


def _arc_embed(model, device, pil: Image.Image) -> np.ndarray:
    import torch
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    t = tf(pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        e = model(t)
    e = e / e.norm(p=2, dim=-1, keepdim=True)
    return e.squeeze(0).float().cpu().numpy()


def _clip_glasses_model():
    import open_clip
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.eval().to(device)
    tok = open_clip.get_tokenizer("ViT-B-32")
    prompts = [CLIP_POS, CLIP_NEG]
    with torch.no_grad():
        text = tok(prompts).to(device)
        tfeat = model.encode_text(text)
        tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    return model, preprocess, tfeat, device


def _clip_glasses_prob(model, preprocess, tfeat, device, pil: Image.Image) -> float:
    import torch
    x = preprocess(pil.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        logits = (feat @ tfeat.T) * 100.0
        probs = logits.softmax(dim=-1).squeeze(0).cpu().numpy()
    return float(probs[0])


def _gather_renders() -> dict[str, list[tuple[float, Path]]]:
    """Return {base: [(scale, path), ...]} sorted by scale."""
    out: dict[str, list[tuple[float, Path]]] = {}
    for name, _, _ in CROSS_BASES:
        vdir = ROOT_DIR / "verify" / name
        if not vdir.exists():
            continue
        items = []
        for p in sorted(vdir.glob("s*.png")):
            # filename: s+0.30.png / s-0.45.png
            stem = p.stem[1:]  # strip 's'
            try:
                s = float(stem)
            except ValueError:
                continue
            items.append((s, p))
        items.sort(key=lambda t: t[0])
        # Add measurement render as s=+1.0 if no verify s=+1.00 exists
        out[name] = items
    return out


def pass1_drift_and_glasses() -> dict:
    renders = _gather_renders()
    arc, dev = _arcface_model()
    clip_model, clip_pre, clip_tfeat, clip_dev = _clip_glasses_model()

    per_base: dict = {}
    for base, items in renders.items():
        print(f"[primary] {base}: {len(items)} renders")
        embs = {}
        glasses = {}
        for s, p in items:
            pil = Image.open(p)
            emb = _arc_embed(arc, dev, pil)
            gp = _clip_glasses_prob(clip_model, clip_pre, clip_tfeat, clip_dev, pil)
            embs[s] = emb
            glasses[s] = gp

        # Baseline s=0.0 — prefer exact match, else nearest to 0
        if 0.0 in embs:
            base_s = 0.0
        else:
            base_s = min(embs.keys(), key=lambda x: abs(x))
        base_emb = embs[base_s]

        drift = {s: float(1.0 - np.dot(base_emb, e)) for s, e in embs.items()}
        per_base[base] = {
            "base_s": base_s,
            "drift": drift,
            "glasses_prob": glasses,
        }
    return per_base


def pass2_geometry() -> dict:
    """cos(δ_mix, attn_base) per (block, step) from each measurement pkl."""
    out: dict = {}
    for name, _, _ in CROSS_BASES:
        pkl = ROOT_DIR / "measurement" / f"{name}_meas.pkl"
        if not pkl.exists():
            continue
        with pkl.open("rb") as f:
            d = pickle.load(f)
        coss = []
        ratios = []  # ‖δ‖ / ‖base‖
        for step, blocks in d["steps"].items():
            for bk, e in blocks.items():
                ab = e["attn_base"]["mean_d"].numpy()
                dm = e["delta_mix"]["mean_d"].numpy()
                ab_n = np.linalg.norm(ab)
                dm_n = np.linalg.norm(dm)
                if ab_n < 1e-9 or dm_n < 1e-9:
                    continue
                coss.append(float(np.dot(ab, dm) / (ab_n * dm_n)))
                ratios.append(float(dm_n / ab_n))
        out[name] = {
            "cos_mean": float(np.mean(coss)),
            "cos_abs_mean": float(np.mean(np.abs(coss))),
            "cos_p95_abs": float(np.percentile(np.abs(coss), 95)),
            "ratio_mean": float(np.mean(ratios)),
            "ratio_p95": float(np.percentile(ratios, 95)),
            "n_entries": len(coss),
        }
    return out


def _plot(per_base: dict) -> None:
    ANAL_DIR.mkdir(parents=True, exist_ok=True)
    n = len(per_base)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3.2*n))
    if n == 1:
        axes = [axes]
    for ax, (base, info) in zip(axes, per_base.items()):
        scales = sorted(info["drift"].keys())
        drift = [info["drift"][s] for s in scales]
        glp = [info["glasses_prob"][s] for s in scales]
        ax2 = ax.twinx()
        ln1, = ax.plot(scales, drift, marker="o", color="#d94040", label="ArcFace drift (1 − cos)")
        ln2, = ax2.plot(scales, glp, marker="s", color="#3a8adf", label="CLIP P(glasses)")
        ax.axvline(info["base_s"], color="#999", ls=":", alpha=0.7)
        ax.set_title(f"{base}   (baseline s={info['base_s']:+.2f})")
        ax.set_xlabel("s")
        ax.set_ylabel("drift", color="#d94040")
        ax2.set_ylabel("P(glasses)", color="#3a8adf")
        ax.set_ylim(-0.02, max(drift) * 1.1 + 0.01)
        ax2.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(handles=[ln1, ln2], loc="upper left", fontsize=9)
    fig.tight_layout()
    out = ANAL_DIR / "primary_metrics.png"
    fig.savefig(out, dpi=120)
    print(f"[primary] wrote {out}")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", default="glasses", choices=list(AXES))
    args = ap.parse_args()
    global AXIS, CROSS_BASES, ROOT_DIR, ANAL_DIR, CLIP_POS, CLIP_NEG
    AXIS = args.axis
    CROSS_BASES = AXES[AXIS]["bases"]
    ROOT_DIR = _axis_dir(AXIS)
    ANAL_DIR = ROOT_DIR / "primary"
    CLIP_POS = AXES[AXIS]["clip_pos"]
    CLIP_NEG = AXES[AXIS]["clip_neg"]
    ANAL_DIR.mkdir(parents=True, exist_ok=True)
    print("[primary] pass 1: drift + glasses")
    per_base = pass1_drift_and_glasses()
    print("[primary] pass 2: geometry")
    geom = pass2_geometry()

    # Summarise + write json
    summary = {"per_base": per_base, "geometry": geom}
    with (ANAL_DIR / "primary_metrics.json").open("w") as f:
        json.dump(summary, f, indent=2, default=lambda o: float(o) if hasattr(o, "item") else str(o))

    # Correlate window width with geometry
    # Observed safe-width from the confirmation doc (rough):
    observed_width = {
        "asian_m": 1.70, "black_f": 1.70, "european_m": 2.70,
        "elderly_latin_m": 1.65, "young_european_f": 2.70, "southasian_f": 2.90,
    }
    print("\n=== geometry vs observed window width ===")
    print(f"{'base':<20} {'cos_mean':>9} {'cos|mean|':>9} {'cos|p95|':>9} {'ratio_mean':>11} {'ratio_p95':>10} {'width':>7}")
    for base, g in geom.items():
        print(f"{base:<20} {g['cos_mean']:>9.4f} {g['cos_abs_mean']:>9.4f} "
              f"{g['cos_p95_abs']:>9.4f} {g['ratio_mean']:>11.4f} "
              f"{g['ratio_p95']:>10.4f} {observed_width.get(base,0):>7.2f}")

    _plot(per_base)

    # Summary per base
    print("\n=== per-base primary metrics ===")
    for base, info in per_base.items():
        ss = sorted(info["drift"].keys())
        gs = info["glasses_prob"]
        dr = info["drift"]
        hi_glass = max(gs.values())
        # Scale where glasses prob peaks
        s_peak = max(gs, key=lambda s: gs[s])
        # Scale where glasses prob exceeds 0.5 with lowest drift
        useful = [s for s in ss if gs[s] > 0.5]
        best = min(useful, key=lambda s: dr[s]) if useful else None
        print(f"{base:<20} peak P(g)={hi_glass:.2f} @ s={s_peak:+.2f}  "
              f"best s(g>0.5, min drift)={best}  "
              f"drift@s=0: {dr[info['base_s']]:.3f}")


if __name__ == "__main__":
    main()
