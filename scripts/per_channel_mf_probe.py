"""Per-input-channel m_f response probe (cheap, m_f space only).

For each of the 58 input channels:
  1. Build a neutral b_61 baseline.
  2. Bump channel i to its p95 value.
  3. Run student → m_f.
  4. Save delta = m_f(bump) - m_f(baseline) as 32x16 heatmap.

Why: student↔teacher cos=0.99 globally hides cell-level errors that produce
visible artefacts (e.g. ghost brows on take 8 f48). This shows which output
cells respond to each input channel — and lets us cross-reference with the
per-cell R² map to spot channels whose responsible m_f cells are weakly fit.

Outputs:
  - <out>/mf_response_channel_<idx>_<name>.png   (delta heatmap)
  - <out>/mf_response_summary.json               (per-channel L2, peak cell, R² of peak cell)
  - <out>/mf_response_grid.png                   (28x4 = 112 panel composite for browse)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.student import MotEncoderStudent           # noqa: E402
from arkit_bridge.llf_csv import ARKIT_BLENDSHAPE_NAMES      # noqa: E402

INPUT_NAMES = list(ARKIT_BLENDSHAPE_NAMES) + [
    "leftEyeYaw", "leftEyePitch", "leftEyeRoll",
    "rightEyeYaw", "rightEyePitch", "rightEyeRoll",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--eval_r2", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    stats = np.load(args.stats)
    b_p95 = stats["b_p95"].astype(np.float32)
    r2 = np.load(args.eval_r2)["r2"]   # (32,16)

    s = MotEncoderStudent().to(args.device).eval()
    s.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    base = torch.zeros((1, 58), device=args.device)
    with torch.no_grad():
        m_base = s(base).squeeze().cpu().numpy()  # (32,16)

    summary = []
    deltas = []
    for i in range(58):
        bp = base.clone()
        bp[0, i] = float(b_p95[i])
        with torch.no_grad():
            m_bump = s(bp).squeeze().cpu().numpy()
        delta = m_bump - m_base                     # (32,16)
        deltas.append(delta)
        flat = delta.reshape(-1)
        peak = int(np.argmax(np.abs(flat)))
        prow, pcol = peak // 16, peak % 16
        summary.append({
            "channel": i,
            "name": INPUT_NAMES[i],
            "b_p95": float(b_p95[i]),
            "delta_l2": float(np.linalg.norm(flat)),
            "delta_peak_abs": float(np.abs(flat[peak])),
            "peak_cell": [prow, pcol],
            "peak_cell_r2": float(r2[prow, pcol]),
            # how much of the response lives in cells with R²<0.5
            "low_r2_energy_frac": float(
                ((delta**2) * (r2 < 0.5)).sum() / max(1e-9, (delta**2).sum())
            ),
        })

    # Summary JSON
    json_path = out / "mf_response_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # Per-channel heatmaps + composite grid
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib missing ({e}); skipping figs")
        return

    deltas = np.stack(deltas)        # (58, 32, 16)
    vmax = float(np.max(np.abs(deltas)))

    # Composite grid: 8 cols × 8 rows = 64 slots (only 58 used)
    cols, rows = 8, 8
    fig, axes = plt.subplots(rows, cols, figsize=(cols*1.4, rows*1.6))
    for i in range(rows*cols):
        ax = axes[i // cols, i % cols]
        ax.set_xticks([]); ax.set_yticks([])
        if i >= 58:
            ax.axis("off"); continue
        ax.imshow(deltas[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"{i}:{INPUT_NAMES[i][:14]}", fontsize=6)
    plt.tight_layout()
    grid_path = out / "mf_response_grid.png"
    plt.savefig(grid_path, dpi=110)
    plt.close(fig)

    # Sorted reports
    by_l2 = sorted(summary, key=lambda d: -d["delta_l2"])
    by_lowr2 = sorted(summary, key=lambda d: -d["low_r2_energy_frac"])

    text = ["=== top-15 channels by m_f delta L2 (most influential) ==="]
    for d in by_l2[:15]:
        text.append(
            f"  ch{d['channel']:>2d} {d['name']:<22s}  L2={d['delta_l2']:.3f}  "
            f"peak[{d['peak_cell'][0]:>2d},{d['peak_cell'][1]:>2d}] R²={d['peak_cell_r2']:+.3f}  "
            f"low_r2_frac={d['low_r2_energy_frac']:.3f}"
        )
    text.append("")
    text.append("=== top-15 channels with most response in LOW-R² cells (risk) ===")
    for d in by_lowr2[:15]:
        text.append(
            f"  ch{d['channel']:>2d} {d['name']:<22s}  low_r2_frac={d['low_r2_energy_frac']:.3f}  "
            f"L2={d['delta_l2']:.3f}  peak[{d['peak_cell'][0]:>2d},{d['peak_cell'][1]:>2d}] "
            f"R²={d['peak_cell_r2']:+.3f}"
        )
    text.append("")
    text.append("=== brow channels detail ===")
    for d in summary:
        if "brow" in d["name"].lower() or "Brow" in d["name"]:
            text.append(
                f"  ch{d['channel']:>2d} {d['name']:<22s}  L2={d['delta_l2']:.3f}  "
                f"peak[{d['peak_cell'][0]:>2d},{d['peak_cell'][1]:>2d}] R²={d['peak_cell_r2']:+.3f}  "
                f"low_r2_frac={d['low_r2_energy_frac']:.3f}"
            )

    txt_path = out / "mf_response_report.txt"
    txt_path.write_text("\n".join(text))
    print("\n".join(text))
    print(f"\nwrote {grid_path}")
    print(f"wrote {json_path}")
    print(f"wrote {txt_path}")


if __name__ == "__main__":
    main()
