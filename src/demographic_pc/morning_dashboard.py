"""One-command morning runner for the overnight Phase-4 broad validation.

Runs in order:
  1. analyze_direction_validation — computes per-atom cos(measured, target)
     across scales; saves `validation_report.json`
  2. analyze_ridge_vs_causal — quick comparison table (already saved
     separately but printed here for cross-reference)
  3. direction_inject_broad_collage — per-atom scale-sweep PNGs for visual
     triage
  4. Short verdict summary: atoms with cos > 0.3 at the largest tested
     scale, grouped by region from atom_region_labels.

Run: `uv run python -m src.demographic_pc.morning_dashboard`
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BROAD = ROOT / "output/demographic_pc/direction_inject_broad"
NMF_DIR = ROOT / "models/blendshape_nmf"


def run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=False)


def main():
    # 1. validation
    run([sys.executable, "-m", "src.demographic_pc.analyze_direction_validation"])
    # 2. ridge-vs-causal reference
    run([sys.executable, "-m", "src.demographic_pc.analyze_ridge_vs_causal"])
    # 3. collages
    run([sys.executable, "-m", "src.demographic_pc.direction_inject_broad_collage"])

    # 4. regional summary
    val_report = BROAD / "validation_report.json"
    regions_json = NMF_DIR / "atom_regions.json"
    if not val_report.exists():
        print("\n[summary] no validation report — run did not finish")
        return
    val = json.loads(val_report.read_text())
    regions = {r["atom"]: r.get("region", "?")
               for r in json.loads(regions_json.read_text())}

    print(f"\n[verdict] {val['wins']}/{val['total']} atoms with cos > 0.3 at max scale\n")
    # Per-atom best-scale cos
    rows: list[tuple[int, float, float, str]] = []
    for atom in val["atoms"]:
        aid = atom["atom"]
        scales = sorted({r["scale"] for r in atom["rows"]})
        if not scales:
            continue
        max_scale = max(scales)
        best = [r["cos"] for r in atom["rows"] if r["scale"] == max_scale]
        mean_cos = sum(best) / len(best) if best else 0.0
        rows.append((aid, max_scale, mean_cos, regions.get(aid, "?")))
    rows.sort(key=lambda r: -r[2])
    print(f"{'atom':>4}  {'max-scale':>9}  {'mean cos':>9}  region")
    for r in rows:
        ind = "✓" if r[2] > 0.3 else " "
        print(f"  {r[0]:>2} {ind} {r[1]:>8.2f}  {r[2]:>+9.3f}  {r[3]}")


if __name__ == "__main__":
    main()
