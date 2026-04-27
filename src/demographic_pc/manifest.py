"""Experiment manifest helper — writes a manifest.json alongside each render
output directory so future sessions can reconstruct what the data is, when it
was gathered, why, and what parameters produced it.

Usage (from a generation script):
    from src.demographic_pc.manifest import write_manifest
    write_manifest(
        output_dir=ROOT,
        name="fluxspace_alpha_interp",
        purpose="Test linearity of mix_b interpolation between Mona Lisa and Joker endpoint prompts across 6 bases × 10 seeds × 11 alpha values.",
        parameters={
            "bases": [b.name for b in BASES_FULL],
            "alphas": ALPHAS,
            "seeds": SEEDS,
            "edit_a": EDIT_A, "edit_b": EDIT_B,
            "scale": SCALE, "start_percent": START_PCT,
        },
        related_to=["bootstrap_v1", "intensity_full"],
    )
"""

from __future__ import annotations

import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=3,
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def write_manifest(output_dir: Path, name: str, purpose: str,
                   parameters: dict[str, Any],
                   related_to: list[str] | None = None,
                   measurement_notes: str | None = None) -> Path:
    """Write {output_dir}/manifest.json with structured metadata.

    Fields:
      name           — short experiment identifier (matches directory name)
      purpose        — 1–3 sentences on WHY this data was gathered
      date_iso       — ISO UTC timestamp at write time
      git_commit     — current repo HEAD short SHA (for reproducing the script)
      script_argv    — how this generation script was invoked
      parameters     — all free parameters of the sweep (bases, scales, seeds…)
      measurement    — notes about how to interpret the output (e.g. blendshape
                       channels expected to be dead, known failure modes)
      related_to     — list of other experiment names this data extends/depends on
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "purpose": purpose,
        "date_iso": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "script_argv": sys.argv,
        "parameters": parameters,
        "measurement": measurement_notes,
        "related_to": related_to or [],
    }
    dest = output_dir / "manifest.json"
    with dest.open("w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return dest
