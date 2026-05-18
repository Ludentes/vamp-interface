"""Drive the depth-ControlNet stacked spike: 3 identities x 4 axes x strength.

For each axis, picks one high-coefficient FFHQ exemplar, rasterizes its
MediaPipe face mesh into a depth map, and drives a stock FLUX Depth ControlNet
with it while identity-only InfuseNet (black spatial control, fixed strength
0.6) holds the face. A neutral-expression depth map gives the per-identity
baseline. Scores identity drift (ArcFace cosine) and expression match
(blendshape cosine vs the exemplar). Resumable: skips a fresh output PNG. See
docs/superpowers/specs/2026-05-18-arkit-depth-controlnet-spike-design.md.
"""
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from arkit_controlnet.eval_spike import arcface_cos, expr_cos
from arkit_controlnet.landmark_control import (
    render_depth_map, select_exemplars, select_neutral,
)
from arkit_controlnet.run_spike import select_identities
from demographic_pc.comfy_flux import ComfyClient

WORKFLOW = Path("comfyui/workflows/arkit_depth_spike.json")
OUT_DIR = Path("exp_output/arkit_depth_spike")
SEED = 2026
DEPTH_STRENGTHS = [0.5, 0.8]
AXES_TO_RUN = ["smile", "pucker", "surprise"]
_CANVAS_W, _CANVAS_H = 864, 1152
_MIN_PNG_BYTES = 1024


def _is_fresh(png: Path) -> bool:
    return png.exists() and png.stat().st_size >= _MIN_PNG_BYTES


def build_workflow(identity_filename: str, depth_filename: str,
                   black_filename: str, strength: float,
                   out_prefix: str) -> dict:
    """Substitute the $$-placeholders in the workflow template."""
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$DEPTH_FILENAME": depth_filename,
        "$$BLACK_FILENAME": black_filename,
        "$$DEPTH_STRENGTH": float(strength),
        "$$SEED": int(SEED),
        "$$OUTPUT_PREFIX": out_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    return _sub(copy.deepcopy(json.loads(WORKFLOW.read_text())))


def _prepare_inputs() -> tuple[dict[str, dict[str, Path]], Path]:
    """Render one depth map per axis (+ neutral) and one black image into
    OUT_DIR. Returns ({axis: {"depth": Path, "exemplar": Path}}, black_png).
    Falls through the k=3 exemplar candidates if MediaPipe re-detection fails.
    """
    inputs: dict[str, dict[str, Path]] = {}
    specs = [(ax, select_exemplars(ax, k=3)) for ax in AXES_TO_RUN]
    specs.append(("neutral", select_neutral(k=3)))
    for axis, candidates in specs:
        for exemplar in candidates:
            try:
                depth = render_depth_map(exemplar)
            except ValueError:
                continue
            depth_png = OUT_DIR / f"depth_{axis}.png"
            cv2.imwrite(str(depth_png), depth)
            inputs[axis] = {"depth": depth_png, "exemplar": exemplar}
            break
        else:
            raise RuntimeError(f"no exemplar yielded a depth map for axis {axis}")
    black_png = OUT_DIR / "black.png"
    cv2.imwrite(str(black_png),
                np.zeros((_CANVAS_H, _CANVAS_W, 3), dtype=np.uint8))
    return inputs, black_png


async def _run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identities = select_identities(3)
    inputs, black_png = _prepare_inputs()
    rows = []
    # neutral first: a collapsed neutral row means the InfuseNet black-image
    # control failed, which makes the depth result inconclusive (see design).
    axis_order = ["neutral", *AXES_TO_RUN]
    async with ComfyClient() as client:
        black_name = await client.upload_image(black_png)
        for ident in identities:
            id_name = await client.upload_image(ident)
            for axis in axis_order:
                spec = inputs[axis]
                depth_name = await client.upload_image(spec["depth"])
                for strength in DEPTH_STRENGTHS:
                    tag = f"{ident.stem}__{axis}__str{strength:.2f}"
                    out_png = OUT_DIR / f"{tag}.png"
                    if not _is_fresh(out_png):
                        wf = build_workflow(id_name, depth_name, black_name,
                                            strength, tag)
                        try:
                            await client.generate(wf, out_png)
                        except Exception as exc:
                            print(f"  FAILED {tag}: {exc}")
                            continue
                    try:
                        af = arcface_cos(out_png, ident)
                        ec = expr_cos(out_png, spec["exemplar"])
                    except Exception as exc:
                        print(f"  METRIC FAILED {tag}: {exc}")
                        af, ec = -1.0, -1.0
                    rows.append({
                        "identity": ident.stem, "axis": axis,
                        "strength": strength, "arcface_cos": af,
                        "expr_cos": ec,
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "metrics.parquet")
    print(df.to_string(index=False))


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
