"""Drive the landmark-control spike: 3 identities x 3 axes x strength band.

For each axis, picks one high-coefficient FFHQ exemplar, renders its MediaPipe
mesh as the InfuseNet control image, and generates each identity under that
control. A neutral-expression control gives the per-identity baseline. Scores
identity drift (ArcFace cosine) and expression match (blendshape cosine vs the
exemplar). Resumable: skips a fresh output PNG. See
docs/superpowers/specs/2026-05-18-arkit-landmark-control-spike-design.md.
"""
import asyncio
import copy
import json
from pathlib import Path
from typing import Any

import cv2
import pandas as pd

from arkit_controlnet.eval_spike import arcface_cos, expr_cos
from arkit_controlnet.landmark_control import (
    render_landmark_mesh, select_exemplars, select_neutral,
)
from arkit_controlnet.run_spike import select_identities
from demographic_pc.comfy_flux import ComfyClient

WORKFLOW = Path("comfyui/workflows/arkit_landmark_spike.json")
OUT_DIR = Path("exp_output/arkit_landmark_spike")
SEED = 2026
STRENGTHS = [0.6, 1.0]
AXES_TO_RUN = ["smile", "pucker", "surprise"]
_MIN_PNG_BYTES = 1024


def _is_fresh(png: Path) -> bool:
    return png.exists() and png.stat().st_size >= _MIN_PNG_BYTES


def build_workflow(identity_filename: str, control_filename: str,
                   strength: float, out_prefix: str) -> dict:
    """Substitute the $$-placeholders in the workflow template."""
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$CONTROL_FILENAME": control_filename,
        "$$STRENGTH": float(strength),
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


def _prepare_controls() -> dict[str, dict[str, Path]]:
    """Render one control mesh PNG per axis (+ neutral) into OUT_DIR.

    Returns {axis: {"mesh": Path, "exemplar": Path}}. Falls through the k=3
    exemplar candidates if MediaPipe re-detection fails on the first.
    """
    controls: dict[str, dict[str, Path]] = {}
    specs = [(ax, select_exemplars(ax, k=3)) for ax in AXES_TO_RUN]
    specs.append(("neutral", select_neutral(k=3)))
    for axis, candidates in specs:
        for exemplar in candidates:
            try:
                mesh = render_landmark_mesh(exemplar)
            except ValueError:
                continue
            mesh_png = OUT_DIR / f"control_{axis}.png"
            cv2.imwrite(str(mesh_png), mesh)
            controls[axis] = {"mesh": mesh_png, "exemplar": exemplar}
            break
        else:
            raise RuntimeError(f"no exemplar yielded a mesh for axis {axis}")
    return controls


async def _run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identities = select_identities(3)
    controls = _prepare_controls()
    rows = []
    async with ComfyClient() as client:
        for ident in identities:
            id_name = await client.upload_image(ident)
            for axis in [*AXES_TO_RUN, "neutral"]:
                ctl = controls[axis]
                ctl_name = await client.upload_image(ctl["mesh"])
                for strength in STRENGTHS:
                    tag = f"{ident.stem}__{axis}__str{strength:.2f}"
                    out_png = OUT_DIR / f"{tag}.png"
                    if not _is_fresh(out_png):
                        wf = build_workflow(id_name, ctl_name, strength, tag)
                        try:
                            await client.generate(wf, out_png)
                        except Exception as exc:
                            print(f"  FAILED {tag}: {exc}")
                            continue
                    try:
                        af = arcface_cos(out_png, ident)
                        ec = expr_cos(out_png, ctl["exemplar"])
                    except Exception as exc:
                        print(f"  METRIC FAILED {tag}: {exc}")
                        af, ec = -1.0, -1.0
                    rows.append({
                        "identity": ident.stem, "axis": axis, "strength": strength,
                        "arcface_cos": af, "expr_cos": ec,
                    })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "metrics.parquet")
    print(df.to_string(index=False))


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
