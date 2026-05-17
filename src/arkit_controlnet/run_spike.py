"""Drive the ARKit-ControlNet spike: 5 identities x smile axis x scale band.

Authors the ComfyUI graph from `comfyui/workflows/arkit_controlnet_spike.json`
by $$-placeholder substitution (project convention — cf.
`scripts/chibi_prompt_sweep.py`) and submits it through
`demographic_pc.comfy_flux.ComfyClient`. Resumable: skips a case whose output
PNG already exists.
"""
import asyncio
import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from arkit_controlnet.axes import AXES
from arkit_controlnet.eval_spike import arcface_cos, bs_delta
from demographic_pc.comfy_flux import ComfyClient

WORKFLOW = Path("comfyui/workflows/arkit_controlnet_spike.json")
OUT_DIR = Path("exp_output/arkit_controlnet_spike")
REVERSE_INDEX = Path("output/reverse_index/reverse_index.parquet")
FFHQ_IMAGES = Path("output/ffhq_images")
COMFY_INPUT = Path.home() / "w" / "ComfyUI" / "input"
SEED = 2026


def select_identities(n: int = 5) -> list[Path]:
    """n FFHQ photos as local PNG paths, spread across FairFace race buckets.

    Maps `image_sha256` to `output/ffhq_images/{sha}.png` (the sha-named image
    dir) — NOT the reverse_index `shard_path`, which points at parquet shards.
    """
    ri = pd.read_parquet(REVERSE_INDEX, columns=["image_sha256", "source", "ff_race"])
    ffhq = ri[ri["source"] == "ffhq"]
    # one per race bucket first (a demographic spread), then the remainder
    ordered = pd.concat([
        ffhq.groupby("ff_race", group_keys=False).head(1),
        ffhq,
    ])
    picks: list[Path] = []
    seen: set[str] = set()
    for sha in ordered["image_sha256"]:
        if sha in seen:
            continue
        seen.add(sha)
        png = FFHQ_IMAGES / f"{sha}.png"
        if png.exists():
            picks.append(png)
        if len(picks) == n:
            break
    if len(picks) < n:
        raise RuntimeError(f"only found {len(picks)} FFHQ images under {FFHQ_IMAGES}, need {n}")
    return picks


def build_workflow(identity_filename: str, axis_name: str, scale: float,
                   out_prefix: str) -> dict:
    """Substitute the $$-placeholders in the workflow template into a graph dict."""
    axis = AXES[axis_name]
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$EDIT_PROMPT_A": axis.edit_prompt_a,
        "$$EDIT_PROMPT_B": axis.edit_prompt_b,
        "$$SCALE": float(scale),
        "$$MIX_B": float(axis.mix_b),
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


async def _run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMFY_INPUT.mkdir(parents=True, exist_ok=True)
    identities = select_identities(5)
    client = ComfyClient()
    rows = []
    for ident in identities:
        # stage the identity image into ComfyUI's input/ dir for the LoadImage node
        staged = COMFY_INPUT / f"arkit_spike_{ident.stem}.png"
        if not staged.exists():
            shutil.copy(ident, staged)
        for axis_name in ["smile"]:                       # primary pass
            for scale in AXES[axis_name].scale_band:
                tag = f"{ident.stem}__{axis_name}__s{scale:.2f}"
                out_png = OUT_DIR / f"{tag}.png"
                if not out_png.exists():
                    wf = build_workflow(staged.name, axis_name, scale, tag)
                    await client.generate(wf, out_png)
                rows.append({
                    "identity": ident.stem, "axis": axis_name, "scale": scale,
                    "arcface_cos": arcface_cos(out_png, ident),
                    "bs_delta": bs_delta(out_png, ident,
                                         AXES[axis_name].target_channels),
                })
    df = pd.DataFrame(rows)
    df.to_parquet(OUT_DIR / "metrics.parquet")
    print(df.to_string(index=False))


def run() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    run()
