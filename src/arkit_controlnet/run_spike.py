"""Drive the ARKit-ControlNet spike: 5 identities x smile axis x scale band.

Authors the ComfyUI graph from `comfyui/workflows/arkit_controlnet_spike.json`
by $$-placeholder substitution (project convention — cf.
`scripts/chibi_prompt_sweep.py`) and submits it through
`demographic_pc.comfy_flux.ComfyClient`. Resumable: skips a case whose output
PNG already exists and is non-empty.
"""
import asyncio
import copy
import json
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
SEED = 2026  # single fixed seed — identity is the variable under test, not seed
_MIN_PNG_BYTES = 1024  # below this, treat an output as a crashed/partial render


def select_identities(n: int = 5) -> list[Path]:
    """n FFHQ photos as local PNG paths, spread across FairFace race buckets.

    Maps `image_sha256` to `output/ffhq_images/{sha}.png` (NOT the reverse_index
    `shard_path`, which points at parquet shards). Restricts to images actually
    present on disk BEFORE the per-race pick, so the demographic spread holds
    even though only a subset of FFHQ is staged locally.
    """
    on_disk = {p.stem for p in FFHQ_IMAGES.glob("*.png")}
    ri = pd.read_parquet(REVERSE_INDEX, columns=["image_sha256", "source", "ff_race"])
    ffhq = ri[(ri["source"] == "ffhq") & (ri["image_sha256"].isin(list(on_disk)))]
    if len(ffhq) < n:
        raise RuntimeError(
            f"only {len(ffhq)} FFHQ images on disk under {FFHQ_IMAGES}, need {n}"
        )
    # one per race bucket first (demographic spread), then fill from the remainder
    spread = ffhq.groupby("ff_race", group_keys=False).head(1)
    ordered = pd.concat([spread, ffhq]).drop_duplicates(subset="image_sha256")
    shas = ordered["image_sha256"].tolist()[:n]
    return [FFHQ_IMAGES / f"{s}.png" for s in shas]


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


def _is_fresh(png: Path) -> bool:
    """A usable prior render — exists and is not a crashed/partial file."""
    return png.exists() and png.stat().st_size >= _MIN_PNG_BYTES


async def _run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    identities = select_identities(5)
    rows = []
    async with ComfyClient() as client:
        for ident in identities:
            # upload the identity image to ComfyUI (host-agnostic; /upload/image)
            id_name = await client.upload_image(ident)
            for axis_name in ["smile"]:                    # primary pass
                for scale in AXES[axis_name].scale_band:
                    tag = f"{ident.stem}__{axis_name}__s{scale:.2f}"
                    out_png = OUT_DIR / f"{tag}.png"
                    if not _is_fresh(out_png):
                        wf = build_workflow(id_name, axis_name, scale, tag)
                        try:
                            await client.generate(wf, out_png)
                        except Exception as exc:
                            print(f"  FAILED {tag}: {exc}")
                            continue
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
