"""Queue paired (photoreal, stylized) generations through the ComfyUI HTTP API.

Iterates over (identity, style, seed) tuples, substitutes parameters into the
Flux+PuLID+Canny+LoRA workflow template, POSTs to /prompt, polls /history for
completion, downloads outputs, and appends rows to the growing manifest parquet.

Resumable: skips any (identity, style, seed) tuple whose output PNG already
exists. Append-only manifest semantics.

Usage (on the Windows machine):
    python scripts/importer_run.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --identities data/importer/identities \\
        --canny      data/importer/cn_canny \\
        --workflow   data/importer/workflows/flux_pulid_canny_lora.api.json \\
        --out        data/importer/refs \\
        --manifest   data/importer/manifest.parquet \\
        --seeds-per-style 3 \\
        --styles     photoreal,chibi,furry,chibi_furry

Styles are defined in STYLE_CONFIGS below; edit to match the LoRA filenames on
the local machine.
"""
from __future__ import annotations

import argparse
import copy
import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests


# Adjust LoRA filenames + weights here to match the local ComfyUI install.
# Filenames must exist in ComfyUI/models/loras/.
STYLE_CONFIGS: dict[str, dict[str, Any]] = {
    "photoreal": {
        # No style LoRA; both slots inactive.
        "lora_a_name": "none",
        "lora_a_strength": 0.0,
        "lora_b_name": "none",
        "lora_b_strength": 0.0,
        "positive_suffix": ", photoreal portrait, neutral expression, plain background, looking at camera, sharp focus, FFHQ studio photograph",
    },
    "chibi": {
        "lora_a_name": "chibi_characters_flux_dev.safetensors",
        "lora_a_strength": 0.9,
        "lora_b_name": "none",
        "lora_b_strength": 0.0,
        "positive_suffix": ", chibi character, cute, big eyes, small body, neutral expression, plain background, full character portrait",
    },
    "furry": {
        "lora_a_name": "anime_furry_style_flux.safetensors",
        "lora_a_strength": 0.9,
        "lora_b_name": "none",
        "lora_b_strength": 0.0,
        "positive_suffix": ", anthropomorphic furry character, anime style, portrait, neutral expression, looking at camera, plain background",
    },
    "chibi_furry": {
        "lora_a_name": "chibi_characters_flux_dev.safetensors",
        "lora_a_strength": 0.7,
        "lora_b_name": "anime_furry_style_flux.safetensors",
        "lora_b_strength": 0.7,
        "positive_suffix": ", chibi anthropomorphic furry character, cute, big eyes, anime style portrait, plain background",
    },
}

NEGATIVE_PROMPT = (
    "hands, body, full body, multiple people, watermark, text, low quality, "
    "blurry, deformed, distorted, signature"
)

PULID_WEIGHT = 0.7   # loose — diversity of identities matters more than tight preservation
CN_STRENGTH  = 0.65  # firm enough to lock pose-skeleton, weak enough to let stylization breathe


@dataclass
class Job:
    identity: str        # e.g. "id_00"
    style: str           # key into STYLE_CONFIGS
    seed: int
    out_path: Path       # final destination on local disk
    workflow: dict       # filled-in workflow JSON ready to POST


def build_workflow(
    template: dict,
    *,
    identity_filename: str,
    canny_filename: str,
    style_cfg: dict[str, Any],
    seed: int,
    output_prefix: str,
    base_prompt: str = "portrait of a person",
) -> dict:
    """Return a deep-copied workflow with placeholders substituted."""
    wf = copy.deepcopy(template)
    positive_prompt = base_prompt + style_cfg["positive_suffix"]

    substitutions = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$CANNY_FILENAME": canny_filename,
        "$$POSITIVE_PROMPT": positive_prompt,
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$LORA_A_NAME": style_cfg["lora_a_name"],
        "$$LORA_A_STRENGTH": float(style_cfg["lora_a_strength"]),
        "$$LORA_B_NAME": style_cfg["lora_b_name"],
        "$$LORA_B_STRENGTH": float(style_cfg["lora_b_strength"]),
        "$$PULID_WEIGHT": PULID_WEIGHT,
        "$$CN_STRENGTH": CN_STRENGTH,
        "$$SEED": int(seed),
        "$$OUTPUT_PREFIX": output_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items() if not k.startswith("_")}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in substitutions:
            return substitutions[node]
        return node

    return _sub(wf)


def queue_prompt(comfy_url: str, workflow: dict, client_id: str) -> str:
    """POST workflow to ComfyUI, return the prompt_id."""
    payload = {"prompt": workflow, "client_id": client_id}
    r = requests.post(f"{comfy_url}/prompt", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait_for_prompt(comfy_url: str, prompt_id: str, timeout: float = 300) -> dict:
    """Poll /history until the prompt completes; return its history entry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = requests.get(f"{comfy_url}/history/{prompt_id}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if prompt_id in data:
                return data[prompt_id]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {prompt_id} did not complete within {timeout}s")


def download_output(comfy_url: str, history_entry: dict, out_path: Path) -> bool:
    """Walk history outputs, find an image, save to out_path. True on success."""
    outputs = history_entry.get("outputs", {})
    for _node_id, node_out in outputs.items():
        for img_info in node_out.get("images", []):
            filename = img_info["filename"]
            subfolder = img_info.get("subfolder", "")
            type_ = img_info.get("type", "output")
            params = {"filename": filename, "subfolder": subfolder, "type": type_}
            r = requests.get(f"{comfy_url}/view", params=params, timeout=30)
            if r.status_code == 200:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(r.content)
                return True
    return False


def append_manifest_row(manifest_path: Path, row: dict) -> None:
    """Append a single row to the manifest parquet (write if missing)."""
    df_new = pd.DataFrame([row])
    if manifest_path.exists():
        df_old = pd.read_parquet(manifest_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(manifest_path, index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--identities", type=Path, default=Path("data/importer/identities"))
    ap.add_argument("--canny",      type=Path, default=Path("data/importer/cn_canny"))
    ap.add_argument("--workflow",   type=Path, default=Path("data/importer/workflows/flux_pulid_canny_lora.api.json"))
    ap.add_argument("--out",        type=Path, default=Path("data/importer/refs"))
    ap.add_argument("--manifest",   type=Path, default=Path("data/importer/manifest.parquet"))
    ap.add_argument("--styles", default="photoreal,chibi,furry,chibi_furry")
    ap.add_argument("--seeds-per-style", type=int, default=3)
    ap.add_argument("--seed-base", type=int, default=20260512)
    ap.add_argument("--comfy-input-dir", type=Path, default=None,
                    help="If ComfyUI runs on this machine, copy identity+canny PNGs into ComfyUI/input/ here.")
    args = ap.parse_args()

    styles = args.styles.split(",")
    for s in styles:
        if s not in STYLE_CONFIGS:
            raise SystemExit(f"unknown style: {s}; available: {list(STYLE_CONFIGS)}")

    template = json.loads(args.workflow.read_text())
    identities = sorted(p.stem for p in args.identities.glob("id_*.png"))
    if not identities:
        raise SystemExit(f"no identities found in {args.identities}")
    print(f"[run] {len(identities)} identities, {len(styles)} styles, "
          f"{args.seeds_per_style} seeds/style "
          f"= {len(identities) * len(styles) * args.seeds_per_style} total jobs")

    # Stage identity + canny PNGs into ComfyUI/input/ if asked
    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        for p in args.identities.glob("id_*.png"):
            shutil.copy2(p, args.comfy_input_dir / p.name)
        for p in args.canny.glob("id_*_canny.png"):
            shutil.copy2(p, args.comfy_input_dir / p.name)
        print(f"[run] staged ID + canny PNGs into {args.comfy_input_dir}")

    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for identity in identities:
        for style in styles:
            for seed_offset in range(args.seeds_per_style):
                seed = args.seed_base + hash((identity, style, seed_offset)) % 10_000_000
                out_path = args.out / style / f"{identity}_seed{seed}.png"
                if out_path.exists():
                    skipped += 1
                    continue

                style_cfg = STYLE_CONFIGS[style]
                workflow = build_workflow(
                    template,
                    identity_filename=f"{identity}.png",
                    canny_filename=f"{identity}_canny.png",
                    style_cfg=style_cfg,
                    seed=seed,
                    output_prefix=f"importer/{style}/{identity}_seed{seed}",
                )

                t_start = time.time()
                try:
                    prompt_id = queue_prompt(args.comfy_url, workflow, client_id)
                    entry = wait_for_prompt(args.comfy_url, prompt_id, timeout=300)
                    ok = download_output(args.comfy_url, entry, out_path)
                except Exception as e:
                    print(f"  [fail] {identity} {style} seed={seed}: {e}")
                    failed += 1
                    continue

                if not ok:
                    print(f"  [fail] {identity} {style} seed={seed}: no image output")
                    failed += 1
                    continue

                duration = time.time() - t_start
                row = {
                    "ts_unix": int(time.time()),
                    "identity": identity,
                    "style": style,
                    "seed": seed,
                    "out_path": str(out_path),
                    "lora_a_name": style_cfg["lora_a_name"],
                    "lora_a_strength": style_cfg["lora_a_strength"],
                    "lora_b_name": style_cfg["lora_b_name"],
                    "lora_b_strength": style_cfg["lora_b_strength"],
                    "pulid_weight": PULID_WEIGHT,
                    "cn_strength": CN_STRENGTH,
                    "duration_s": round(duration, 2),
                    "workflow_version": "v1_2026-05-12",
                }
                append_manifest_row(args.manifest, row)
                done += 1
                if done % 5 == 0:
                    rate = done / max(1e-6, time.time() - t0) * 60
                    print(f"  [ok] {identity} {style} seed={seed} ({duration:.1f}s) — "
                          f"{done} done, {skipped} skipped, {failed} failed, {rate:.1f}/min")

    print(f"[run] complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.time() - t0)/60:.1f} min")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
