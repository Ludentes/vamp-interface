"""Generate a fresh 20-portrait identity pool with Flux-Krea via ComfyUI.

Strata: gender (M/F) × age (20-29, 30-39, 40-49, 50-59) × race
(Black, White, East Asian, South Asian, Latino, Middle Eastern).

Writes PNGs to data/importer/identities_flux/id_NN.png (1024x1024) plus a
manifest.csv matching the v1 schema (id_idx, filename, sha256, gender, age_bin,
race). Resumable: skips existing id_NN.png.

Usage:
    python scripts/importer_build_id_pool_flux.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --out data/importer/identities_flux \\
        --seed-base 30000000
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import time
import uuid
from pathlib import Path
from typing import Any

import requests


# 20 stratified demographic targets. Order matters for id_00..id_19 assignment.
DEMOGRAPHICS: list[dict[str, str]] = [
    # 8 × young (20-29)
    {"gender": "F", "age_bin": "20-29", "race": "Black"},
    {"gender": "M", "age_bin": "20-29", "race": "Black"},
    {"gender": "F", "age_bin": "20-29", "race": "White"},
    {"gender": "M", "age_bin": "20-29", "race": "White"},
    {"gender": "F", "age_bin": "20-29", "race": "East Asian"},
    {"gender": "M", "age_bin": "20-29", "race": "East Asian"},
    {"gender": "F", "age_bin": "20-29", "race": "South Asian"},
    {"gender": "M", "age_bin": "20-29", "race": "Latino"},
    # 6 × 30-39
    {"gender": "F", "age_bin": "30-39", "race": "Middle Eastern"},
    {"gender": "M", "age_bin": "30-39", "race": "South Asian"},
    {"gender": "F", "age_bin": "30-39", "race": "Latino"},
    {"gender": "M", "age_bin": "30-39", "race": "White"},
    {"gender": "F", "age_bin": "30-39", "race": "East Asian"},
    {"gender": "M", "age_bin": "30-39", "race": "Middle Eastern"},
    # 4 × 40-49
    {"gender": "F", "age_bin": "40-49", "race": "White"},
    {"gender": "M", "age_bin": "40-49", "race": "Black"},
    {"gender": "F", "age_bin": "40-49", "race": "South Asian"},
    {"gender": "M", "age_bin": "40-49", "race": "Latino"},
    # 2 × 50-59
    {"gender": "F", "age_bin": "50-59", "race": "East Asian"},
    {"gender": "M", "age_bin": "50-59", "race": "White"},
]
assert len(DEMOGRAPHICS) == 20

RACE_PHRASE = {
    "Black": "a Black",
    "White": "a White",
    "East Asian": "an East Asian",
    "South Asian": "a South Asian",
    "Latino": "a Latino",
    "Middle Eastern": "a Middle Eastern",
}
GENDER_NOUN = {"F": "woman", "M": "man"}
AGE_PHRASE = {
    "20-29": "in their mid 20s",
    "30-39": "in their mid 30s",
    "40-49": "in their mid 40s",
    "50-59": "in their mid 50s",
}


def build_prompt(demo: dict[str, str]) -> str:
    return (
        f"A photoreal studio portrait of {RACE_PHRASE[demo['race']]} {GENDER_NOUN[demo['gender']]} "
        f"{AGE_PHRASE[demo['age_bin']]}, neutral expression, looking at camera, "
        "soft front lighting, plain neutral background, sharp focus, FFHQ aligned, head and shoulders, "
        "no glasses, no jewelry, no hat, natural hair"
    )


NEGATIVE = "watermark, text, signature, logo, blurry, low quality, deformed, multiple people, hands"


WORKFLOW_TEMPLATE: dict[str, dict[str, Any]] = {
    "1": {
        "class_type": "UNETLoader",
        "inputs": {
            "unet_name": "FLUX1\\flux1-krea-dev_fp8_scaled.safetensors",
            "weight_dtype": "default",
        },
    },
    "2": {
        "class_type": "DualCLIPLoader",
        "inputs": {
            "clip_name1": "t5\\t5xxl_fp8_e4m3fn.safetensors",
            "clip_name2": "clip_l.safetensors",
            "type": "flux",
        },
    },
    "3": {"class_type": "VAELoader", "inputs": {"vae_name": "FLUX1\\ae.safetensors"}},
    "11": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "$$POSITIVE_PROMPT", "clip": ["2", 0]},
    },
    "12": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "$$NEGATIVE_PROMPT", "clip": ["2", 0]},
    },
    "14": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1024, "height": 1024, "batch_size": 1},
    },
    "15": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["11", 0],
            "negative": ["12", 0],
            "latent_image": ["14", 0],
            "seed": "$$SEED",
            "steps": 25,
            "cfg": 1.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 1.0,
        },
    },
    "16": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["15", 0], "vae": ["3", 0]},
    },
    "17": {
        "class_type": "SaveImage",
        "inputs": {"images": ["16", 0], "filename_prefix": "$$OUTPUT_PREFIX"},
    },
}


def build_workflow(positive: str, seed: int, output_prefix: str) -> dict:
    wf = copy.deepcopy(WORKFLOW_TEMPLATE)
    subs = {
        "$$POSITIVE_PROMPT": positive,
        "$$NEGATIVE_PROMPT": NEGATIVE,
        "$$SEED": int(seed),
        "$$OUTPUT_PREFIX": output_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    return _sub(wf)


def queue(comfy_url: str, wf: dict, client_id: str) -> str:
    r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf, "client_id": client_id}, timeout=30)
    r.raise_for_status()
    return r.json()["prompt_id"]


def wait(comfy_url: str, pid: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = requests.get(f"{comfy_url}/history/{pid}", timeout=10)
        if r.status_code == 200:
            data = r.json()
            if pid in data:
                return data[pid]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {pid} did not complete within {timeout}s")


def download(comfy_url: str, entry: dict, out_path: Path) -> bool:
    for _, node_out in entry.get("outputs", {}).items():
        for img in node_out.get("images", []):
            params = {
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output"),
            }
            r = requests.get(f"{comfy_url}/view", params=params, timeout=30)
            if r.status_code == 200:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(r.content)
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--out", type=Path, default=Path("data/importer/identities_flux"))
    ap.add_argument("--seed-base", type=int, default=30000000)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    client_id = str(uuid.uuid4())
    manifest_path = args.out / "manifest.csv"

    rows = []
    t0 = time.time()
    for i, demo in enumerate(DEMOGRAPHICS):
        out_png = args.out / f"id_{i:02d}.png"
        if out_png.exists():
            sha = hashlib.sha256(out_png.read_bytes()).hexdigest()
            print(f"  [skip] id_{i:02d} exists")
        else:
            seed = args.seed_base + i * 7919  # spread seeds
            prompt = build_prompt(demo)
            wf = build_workflow(prompt, seed, f"identities_flux/id_{i:02d}")
            t_start = time.time()
            try:
                pid = queue(args.comfy_url, wf, client_id)
                entry = wait(args.comfy_url, pid, timeout=300)
                if not download(args.comfy_url, entry, out_png):
                    print(f"  [fail] id_{i:02d}: no image output")
                    continue
                sha = hashlib.sha256(out_png.read_bytes()).hexdigest()
                print(f"  [ok]   id_{i:02d} ({time.time()-t_start:.1f}s) {demo['gender']} {demo['age_bin']} {demo['race']}")
            except Exception as e:
                print(f"  [fail] id_{i:02d}: {e}")
                continue
        rows.append({
            "id_idx": i,
            "filename": f"id_{i:02d}.png",
            "sha256": sha,
            "gender": demo["gender"],
            "age_bin": demo["age_bin"],
            "race": demo["race"],
        })

    if rows:
        with manifest_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["id_idx", "filename", "sha256", "gender", "age_bin", "race"])
            w.writeheader()
            w.writerows(rows)
        print(f"[pool] wrote {len(rows)} identities + {manifest_path} in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
