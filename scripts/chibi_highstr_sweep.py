"""Chibi high-strength + conditioner-ablation sweep.

The prompt sweep showed the creepy chibi is a *partial-stylization*
artifact: photoreal skin on neotenic proportions. Cranking the chibi LoRA
commits to the figurine read and exits the uncanny valley -- ArcFace
detection fell monotonically with strength (77% -> 65%), the eyeball
agreed. This sweep pushes strength past the prompt sweep's 0.85 ceiling
and ablates each conditioner to see what each one actually contributes.

Fixed at the prompt-sweep winner: T5_combo treatment + S1_tuned schedule.

Grid (chibi style only):
  identity   id_03, id_08, id_14 (reference), id_15
  strength   chibi LoRA strength_model -- high range
  ablation   full     CN 0.5 + PuLID 0.7   (production config)
             no_cn    CN 0.0               (PuLID-only: identity, no pose lock)
             no_pulid PuLID 0.0            (CN-only: pose, no identity)
  2 seeds per cell (deterministic: seed_base + render_index * 7919)

The ablation is a *measurement*, not a production proposal -- CN stays on
in production as the photoreal<->chibi label-transfer contract. Disabling
is done by zeroing CN strength / PuLID weight, so the workflow template is
untouched.

Manifest = the full deterministic grid, written up front. Resumable
(skip-if-exists) with atomic PNG writes.

Usage (on the ComfyUI box, cwd = data/importer):
    python scripts/chibi_highstr_sweep.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow workflows/flux_pulid_canny_lora.api.json \\
        --id-dir identities_flux --canny-dir cn_canny_flux \\
        --out refs_highstr --manifest manifest_highstr.parquet \\
        --comfy-input-dir C:/comfy/ComfyUI/input
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
import requests

WORKFLOW_VERSION = "highstr_2026-05-15"
LORA_CHIBI = "Ksbt_000001750.safetensors"
LORA_FURRY = "Anime_Furry_Style_Flux.safetensors"
NEGATIVE_PROMPT = (
    "hands, body, full body, multiple people, watermark, text, low quality, "
    "blurry, deformed, distorted, signature"
)

# Prompt-sweep winner: baseline chibi prompt + the full additive adult suffix.
BASE_PROMPT = ("portrait of a person, chibi character, cute, big eyes, small body, "
               "neutral expression, plain background, full character portrait")
T5_COMBO = (", defined jawline, prominent chin, mature facial structure, high cheekbones"
            ", wearing glasses, adult hairstyle, grown-up outfit"
            ", adult, grown-up person, mature, in their thirties"
            ", designer collectible figure, matte vinyl finish, painted toy eyes")
PROMPT = BASE_PROMPT + T5_COMBO

# S1_tuned schedule: CN releases after structural phase, PuLID mid-window.
CN_START, CN_END = 0.0, 0.4
PULID_START, PULID_END = 0.1, 0.7

# ablation -> (cn_strength, pulid_weight)
ABLATIONS: dict[str, tuple[float, float]] = {
    "full": (0.5, 0.7),
    "no_cn": (0.0, 0.7),
    "no_pulid": (0.5, 0.0),
}
STRENGTHS = [0.85, 1.00, 1.15]
IDENTITIES = [3, 8, 14, 15]  # id_14 = reference
SEEDS_PER_CELL = 2

# Nodes whose literals the sweep overwrites -- asserted at startup so a
# re-exported (renumbered) workflow fails loud.
SCHEDULE_NODES = {"8": "ApplyPulidFlux", "13": "ControlNetApplyAdvanced"}


def build_grid() -> list[dict]:
    """Deterministic, ordered grid. cell index drives the seed -> reproducible."""
    rows: list[dict] = []
    cell = 0
    for idx in IDENTITIES:
        for aname in ABLATIONS:
            for strength in STRENGTHS:
                for _ in range(SEEDS_PER_CELL):
                    seed = 60_000_000 + cell * 7919
                    cn_str, pulid_w = ABLATIONS[aname]
                    str_tag = f"str{int(round(strength * 100)):03d}"
                    identity = f"id_{idx:02d}"
                    stem = f"{identity}_{aname}_{str_tag}_seed{seed}"
                    rows.append({
                        "render": f"{stem}.png", "stem": stem, "id_idx": idx,
                        "identity": identity, "ablation": aname,
                        "strength": strength, "seed": seed,
                        "cn_strength": cn_str, "pulid_weight": pulid_w,
                        "cn_start": CN_START, "cn_end": CN_END,
                        "pulid_start": PULID_START, "pulid_end": PULID_END,
                        "prompt": PROMPT, "workflow_version": WORKFLOW_VERSION,
                    })
                    cell += 1
    return rows


def build_workflow(template: dict, cell: dict, output_prefix: str) -> dict:
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": f"{cell['identity']}.png",
        "$$CANNY_FILENAME": f"{cell['identity']}_canny.png",
        "$$POSITIVE_PROMPT": cell["prompt"],
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$SEED": int(cell["seed"]),
        "$$PULID_WEIGHT": float(cell["pulid_weight"]),
        "$$CN_STRENGTH": float(cell["cn_strength"]),
        "$$LORA_A_NAME": LORA_CHIBI,
        "$$LORA_A_STRENGTH": float(cell["strength"]),
        "$$LORA_B_NAME": LORA_FURRY,
        "$$LORA_B_STRENGTH": 0.0,
        "$$OUTPUT_PREFIX": output_prefix,
    }

    def _sub(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items() if not k.startswith("_")}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    wf = _sub(copy.deepcopy(template))
    wf["13"]["inputs"]["start_percent"] = cell["cn_start"]
    wf["13"]["inputs"]["end_percent"] = cell["cn_end"]
    wf["8"]["inputs"]["start_at"] = cell["pulid_start"]
    wf["8"]["inputs"]["end_at"] = cell["pulid_end"]
    return wf


def _retry(fn, *, tries: int = 3, what: str = ""):
    """Run fn() with a few retries -- HTTP blips over a long unattended run."""
    for attempt in range(tries):
        try:
            return fn()
        except (requests.RequestException, ConnectionError) as e:
            if attempt == tries - 1:
                raise
            print(f"  [retry {attempt+1}/{tries}] {what}: {e}")
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


def queue(sess: requests.Session, url: str, wf: dict, client_id: str) -> str:
    def _do():
        r = sess.post(f"{url}/prompt", json={"prompt": wf, "client_id": client_id},
                      timeout=30)
        r.raise_for_status()
        return r.json()["prompt_id"]
    return _retry(_do, what="queue")


def wait(sess: requests.Session, url: str, pid: str, timeout: float = 420) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = _retry(lambda: sess.get(f"{url}/history/{pid}", timeout=10), what="history")
        if r is not None and r.status_code == 200 and pid in r.json():
            return r.json()[pid]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {pid} did not complete within {timeout}s")


def download(sess: requests.Session, url: str, entry: dict, out_path: Path) -> bool:
    status = entry.get("status", {})
    if status.get("status_str") not in (None, "success"):
        print(f"  [fail] {out_path.stem}: ComfyUI status={status.get('status_str')}")
        return False
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            r = _retry(lambda: sess.get(f"{url}/view", params={
                "filename": img["filename"], "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output")}, timeout=30), what="view")
            if r is None or r.status_code != 200 or r.content[:8] != b"\x89PNG\r\n\x1a\n":
                continue
            out_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".png.tmp")
            tmp.write_bytes(r.content)
            os.replace(tmp, out_path)  # atomic -- no truncated PNG masquerades as done
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--id-dir", type=Path, default=Path("identities_flux"))
    ap.add_argument("--canny-dir", type=Path, default=Path("cn_canny_flux"))
    ap.add_argument("--out", type=Path, default=Path("refs_highstr"))
    ap.add_argument("--manifest", type=Path, default=Path("manifest_highstr.parquet"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    args = ap.parse_args()

    template = json.loads(args.workflow.read_text())
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(f"workflow node {nid} is {got!r}, expected {cls!r} -- "
                             f"injection would target the wrong node")

    chibi_dir = args.out / "chibi"
    chibi_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    pd.DataFrame([{k: v for k, v in c.items() if k != "stem"} for c in grid]
                 ).to_parquet(args.manifest, index=False)
    print(f"[sweep] {len(grid)} cells -> manifest {args.manifest}")

    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        for idx in IDENTITIES:
            for src in (args.id_dir / f"id_{idx:02d}.png",
                        args.canny_dir / f"id_{idx:02d}_canny.png"):
                if src.exists():
                    shutil.copy2(src, args.comfy_input_dir / src.name)
        print(f"[sweep] staged ID + canny PNGs into {args.comfy_input_dir}")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for cell in grid:
        out_png = chibi_dir / cell["render"]
        if out_png.exists():
            skipped += 1
            continue

        wf = build_workflow(template, cell, output_prefix=f"highstr/chibi/{cell['stem']}")
        t_start = time.time()
        try:
            pid = queue(sess, args.comfy_url, wf, client_id)
            entry = wait(sess, args.comfy_url, pid)
            if not download(sess, args.comfy_url, entry, out_png):
                failed += 1
                continue
        except Exception as e:
            print(f"  [fail] {cell['stem']}: {e}")
            failed += 1
            continue

        done += 1
        rate = done / max(time.time() - t0, 1) * 60
        print(f"  [ok] {cell['stem']} ({time.time()-t_start:.1f}s) "
              f"- {done} done, {skipped} skipped, {failed} failed, {rate:.1f}/min")

    print(f"[sweep] complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
