"""Chibi grown-up sweep: which prompt additions + schedule make a chibi read adult.

The chibi LoRA neotenises by construction -- the goal is NOT to strip its
`cute, big eyes, small body` tokens (those produced the good reference
render id_14) but to *add* grown-up characteristics on top so the adult
read becomes reliable instead of a seed lottery.

Grid (chibi style only):
  treatment  T0..T5  -- baseline chibi prompt + an additive adult suffix
  schedule   S0/S1   -- CN + PuLID step windows (default vs structural-tuned)
  strength   chibi LoRA strength_model
  identity   a 4-id demographic spread incl. id_14 (the reference)
  one seed per cell (deterministic: seed_base + cell_index * 7919); the
  winning cell gets a separate multi-seed confirm run.

CN stays on throughout -- it is the photoreal<->chibi label-transfer
contract; only its *window* is swept, never its presence.

The manifest describes the full deterministic grid and is written up front,
so a crash mid-run never loses it; which PNGs exist on disk is the record
of progress. Resumable (skip-if-exists) with atomic PNG writes.

Usage (on the ComfyUI box, cwd = data/importer):
    python scripts/chibi_prompt_sweep.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow data/importer/workflows/flux_pulid_canny_lora.api.json \\
        --id-dir identities_flux --canny-dir cn_canny_flux \\
        --out refs_sweep --manifest manifest_sweep.parquet \\
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

WORKFLOW_VERSION = "sweep_2026-05-15"
LORA_CHIBI = "Ksbt_000001750.safetensors"
LORA_FURRY = "Anime_Furry_Style_Flux.safetensors"
CN_STRENGTH = 0.5
PULID_WEIGHT = 0.7
NEGATIVE_PROMPT = (
    "hands, body, full body, multiple people, watermark, text, low quality, "
    "blurry, deformed, distorted, signature"
)

# Baseline chibi prompt -- the exact string that produced the id_14 reference.
BASE_PROMPT = ("portrait of a person, chibi character, cute, big eyes, small body, "
               "neutral expression, plain background, full character portrait")

# Additive adult-characteristic suffixes. T0 is the unmodified control.
TREATMENTS: dict[str, str] = {
    "T0_baseline": "",
    "T1_face": ", defined jawline, prominent chin, mature facial structure, high cheekbones",
    "T2_styling": ", wearing glasses, adult hairstyle, grown-up outfit",
    "T3_age": ", adult, grown-up person, mature, in their thirties",
    "T4_toy": ", designer collectible figure, matte vinyl finish, painted toy eyes",
}
TREATMENTS["T5_combo"] = (TREATMENTS["T1_face"] + TREATMENTS["T2_styling"]
                          + TREATMENTS["T3_age"] + TREATMENTS["T4_toy"])

# (cn_start, cn_end, pulid_start, pulid_end)
SCHEDULES: dict[str, tuple[float, float, float, float]] = {
    "S0_default": (0.0, 0.7, 0.0, 1.0),
    "S1_tuned": (0.0, 0.4, 0.1, 0.7),  # CN releases after structural phase; PuLID mid-window
}

STRENGTHS = [0.55, 0.70, 0.85]
IDENTITIES = [3, 8, 14, 15]  # id_14 = reference; spread of gender + age

# Nodes whose schedule literals the sweep overwrites -- asserted at startup so a
# re-exported (renumbered) workflow fails loud instead of running with no
# schedule variation.
SCHEDULE_NODES = {"8": "ApplyPulidFlux", "13": "ControlNetApplyAdvanced"}


def build_grid() -> list[dict]:
    """Deterministic, ordered grid. cell index drives the seed -> reproducible."""
    rows: list[dict] = []
    cell = 0
    for idx in IDENTITIES:
        for tname in TREATMENTS:
            for sname in SCHEDULES:
                for strength in STRENGTHS:
                    seed = 50_000_000 + cell * 7919
                    cn_s, cn_e, pu_s, pu_e = SCHEDULES[sname]
                    str_tag = f"str{int(round(strength * 100)):03d}"
                    identity = f"id_{idx:02d}"
                    stem = f"{identity}_{tname}_{sname}_{str_tag}_seed{seed}"
                    rows.append({
                        "render": f"{stem}.png", "stem": stem, "id_idx": idx,
                        "identity": identity, "treatment": tname, "schedule": sname,
                        "strength": strength, "seed": seed,
                        "cn_start": cn_s, "cn_end": cn_e,
                        "pulid_start": pu_s, "pulid_end": pu_e,
                        "prompt": BASE_PROMPT + TREATMENTS[tname],
                        "workflow_version": WORKFLOW_VERSION,
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
        "$$PULID_WEIGHT": PULID_WEIGHT,
        "$$CN_STRENGTH": CN_STRENGTH,
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
    """Run fn() with a few retries -- HTTP blips over a 2h unattended run."""
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
    ap.add_argument("--out", type=Path, default=Path("refs_sweep"))
    ap.add_argument("--manifest", type=Path, default=Path("manifest_sweep.parquet"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    args = ap.parse_args()

    template = json.loads(args.workflow.read_text())
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(f"workflow node {nid} is {got!r}, expected {cls!r} -- "
                             f"schedule injection would target the wrong node")

    chibi_dir = args.out / "chibi"
    chibi_dir.mkdir(parents=True, exist_ok=True)

    # Manifest = the full deterministic grid, written up front. A crash mid-run
    # cannot lose it; disk PNGs are the progress record.
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

        wf = build_workflow(template, cell, output_prefix=f"sweep/chibi/{cell['stem']}")
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
