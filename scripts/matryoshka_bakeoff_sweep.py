"""Matryoshka fast-model bake-off sweep.

Generates the SAME generic matryoshka doll across several base models and
logs wall-clock render time per cell. Identity is NOT generated here -- the
downstream inswapper stage owns identity (see the bake-off design doc). So
every arm only needs Canny structure + a folk-art prompt; the Z-Image arm
has no Canny ControlNet and runs prompt-only.

Each arm is its own ComfyUI API workflow under --workflow-dir. The workflow
bakes its model-specific steps / sampler / cfg as literals; the harness only
injects the shared placeholders (prompt, seed, output prefix, and -- for
Canny arms -- the doll template + CN schedule).

Grid: arms x SEEDS_PER_ARM. Manifest written at the end with elapsed_s per
cell. Resumable (skip-if-exists), atomic PNG writes.

Usage (cwd = data/importer):
    python scripts/matryoshka_bakeoff_sweep.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow-dir ../../comfyui/workflows \\
        --template refs/matryoshka/template_canny.png \\
        --out refs_matryoshka_bakeoff --manifest manifest_matryoshka_bakeoff.parquet \\
        --comfy-input-dir C:/comfy/ComfyUI/input \\
        --arm sdxl_lightning --limit 1     # smoke test one arm
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

WORKFLOW_VERSION = "matryoshka_bakeoff_2026-05-18"
CANNY_TEMPLATE = "matryoshka_template_canny.png"
NEGATIVE_PROMPT = (
    "hands, body, full body, multiple people, watermark, text, low quality, "
    "blurry, deformed, distorted, signature"
)
BASE_PROMPT = ("a Russian matryoshka nesting doll, single doll, painted "
               "wooden figure, glossy lacquer finish, floral folk-art shawl "
               "and apron, flat painted face, plain background, centered")
STYLE_SUFFIX = (", traditional khokhloma painting, rosy painted cheeks, "
                "hand-painted detail")
PROMPT = BASE_PROMPT + STYLE_SUFFIX

# Canny doll-form structure schedule -- shared by every Canny arm.
CN_STRENGTH = 0.5
CN_START, CN_END = 0.0, 0.5
SEEDS_PER_ARM = 2

# Bake-off arms. Each names a workflow file in --workflow-dir, whether it
# consumes the Canny template, and the nodes the harness asserts on so a
# re-exported (renumbered) workflow fails loud rather than mis-injecting.
ARMS: dict[str, dict] = {
    "flux_krea": {
        "workflow": "matryoshka_flux_krea.api.json",
        "has_canny": True,
        "schedule": {"8": "ControlNetApplyAdvanced", "10": "KSampler"},
    },
    "flux_schnell": {
        "workflow": "matryoshka_flux_schnell.api.json",
        "has_canny": True,
        "schedule": {"6": "ControlNetApplyAdvanced", "8": "KSampler"},
    },
    "sdxl_lightning": {
        "workflow": "matryoshka_sdxl_lightning.api.json",
        "has_canny": True,
        "schedule": {"7": "ControlNetApplyAdvanced", "9": "KSampler"},
    },
    "zimage_turbo": {
        "workflow": "matryoshka_zimage_turbo.api.json",
        "has_canny": False,  # no Z-Image Canny CN -> prompt-only arm
        "schedule": {"7": "ModelSamplingAuraFlow", "8": "KSampler"},
    },
}


def build_grid(arms: list[str]) -> list[dict]:
    rows: list[dict] = []
    cell = 0
    for arm in arms:
        for _ in range(SEEDS_PER_ARM):
            seed = 73_000_000 + cell * 7919
            stem = f"{arm}_seed{seed}"
            rows.append({
                "render": f"{stem}.png", "stem": stem, "arm": arm,
                "seed": seed, "has_canny": ARMS[arm]["has_canny"],
                "workflow": ARMS[arm]["workflow"], "prompt": PROMPT,
                "cn_strength": CN_STRENGTH, "cn_start": CN_START,
                "cn_end": CN_END, "workflow_version": WORKFLOW_VERSION,
            })
            cell += 1
    return rows


def build_workflow(template: dict, cell: dict, output_prefix: str) -> dict:
    subs: dict[str, Any] = {
        "$$POSITIVE_PROMPT": cell["prompt"],
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$SEED": int(cell["seed"]),
        "$$CANNY_FILENAME": CANNY_TEMPLATE,
        "$$CN_STRENGTH": float(cell["cn_strength"]),
        "$$CN_START": float(cell["cn_start"]),
        "$$CN_END": float(cell["cn_end"]),
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

    return _sub(copy.deepcopy(template))


def _retry(fn, *, tries: int = 3, what: str = ""):
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


def comfy_exec_seconds(entry: dict) -> float | None:
    """Server-side generation time from ComfyUI's history timestamps.

    Spans execution_start -> execution_success, so it covers the FULL
    generation: checkpoint load (cold first render of an arm), text encode,
    sampling, and VAE decode -- everything but our HTTP download. Returns
    None if the timestamps are absent.
    """
    msgs = entry.get("status", {}).get("messages", [])
    start = end = None
    for m in msgs:
        if not (isinstance(m, list) and len(m) == 2):
            continue
        name, payload = m
        ts = payload.get("timestamp") if isinstance(payload, dict) else None
        if ts is None:
            continue
        if name == "execution_start":
            start = ts
        elif name in ("execution_success", "execution_error"):
            end = ts
    if start is None or end is None:
        return None
    return round((end - start) / 1000.0, 2)


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
            os.replace(tmp, out_path)
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow-dir", type=Path, required=True,
                    help="dir holding the per-arm *.api.json workflows")
    ap.add_argument("--template", type=Path,
                    default=Path("refs/matryoshka/template_canny.png"))
    ap.add_argument("--out", type=Path, default=Path("refs_matryoshka_bakeoff"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("manifest_matryoshka_bakeoff.parquet"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    ap.add_argument("--arm", action="append", choices=sorted(ARMS),
                    help="run only this arm (repeatable); default = all")
    ap.add_argument("--limit", type=int, default=0,
                    help="render at most N cells (0 = all); for smoke tests")
    args = ap.parse_args()

    arms = args.arm or list(ARMS)

    # Load + validate each arm's workflow up front.
    templates: dict[str, dict] = {}
    for arm in arms:
        wf_path = args.workflow_dir / ARMS[arm]["workflow"]
        tpl = json.loads(wf_path.read_text())
        for nid, cls in ARMS[arm]["schedule"].items():
            got = tpl.get(nid, {}).get("class_type")
            if got != cls:
                raise SystemExit(f"{arm}: workflow node {nid} is {got!r}, "
                                 f"expected {cls!r}")
        templates[arm] = tpl

    chibi_dir = args.out / "chibi"
    chibi_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid(arms)
    print(f"[bakeoff] {len(grid)} cells over arms: {', '.join(arms)}")

    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        if args.template.exists():
            shutil.copy2(args.template, args.comfy_input_dir / CANNY_TEMPLATE)
        else:
            raise SystemExit(f"doll template not found: {args.template}")
        print(f"[bakeoff] staged Canny template into {args.comfy_input_dir}")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()
    results: list[dict] = []
    seen_arms: set[str] = set()  # first render of an arm is a cold model load

    for cell in grid:
        if args.limit and (done + failed) >= args.limit:
            break
        out_png = chibi_dir / cell["render"]
        if out_png.exists():
            skipped += 1
            results.append({**{k: v for k, v in cell.items() if k != "stem"},
                            "wall_s": None, "comfy_exec_s": None,
                            "is_cold": None, "status": "skipped"})
            continue

        is_cold = cell["arm"] not in seen_arms
        seen_arms.add(cell["arm"])
        wf = build_workflow(templates[cell["arm"]], cell,
                            output_prefix=f"matryoshka_bakeoff/{cell['stem']}")
        t_start = time.time()
        status = "ok"
        comfy_s: float | None = None
        try:
            pid = queue(sess, args.comfy_url, wf, client_id)
            entry = wait(sess, args.comfy_url, pid)
            comfy_s = comfy_exec_seconds(entry)
            if not download(sess, args.comfy_url, entry, out_png):
                status = "failed"
        except Exception as e:
            print(f"  [fail] {cell['stem']}: {e}")
            status = "failed"
        wall = round(time.time() - t_start, 2)

        results.append({**{k: v for k, v in cell.items() if k != "stem"},
                        "wall_s": wall, "comfy_exec_s": comfy_s,
                        "is_cold": is_cold, "status": status})
        if status == "ok":
            done += 1
            tag = "cold" if is_cold else "warm"
            print(f"  [ok] {cell['stem']} ({tag}: gen={comfy_s}s wall={wall}s) "
                  f"- {done} done, {skipped} skipped, {failed} failed")
        else:
            failed += 1

    pd.DataFrame(results).to_parquet(args.manifest, index=False)
    print(f"[bakeoff] complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.time()-t0)/60:.1f} min -> manifest {args.manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
