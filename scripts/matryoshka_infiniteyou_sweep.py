"""Matryoshka InfiniteYou sweep: identity-preserving single nesting doll.

InfiniteYou (InfuseNet) replaces PuLID for identity. InfuseNet chains after a
fixed Canny doll-silhouette ControlNet, so doll structure and face identity
compose. Base = FLUX.1-dev fp8, sim_stage1 variant.

Grid: identity x infusenet_strength x infusenet_end x 2 seeds = 48 cells.
Canny is the SAME doll-silhouette template for every cell (structure tool).
Manifest written up front, resumable (skip-if-exists), atomic PNG writes.

Usage (cwd = data/importer):
    python scripts/matryoshka_infiniteyou_sweep.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow ../../comfyui/workflows/flux_infiniteyou_canny.api.json \\
        --id-dir identities_flux --template refs/matryoshka/template_canny.png \\
        --out refs_matryoshka_infu --manifest manifest_matryoshka_infu.parquet \\
        --comfy-input-dir /home/newub/w/ComfyUI/input
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

WORKFLOW_VERSION = "matryoshka_infu_2026-05-17"
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

CN_STRENGTH = 0.5
CN_START, CN_END = 0.0, 0.5
INFU_START = 0.0

INFU_STRENGTHS = [0.6, 0.8, 1.0]
INFU_ENDS = [0.5, 0.8]
IDENTITIES = [3, 8, 14, 15]
SEEDS_PER_CELL = 2

# Nodes whose literals the sweep overwrites -- asserted at startup so a
# re-exported (renumbered) workflow fails loud.
SCHEDULE_NODES = {"8": "ControlNetApplyAdvanced", "14": "InfuseNetApply",
                  "16": "KSampler"}


def build_grid() -> list[dict]:
    rows: list[dict] = []
    cell = 0
    for idx in IDENTITIES:
        for st in INFU_STRENGTHS:
            for en in INFU_ENDS:
                for _ in range(SEEDS_PER_CELL):
                    seed = 71_000_000 + cell * 7919
                    st_tag = f"st{int(round(st * 100)):03d}"
                    en_tag = f"en{int(round(en * 100)):03d}"
                    identity = f"id_{idx:02d}"
                    stem = f"{identity}_{st_tag}_{en_tag}_seed{seed}"
                    rows.append({
                        "render": f"{stem}.png", "stem": stem, "id_idx": idx,
                        "identity": identity, "infu_strength": st,
                        "infu_end": en, "infu_start": INFU_START,
                        "cn_strength": CN_STRENGTH, "cn_start": CN_START,
                        "cn_end": CN_END, "seed": seed, "prompt": PROMPT,
                        "workflow_version": WORKFLOW_VERSION,
                    })
                    cell += 1
    return rows


def build_workflow(template: dict, cell: dict, output_prefix: str) -> dict:
    subs: dict[str, Any] = {
        "$$IDENTITY_FILENAME": f"{cell['identity']}.png",
        "$$CANNY_FILENAME": CANNY_TEMPLATE,
        "$$POSITIVE_PROMPT": cell["prompt"],
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$SEED": int(cell["seed"]),
        "$$CN_STRENGTH": float(cell["cn_strength"]),
        "$$CN_START": float(cell["cn_start"]),
        "$$CN_END": float(cell["cn_end"]),
        "$$INFU_STRENGTH": float(cell["infu_strength"]),
        "$$INFU_START": float(cell["infu_start"]),
        "$$INFU_END": float(cell["infu_end"]),
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
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--id-dir", type=Path, default=Path("identities_flux"))
    ap.add_argument("--template", type=Path,
                    default=Path("refs/matryoshka/template_canny.png"))
    ap.add_argument("--out", type=Path, default=Path("refs_matryoshka_infu"))
    ap.add_argument("--manifest", type=Path,
                    default=Path("manifest_matryoshka_infu.parquet"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=0,
                    help="render at most N cells (0 = all); for smoke tests")
    args = ap.parse_args()

    template = json.loads(args.workflow.read_text())
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(f"workflow node {nid} is {got!r}, expected {cls!r}")

    chibi_dir = args.out / "chibi"
    chibi_dir.mkdir(parents=True, exist_ok=True)

    grid = build_grid()
    pd.DataFrame([{k: v for k, v in c.items() if k != "stem"} for c in grid]
                 ).to_parquet(args.manifest, index=False)
    print(f"[sweep] {len(grid)} cells -> manifest {args.manifest}")

    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        if args.template.exists():
            shutil.copy2(args.template, args.comfy_input_dir / CANNY_TEMPLATE)
        else:
            raise SystemExit(f"doll template not found: {args.template}")
        for idx in IDENTITIES:
            src = args.id_dir / f"id_{idx:02d}.png"
            if src.exists():
                shutil.copy2(src, args.comfy_input_dir / src.name)
        print(f"[sweep] staged template + ID PNGs into {args.comfy_input_dir}")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for cell in grid:
        if args.limit and (done + failed) >= args.limit:
            break
        out_png = chibi_dir / cell["render"]
        if out_png.exists():
            skipped += 1
            continue

        wf = build_workflow(template, cell,
                            output_prefix=f"matryoshka_infu/{cell['stem']}")
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
