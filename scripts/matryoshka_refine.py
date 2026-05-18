"""Matryoshka swap-target refine pass.

Takes generated doll PNGs (e.g. the bake-off renders) and re-diffuses only the
face region into a face-like structure, so the downstream inswapper swap has a
detectable, in-distribution target. Identity is NOT added here -- the inpaint
prompt is identity-blind; the swap stays the identity vehicle.

Per doll: detect the face -> build a feathered mask -> run the arm's inpaint
workflow at the given denoise -> save to <out>/d<NNN>/<name>.png. Resumable
(skip-if-exists), atomic writes. Dolls with no detectable face are copied
through unrefined.

Usage (cwd = repo root; run on the box where ComfyUI is bound):
    python scripts/matryoshka_refine.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --renders exp_output/matryoshka_bakeoff/renders \\
        --out exp_output/matryoshka_bakeoff/refined \\
        --comfy-input-dir /home/newub/w/ComfyUI/input \\
        --denoise 0.4 --denoise 0.55 --denoise 0.7
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import cv2
import requests

from face_region import build_face_mask

REFINE_PROMPT = (
    "a realistic painted human face on a wooden doll, soft three-dimensional "
    "shading, correct facial proportions, defined nose and lips, "
    "natural-sized eyes, rosy cheeks, centered frontal face"
)
NEGATIVE_PROMPT = (
    "flat painted face, oversized eyes, cartoon, deformed, distorted, "
    "watermark, text, low quality, blurry"
)
REFINE_SEED = 88_000_001

# arm -> (inpaint workflow file, steps, sampler, scheduler).
# Step counts are RAISED above the bake-off generation recipe: partial-denoise
# img2img (denoise < 1) on a distilled few-step model only runs the tail
# `steps * denoise` of the schedule, so the bake-off's 6/4-step counts leave
# too few effective steps. Z-Image runs 12 (a bake-off-validated count); SDXL
# runs 8 to match the 8-step Lightning LoRA the workflow loads.
ARMS: dict[str, dict] = {
    "zimage_turbo": {"workflow": "matryoshka_zimage_inpaint.api.json",
                     "steps": 12, "sampler": "euler", "scheduler": "simple"},
    "sdxl_lightning": {"workflow": "matryoshka_sdxl_inpaint.api.json",
                       "steps": 8, "sampler": "dpmpp_sde",
                       "scheduler": "sgm_uniform"},
}


def arm_of(filename: str) -> str | None:
    """Arm name a render filename belongs to, or None if not a refine arm."""
    for arm in ARMS:
        if filename.startswith(arm):
            return arm
    return None


def denoise_subdir(denoise: float) -> str:
    """Stable subdir name for a denoise value: 0.55 -> 'd055'."""
    return f"d{int(round(denoise * 100)):03d}"


def build_workflow(template: dict, *, doll_file: str, mask_file: str,
                   denoise: float, arm_cfg: dict, output_prefix: str) -> dict:
    subs: dict[str, Any] = {
        "$$POSITIVE_PROMPT": REFINE_PROMPT,
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$DOLL_FILENAME": doll_file,
        "$$MASK_FILENAME": mask_file,
        "$$DENOISE": float(denoise),
        "$$SEED": REFINE_SEED,
        "$$STEPS": int(arm_cfg["steps"]),
        "$$SAMPLER": arm_cfg["sampler"],
        "$$SCHEDULER": arm_cfg["scheduler"],
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


def wait(sess: requests.Session, url: str, pid: str, timeout: float = 300) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = _retry(lambda: sess.get(f"{url}/history/{pid}", timeout=10),
                   what="history")
        if r is not None and r.status_code == 200 and pid in r.json():
            return r.json()[pid]
        time.sleep(1.0)
    raise TimeoutError(f"prompt {pid} did not complete within {timeout}s")


def download(sess: requests.Session, url: str, entry: dict,
             out_path: Path) -> bool:
    status = entry.get("status", {})
    if status.get("status_str") not in (None, "success"):
        print(f"  [fail] {out_path.stem}: status={status.get('status_str')}")
        return False
    for node_out in entry.get("outputs", {}).values():
        for img in node_out.get("images", []):
            r = _retry(lambda: sess.get(f"{url}/view", params={
                "filename": img["filename"],
                "subfolder": img.get("subfolder", ""),
                "type": img.get("type", "output")}, timeout=30), what="view")
            if r is None or r.status_code != 200 or \
                    r.content[:8] != b"\x89PNG\r\n\x1a\n":
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
    ap.add_argument("--renders", type=Path, required=True,
                    help="dir of generated doll PNGs to refine")
    ap.add_argument("--out", type=Path, required=True,
                    help="output root; refined dolls go to <out>/d<NNN>/")
    ap.add_argument("--workflow-dir", type=Path,
                    default=Path("comfyui/workflows"))
    ap.add_argument("--comfy-input-dir", type=Path, required=True,
                    help="ComfyUI input dir to stage doll + mask into")
    ap.add_argument("--denoise", type=float, action="append", required=True,
                    help="denoise value(s) to refine at (repeatable)")
    ap.add_argument("--limit", type=int, default=0,
                    help="refine at most N dolls per denoise (0 = all)")
    args = ap.parse_args()

    templates: dict[str, dict] = {}
    for arm, cfg in ARMS.items():
        tpl = json.loads((args.workflow_dir / cfg["workflow"]).read_text())
        classes = {n["class_type"] for n in tpl.values()}
        if "SetLatentNoiseMask" not in classes:
            raise SystemExit(f"{arm}: {cfg['workflow']} has no SetLatentNoiseMask "
                             f"-- regenerate with build_inpaint_workflow.py")
        templates[arm] = tpl

    args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
    dolls = sorted(p for p in args.renders.glob("*.png")
                   if arm_of(p.name) is not None)
    print(f"[refine] {len(dolls)} dolls x {len(args.denoise)} denoise values")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done = skipped = failed = passthrough = 0

    # Outer loop = doll, inner loop = denoise. Because `dolls` is arm-sorted,
    # this runs every cell for one arm before the next, so ComfyUI swaps the
    # checkpoint once for the whole run instead of once per denoise value.
    for n, doll_path in enumerate(dolls):
        if args.limit and n >= args.limit:
            break
        # Each denoise gets its own output subdir; skip the doll entirely if
        # every rung already exists (resumable, no model load / no detect).
        targets = [(d, args.out / denoise_subdir(d) / doll_path.name)
                   for d in args.denoise]
        pending = [(d, p) for d, p in targets if not p.exists()]
        skipped += len(targets) - len(pending)
        if not pending:
            continue

        arm = arm_of(doll_path.name)
        assert arm is not None  # filtered above
        doll_bgr = cv2.imread(str(doll_path))
        if doll_bgr is None:
            print(f"  [fail] unreadable {doll_path.name}")
            failed += len(pending)
            continue

        # Mask + comfy-input staging are denoise-independent: do them once.
        mask = build_face_mask(doll_bgr)
        if mask is None:
            # no detectable face -> pass the doll through unrefined
            for _d, out_png in pending:
                out_png.parent.mkdir(parents=True, exist_ok=True)
                tmp = out_png.with_suffix(".png.tmp")
                cv2.imwrite(str(tmp), doll_bgr)
                os.replace(tmp, out_png)
                passthrough += 1
            continue

        stem = doll_path.stem
        doll_file = f"refine_{stem}.png"
        mask_file = f"refine_{stem}_mask.png"
        cv2.imwrite(str(args.comfy_input_dir / doll_file), doll_bgr)
        cv2.imwrite(str(args.comfy_input_dir / mask_file), mask)

        for denoise, out_png in pending:
            wf = build_workflow(templates[arm], doll_file=doll_file,
                                mask_file=mask_file, denoise=denoise,
                                arm_cfg=ARMS[arm],
                                output_prefix=f"matryoshka_refine/{stem}")
            try:
                pid = queue(sess, args.comfy_url, wf, client_id)
                entry = wait(sess, args.comfy_url, pid)
                ok = download(sess, args.comfy_url, entry, out_png)
            except Exception as e:
                print(f"  [fail] {stem} d={denoise}: {e}")
                ok = False
            if ok:
                done += 1
                print(f"  [ok] {denoise_subdir(denoise)}/{doll_path.name} "
                      f"({done} done, {skipped} skipped, {passthrough} "
                      f"passthrough, {failed} failed)")
            else:
                failed += 1

    print(f"[refine] complete: {done} done, {skipped} skipped, "
          f"{passthrough} passthrough, {failed} failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
