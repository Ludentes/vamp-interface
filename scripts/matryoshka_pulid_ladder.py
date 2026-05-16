"""Matryoshka PuLID-weight ladder: does identity survive the matryoshka prompt?

Option-A probe (identity injected during generation). The Phase-0 sweep capped
PuLID weight at 0.8 and produced generic dolls. This ladder overcranks PuLID
to 4.0 to confirm there is ANY identity signal that survives the flat-painted
matryoshka style, then walks the weight back down to find where resemblance
disappears.

Single identity (id_03 -- our inswapper anchor), single seed, CN held at 0.5
so the doll silhouette stays fixed; PuLID runs the whole schedule (start 0.0,
end 1.0) for maximum identity. Resumable (skip-if-exists).

Run ON the ComfyUI box (ComfyUI binds 127.0.0.1), cwd = data/importer:
    python ../../scripts/matryoshka_pulid_ladder.py \\
        --comfy-url http://127.0.0.1:8188 \\
        --workflow workflows/flux_pulid_canny_lora.api.json \\
        --id-dir identities_flux --template refs/matryoshka/template_canny.png \\
        --out refs_matryoshka --comfy-input-dir C:/comfy/ComfyUI/input
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
import uuid
from pathlib import Path

import requests

from matryoshka_sweep import (
    CANNY_TEMPLATE,
    PROMPT,
    SCHEDULE_NODES,
    build_workflow,
    download,
    queue,
    wait,
)

# overcrank first (confirm any identity signal), then walk down past the
# Phase-0 ceiling of 0.8 to find where resemblance dies.
PULID_WEIGHTS = [4.0, 3.0, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4]
IDENTITY = "id_03"          # our inswapper anchor -- keeps the comparison honest
SEED = 70_000_000           # one fixed seed: weight is the only variable
CN_STRENGTH = 0.5           # hold the doll silhouette
CN_START, CN_END = 0.0, 0.5
PULID_START, PULID_END = 0.0, 1.0   # identity over the whole schedule


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy-url", default="http://127.0.0.1:8188")
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--id-dir", type=Path, default=Path("identities_flux"))
    ap.add_argument("--template", type=Path,
                    default=Path("refs/matryoshka/template_canny.png"))
    ap.add_argument("--out", type=Path, default=Path("refs_matryoshka"))
    ap.add_argument("--comfy-input-dir", type=Path, default=None)
    args = ap.parse_args()

    template = json.loads(args.workflow.read_text())
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(f"workflow node {nid} is {got!r}, expected {cls!r}")

    out_dir = args.out / "pulid_ladder"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.comfy_input_dir is not None:
        args.comfy_input_dir.mkdir(parents=True, exist_ok=True)
        if not args.template.exists():
            raise SystemExit(f"doll template not found: {args.template}")
        shutil.copy2(args.template, args.comfy_input_dir / CANNY_TEMPLATE)
        src = args.id_dir / f"{IDENTITY}.png"
        if not src.exists():
            raise SystemExit(f"identity PNG not found: {src}")
        shutil.copy2(src, args.comfy_input_dir / src.name)
        print(f"[ladder] staged template + {IDENTITY}.png into {args.comfy_input_dir}")

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    done, skipped, failed = 0, 0, 0
    t0 = time.time()

    for pw in PULID_WEIGHTS:
        stem = f"{IDENTITY}_pulidladder_pw{int(round(pw * 100)):03d}_seed{SEED}"
        out_png = out_dir / f"{stem}.png"
        if out_png.exists():
            skipped += 1
            print(f"  [skip] {stem}")
            continue

        cell = {
            "identity": IDENTITY, "prompt": PROMPT, "seed": SEED,
            "pulid_weight": pw, "cn_strength": CN_STRENGTH,
            "cn_start": CN_START, "cn_end": CN_END,
            "pulid_start": PULID_START, "pulid_end": PULID_END,
        }
        wf = build_workflow(template, cell, output_prefix=f"matryoshka/ladder/{stem}")
        t_start = time.time()
        try:
            pid = queue(sess, args.comfy_url, wf, client_id)
            entry = wait(sess, args.comfy_url, pid)
            if not download(sess, args.comfy_url, entry, out_png):
                failed += 1
                print(f"  [fail] {stem}: no image in history")
                continue
        except Exception as e:
            failed += 1
            print(f"  [fail] {stem}: {e}")
            continue

        done += 1
        print(f"  [ok] pw={pw:.2f} {stem} ({time.time()-t_start:.1f}s)")

    print(f"[ladder] complete: {done} done, {skipped} skipped, {failed} failed "
          f"in {(time.time()-t0)/60:.1f} min -> {out_dir}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
