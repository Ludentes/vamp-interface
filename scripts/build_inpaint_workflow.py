"""Transform a matryoshka bake-off generation workflow into an inpaint one.

A generation workflow synthesizes a doll from an empty latent. The face-region
refine pass instead re-diffuses only the masked face of an existing doll. This
module performs that transform deterministically so the inpaint workflows stay
in sync with the generation workflows they derive from -- rather than being
hand-authored and drifting.

Run as a script to (re)generate the committed inpaint workflow JSONs:
    python scripts/build_inpaint_workflow.py
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

_WF_DIR = Path(__file__).resolve().parents[1] / "comfyui" / "workflows"
_ARMS = {
    "matryoshka_zimage_turbo.api.json": "matryoshka_zimage_inpaint.api.json",
    "matryoshka_sdxl_lightning.api.json": "matryoshka_sdxl_inpaint.api.json",
}


def build_inpaint_workflow(base_wf: dict) -> dict:
    """Return a copy of base_wf rewired for masked face-region img2img.

    The empty-latent node feeding KSampler.latent_image is replaced by
    LoadImage(doll) -> VAEEncode -> SetLatentNoiseMask(mask). KSampler.denoise
    becomes the placeholder "$$DENOISE". A ControlNetApplyAdvanced feeding the
    KSampler conditioning is bypassed (its own positive/negative sources are
    wired straight into KSampler). New nodes load "$$DOLL_FILENAME" /
    "$$MASK_FILENAME". Nodes left unreachable by the rewire -- the empty-latent
    and the bypassed ControlNet chain (incl. the Canny LoadImage) -- are
    pruned: ComfyUI still validates such nodes and warns on their
    unsubstituted "$$" placeholders, so they cannot be left in the graph.
    """
    wf = copy.deepcopy(base_wf)
    ks_id = next(i for i, n in wf.items() if n["class_type"] == "KSampler")
    ks = wf[ks_id]

    vae_dec = next(n for n in wf.values() if n["class_type"] == "VAEDecode")
    vae_ref = list(vae_dec["inputs"]["vae"])

    for slot in ("positive", "negative"):
        src = wf[ks["inputs"][slot][0]]
        if src["class_type"] == "ControlNetApplyAdvanced":
            ks["inputs"][slot] = list(src["inputs"][slot])

    nid = max(int(i) for i in wf) + 1
    doll_id, mask_id, enc_id, snm_id = (str(nid + k) for k in range(4))
    wf[doll_id] = {"class_type": "LoadImage",
                   "inputs": {"image": "$$DOLL_FILENAME"}}
    wf[mask_id] = {"class_type": "LoadImageMask",
                   "inputs": {"image": "$$MASK_FILENAME", "channel": "red"}}
    wf[enc_id] = {"class_type": "VAEEncode",
                  "inputs": {"pixels": [doll_id, 0], "vae": vae_ref}}
    wf[snm_id] = {"class_type": "SetLatentNoiseMask",
                  "inputs": {"samples": [enc_id, 0], "mask": [mask_id, 0]}}
    ks["inputs"]["latent_image"] = [snm_id, 0]
    ks["inputs"]["denoise"] = "$$DENOISE"
    return _prune_unreachable(wf)


def _prune_unreachable(wf: dict) -> dict:
    """Drop nodes not reachable from any output (SaveImage) node.

    Walks input edges -- the [node_id, slot] pairs in each node's "inputs" --
    backward from every SaveImage node, then keeps only the visited set.
    """
    reachable: set[str] = set()
    stack = [i for i, n in wf.items() if n["class_type"] == "SaveImage"]
    while stack:
        nid = stack.pop()
        if nid in reachable or nid not in wf:
            continue
        reachable.add(nid)
        for val in wf[nid].get("inputs", {}).values():
            if (isinstance(val, list) and len(val) == 2
                    and isinstance(val[0], str)):
                stack.append(val[0])
    return {i: n for i, n in wf.items() if i in reachable}


def main() -> int:
    for src, dst in _ARMS.items():
        base = json.loads((_WF_DIR / src).read_text())
        out = build_inpaint_workflow(base)
        (_WF_DIR / dst).write_text(json.dumps(out, indent=2) + "\n")
        print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
