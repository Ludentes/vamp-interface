"""Generate one flat-lit, front-facing, chibi-styled face portrait per anchor,
via the project Flux + PuLID pipeline. Resumable: skip if the PNG exists.

ENTRY POINT — what the matryoshka thread actually does
------------------------------------------------------
The project does NOT call Flux as a Python module. Generation is driven
entirely through the **ComfyUI REST API**: a workflow-template JSON
(`data/importer/workflows/flux_pulid_canny_lora.api.json`) has `$$`-prefixed
placeholders substituted per render, the graph is POSTed to `/prompt`, polled
on `/history/{id}`, and the result PNG fetched from `/view`. This wrapper
reuses the exact `queue` / `wait` / `download` helpers and the
`SCHEDULE_NODES` startup assertion from `scripts/matryoshka_sweep.py` (which
`scripts/matryoshka_pulid_ladder.py` and `scripts/chibi_prompt_sweep.py` both
build on) so it stays bit-identical to the proven path.

IDENTITY INPUT — image, not an embedding vector
-----------------------------------------------
The Task-4 plan skeleton names the identity parameter `embedding`. The real
matryoshka pipeline has no embedding-vector path: identity is a **reference
image PNG**. In the workflow, `LoadImage` (node 7) reads `$$IDENTITY_FILENAME`
and feeds it into `ApplyPulidFlux` (node 8); PuLID-Flux derives its own face
embedding internally via `PulidFluxInsightFaceLoader` + `PulidFluxEvaClipLoader`.
There is nowhere to inject a precomputed vector without changing the graph.

So this wrapper's identity parameter is `identity_image`: a path to a face
reference PNG. The caller must stage that PNG into the ComfyUI input directory
(same as `chibi_prompt_sweep.py` does with `--comfy-input-dir`) so the
`LoadImage` node can find it by basename. The parameter is honest about what
the pipeline actually consumes; it is never silently ignored.
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

import requests

# Reuse the proven ComfyUI-API helpers from the matryoshka driver. The scripts/
# dir is not an installed package, so add it to sys.path on import.
_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from matryoshka_sweep import (  # noqa: E402
    NEGATIVE_PROMPT,
    SCHEDULE_NODES,
    download,
    queue,
    wait,
)

# Flat-lit, front-facing, chibi-styled face. Matches the plan's intent.
PROMPT = ("chibi character face portrait, front view, big round eyes, "
          "soft flat even studio lighting, no harsh shadows, matte skin, "
          "centered, neutral expression, plain background")

# Chibi style LoRA — the same file the chibi prompt sweep used to produce the
# id_14 reference render (see scripts/chibi_prompt_sweep.py).
LORA_CHIBI = "Ksbt_000001750.safetensors"
LORA_FURRY = "Anime_Furry_Style_Flux.safetensors"  # inert, strength 0.0
LORA_CHIBI_STRENGTH = 0.70

# PuLID identity weight + schedule. 0.7 is the matryoshka/chibi-sweep default —
# enough identity signal without overpowering the flat chibi style.
PULID_WEIGHT = 0.7
PULID_START, PULID_END = 0.0, 1.0

# Canny ControlNet holds the front-facing chibi face structure. Released after
# the structural phase, matching the matryoshka driver's CN window.
CN_STRENGTH = 0.5
CN_START, CN_END = 0.0, 0.5

WIDTH = HEIGHT = 1024

# Default workflow template, relative to repo root.
DEFAULT_WORKFLOW = (
    Path(__file__).resolve().parents[2]
    / "data" / "importer" / "workflows" / "flux_pulid_canny_lora.api.json"
)


def _build_workflow(template: dict, *, identity_filename: str,
                     canny_filename: str, prompt: str, seed: int,
                     output_prefix: str) -> dict:
    """Substitute the workflow's $$ placeholders and set the CN/PuLID windows.

    Mirrors scripts/matryoshka_sweep.build_workflow but keeps a single fixed
    identity + canny pair (no grid). The EmptyLatentImage node is forced to
    1024x1024 so output size is honored regardless of the template's literal.
    """
    import copy

    subs = {
        "$$IDENTITY_FILENAME": identity_filename,
        "$$CANNY_FILENAME": canny_filename,
        "$$POSITIVE_PROMPT": prompt,
        "$$NEGATIVE_PROMPT": NEGATIVE_PROMPT,
        "$$SEED": int(seed),
        "$$PULID_WEIGHT": float(PULID_WEIGHT),
        "$$CN_STRENGTH": float(CN_STRENGTH),
        "$$LORA_A_NAME": LORA_CHIBI,
        "$$LORA_A_STRENGTH": float(LORA_CHIBI_STRENGTH),
        "$$LORA_B_NAME": LORA_FURRY,
        "$$LORA_B_STRENGTH": 0.0,
        "$$OUTPUT_PREFIX": output_prefix,
    }

    def _sub(node):
        if isinstance(node, dict):
            return {k: _sub(v) for k, v in node.items() if not k.startswith("_")}
        if isinstance(node, list):
            return [_sub(v) for v in node]
        if isinstance(node, str) and node in subs:
            return subs[node]
        return node

    wf = _sub(copy.deepcopy(template))
    # Schedule literals — node "13" is CN, node "8" is PuLID (SCHEDULE_NODES).
    wf["13"]["inputs"]["start_percent"] = CN_START
    wf["13"]["inputs"]["end_percent"] = CN_END
    wf["8"]["inputs"]["start_at"] = PULID_START
    wf["8"]["inputs"]["end_at"] = PULID_END
    # Force 1024x1024 output.
    wf["14"]["inputs"]["width"] = WIDTH
    wf["14"]["inputs"]["height"] = HEIGHT
    return wf


def generate_portrait(anchor_id: str, identity_image, out_dir: str | Path,
                      seed: int, *,
                      canny_image: str | Path | None = None,
                      comfy_url: str = "http://127.0.0.1:8188",
                      workflow: str | Path = DEFAULT_WORKFLOW,
                      timeout: float = 420.0) -> Path:
    """Generate ``<out_dir>/<anchor_id>.png`` if absent. Returns the PNG path.

    Resumable: if the PNG already exists, returns it immediately without
    touching ComfyUI (project rule ``feedback_resumable_generation``).

    Parameters
    ----------
    anchor_id
        Job-anchor id; names the output PNG.
    identity_image
        Path to the face reference PNG for PuLID. The pipeline takes an
        *image*, not an embedding vector — see the module docstring. Its
        basename must already be staged in the ComfyUI input directory so the
        workflow's ``LoadImage`` node can resolve it.
    out_dir
        Directory the portrait PNG is written into.
    seed
        Generation seed (fixed per anchor → deterministic face).
    canny_image
        Optional face-structure Canny PNG (basename staged in the ComfyUI
        input dir). Defaults to ``<identity_image stem>_canny.png``.
    comfy_url
        ComfyUI REST endpoint (binds 127.0.0.1 on the ComfyUI box).
    workflow
        Path to the Flux+PuLID+Canny workflow-template JSON.
    timeout
        Per-render wait budget passed to ``matryoshka_sweep.wait``.
    """
    out = Path(out_dir) / f"{anchor_id}.png"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    identity_image = Path(identity_image)
    canny_filename = (Path(canny_image).name if canny_image is not None
                      else f"{identity_image.stem}_canny.png")

    template = json.loads(Path(workflow).read_text())
    # Fail loud if a re-exported workflow renumbered the schedule nodes.
    for nid, cls in SCHEDULE_NODES.items():
        got = template.get(nid, {}).get("class_type")
        if got != cls:
            raise SystemExit(
                f"workflow node {nid} is {got!r}, expected {cls!r} — "
                f"schedule/identity injection would target the wrong node")

    wf = _build_workflow(
        template,
        identity_filename=identity_image.name,
        canny_filename=canny_filename,
        prompt=PROMPT,
        seed=seed,
        output_prefix=f"chibi_portrait/{anchor_id}",
    )

    sess = requests.Session()
    client_id = str(uuid.uuid4())
    pid = queue(sess, comfy_url, wf, client_id)
    entry = wait(sess, comfy_url, pid, timeout=timeout)
    if not download(sess, comfy_url, entry, out):
        raise RuntimeError(f"chibi portrait for {anchor_id!r}: no image in history")
    return out
