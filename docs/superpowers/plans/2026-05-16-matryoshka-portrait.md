# Matryoshka Portrait Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Given a person's photo, render a single identity-preserving Russian matryoshka doll, with a sweep + scorer to pick the best generation cell.

**Architecture:** Reuse the chibi hero-doll node graph — Flux-Krea + PuLID (identity) + Canny ControlNet of a doll-silhouette template (structure) + style prompt. Two new scripts cloned from the chibi sweep: a resumable sweep runner and a scorer. A Phase 0 prompt-only baseline gates whether a matryoshka style LoRA is needed; a Phase 0.5 generator bake-off picks which generator the full sweep runs on.

**Tech Stack:** Python 3.12 / uv, ComfyUI HTTP API at 127.0.0.1:8188, insightface buffalo_l, MediaPipe FaceLandmarker, open-clip for the matryoshka-ness metric.

Spec: `docs/superpowers/specs/2026-05-16-matryoshka-portrait-design.md`

---

## File Structure

- `data/importer/refs/matryoshka/template_canny.png` — Canny edge map of one generic matryoshka silhouette (the structural ControlNet input). Created once.
- `scripts/matryoshka_sweep.py` — sweep runner. Clone of `scripts/chibi_highstr_sweep.py` with the matryoshka grid. One responsibility: drive ComfyUI over the grid, write a deterministic manifest, download PNGs atomically.
- `scripts/score_matryoshka.py` — scorer. Clone of `scripts/score_highstr.py` plus a CLIP matryoshka-ness metric. One responsibility: score each render and report marginals.
- `tests/test_matryoshka_grid.py` — unit tests for the deterministic grid + workflow substitution (the only ComfyUI-free testable logic).

ComfyUI-touching code (queue/wait/download) is not unit-tested — it is verified by the Phase 0 / Phase 1 run gates and reviewed by `superpowers:code-reviewer` per the project's standing rule.

---

### Task 1: Doll-silhouette Canny template

**Files:**
- Create: `data/importer/refs/matryoshka/template_canny.png`
- Create: `scripts/make_matryoshka_template.py`

- [ ] **Step 1: Write the template generator**

A matryoshka silhouette is a smooth "bowling-pin" form: a hemispherical head blending into an ovoid body, vertical symmetry, flat base. Draw it as a filled polygon/ellipse composite on a 1024×1024 white canvas, then Canny it so ControlNet sees only the outline.

```python
"""Generate the Canny doll-silhouette template for the matryoshka sweep.

One generic matryoshka outline: hemispherical head merging into an ovoid
body, flat base, vertically symmetric. ControlNet consumes the edge map so
the render keeps the doll form instead of drifting into a portrait.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "importer" / "refs" / "matryoshka" / "template_canny.png"
SIZE = 1024


def main() -> int:
    canvas = np.full((SIZE, SIZE), 255, dtype=np.uint8)
    cx = SIZE // 2
    # body: tall ellipse, lower 2/3 of the canvas
    cv2.ellipse(canvas, (cx, int(SIZE * 0.62)), (int(SIZE * 0.30), int(SIZE * 0.34)),
                0, 0, 360, 0, -1)
    # head: smaller ellipse overlapping the body top, slightly narrower
    cv2.ellipse(canvas, (cx, int(SIZE * 0.30)), (int(SIZE * 0.22), int(SIZE * 0.24)),
                0, 0, 360, 0, -1)
    # flat base: clip the bottom of the body to a straight edge
    cv2.rectangle(canvas, (0, int(SIZE * 0.92)), (SIZE, SIZE), 255, -1)
    edges = cv2.Canny(canvas, 50, 150)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), edges)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/make_matryoshka_template.py`
Expected: `wrote .../template_canny.png`

- [ ] **Step 3: Eyeball the template**

Open `data/importer/refs/matryoshka/template_canny.png`. Expected: a single white doll outline on black — head-on-body bowling-pin shape, flat base, no interior noise. If the head/body proportions look wrong, adjust the ellipse constants and re-run.

- [ ] **Step 4: Commit**

```bash
git add scripts/make_matryoshka_template.py data/importer/refs/matryoshka/template_canny.png
git commit -m "feat(matryoshka): doll-silhouette Canny template"
```

---

### Task 2: Sweep runner — deterministic grid (TDD)

**Files:**
- Create: `scripts/matryoshka_sweep.py`
- Test: `tests/test_matryoshka_grid.py`

This task builds and tests only the ComfyUI-free logic: `build_grid` and `build_workflow`. Clone `scripts/chibi_highstr_sweep.py` wholesale, then change the constants below — the `_retry`/`queue`/`wait`/`download`/`main` plumbing is copied verbatim.

Grid for v1 (single doll, standalone feature):

```python
WORKFLOW_VERSION = "matryoshka_2026-05-16"
LORA_CHIBI = "Ksbt_000001750.safetensors"   # placeholder; LORA_B_STRENGTH stays 0.0

BASE_PROMPT = ("a Russian matryoshka nesting doll, single doll, painted "
               "wooden figure, glossy lacquer finish, floral folk-art shawl "
               "and apron, flat painted face, plain background, centered")
# additive identity-styling suffix; the painted-face read is the risk axis
STYLE_SUFFIX = ", traditional khokhloma painting, rosy painted cheeks, hand-painted detail"
PROMPT = BASE_PROMPT + STYLE_SUFFIX

CN_START, CN_END = 0.0, 0.5          # doll-form structure, released after layout
PULID_STARTS = [0.0, 0.1]            # identity entry point
PULID_WEIGHTS = [0.4, 0.6, 0.8]      # identity vs flat-painted-style tension
PULID_END = 0.7
CN_STRENGTHS = [0.0, 0.5]            # 0.0 == prompt-only (ablation/baseline arm)
IDENTITIES = [3, 8, 14, 15]
SEEDS_PER_CELL = 2
```

`build_grid` iterates `IDENTITIES × CN_STRENGTHS × PULID_WEIGHTS × PULID_STARTS × SEEDS_PER_CELL` (96 cells). Seed = `70_000_000 + cell * 7919`. Stem = `id_{idx:02d}_cn{int(cn*100):03d}_pw{int(pw*100):03d}_ps{int(ps*100):03d}_seed{seed}`. Each row carries `cn_strength, pulid_weight, pulid_start, pulid_end (=PULID_END), cn_start, cn_end, prompt`.

`build_workflow` is copied unchanged from `chibi_highstr_sweep.py` — it already substitutes `$$CN_STRENGTH`, `$$PULID_WEIGHT`, `start_at`/`end_at` on node 8, `start_percent`/`end_percent` on node 13. The canny filename substitution becomes the shared template: `"$$CANNY_FILENAME": "matryoshka_template_canny.png"` (same template for every cell — copied into the ComfyUI input dir once in `main`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_matryoshka_grid.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matryoshka_sweep as ms


def test_grid_size():
    grid = ms.build_grid()
    assert len(grid) == 96  # 4 id x 2 cn x 3 pw x 2 ps x 2 seeds


def test_seeds_and_stems_unique():
    grid = ms.build_grid()
    assert len({c["seed"] for c in grid}) == 96
    assert len({c["stem"] for c in grid}) == 96


def test_seed_is_deterministic():
    grid = ms.build_grid()
    assert grid[0]["seed"] == 70_000_000
    assert grid[1]["seed"] == 70_000_000 + 7919


def test_cn0_cell_is_prompt_only():
    grid = ms.build_grid()
    cn0 = [c for c in grid if c["cn_strength"] == 0.0]
    assert len(cn0) == 48
    assert all(c["prompt"] == ms.PROMPT for c in cn0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_matryoshka_grid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'matryoshka_sweep'`

- [ ] **Step 3: Create the script**

Copy `scripts/chibi_highstr_sweep.py` to `scripts/matryoshka_sweep.py`. Replace the constants block (lines 51–84 region) with the grid constants above. Replace `build_grid` so it produces the 96-cell grid described above. Replace `ABLATIONS`/`STRENGTHS` usages. In `build_workflow` set `$$CANNY_FILENAME` to the fixed template name and `$$LORA_A_STRENGTH` to `0.0` (no style LoRA in v1). Update the docstring to describe the matryoshka v1 single-doll sweep. Default `--out refs_matryoshka`, `--manifest manifest_matryoshka.parquet`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_matryoshka_grid.py -v`
Expected: PASS — all 4 tests.

- [ ] **Step 5: Code review**

Dispatch `superpowers:code-reviewer` on `scripts/matryoshka_sweep.py` against the spec. Fix any correctness findings (grid construction, workflow substitution, seed determinism). Per the project's standing rule, no non-trivial script is "done" before this review.

- [ ] **Step 6: Commit**

```bash
git add scripts/matryoshka_sweep.py tests/test_matryoshka_grid.py
git commit -m "feat(matryoshka): single-doll sweep runner + grid tests"
```

---

### Task 3: Scorer

**Files:**
- Create: `scripts/score_matryoshka.py`

Clone `scripts/score_highstr.py`. Keep `score_png` (buffalo_l age/gender, ArcFace detection, MediaPipe). Changes:

- `SWEEP = IMPORTER / "refs_matryoshka" / "chibi"` → use the matryoshka out dir; `MANIFEST = manifest_matryoshka.parquet`; `OUT = exp_output/matryoshka_score/features.parquet`.
- Remove the `id_14` reference-vector / `ref_dist` block — v1 has no hand-picked reference render.
- Add a CLIP matryoshka-ness metric:

```python
import open_clip
import torch

_CLIP_PROMPTS = ["a Russian matryoshka nesting doll",
                 "a ceramic figurine", "a chibi character", "a photo of a person"]


def make_clip():
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k")
    tok = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        text = model.encode_text(tok(_CLIP_PROMPTS))
        text /= text.norm(dim=-1, keepdim=True)
    return model, preprocess, text


def clip_matryoshka(model, preprocess, text, path) -> float:
    """Softmax prob that the render reads as a matryoshka vs distractors."""
    from PIL import Image
    img = preprocess(Image.open(path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(img)
        feat /= feat.norm(dim=-1, keepdim=True)
        probs = (100.0 * feat @ text.T).softmax(dim=-1)
    return float(probs[0, 0])  # index 0 == matryoshka prompt
```

Add `clip_matryoshka` to each scored row. Report marginals by `cn_strength`, `pulid_weight`, `pulid_start`: ArcFace detection rate (uncanny proxy, low = good), MediaPipe rate, mean `clip_matryoshka`, mean apparent age + gender-match rate vs the manifest's source identity. Add a `cn_strength × pulid_weight` pivot of `clip_matryoshka`.

- [ ] **Step 1: Write the script** — per the description above.

- [ ] **Step 2: Smoke-test the CLIP loader**

Run: `uv run python -c "import scripts.score_matryoshka as s; s.make_clip(); print('clip ok')"` (from repo root, or adjust import). Expected: `clip ok` — confirms `open_clip` weights download and load before the full scoring run. If `open_clip` is missing: `uv add open_clip_torch`.

- [ ] **Step 3: Code review**

Dispatch `superpowers:code-reviewer` on `scripts/score_matryoshka.py`. Fix correctness findings.

- [ ] **Step 4: Commit**

```bash
git add scripts/score_matryoshka.py
git commit -m "feat(matryoshka): scorer with CLIP matryoshka-ness metric"
```

---

### Task 4: Phase 0 baseline run + gate

**Files:** none created — this is an experiment gate.

- [ ] **Step 1: Stage the template on the ComfyUI box**

Copy `data/importer/refs/matryoshka/template_canny.png` to the ComfyUI input dir as `matryoshka_template_canny.png` (the sweep runner's `--comfy-input-dir` staging step handles this if wired; otherwise scp it).

- [ ] **Step 2: Run the prompt-only arm only**

The `cn_strength == 0.0` cells (48 of them) are the prompt-only baseline. Run the full sweep — it produces both arms — or temporarily restrict `CN_STRENGTHS = [0.0]` for a fast Phase 0. On the ComfyUI box, cwd `data/importer`:

```
python scripts/matryoshka_sweep.py --comfy-url http://127.0.0.1:8188 \
  --workflow workflows/flux_pulid_canny_lora.api.json \
  --id-dir identities_flux --canny-dir cn_canny_flux \
  --out refs_matryoshka --manifest manifest_matryoshka.parquet \
  --comfy-input-dir C:/comfy/ComfyUI/input
```

- [ ] **Step 3: Eyeball gate**

Build a contact sheet of the prompt-only renders across the 4 identities. **Gate question:** does prompt-only Flux-Krea read as a matryoshka *and* is the face recognizable as the person? 
- If yes → proceed to Task 5 (bake-off); no LoRA needed.
- If the doll form is fine but the painted-face style is unconvincing → a matryoshka style LoRA is needed; stop and open a Phase 2 LoRA-training task (out of scope for this plan — record the finding in the spec and a research doc).

---

### Task 5: Generator bake-off (Phase 0.5)

**Files:**
- Create: `exp_output/matryoshka_bakeoff/` (renders, one subfolder per arm)
- Create: `docs/research/2026-05-16-matryoshka-generator-bakeoff.md`

Run the same 4 identities × the matryoshka prompt through 5 architecturally distinct arms, score with `score_matryoshka.py`, record wall-clock latency per render, and pick the generator the full sweep (Task 6) runs on. Each arm produces 4 PNGs (one per identity, fixed seed) into `exp_output/matryoshka_bakeoff/<arm>/`.

Arms:
- `flux_krea_pulid` — current pipeline; `matryoshka_sweep.py` restricted to one cell (`cn_strength=0.5, pulid_weight=0.6, pulid_start=0.1`).
- `flux_schnell_pulid` — same ComfyUI graph, checkpoint swapped to FLUX.1-schnell, KSampler steps 4, PuLID-for-Schnell weights.
- `flux2_klein` — FLUX.2 [klein] 4B; person photo passed as a reference image, no PuLID node, no ControlNet. Needs the klein ComfyUI nodes installed.
- `sdxl_lightning_ipa` — SDXL-Lightning 8-step + IP-Adapter (face) + Canny ControlNet on the doll template.
- `zimage_inswapper` — Z-Image-Turbo (8-step) + ControlNet on the doll template for the *generic* doll, then `inswapper_128` swaps the identity face in as a post-process. **No output-side detection:** source face + kps detected on the raw input photo; target kps come from a fixed canonical layout for the doll head box the Canny template enforces (optionally render the doll with a blank face oval to make the target region fully deterministic). This keeps the swap robust no matter how stylized the doll face is.

- [ ] **Step 1: Stand up each arm's workflow**

For arms 1–2 reuse `flux_pulid_canny_lora.api.json` (arm 2 swaps the checkpoint loader + step count). For arms 3–5 install the model + nodes on the ComfyUI box and save one API-format workflow JSON per arm under `data/importer/workflows/matryoshka_<arm>.api.json`. Verify each renders one test image before the batch.

- [ ] **Step 2: Render all arms**

For each arm, render the 4 identities at a fixed seed (`80_000_000`). Save to `exp_output/matryoshka_bakeoff/<arm>/id_NN.png`. Capture per-render wall-clock latency (the ComfyUI `/history` entry carries execution timing) into `exp_output/matryoshka_bakeoff/latency.csv` with columns `arm,identity,seconds`.

- [ ] **Step 3: Score every arm**

Point `score_matryoshka.py` at each arm folder (or add an `--in-dir` argument). Collect per-arm: mean `clip_matryoshka`, ArcFace detection rate, MediaPipe rate, gender-match rate, mean latency.

- [ ] **Step 4: Eyeball + pick**

Build a 5-row (arm) × 4-col (identity) montage → `exp_output/matryoshka_bakeoff/montage.png`. **The leveling metric is the human eyeball rating + CLIP matryoshka-ness — NOT ArcFace detection**, which is degenerate for the `zimage_inswapper` arm (a swap manufactures a detectable photoreal face by construction). Pick the arm with the best recognizability × matryoshka-ness at acceptable latency.

- [ ] **Step 5: Record + commit**

Write the bake-off table, the montage reference, and the chosen generator + reasoning to `docs/research/2026-05-16-matryoshka-generator-bakeoff.md`.

```bash
git add docs/research/2026-05-16-matryoshka-generator-bakeoff.md
git commit -m "docs(matryoshka): generator bake-off results + chosen generator"
```

---

### Task 6: Full sweep, score, pick winner

**Files:** none created — analysis.

The full sweep runs on the **bake-off winner** from Task 5. If the winner is not `flux_krea_pulid`, point `matryoshka_sweep.py --workflow` at that arm's workflow JSON; the grid axes (`cn_strength`, `pulid_weight`, `pulid_start`) still apply for the Flux/SDXL arms. If the winner is `flux2_klein` (no PuLID/CN), the sweep axes collapse to reference-strength + prompt variants — adjust `build_grid` accordingly before running.

- [ ] **Step 1: Run the full 96-cell sweep** (if Phase 0 ran only `CN_STRENGTHS=[0.0]`, restore `[0.0, 0.5]` and re-run — skip-if-exists keeps the baseline renders).

- [ ] **Step 2: Score**

Run: `uv run python scripts/score_matryoshka.py`
Expected: `=== N cells → exp_output/matryoshka_score/features.parquet ===` plus the marginal tables.

- [ ] **Step 3: Build a strength×PuLID-weight montage**

Build a `cn_strength × pulid_weight` montage PNG for one identity (id_14), first seed per cell — same form as the chibi `highstr_id14_grid.png`. Save to `exp_output/matryoshka_score/montage_id14.png`.

- [ ] **Step 4: Pick the winning cell**

From the scorer output + the montage, pick the cell that **maximises CLIP matryoshka-ness and recognizability while keeping ArcFace detection low** (committed to the doll read, not a photoreal face pasted on a doll). Record the winning `(cn_strength, pulid_weight, pulid_start)` and the reasoning in a dated research doc `docs/research/2026-05-16-matryoshka-v1-sweep.md`, and create the `docs/research/_topics/matryoshka-portrait.md` topic index.

- [ ] **Step 5: Commit**

```bash
git add docs/research/2026-05-16-matryoshka-v1-sweep.md docs/research/_topics/matryoshka-portrait.md
git commit -m "docs(matryoshka): v1 single-doll sweep results + winning cell"
```

---

## Self-Review

**Spec coverage:**
- Subject extraction → Task 2 (insightface ID dir, reused via `--id-dir`). ✓
- Structure (Canny doll template) → Task 1. ✓
- Render (Flux-Krea + PuLID + style) → Task 2 sweep. ✓
- Metric: recognizability → Task 6 eyeball + Task 3 gender/age proxy. ✓
- Metric: matryoshka-ness CLIP → Task 3. ✓
- Metric: uncanny ArcFace detection → Task 3. ✓
- Metric: MediaPipe floor → Task 3. ✓
- Spec Phase 0 baseline → Task 4. ✓
- Spec Phase 0.5 generator bake-off (5 arms) → Task 5. ✓
- Spec Phase 1 sweep → Tasks 2,3,6. ✓
- Phase 2 (set, LoRA training) explicitly deferred — Task 4 Step 3 records the LoRA trigger condition. ✓

**Placeholder scan:** no TBD/TODO; the one "placeholder" `LORA_CHIBI` constant is explicitly inert (`LORA_A_STRENGTH = 0.0`) and named as such. ✓

**Type consistency:** `build_grid`/`build_workflow` names match `chibi_highstr_sweep.py`; `make_clip`/`clip_matryoshka` signatures consistent between definition and use in Task 3. Grid cell keys (`cn_strength`, `pulid_weight`, `pulid_start`, `pulid_end`) consistent between Task 2 and the Task 3 scorer marginals. ✓
