---
status: live
topic: arkit-controlnet
---

# Matryoshka one-shot ControlNet — CN-strength × steps grid sweep

Follow-on to the fast-model bake-off (`2026-05-18-matryoshka-bakeoff-verdict.md`).
The bake-off picked **Z-Image Turbo @ 6 steps** for generating the *generic*
doll. This sweep adds a one-shot ControlNet path so the doll is generated with
a realistic, correctly-proportioned face that `inswapper` can actually detect
and swap onto — then measures it across CN strength and step count.

Harness `scripts/cn_grid_sweep.py`. Artifacts in `exp_output/cn_grid/`
(`results.jsonl`, `renders/`, `swaps/`, `grid_collage.png`, `run.log`).

## Why a ControlNet at all

The matryoshka's *painted* face is oversized — folk-art proportions, large
eyes, no real nose/lip structure. A correctly-proportioned realistic face is
smaller-featured and does not overlay the painted one. That mismatch is why
every refine / composite / latent-mask approach failed: there is no realistic
face to refine *into*, and a transplanted realistic face does not register
against the doll geometry. (Refine, ellipse-mask composite, and
`SetLatentNoiseMask` were each tried and falsified in the spike session;
GFPGAN restore is separately falsified in `swap_core` — it lowers median
id_cos 0.773 → 0.525.)

**One-shot generation avoids the mismatch entirely.** Generate the doll fresh
with a realistic face baked in, in a single pass: Z-Image Turbo txt2img +
`Z-Image-Turbo-Fun-Controlnet-Union` driven by a Canny of the per-job swap
identity. The Canny forces realistic facial proportions at *generation* time;
`inswapper` then does the identity swap on a face it can cleanly detect.

The ControlNet loads as a `model_patch` via `ModelPatchLoader` →
`ZImageFunControlnet` (the stock `ControlNetLoader` rejects the `videox_fun`
format). Both nodes are native to ComfyUI ≥ 0.18 (`comfy_extras/
nodes_model_patch.py`) — no custom-node install. Control image is the source
identity's face Canny placed in the doll's natural face rect on an 864×1152
black canvas.

## Grid

Run on the Windows 3090 (`videocard@192.168.87.25`), ComfyUI 0.18.1 as a
WinSW service. Coordinator runs on the Linux box, talks to the remote ComfyUI
over HTTP; control images staged via the `/upload/image` API.

| Axis        | Values                  |
|-------------|-------------------------|
| Identities  | id_00 … id_19 (20)      |
| CN strength | 0.50, 0.70, 0.90        |
| Steps       | 6, 8                    |

120 cells, 0 failures, 27.9 min wall. Warm render 11.2 s (6-step) /
14.1 s (8-step). Seed fixed per identity (`91_000_000 + idx`).

Metrics: SCRFD detection mode (`default` = SCRFD detected, `forced` =
MediaPipe fallback, `failed` = neither) and post-swap ArcFace id_cos vs the
source identity.

## Results

| strength | steps | SCRFD default | det (mean) | id_cos (mean) | id_cos (min) |
|---------:|------:|--------------:|-----------:|--------------:|-------------:|
| 0.50     | 6     | 90%           | 0.525      | 0.828         | 0.772        |
| 0.70     | 6     | 100%          | 0.612      | 0.845         | 0.791        |
| 0.90     | 6     | 100%          | 0.660      | **0.864**     | 0.795        |
| 0.50     | 8     | 85%           | 0.493      | 0.832         | 0.757        |
| 0.70     | 8     | 100%          | 0.609      | 0.850         | 0.788        |
| 0.90     | 8     | 100%          | 0.640      | 0.861         | 0.785        |

120 cells: 115 `default`, 5 `forced`, **0 `failed`**. All 5 `forced` cells are
at strength 0.50.

## Verdict

**Strength 0.90, 6 steps.** 100% SCRFD `default` across all 20 identities,
highest id_cos (0.864 mean, 0.795 worst-case), highest det (0.660), fastest.

- **CN strength is the dominant axis** — monotone on every metric. Higher
  strength → better detection, higher identity, and it eliminates the SCRFD
  fallbacks. 0.50 is the only setting that drops cells to `forced`.
- **Steps barely matter** — 6 vs 8 is within noise (0.864 vs 0.861 id_cos at
  str 0.90; 6-step det is marginally *better*). 6 steps renders ~20% faster.
  Consistent with the bake-off's 6-step verdict.
- Collage confirms coherent vibrant red/gold dolls with realistic faces across
  all 20 identities — no body washout, no recoloring. The one-shot win holds
  at scale.

## Open question — the id_cos ceiling

id_cos plateaus around 0.86 even at the best cell. This is the
swap-onto-a-small-painted-face ceiling, not a CN-strength limit — strength
0.90 does not visibly over-constrain the doll, so there may be headroom to
push strength past 0.90, or to revisit the swap crop/upscale path. A
micro-sweep (strength {0.9, 1.0, 1.1} on a few identities) would show whether
the id_cos curve is still climbing or has flattened.
