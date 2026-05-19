---
status: live
topic: arkit-controlnet
---

# Face-swapper landscape — newer alternatives to inswapper_128 / ReSwapper

The matryoshka pipeline's swap stage uses InsightFace `inswapper_128` (the
swap result currently "looks fine" — id_cos plateaus ~0.86, see
`2026-05-18-matryoshka-cn-grid-sweep.md`). This is an options survey, not a
verdict: what newer open-source swappers exist as of mid-2026, and which could
slot into `swap_core`'s InSwapper ONNX contract.

## Baseline facts

`inswapper_128` — ONNX, 128×128 swap region, strong ArcFace fidelity, trivial
integration, but **no redistribution license** (DeepInsight takes down HF
mirrors) and only 128px.

**`inswapper_256` / `inswapper_512` do not exist as legitimate weights.**
DeepInsight never released them; the upstream requests (insightface #2270,
#2373) were declined. `deepinsight/inswapper-512-live` is a closed demo, not a
download. Any community file with that name is mislabelled or fake — not an
option.

## Viable options

| Swapper | Year / state | Res | License | Weights | Integration |
|---|---|---|---|---|---|
| **HyperSwap** (FaceFusion Labs) | 2025, actively maintained | 256 | ResearchRAIL-MS (non-commercial research; commercial = contact FFLabs) | public ONNX, HF `facefusion/models-3.3.0` | **drop-in** — same ONNX API as inswapper, 3 variants `1a/1b/1c_256` |
| **ReSwapper** (somanchiu) | opset bump Nov 2024, minimally maintained (228★, AGPL-3.0) | 128 + **256** (512 is a TODO, unreleased) | AGPL-3.0 (clean, redistributable) | public, `.pth` + `.onnx`, HF | easy — ONNX or PyTorch, light deps |
| **REFace** (Sanoojan, WACV 2025 Oral) | checkpoint Sep 2024, 146★ | SD v1.4 backbone | MIT + OpenRAIL-M + CelebAMask-HQ **non-commercial** | public, ~1–2 GB w/ 5 dependency models | **heavy** — full SD env, multi-step DDIM |
| **DreamID-V** (ByteDance) | code+weights Jan 2026, actively maintained | 512, **video** DiT (Wan 1.3B) | **Apache-2.0** | genuinely released, ComfyUI nodes exist | heavy video DiT — architecturally mismatched to single-image swap |

Notes:

- **HyperSwap-256** is the strongest practical upgrade — 2× inswapper
  resolution, identical ONNX call shape (already supported by ReActor /
  ComfyUI), public weights, alive. Only blocker is the non-commercial research
  license. Quality is rated close to inswapper at 2× the resolution; FaceFusion
  docs still call `inswapper_128_fp16` highest-quality for *final* renders.
- **ReSwapper-256** is the clean-license fallback (AGPL-3.0, redistributable)
  and the only one besides HyperSwap that fits `swap_core` without a new
  loader. Fidelity is judged slightly below inswapper (visual A/B only, no
  published ArcFace numbers) — its value is the license + training code.
- **REFace** is the diffusion-grade-blending option if quality demands escalate
  and a full SD environment is acceptable. Non-commercial.
- **DreamID-V** is the only Apache-licensed diffusion swapper with real weights,
  but it is a *video* pipeline (Wan T2V backbone) — note it for an escalation
  path, do not pull it for a still-image stage.

## Dead / dormant / wrong-task — do not pursue

- **GHOST** (ai-forever/sber-swap) — abandoned, last release Jan 2022, GAN-era.
  Empirically falsified 2026-05-19 (0.536 id_cos — see bake-off below).
- **GHOST-2.0** — Feb 2025, Apache-2.0, but it is **head-swap** (transfers whole
  head incl. hair) with a DECA/EMOCA/BlazeFace dependency chain — wrong tool.
- **SimSwap** — 5.2k★ but dormant since 2021, CC-BY-NC-4.0. **SimSwap++** TPAMI
  2024 paper has **no code**.
- **DeepFaceLive** — GitHub-archived Nov 2024 (read-only). **DeepFaceLab** —
  per-identity training, not one-shot; wrong workflow.
- **DiffSwap** (CVPR 2023) — stale (20 open issues / 0 PRs), license-ambiguous;
  REFace supersedes it.
- **DreamID** (image version) — no code/weights released; repo is a project page
  pointing at ByteDance's commercial Dreamina platform.
- **DiffFace, HifiFace, e4s, MegaFS, BlendFace, Face Transformer** — research-
  grade GAN-era (2022–23), small/unmaintained, no production swapper. BlendFace
  is really an identity-encoder redesign for use *inside* other swappers.

## What fits our code without a loader rewrite

`swap_core` is built around the InSwapper ONNX contract (`get(img, face,
paste_back=True)` + ArcFace `normed_embedding`). **HyperSwap-256** and
**ReSwapper-256** are the only two that slot in with just a different `.onnx`
path. Everything else (REFace, DreamID-V, SimSwap, GHOST) needs a new adapter.

## Bake-off results (2026-05-18)

Ran the side-by-side. Harness `scripts/swapper_bakeoff.py`: the 20 best-cell
doll renders (CN grid strength 0.90 / 6 steps) as fixed targets, each importer
identity swapped onto its own doll with every backend. The detect / crop /
upscale / collapse / feathered paste-back path (`swap_core.swap_identity`) is
held constant — only the swapper object changes. Each backend is wrapped to the
InSwapper `get(img, target, source, paste_back=True)` signature; HyperSwap is a
faithful port of FaceFusion's hyperswap inference (arcface_128 warp template,
[-1,1] norm, L2-normed source embedding, model-emitted mask). CPU swap.
Artifacts in `exp_output/swapper_bakeoff/` (`results.jsonl`, `swaps/`,
`collage.png`).

| backend           | SCRFD default | id_cos mean | min   | max   | swap_s |
|-------------------|--------------:|------------:|------:|------:|-------:|
| **inswapper_128** | 100%          | **0.864**   | 0.795 | 0.919 | 1.53   |
| hyperswap_1c_256  | 100%          | 0.796       | 0.718 | 0.889 | 1.59   |
| hyperswap_1b_256  | 100%          | 0.790       | 0.704 | 0.878 | 1.15   |
| hyperswap_1a_256  | 100%          | 0.743       | 0.611 | 0.854 | 1.12   |

**Metric verdict (ArcFace id_cos): `inswapper_128`.** It wins identity — 0.864
vs 0.796 (1c) vs 0.790 (1b) vs 0.743 (1a), on every one of the 20 identities.
All four detect 100%.

- **HyperSwap quality tier (1a → 1b → 1c) does not buy *identity*.** 1c is the
  max-realism tier and is only +0.006 id_cos over 1b (within noise). The tier
  improves sharpness/realism, not ArcFace similarity — the skin-tone wash that
  caps HyperSwap on the metric is a model-family trait, not a tier knob. But
  1c's visual output *is* the sharpest of the family at 256px (2× inswapper).

- **ReSwapper-256 — falsified.** Loaded clean (emap present, INSwapper-contract,
  output well-aligned and coherent), but does not carry identity onto the small
  painted doll face: ArcFace cos ≈0.2 on the recognition model, ≈0.38 via the
  SCRFD-redetect path — vs 0.86 for inswapper. Tested all three source-latent
  conventions (emap, raw-normed, emapᵀ); emap is correct and still the worst-
  performing backend by far. Not an integration bug — the reimplementation is
  simply too weak here. Dropped from the run.
- **HyperSwap is coherent but washes identity.** 2× the resolution of
  inswapper, 100% detection, photoreal output — but it averages the source
  toward a smoother prior and notably loses skin tone (clear in `collage.png`:
  dark-skinned sources come back markedly lighter). That costs ~0.07 id_cos.
- **GHOST (sber-swap) — falsified (2026-05-19).** Tested after a survey of
  swapper *training* code (HyperSwap ships training code but no resumable
  checkpoint; GHOST-1 publishes generator+discriminator weights). Gated GHOST
  before any training analysis: ran the released `G_unet_2blocks` AEI_Net
  generator natively (vendored model code under torch 2.x, no GHOST env) on 5
  CN-grid dolls — `scripts/ghost_gate.py`, `exp_output/ghost_gate/`. Result:
  **0.536 mean id_cos** (0.40–0.68), vs 0.864 inswapper / 0.796 hyperswap_1c.
  Output is correctly aligned and coherent but soft, low-detail, and
  identity-weak — a 2022-era 224/256px GAN swapper — and washes skin tone
  *harder* than HyperSwap. Not an integration bug (recipe + collage audited).
  GHOST-2.0 not tested: it is head-swap, the wrong task. The training analysis
  is moot — no swapper-training route is worth pursuing.
- **The ~0.86 ceiling is target-side, confirmed.** A 2× higher-resolution
  swapper does *worse* on id_cos, not better. The limit is the small painted
  doll face as a swap target, not inswapper's 128px crop. Pushing the *metric*
  further means generation-time injection (PuLID / InfiniteYou), not a bigger
  swapper.

## Decision (2026-05-18): adopt HyperSwap 1c as the pipeline default

Despite the lower id_cos, **HyperSwap 1c is now the default swap backend**
(`swap_core.DEFAULT_SWAPPER`). The call: id_cos is a proxy, and on visual
inspection of the swaps 1c is the better result — sharper, 256px (2× inswapper),
more photoreal on the doll face. The ~0.07 id_cos gap reflects HyperSwap's
skin-tone averaging, a known and accepted trait, not a broken swap; every 1c
swap detects and reads as the right person. For a visualisation pipeline whose
output is *looked at*, render quality on the doll outweighs a proxy metric.

The swap op is CPU-only and not the pipeline bottleneck (generation is ~7 s on
GPU), so 1c's ~0.44 s/image over 1b costs nothing in wall time.

`inswapper_128` remains fully supported — `load_swapper()` dispatches on the
model filename, so passing an `inswapper_128.onnx` path still loads it (the
bake-off harness relies on this for A/B). 1b is the faster HyperSwap tier if
the ~0.44 s ever matters; on identity 1b ≈ 1c.

Wiring: `HyperSwap` class + `DEFAULT_SWAPPER` live in `scripts/swap_core.py`;
`load_swapper(model_path=None)` defaults to `hyperswap_1c_256.onnx` (staged at
`~/w/ComfyUI/models/insightface/` next to inswapper). `matryoshka_swap_sweep.py`
and `matryoshka_bakeoff_swap_test.py` default `--swapper` to it.
