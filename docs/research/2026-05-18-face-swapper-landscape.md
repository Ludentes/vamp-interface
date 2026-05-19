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

If the swap is ever revisited, the cheap experiment is a `cn_grid_sweep`-style
side-by-side of `inswapper_128` vs HyperSwap-256 vs ReSwapper-256 on the 20
importer identities — same SCRFD-mode + id_cos metrics.
