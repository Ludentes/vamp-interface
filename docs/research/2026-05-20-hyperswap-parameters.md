---
status: live
topic: photobooth-sweep
---

# HyperSwap inference parameters — survey (2026-05-20)

## TL;DR

HyperSwap's ONNX has **exactly two runtime inputs**: `source` (1×512 ArcFace embedding) and `target` (1×3×256×256 image). There are no hidden alpha/weight/style scalar inputs. **All identity-strength control lives outside the model.** FaceFusion exposes three swap-stage knobs: `--face-swapper-model` (1a/1b/1c quality variants), `--face-swapper-pixel-boost` (tile-then-stitch supersampling), and `--face-swapper-weight` (linear interpolation between source and target ArcFace embeddings *before* the ONNX call). The weight knob is the only one we don't currently use, and it is exactly what we want: it lets us blend the identity embedding toward the target's own face. The mapping is `w ∈ [0,1] → α ∈ [+0.35, -0.35]` with default `0.5 → α=0` (pure source). Sign convention: the embedding becomes `(1-α)·source + α·target`, so **lower** CLI `w` blends the embedding toward the target matryoshka (weaker identity transfer), and **higher** CLI `w` extrapolates away from the target (amplified identity, more photoreal). For our over-photorealistic problem we want `w < 0.5` — sweep `{0.5, 0.4, 0.3, 0.2, 0.1, 0.0}`. No ONNX re-export needed — just a runtime numpy interp. No paper exists yet (HyperSwap is a 2025 FaceFusion Labs release with a model card / DeepWiki page, not an arXiv paper).

## Architecture (from paper / code)

No arXiv paper has been published; HyperSwap is documented only via the FaceFusion Labs repo and its DeepWiki mirror ([facefusion/facefusion-labs](https://deepwiki.com/facefusion/facefusion-labs/2-hyperswap-system)). What's confirmed:

- Generator-discriminator (GAN) family, **not** diffusion. Trained against an ArcFace identity loss + adversarial loss.
- Identity enters as a **single 512-D ArcFace embedding** fed to the generator. The architecture is consistent with INSwapper-family AdaIN-style modulation (vendor explicitly compares to inswapper); the exact injection (AdaIN vs FiLM vs cross-attn) is not documented publicly, but is irrelevant for our use case because *only* the embedding is exposed at the ONNX boundary.
- Trained at 256² with warp template `arcface_128` (note: 128-point arcface template upscaled to 256-crop, per `core.py: template='arcface_128', size=(256,256)`).
- Export: ONNX IR v10, opset 15.
- License: ResearchRAIL-MS (research/non-commercial; contact FF Labs for commercial).

## Parameters

### face_swapper_weight  (THE knob we missed)

**Found.** Source: [`facefusion/processors/modules/face_swapper/core.py:balance_source_embedding`](https://github.com/facefusion/facefusion/tree/master/facefusion/processors/modules/face_swapper).

```python
def balance_source_embedding(source_embedding, target_embedding):
    face_swapper_weight = state_manager.get_item('face_swapper_weight')
    face_swapper_weight = numpy.interp(face_swapper_weight, [0, 1], [0.35, -0.35]).astype(numpy.float32)
    # hyperswap branch: L2-normalize target_embedding
    target_embedding = target_embedding / numpy.linalg.norm(target_embedding)
    source_embedding = source_embedding * (1 - face_swapper_weight) + target_embedding * face_swapper_weight
    return source_embedding
```

- **CLI range:** 0.0 – 1.0 in 0.05 increments (default 0.5).
- **Effective α mapping** (formula: `embedding = source·(1-α) + target·α`):
  - `w=0.0 → α=+0.35`: `0.65·source + 0.35·target` → blend **toward** target = identity-weakened, matryoshka-deferred.
  - `w=0.5 → α=0`: pure source (no-op default; this is what we currently use).
  - `w=1.0 → α=-0.35`: `1.35·source − 0.35·target` → push **away** from target = amplified identity, more photoreal.
  - **Sign convention to remember:** lower CLI `w` weakens the swap (more matryoshka), higher CLI `w` strengthens it (more photoreal). This is the direction we want for our over-photorealistic problem: **sweep `w < 0.5`**.
- **Runtime input or re-export?** Runtime. The interpolation happens in numpy *before* the ONNX call. The ONNX itself never sees `face_swapper_weight`. We can replicate by computing `src = (1-α)·source_normed + α·target_normed; src /= ||src||` (re-normalize is safe — both inputs were unit-norm) and passing `src` into the existing `self.sess.run`.
- **Used by FF only when source-pure embedding is replaced by an averaged source.** Our case is single-source: the no-op comment in `swap_core.py:74` is correct *at default 0.5* but **misses that any other value is a non-trivial source↔target embedding mix** that we can dial.
- **The target embedding required for this is the ArcFace embedding of the target's own face** — InsightFace already gives this on `target_face.normed_embedding`; we'd need to pass it through.

### pixel boost

**Found.** [`--face-swapper-pixel-boost`](https://docs.facefusion.io/usage/cli-arguments/processors/face-swapper), default `128x128`, choices `{256,384,512,768,1024}²`.

Mechanism (`pixel_boost.py`): the crop is taken at `pixel_boost_size` (e.g. 768²), then **tiled into N×N 256² tiles** via `implode_pixel_boost`, each tile is independently swapped, then re-stitched via `explode_pixel_boost`. It's a supersampling trick — gives finer detail at higher cost; **does not change identity strength**. For HyperSwap 1c at 256 native, the relevant values are `256x256` (1 tile, baseline) up to `1024x1024` (16 tiles).

Useful for sharpness but orthogonal to the matryoshka problem.

### Mask / feather inside the model

The ONNX outputs a second tensor `mask` (1×1×256×256) — this is the **model's own** alpha mask. We already use it (`face_mask = mask[0,0]`). There are no feather/erode knobs *inside* the model; that's the entire built-in mask control.

FaceFusion's `face_mask_*` controls (`box`, `occlusion`, `area`, `region`, `face_mask_blur`, `face_mask_padding`) are all **post-swap** mask combiners applied to the paste-back blend — they don't influence what the swapper paints. We're already doing the equivalent in `swap_core.py:109-111` (single-mask blend, no box/occlusion/area/region).

### Quality variants (1a/1b/1c)

The three variants are the same architecture / same I/O signature ([core.py confirms identical `'type': 'hyperswap'`, identical `template`, `size`, `mean`, `std`](https://github.com/facefusion/facefusion/tree/master/facefusion/processors/modules/face_swapper)). Differences are training-budget / capacity ("1a for speed, 1c for quality" per [FF release notes](https://deepwiki.com/facefusion/facefusion-labs/2-hyperswap-system)). No 1d or higher variants exist as of FaceFusion 3.5.4 (May 2026 docs).

Swapping among them is a drop-in: same `source` / `target` shapes.

### ONNX input/output tensor list

Confirmed by direct `onnxruntime.InferenceSession.get_inputs()/get_outputs()` on `hyperswap_1c_256.onnx` (sha matches HF `facefusion/models-3.3.0`, 402 MB):

```
INPUTS:
  source   [1, 512]          float32   # L2-normalized ArcFace embedding
  target   [1, 3, 256, 256]  float32   # arcface_128-warped crop, [-1, 1] range, RGB, CHW
OUTPUTS:
  output   [1, 3, 256, 256]  float32   # swapped face, [-1, 1] range, RGB, CHW
  mask     [1, 1, 256, 256]  float32   # alpha mask, [0, 1]
```

No third "weight" / "alpha" / "style" input. No optional inputs. **The two tensors are the entire runtime contract.**

## What we cannot control (limits of the model)

Things that would require retraining or model replacement, not parameter tuning:

- **Stylization / non-photoreal output.** HyperSwap's training corpus is photoreal; its generator's prior collapses stylized targets toward photoreal skin/shading. No knob changes that.
- **Resolution above 256.** Output is locked to 256². Pixel boost is a multi-tile workaround, not a true higher-res path.
- **Identity-feature isolation.** Cannot isolate "keep eyes, swap nose" — embedding is monolithic.
- **Lighting / pose conditioning.** Both come implicitly from `target` image content; not steerable.
- **Expression preservation strength** — implicit in the generator's identity-vs-target trade-off, not a knob.

## Recommended next move

**Sweep `face_swapper_weight ∈ {0.5, 0.4, 0.3, 0.2, 0.1, 0.0}`** (note: descending — lower w = more matryoshka deference, which is what we want). Implementation: edit `HyperSwap.get()` in `scripts/swap_core.py` to (a) accept a `weight` arg, (b) take `target_face.normed_embedding` (already available from the detector — we currently throw it away), (c) interpolate per the formula above, (d) re-normalize. About 8 lines of code.

Expected behavior: as `w → 0.0`, the swap output should drift *toward* the matryoshka's own painted face — i.e. less crisp identity transfer, more deference to the target's painted features. This is the direction we want for the over-photorealistic / washed-matryoshka failure mode.

If `w=0.1–0.3` already plateaus before producing acceptable matryoshka deference, the embedding-mix range `[+0.35, -0.35]` is too narrow — we can break out of FaceFusion's clamped α range and use unclamped values (e.g. `α=0.5–0.8` for stronger toward-target mix). If even that plateaus the model itself is the ceiling. Fall-back options ranked: (i) try `simswap_512` (different training distribution, looser identity), (ii) try `ghost_2_256` (Apache-licensed, different prior), (iii) stop trying to make a photoreal swapper do stylized — re-enter the [face-swapper-landscape](2026-05-18-face-swapper-landscape.md) survey for stylized-tolerant alternatives.

Independent secondary lever once weight is dialed: **bypass the model's `mask` output and use a tighter, more eroded mask** so less of the swap pixels reach the matryoshka. That's a paste-back tweak, not an inference parameter, but it stacks with weight.

## Cross-references

- `[[reference-comfyui-shard-runbook]]`
- `2026-05-18-face-swapper-landscape.md`
- `2026-05-20-photobooth-phase2-findings.md`
- Source: [`facefusion/processors/modules/face_swapper/core.py`](https://github.com/facefusion/facefusion/tree/master/facefusion/processors/modules/face_swapper) (functions `balance_source_embedding`, `forward_swap_face`, `prepare_source_embedding`)
- Docs: [Face Swapper CLI args](https://docs.facefusion.io/usage/cli-arguments/processors/face-swapper)
- Model card / arch summary: [DeepWiki HyperSwap System](https://deepwiki.com/facefusion/facefusion-labs/2-hyperswap-system)
