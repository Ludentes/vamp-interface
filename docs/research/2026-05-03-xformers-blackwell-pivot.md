---
status: live
topic: neural-deformation-control
---

# xformers on Blackwell: source build is a no-op, use flash-attn instead

**Date:** 2026-05-03
**Context:** While optimizing PersonaLive for the RTX 5090 (sm_120), we ran the standard "source-build xformers with `TORCH_CUDA_ARCH_LIST=12.0`" recipe documented in late-2025 community guides. It compiled and installed cleanly but produced an xformers package with **no attention backends at all**. This doc captures why, so we (and the office Windows 5090 setup, and ComfyUI) don't repeat the wasted cycle.

## TL;DR

- xformers `main` (post-0.0.35) has **structurally removed its bundled CUDA fmha kernels.** Only `sparse24`, `indexing`, and `nvcc_info` `.cu` files remain in `xformers/csrc/`. There are 11 `.cu` files total in the source tree.
- The release notes for v0.0.35 (Feb 2025) say "Rely on upstream FA3" — they delegated all flash/cutlass attention to the standalone `flash-attn` package by Dao-AILab.
- A from-source build with `TORCH_CUDA_ARCH_LIST=12.0` produces a working `xformers` import with sparse24 ops on Blackwell, but **no `memory_efficient_attention_*` backends.**
- Late-2025 community guides recommending `pip install -v git+...xformers.git@main` for sm_120 are now misleading — they predate this restructuring.
- **Correct path for Blackwell attention acceleration: `pip install flash-attn` separately.** xformers will dispatch to it at call time if available.

## What we observed

After installing the published `xformers==0.0.35` wheel (cu128 channel) into PersonaLive's venv on the RTX 5090, `enable_xformers_memory_efficient_attention()` failed at probe time:

```
Failed to enable xformers: No operator found for `memory_efficient_attention_forward` with inputs:
     query/key/value : shape=(1, 2, 1, 40) (torch.float32)  # internal probe shape
`fa3F@0.0.0` not supported because: requires capability <= (9, 0) but GPU has (12, 0)
`fa2F@2.5.7-pt` not supported because: dtype=torch.float32 (only fp16/bf16)
`cutlassF-pt` not supported because: requires capability <= (9, 0) but GPU has (12, 0)
```

PersonaLive's try/except swallows this and falls through to PyTorch SDPA. SDPA on fp16 inputs gives 7.3 FPS for 100-frame demo on the 5090 (40% of the paper's 4090 claim of 15-20 FPS).

Two layers to unpack the error:

1. **`fa3F` and `cutlassF` are blocked by capability check, not absence of code.** The published wheels were built with `TORCH_CUDA_ARCH_LIST` capping at sm_90. The kernel sources exist (in the wheel) but weren't compiled for sm_120 in the wheel-build CI.
2. **`fa2F` is rejected only by dtype.** The probe at enable time uses fp32; fa2 only accepts fp16/bf16. So `fa2F` is *eligible* on sm_120 — it just fails the probe. At actual fp16 inference time it might be selected.

But this is moot for *building* a fix:

## What source-build actually produces

```
TORCH_CUDA_ARCH_LIST="12.0" MAX_JOBS=4 FORCE_CUDA=1 \
  uv pip install -v --no-build-isolation \
  "xformers @ git+https://github.com/facebookresearch/xformers.git@main"
```

Build completes in ~90 s (suspiciously fast). 13 ninja compile steps total. Resulting `xformers.info`:

```
xFormers 0.0.35+ca6d2aa.d20260503
indexing.scaled_index_addF: available
sp24.sparse24_*: available  (multiple)
is_triton_available: True
```

**No `memory_efficient_attention_*`, no `fa*`, no `cutlassF` — they don't exist in this build because they don't exist in the source tree.** Inventory of `xformers/csrc/`:

```
xformers/csrc/
├── attention/
│   ├── attention.cpp          # CPU stub only
│   ├── hip_decoder/           # AMD ROCm
│   └── hip_fmha/              # AMD ROCm
├── nvcc_info.cu
├── pt_stable_utils.cu
├── sparse24/                  # 9 .cu files for 2:4 sparsity
└── ...
```

**No CUDA fmha directory.** `third_party/` has only `cutlass` (header-only) and `composable_kernel_tiled` (AMD). No `flash-attention` submodule.

## What changed and when

xformers historically bundled three attention backends:

1. `cutlassF` — based on NVIDIA CUTLASS templates, xformers' own fork
2. `fa2F` — Flash-Attention 2 forward, vendored sources
3. `fa3F` — Flash-Attention 3 forward (Hopper), vendored sources

Sometime around xformers 0.0.34/0.0.35 (early 2025), the team migrated to:

- **PyTorch stable ABI only** (`-DTORCH_STABLE_ONLY` is unconditionally added to all CUDA builds — no env var to opt out per current `setup.py`).
- **External attention dependency** — `flash-attn` and the cutlass backend either get loaded from upstream packages or fall back to PyTorch SDPA.

The `xformers/ops/fmha/{flash,flash3,cutlass,cutlass_blackwell}.py` files still exist as **Python dispatchers** — they wrap calls to whichever attention package is present. If `flash-attn` isn't installed, `fa2F` reports "operator wasn't built" (which we saw above for `fa3F`).

## Implications

1. **The "source-build xformers for sm_120" path documented in late-2025 community guides is obsolete.** It worked when xformers still vendored its own kernels; it doesn't now.
2. **Our wasted compile cycle was educational but should have been a 5-min check first.** Before any source-build, run `find xformers/csrc -name "*.cu" | wc -l` on the checkout. If <50, the kernels aren't there.
3. **Correct Blackwell attention path:**
   - Install `flash-attn` separately. As of May 2026, `flash-attn` builds for Blackwell (sm_120) require either nightly PyTorch or recent stable + manual source build with `FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0`.
   - For fa3 specifically (Hopper-targeted), Blackwell support is in upstream `flash-attn` issue #1987 — track for native release.
   - Alternative: stick with PyTorch SDPA (built-in `torch.nn.functional.scaled_dot_product_attention`), which on torch 2.11+cu128 dispatches to FlashAttention-2 internally for fp16/bf16 inputs on Blackwell. May be the easiest path with minimal install surface.

## Same applies to ComfyUI

ComfyUI's xformers integration goes through the same dispatch path. On Blackwell:

- Don't bother source-building xformers.
- Either install `flash-attn` separately (and let xformers dispatch), or launch ComfyUI with `--use-pytorch-cross-attention` (uses SDPA directly, bypasses xformers entirely).
- The second option is what the upstream PyTorch issue #164342 thread recommends for Blackwell users.

## Action items

- [ ] Try `flash-attn` install in PersonaLive's venv. Pre-built wheel first; if absent, source build with `FLASH_ATTENTION_FORCE_BUILD=TRUE TORCH_CUDA_ARCH_LIST=12.0`.
- [ ] Re-run smoke test, compare to 7.3 FPS SDPA baseline.
- [ ] If flash-attn gives meaningful speedup, copy the wheel into ComfyUI's venv too.
- [ ] If it doesn't, the next step is the TensorRT path (PersonaLive ships TRT engines under `pretrained_weights/tensorrt/`). TRT is the path that almost certainly produced the paper's 15-20 FPS number.

## Reference timeline

| Date | Event |
|------|-------|
| Late 2024 | xformers ≤ 0.0.32 still vendors fmha CUDA kernels. Community guides recommend `TORCH_CUDA_ARCH_LIST=12.0` source-build for sm_120 |
| Feb 2025 | xformers 0.0.34: migrate to PyTorch stable ABI |
| Feb 2025 | xformers 0.0.35: "Rely on upstream FA3", attention kernels structurally removed from `csrc/` |
| 2025 | PRs #1254, #1262, #1285 add Blackwell capability flags to wheel-build CI — but only enable the *Python dispatchers*, not new kernel sources |
| May 2026 | We hit this on the 5090 |

## Sources

- [xformers PR #1254 — capability 120](https://github.com/facebookresearch/xformers/pull/1254)
- [xformers PR #1262 — Blackwell support](https://github.com/facebookresearch/xformers/pull/1262)
- [xformers PR #1285 — precompiled wheels](https://github.com/facebookresearch/xformers/pull/1285)
- [xformers issue #1356 — capability (12,0) too new](https://github.com/facebookresearch/xformers/issues/1356)
- [PyTorch issue #164342 — sm_120 in stable](https://github.com/pytorch/pytorch/issues/164342)
- [flash-attention issue #1987 — Blackwell support](https://github.com/Dao-AILab/flash-attention/issues/1987)
- xformers `setup.py` line 225 — unconditional `TORCH_STABLE_ONLY`
