# Importer corpus — operational notes

Off-piste paired (photoreal, stylized) generation corpus for the cases where
the FFHQ-StyleGAN2 importer bridge breaks: **chibi**, **anime-furry**, and
the **combination** that's maximally hard for LP / PersonaLive today.

This is the Track-2-direct path, not the bridge. Each (identity, style, seed)
tuple produces one image; pairing happens *across* the manifest by shared
`identity` and shared `seed`, so the photoreal partner of any styled image
is recoverable by `WHERE identity = X AND seed = Y AND style = "photoreal"`.

**Strategic posture:** 2–4 week drip-feed, not a one-shot spike. The corpus
grows nightly while the FFHQ-class bridge experiments run in parallel; when
the bridge hits its capability ceiling we already have off-piste training
data ready for LP-G LoRA fits.

## Layout

```
data/importer/
  identities/                   # 20 FFHQ portraits (stratified by gender/age/race)
    id_00.png ... id_19.png
    manifest.csv                # sha256 + demographics per ID
  cn_canny/                     # one Canny edge map per identity (pose-lock CN input)
    id_00_canny.png ...
  workflows/
    flux_pulid_canny_lora.api.json   # ComfyUI API-format template
  refs/                         # output PNGs, partitioned by style
    photoreal/id_NN_seedXXXX.png
    chibi/id_NN_seedXXXX.png
    furry/id_NN_seedXXXX.png
    chibi_furry/id_NN_seedXXXX.png
  manifest.parquet              # growing per-generation manifest (append-only)
  README.md                     # you are here
```

## Manifest schema (`manifest.parquet`)

| column | meaning |
|---|---|
| `ts_unix` | generation timestamp |
| `identity` | `id_NN` from the FFHQ pool |
| `style` | one of `photoreal | chibi | furry | chibi_furry` |
| `seed` | reproducibility seed (deterministic from identity+style+offset) |
| `out_path` | absolute or repo-relative path to the PNG |
| `lora_a_name`, `lora_a_strength` | first LoRA slot |
| `lora_b_name`, `lora_b_strength` | second LoRA slot |
| `pulid_weight` | PuLID injection weight |
| `cn_strength` | ControlNet strength |
| `duration_s` | wall-clock generation time |
| `workflow_version` | track schema changes per recipe revision |

## Pair recovery (post-generation)

Identity-paired tuples for LP-G LoRA training:

```python
import pandas as pd
m = pd.read_parquet("data/importer/manifest.parquet")
photoreal = m[m.style == "photoreal"][["identity", "seed", "out_path"]]
styled    = m[m.style == "chibi"]    [["identity", "seed", "out_path"]]
pairs = photoreal.merge(styled, on=["identity", "seed"], suffixes=("_photo", "_styled"))
```

`pairs` is the training corpus. Filter by ArcFace cosine identity check before
feeding into LP-G LoRA training — expect 30–50% retention on chibi, 10–20%
on furry, ~5% on `chibi_furry`. Low retention is *expected* and *informative*
for this off-piste corpus.

## Windows operator setup

### ComfyUI custom nodes required

Install via ComfyUI-Manager or by `git clone` into `ComfyUI/custom_nodes/`:

- **PuLID-Flux** — `balazik/ComfyUI-PuLID-Flux`. Provides `PulidFluxModelLoader`, `PulidFluxInsightFaceLoader`, `PulidFluxEvaClipLoader`, `ApplyPulidFlux`.
- **Flux ControlNet** — InstantX or XLabs Flux ControlNet nodes. Provides `ControlNetLoader` compatible with Flux + the Canny variant.
- Standard ComfyUI core (LoraLoaderModelOnly, UNETLoader, DualCLIPLoader, VAELoader, KSampler, etc.) is bundled.

### Models to drop in

| ComfyUI dir | File | Source |
|---|---|---|
| `models/unet/` | `flux1-dev.safetensors` | Black Forest Labs |
| `models/clip/` | `t5xxl_fp16.safetensors`, `clip_l.safetensors` | ComfyUI examples page |
| `models/vae/` | `ae.safetensors` | Black Forest Labs |
| `models/pulid/` | `pulid_flux_v0.9.1.safetensors` | huggingface.co/guozinan/PuLID |
| `models/insightface/` | antelopev2 model bundle | auto-downloaded by PuLID-Flux on first use |
| `models/controlnet/` | `flux-canny-controlnet-v3.safetensors` | InstantX or XLabs |
| `models/loras/` | `chibi_characters_flux_dev.safetensors` | from Civitai Downloads — rename to match |
| `models/loras/` | `anime_furry_style_flux.safetensors` | from Civitai Downloads — rename to match |

The two LoRA filenames must match the `lora_a_name` / `lora_b_name` entries in
`scripts/importer_run.py::STYLE_CONFIGS`. Rename either the files or those
config entries to align.

### Staging the identity + canny PNGs

ComfyUI's `LoadImage` node reads from `ComfyUI/input/`. Two options:

1. **Auto-stage** — pass `--comfy-input-dir C:\path\to\ComfyUI\input` to
   `scripts/importer_run.py` and the runner copies all PNGs in for you.
2. **Manual** — copy `data/importer/identities/*.png` and
   `data/importer/cn_canny/*.png` into `ComfyUI/input/` once before running.

### Running the night batch

Start ComfyUI normally (`python main.py`, default port 8188), then:

```powershell
cd <repo>
python scripts/importer_run.py `
    --comfy-url http://127.0.0.1:8188 `
    --comfy-input-dir "C:\path\to\ComfyUI\input" `
    --seeds-per-style 3 `
    --styles photoreal,chibi,furry,chibi_furry
```

20 IDs × 4 styles × 3 seeds = **240 generations** ≈ 2–4 hours on a 4090,
3–5 hours on a 3090, 5–8 hours on a 3080. Resumable — kill any time, re-run
to pick up.

### What to check after the first ~30 generations

- **PuLID identity preservation:** open 5 random (`photoreal`, `chibi`) pairs in
  a viewer side-by-side. Do they look like the same person? Expect "loosely
  the same person" — sharper preservation isn't the bar here, recognizable
  identity drift is.
- **CN pose lock:** photoreal partner should match the FFHQ identity portrait's
  head pose. If wildly off, raise `CN_STRENGTH` from 0.65 → 0.75.
- **LoRA strength sanity:** if chibi looks "barely chibi," raise
  `lora_a_strength` from 0.9 → 1.05. If "broken / artifacts," drop to 0.75.
- **Duration per generation:** should be 30–60s on a 4090. >2 min/gen means
  PuLID + CN are fighting; consider dropping PuLID to a CUDA provider check
  or shrinking resolution to 768² for the first night.

### Nightly cadence (recommendation)

Run one batch per night, vary `--seed-base` per night so seeds don't collide:

```
night 1: --seed-base 20260512   (initial 240 generations)
night 2: --seed-base 20260513   (next 240, accumulates to ~480 total)
night 3: --seed-base 20260514   (~720)
...
```

By night 7 the manifest holds ~1700 generations, of which ~25–40% will
survive the post-hoc ArcFace identity filter — enough to spike a per-style
LP-G LoRA fit and see whether the Track-2-direct path produces a working
stylized renderer on these out-of-bridge cases.

### Pulling the corpus back for training

The Windows machine writes to `data/importer/refs/` and `data/importer/manifest.parquet`
on local disk. To consolidate onto the Linux training machine:

```bash
# from Linux side
rsync -avh windows:/path/to/repo/data/importer/refs/ \
            /home/newub/w/vamp-interface/data/importer/refs/
rsync -avh windows:/path/to/repo/data/importer/manifest.parquet \
            /home/newub/w/vamp-interface/data/importer/manifest.parquet
```

Or via shared drive / SMB / git-lfs (large files only — manifest itself is fine in git).
