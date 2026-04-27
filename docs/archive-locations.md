---
status: live
topic: archived-threads
---

# External archive — rebalance + attention pkls

## What lives where

Archive root (external USB drive, not always mounted):
`/media/newub/Seagate Hub/vamp-interface-archive/`

Layout mirrors the project tree. A file at
`vamp-interface-archive/crossdemo/anger/rebalance/asian_m/foo.pkl`
corresponds to the repo path
`output/demographic_pc/fluxspace_metrics/crossdemo/anger/rebalance/asian_m/foo.pkl`.

## What was archived (2026-04-22)

All `.pkl` files (per-render FluxSpace measurement caches, ~112 MB each) from:

| Source dir | N pkls | Size |
|---|---|---|
| `crossdemo/anger/rebalance/` | 300 | 33 GB |
| `crossdemo/surprise/rebalance/` | 300 | 33 GB |
| `crossdemo/disgust/rebalance/` | 300 | 33 GB |
| `crossdemo/pucker/rebalance/` | 300 | 33 GB |
| `crossdemo/lip_press/rebalance/` | 300 | 33 GB |
| `crossdemo/smile/smile_inphase/` | 330 | 38 GB |
| `crossdemo/smile/jaw_inphase/` | 330 | 38 GB |
| `crossdemo/smile/alpha_interp_attn/` | 660 | 76 GB |
| **total** | **2820** | **~315 GB** |

PNGs, `blendshapes.json`, and `measurement/` content stayed on local disk.

## Why they were safe to archive

All these `.pkl` sources had been compressed to fp16 via
`cache_attn_features.py` into `models/blendshape_nmf/attn_cache/{tag}/{delta_mix,attn_base}.npy + meta.json`
before archiving. The cache preserves `delta_mix.mean_d` and
`attn_base.mean_d` at every (step, block) — the only fields downstream
(`fit_nmf_directions_resid.py`, phase-3 direction-ridge, any future
direction-injection validation) reads. The raw pkls additionally hold
`rms_d`, `steered_at_scale`, and `ab_half_diff` per block, which no
current script consumes.

## Restoring

Re-mount the drive, then:

```bash
ARCHIVE="/media/newub/Seagate Hub/vamp-interface-archive"
REPO_CROSSDEMO="output/demographic_pc/fluxspace_metrics/crossdemo"
rsync -a --info=progress2 "$ARCHIVE/crossdemo/" "$REPO_CROSSDEMO/"
```

The layout mirror means this is a pure overlay — it won't overwrite
anything still on local disk.

## Reason for the archive

Disk was at 11 GB free / 100% used after the 2026-04-22 rebalance
render finished (per-pkl estimate was off by 15× — see
`memory/feedback_disk_preflight.md`). Moving the cached-and-redundant
raw pkls freed ~315 GB without deleting anything.

Useful lifecycle pattern for future bulk renders:
1. Render → raw pkls on local disk
2. Run `cache_attn_features.py` on the new source
3. Verify cache loads (smoke test downstream ridge)
4. Move raw pkls to archive
