# Runbook: Embedding Visualization (Embedding Atlas)

Interactive 2D exploration of the job posting corpus via PaCMAP layout + Embedding Atlas.

---

## Quick start

```bash
# View the pre-built full corpus layout (25k jobs)
uv run embedding-atlas output/full_layout.parquet \
  --x x --y y --text text \
  --disable-projection \
  --host 0.0.0.0 --port 8765

# Open in browser
open http://localhost:8765
```

Requires: DB not needed (parquet is pre-built). ComfyUI not needed.

---

## Rebuild layout from DB

Run when new jobs have been embedded or you want fresh coordinates.

```bash
# Requires DB running (see runbooks/telejobs-db.md)
uv run src/build_full_layout.py

# Then launch
uv run embedding-atlas output/full_layout.parquet \
  --x x --y y --text text \
  --disable-projection \
  --host 0.0.0.0 --port 8765
```

Options:

```bash
# Pre-cutoff jobs included
uv run src/build_full_layout.py --all

# Tighter local structure (more neighbours = slower, better global)
uv run src/build_full_layout.py --n-neighbors 25

# Custom output path
uv run src/build_full_layout.py --out output/layout_experiment.parquet
```

---

## What you can do in the UI

| Goal | How |
|------|-----|
| See fraud geography | Color by `sus_level` |
| Find coordinated channels | Color by `source_name` |
| Validate work-type clusters | Color by `work_type` |
| Find contact-linked jobs | Color by `contact_telegram` |
| Read a posting | Click any point → tooltip shows `text` |
| Find dense fraud regions | Enable density contours |
| Find nearest neighbours | Click point → "Find similar" |

---

## Small dataset (543-job test set with face paths)

```bash
# Build layout for test dataset only
uv run src/build_pacmap_layout.py --face-version flux_v3

# View it
uv run embedding-atlas output/pacmap_layout.json ... # JSON not supported by CLI
# Convert first:
python3 -c "
import json, pandas as pd
d = json.load(open('output/pacmap_layout.json'))
df = pd.DataFrame([{k:v for k,v in p.items() if k not in ('sus_factors','face_path')} for p in d['points']])
df.to_parquet('output/test_layout.parquet', index=False)
"
uv run embedding-atlas output/test_layout.parquet \
  --x x --y y --text text \
  --disable-projection \
  --host 0.0.0.0 --port 8765
```

---

## Layout algorithm

**PaCMAP** (not UMAP). Reasons:
- Preserves both local and global structure (UMAP only local)
- Deterministic: same seed → same layout every run
- FAISS KNN backend, fast at 25k points (~2 min on CPU)

Parameters used: `n_neighbors=15`, `seed=42`, `apply_pca=True` (auto, reduces to 100-d first).

Position reflects **semantic similarity of raw job text only**. Sus level, fraud label, source channel — none of these influence the layout. If fraud clusters appear spatially, it means the fraud language is genuinely encoded in text similarity.

---

## Files

| File | What |
|------|------|
| `src/build_full_layout.py` | Fetch from DB → PaCMAP → Parquet |
| `src/build_pacmap_layout.py` | Test dataset (543 jobs) → PaCMAP → JSON |
| `output/full_layout.parquet` | Pre-built full corpus layout (gitignored) |
| `output/pacmap_layout.json` | Pre-built test layout (gitignored) |
