# Phase 4 — Evaluation Harness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the new v8 pipeline preserves the qwen manifold structure better than the v6 (archetype blend) and v7 (pooled CLIP-L replace) baselines. Produce a report that either justifies scaling v8 to 2k+ or identifies what needs to iterate.

**Architecture:** Freeze a 50-job evaluation sample (30 cluster centres + 10 boundary + 10 sus-extremes). Generate the same 50 faces under each of three pipelines (v6/v7/v8). Extract FaceNet-512 embeddings from each face. Compute six metrics per pipeline (M1–M6 from spec §8). Print a comparison table and write a JSON report.

**Tech Stack:** Python 3.12 + uv, facenet-pytorch (for FaceNet), Pillow, ComfyUI, existing `generate_dataset` + `rbf_conditioning` + `generate_v8`.

**Working directory:** `/home/newub/w/telejobs/tools/face-pipeline`

**Prerequisite:** Phase 3 complete — `generate_v8.py` produces valid distinct faces in smoke test.

**Gate (v8 accepted if):**
- M3 (Pearson r vs qwen) improves by ≥ 0.10 over BOTH v6 and v7
- M4 (intra/inter cluster ratio) improves over both baselines
- M5 (boundary smoothness) ≥ 7/10
- M6 (sus separation) does not regress vs baselines

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/build_eval_sample.py` | Create | Deterministically pick 50 jobs, save to JSON |
| `src/eval_generate.py` | Create | Generate the 50-face batch under a given pipeline (v6/v7/v8) |
| `src/eval_facenet.py` | Create | Extract FaceNet embeddings from the generated PNGs |
| `src/eval_metrics.py` | Create | Compute M1–M6 from a face-embedding matrix + metadata |
| `src/eval_report.py` | Create | Run all three pipelines + metrics, print comparison, save JSON |
| `data/eval_sample.json` | Create | Frozen 50 jobs (written once, not regenerated after creation) |
| `output/eval_results/v{6,7,8}/` | Create | Generated faces + per-pipeline metrics JSON |
| `output/eval_results/comparison.json` | Create | Final three-way report |

Five small files instead of one big one — each has one job, easier to test and rerun independently.

---

## Task 1: Build and freeze the evaluation sample

**Files:**
- Create: `src/build_eval_sample.py`
- Create: `data/eval_sample.json`

- [ ] **Step 1: Create `src/build_eval_sample.py`**

```python
#!/usr/bin/env python3
"""
build_eval_sample.py — Build the frozen 50-job evaluation sample.

Selects:
  - 30 cluster-centre jobs (3 per cluster, for 10 largest clusters)
  - 10 boundary jobs (ambiguous between two nearest clusters in qwen space)
  - 10 sus-extremes (5 sus≤20 + 5 sus≥90)

Output: data/eval_sample.json  — frozen after first creation (do not regenerate).

Usage:
    uv run src/build_eval_sample.py
    uv run src/build_eval_sample.py --force       # overwrite existing sample
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import torch

DB_DSN         = "postgresql://USER:PASS@HOST:PORT/DB"
LAYOUT_PATH    = Path("/home/newub/w/vamp-interface/output/full_layout.parquet")
CENTROIDS_PATH = Path("data/cluster_centroids.pt")
OUT_PATH       = Path("data/eval_sample.json")

SAMPLE_SEED             = 42
CLUSTER_CENTRE_CLUSTERS = 10   # top N largest clusters
CLUSTER_CENTRE_PER      = 3    # jobs per cluster
BOUNDARY_COUNT          = 10
SUS_EXTREME_LOW_COUNT   = 5
SUS_EXTREME_HIGH_COUNT  = 5


def parse_embedding(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


def load_candidates() -> pd.DataFrame:
    """Load all jobs with cluster, embedding, sus_level."""
    df_layout = pd.read_parquet(LAYOUT_PATH, columns=["id", "cluster_coarse"])

    conn = psycopg2.connect(DB_DSN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id::text, embedding, sus_level FROM jobs "
            "WHERE embedding IS NOT NULL AND sus_level IS NOT NULL"
        )
        rows = list(cur.fetchall())
    conn.close()

    db_df = pd.DataFrame(rows)
    merged = df_layout.merge(db_df, on="id", how="inner")
    return merged


def pick_cluster_centres(
    candidates: pd.DataFrame,
    centroids: torch.Tensor,
    cluster_keys: list[str],
    rng: np.random.Generator,
) -> list[dict]:
    """For each of the top N largest clusters, pick the CLUSTER_CENTRE_PER closest to centroid."""
    counts = candidates["cluster_coarse"].value_counts()
    top_clusters = counts.head(CLUSTER_CENTRE_CLUSTERS).index.tolist()
    print(f"  Top {CLUSTER_CENTRE_CLUSTERS} clusters by size: {top_clusters}")

    selected: list[dict] = []
    for cluster_id in top_clusters:
        if cluster_id not in cluster_keys:
            print(f"    {cluster_id} not in centroids — skipping")
            continue
        centroid_idx = cluster_keys.index(cluster_id)
        centroid = centroids[centroid_idx].numpy()

        cluster_jobs = candidates[candidates["cluster_coarse"] == cluster_id].copy()
        distances = []
        for _, row in cluster_jobs.iterrows():
            emb = np.array(parse_embedding(row["embedding"]), dtype=np.float32)
            d = float(np.sum((emb - centroid) ** 2))
            distances.append(d)
        cluster_jobs = cluster_jobs.assign(_dist=distances).sort_values("_dist")

        for _, row in cluster_jobs.head(CLUSTER_CENTRE_PER).iterrows():
            selected.append({
                "id":             row["id"],
                "cluster_coarse": cluster_id,
                "sus_level":      int(row["sus_level"]),
                "category":       "cluster_centre",
                "embedding":      parse_embedding(row["embedding"]),
            })
    return selected


def pick_boundary_jobs(
    candidates: pd.DataFrame,
    centroids: torch.Tensor,
    cluster_keys: list[str],
    already_picked: set[str],
    rng: np.random.Generator,
) -> list[dict]:
    """Pick BOUNDARY_COUNT jobs whose two nearest centroids are roughly equidistant.

    Ratio = dist_to_nearest / dist_to_2nd_nearest. ~1.0 means on the boundary.
    We select from the ratio-closest-to-1.0 jobs, drawn at random from the top 200.
    """
    centroids_np = centroids.numpy()
    ratios: list[tuple[float, dict]] = []

    for _, row in candidates.iterrows():
        if row["id"] in already_picked:
            continue
        emb = np.array(parse_embedding(row["embedding"]), dtype=np.float32)
        dists = np.sum((centroids_np - emb) ** 2, axis=1)
        order = np.argsort(dists)
        d1, d2 = float(dists[order[0]]), float(dists[order[1]])
        if d2 == 0:
            continue
        ratio = d1 / d2  # ∈ (0, 1], higher = more boundary-like
        ratios.append((
            ratio,
            {
                "id":             row["id"],
                "cluster_coarse": row["cluster_coarse"],
                "sus_level":      int(row["sus_level"]),
                "category":       "boundary",
                "nearest_cluster":     cluster_keys[int(order[0])],
                "second_nearest_cluster": cluster_keys[int(order[1])],
                "boundary_ratio": round(ratio, 4),
                "embedding":      parse_embedding(row["embedding"]),
            }
        ))

    # Top 200 most boundary-like, then random sample of BOUNDARY_COUNT
    ratios.sort(key=lambda t: -t[0])
    pool = ratios[:200]
    idxs = rng.choice(len(pool), size=min(BOUNDARY_COUNT, len(pool)), replace=False)
    return [pool[int(i)][1] for i in idxs]


def pick_sus_extremes(
    candidates: pd.DataFrame,
    already_picked: set[str],
    rng: np.random.Generator,
) -> list[dict]:
    """Pick 5 sus≤20 and 5 sus≥90 jobs, deterministic random."""
    low = candidates[(candidates["sus_level"] <= 20) & (~candidates["id"].isin(already_picked))]
    high = candidates[(candidates["sus_level"] >= 90) & (~candidates["id"].isin(already_picked))]

    def sample(df: pd.DataFrame, n: int, category: str) -> list[dict]:
        if len(df) < n:
            raise SystemExit(f"Not enough {category} jobs: need {n}, have {len(df)}")
        idxs = rng.choice(len(df), size=n, replace=False)
        out: list[dict] = []
        for i in idxs:
            row = df.iloc[int(i)]
            out.append({
                "id":             row["id"],
                "cluster_coarse": row["cluster_coarse"],
                "sus_level":      int(row["sus_level"]),
                "category":       category,
                "embedding":      parse_embedding(row["embedding"]),
            })
        return out

    return sample(low,  SUS_EXTREME_LOW_COUNT,  "sus_low") + \
           sample(high, SUS_EXTREME_HIGH_COUNT, "sus_high")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Overwrite existing sample")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"ERROR: {OUT_PATH} already exists. Use --force to overwrite.")
        print("The evaluation sample is frozen after first creation (see spec §8).")
        sys.exit(1)

    print("Phase 4 — Build evaluation sample")
    print()

    rng = np.random.default_rng(SAMPLE_SEED)

    print("── Loading candidates ────────────────────────────")
    candidates = load_candidates()
    print(f"  {len(candidates)} candidates")

    print("  Loading centroids ...")
    cen_data = torch.load(CENTROIDS_PATH, weights_only=True)
    centroids: torch.Tensor = cen_data["centroids"]
    cluster_keys: list[str] = cen_data["cluster_keys"]

    print("\n── Picking cluster centres ───────────────────────")
    centres = pick_cluster_centres(candidates, centroids, cluster_keys, rng)
    print(f"  {len(centres)} cluster-centre jobs")

    already = {j["id"] for j in centres}

    print("\n── Picking boundary jobs ─────────────────────────")
    boundary = pick_boundary_jobs(candidates, centroids, cluster_keys, already, rng)
    print(f"  {len(boundary)} boundary jobs")
    already |= {j["id"] for j in boundary}

    print("\n── Picking sus extremes ──────────────────────────")
    extremes = pick_sus_extremes(candidates, already, rng)
    print(f"  {len(extremes)} sus-extreme jobs")

    sample = centres + boundary + extremes
    print(f"\nTotal sample: {len(sample)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "seed":            SAMPLE_SEED,
        "cluster_centres": len(centres),
        "boundary":        len(boundary),
        "sus_extremes":    len(extremes),
        "jobs":            sample,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Build the sample**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/build_eval_sample.py`

Expected: prints 30 cluster centres + 10 boundary + 10 sus extremes = 50 total. Saves `data/eval_sample.json`.

- [ ] **Step 3: Verify sample integrity**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && python3 -c "import json; d=json.load(open('data/eval_sample.json')); print('jobs:', len(d['jobs'])); print('categories:', set(j['category'] for j in d['jobs'])); print('unique ids:', len({j['id'] for j in d['jobs']}))"`

Expected: `jobs: 50`, categories include `cluster_centre`, `boundary`, `sus_low`, `sus_high`, unique ids = 50.

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/build_eval_sample.py data/eval_sample.json
git commit -m "$(cat <<'EOF'
feat(phase4): frozen 50-job evaluation sample

30 cluster centres (3 per top-10 cluster) + 10 boundary jobs (most
ambiguous between two nearest centroids) + 10 sus extremes (5 low, 5 high).
Deterministic with seed=42. Refuses to overwrite without --force.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Eval face generator

**Files:**
- Create: `src/eval_generate.py`

- [ ] **Step 1: Create the file**

```python
#!/usr/bin/env python3
"""
eval_generate.py — Generate the 50 evaluation faces under a single pipeline.

Given a pipeline name (v6/v7/v8), generates a face for each job in
data/eval_sample.json using that pipeline's conditioning logic, saves to
output/eval_results/<pipeline>/<job_id>.png.

Pipelines:
  v6: archetype ConditioningAverage (text RBF)        — conditioning_nodes()
  v7: direct CLIP-L pooled replace + per-job blend    — direct_conditioning_nodes()
  v8: cluster prototype T5 blend                      — cluster_prototype_conditioning_nodes()

All use Flux.1-dev img2img, same anchor, denoise=0.80, seed=hash(job_id).
LoRAs scale with sus_factor. Resumes from existing PNGs.

Usage:
    uv run src/eval_generate.py --pipeline v8
    uv run src/eval_generate.py --pipeline v6 --pipeline v7 --pipeline v8   # all three
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import (  # type: ignore[import-untyped]
    ComfyClient, flux_v6_workflow,
    FLUX_SAMPLER, FLUX_SCHEDULER,
    ANCHOR_PATH,
)
from rbf_conditioning import RBFConditioner  # type: ignore[import-untyped]

COMFY_URL       = "http://localhost:8188"
EVAL_SAMPLE     = Path("data/eval_sample.json")
OUT_BASE        = Path("output/eval_results")
DEV_CHECKPOINT  = "FLUX1/flux1-krea-dev_fp8_scaled.safetensors"
DEV_STEPS       = 20
DEV_GUIDANCE    = 3.5
DENOISE         = 0.80


def build_conditioning(pipeline: str, rbf: RBFConditioner, job: dict) -> tuple[dict, str]:
    """Dispatch to the right conditioning method based on pipeline name."""
    sus_factor = (job["sus_level"] / 100.0) ** 0.8

    if pipeline == "v6":
        return rbf.conditioning_nodes(
            embedding=job["embedding"],
            sus_factor=sus_factor,
            clip_ref=["3", 0],
            start_node_id=20,
        )
    if pipeline == "v7":
        return rbf.direct_conditioning_nodes(
            embedding=job["embedding"],
            sus_factor=sus_factor,
            clip_ref=["3", 0],
            start_node_id=20,
            job_id=job["id"],
            alpha=0.7,
        )
    if pipeline == "v8":
        return rbf.cluster_prototype_conditioning_nodes(
            embedding=job["embedding"],
            clip_ref=["3", 0],
            start_node_id=20,
        )
    raise ValueError(f"unknown pipeline: {pipeline}")


async def generate_for_pipeline(pipeline: str, jobs: list[dict], comfy: ComfyClient, anchor_name: str) -> None:
    out_dir = OUT_BASE / pipeline
    out_dir.mkdir(parents=True, exist_ok=True)

    rbf = RBFConditioner(temperature=1.5)
    if pipeline == "v8" and not rbf.cluster_prototypes:
        sys.exit("ERROR: v8 requires data/cluster_prototypes.json (run Phase 1 first)")

    done = {p.stem for p in out_dir.glob("*.png")}
    to_generate = [j for j in jobs if j["id"] not in done]
    print(f"[{pipeline}] {len(done)} already done, {len(to_generate)} to generate")

    for i, job in enumerate(to_generate):
        sus_factor = (job["sus_level"] / 100.0) ** 0.8
        seed = abs(hash(job["id"])) % (2 ** 32)
        cond_nodes, final_cond_id = build_conditioning(pipeline, rbf, job)

        workflow = flux_v6_workflow(
            checkpoint=DEV_CHECKPOINT,
            image_name=anchor_name,
            seed=seed,
            steps=DEV_STEPS,
            guidance=DEV_GUIDANCE,
            sampler=FLUX_SAMPLER,
            scheduler=FLUX_SCHEDULER,
            denoise=DENOISE,
            prefix=f"eval_{pipeline}_{job['category']}",
            conditioning_nodes=cond_nodes,
            final_cond_node_id=final_cond_id,
            lora_cursed=sus_factor * 1.0,
            lora_eerie=sus_factor * 0.75,
            gain=1.0,
        )

        dest = out_dir / f"{job['id']}.png"
        await comfy.generate(workflow, dest)
        print(f"  [{pipeline}][{i+1:2d}/{len(to_generate)}] {job['category']:14s} "
              f"{job['cluster_coarse']:5s} sus={job['sus_level']:3d} → {dest.name}")


async def main_async(pipelines: list[str]) -> None:
    if not EVAL_SAMPLE.exists():
        sys.exit(f"ERROR: {EVAL_SAMPLE} not found. Run build_eval_sample.py first.")
    data = json.loads(EVAL_SAMPLE.read_text(encoding="utf-8"))
    jobs: list[dict] = data["jobs"]
    print(f"Loaded {len(jobs)} eval jobs from {EVAL_SAMPLE}")

    async with ComfyClient(COMFY_URL) as comfy:
        anchor_name = await comfy.upload_image(ANCHOR_PATH)
        print(f"Anchor: {anchor_name}\n")
        for pipeline in pipelines:
            await generate_for_pipeline(pipeline, jobs, comfy, anchor_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="append", required=True,
                        choices=["v6", "v7", "v8"],
                        help="Pipeline(s) to run. Can specify multiple times.")
    args = parser.parse_args()
    asyncio.run(main_async(args.pipeline))
```

- [ ] **Step 2: Smoke test with v8 only (fastest)**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/eval_generate.py --pipeline v8`

Expected: generates 50 PNGs in `output/eval_results/v8/`. Takes ~17 min (50 × 20s).

- [ ] **Step 3: Run v6 and v7 baselines**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/eval_generate.py --pipeline v6 --pipeline v7`

Expected: generates 50 PNGs each in `output/eval_results/v6/` and `output/eval_results/v7/`. Takes ~35 min total.

- [ ] **Step 4: Verify all 150 faces exist**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && for p in v6 v7 v8; do echo -n "$p: "; ls output/eval_results/$p/*.png 2>/dev/null | wc -l; done`

Expected: each pipeline shows 50.

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/eval_generate.py
git commit -m "$(cat <<'EOF'
feat(phase4): eval face generator — runs 50 eval jobs under each pipeline

eval_generate.py dispatches to conditioning_nodes (v6), direct_conditioning_nodes (v7),
or cluster_prototype_conditioning_nodes (v8). Same anchor, denoise, seed, LoRAs —
only the conditioning subgraph differs. Resumes from existing PNGs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Do NOT commit the actual eval PNGs — they're regenerable. Just the script.

---

## Task 3: FaceNet embedding extraction

**Files:**
- Create: `src/eval_facenet.py`
- Modify: `pyproject.toml` (add `facenet-pytorch` if missing)

- [ ] **Step 1: Check if facenet-pytorch is installed**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run python -c "import facenet_pytorch; print(facenet_pytorch.__version__)"`

If printed, skip to step 3. If `ModuleNotFoundError`, continue.

- [ ] **Step 2: Add facenet-pytorch**

Run:
```bash
cd /home/newub/w/telejobs/tools/face-pipeline
uv add facenet-pytorch
```

Verify: `uv run python -c "from facenet_pytorch import InceptionResnetV1; print('ok')"` prints `ok`.

- [ ] **Step 3: Create `src/eval_facenet.py`**

```python
#!/usr/bin/env python3
"""
eval_facenet.py — Extract FaceNet-512 embeddings from generated eval faces.

For each pipeline in output/eval_results/{v6,v7,v8}/, loads all PNGs,
resizes to 160×160, runs InceptionResnetV1(classify=False) (vggface2),
saves an L2-normalized [50, 512] embedding matrix per pipeline.

Output: output/eval_results/<pipeline>/embeddings.npz
  arrays: ids (str), embeddings (float32, [N, 512])

Usage:
    uv run src/eval_facenet.py --pipeline v6 --pipeline v7 --pipeline v8
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from torchvision import transforms

OUT_BASE = Path("output/eval_results")


def load_model(device: str) -> InceptionResnetV1:
    print(f"  Loading InceptionResnetV1 (vggface2) on {device} ...")
    model = InceptionResnetV1(pretrained="vggface2", classify=False).eval()
    model = model.to(device)
    return model


def preprocess(img_path: Path) -> torch.Tensor:
    """Resize to 160×160, normalize to [-1, 1] per facenet-pytorch convention."""
    img = Image.open(img_path).convert("RGB").resize((160, 160), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    # FaceNet preprocessing: (x * 2) - 1 → [-1, 1]
    arr = (arr * 2.0) - 1.0
    # HWC → CHW
    return torch.from_numpy(arr.transpose(2, 0, 1))


def extract(pipeline: str, model: InceptionResnetV1, device: str) -> None:
    in_dir = OUT_BASE / pipeline
    pngs = sorted(in_dir.glob("*.png"))
    if not pngs:
        print(f"  [{pipeline}] no PNGs found in {in_dir}")
        return

    print(f"  [{pipeline}] processing {len(pngs)} faces ...")
    tensors = torch.stack([preprocess(p) for p in pngs]).to(device)
    with torch.no_grad():
        embs = model(tensors)
    embs = F.normalize(embs, dim=-1).cpu().numpy().astype(np.float32)

    ids = np.array([p.stem for p in pngs])
    out_path = in_dir / "embeddings.npz"
    np.savez(out_path, ids=ids, embeddings=embs)
    print(f"    saved {out_path}  shape={embs.shape}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="append", required=True,
                        choices=["v6", "v7", "v8"])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)

    for pipeline in args.pipeline:
        extract(pipeline, model, device)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run on all three pipelines**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/eval_facenet.py --pipeline v6 --pipeline v7 --pipeline v8`

Expected: downloads vggface2 weights on first run (~110 MB), processes 50 faces per pipeline, saves `embeddings.npz` in each pipeline directory.

- [ ] **Step 5: Verify shapes**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && python3 -c "
import numpy as np
for p in ['v6', 'v7', 'v8']:
    d = np.load(f'output/eval_results/{p}/embeddings.npz')
    print(p, d['ids'].shape, d['embeddings'].shape)
"`

Expected: each line shows `v? (50,) (50, 512)`.

- [ ] **Step 6: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add pyproject.toml uv.lock src/eval_facenet.py
git commit -m "$(cat <<'EOF'
feat(phase4): facenet embedding extraction for eval faces

eval_facenet.py loads InceptionResnetV1(vggface2), processes all PNGs
in output/eval_results/<pipeline>/, saves L2-normalized [N, 512] embeddings.

Adds facenet-pytorch dependency.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Metrics computation (M1–M6)

**Files:**
- Create: `src/eval_metrics.py`

- [ ] **Step 1: Create the file**

```python
#!/usr/bin/env python3
"""
eval_metrics.py — Compute M1-M6 metrics for a single pipeline's eval output.

M1. FaceNet cosine distance matrix (50×50)
M2. qwen cosine distance matrix (50×50)
M3. Pearson r between flattened M1 and M2
M4. Intra/inter cluster ratio (only cluster_centre category)
M5. Boundary smoothness pass rate (boundary category)
M6. Sus separation ratio (sus_extreme category)

Input: eval_sample.json (for job metadata + qwen embeddings)
       output/eval_results/<pipeline>/embeddings.npz (for face embeddings)

Output: output/eval_results/<pipeline>/metrics.json

Usage:
    uv run src/eval_metrics.py --pipeline v8
    uv run src/eval_metrics.py --pipeline v6 --pipeline v7 --pipeline v8
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

EVAL_SAMPLE = Path("data/eval_sample.json")
OUT_BASE    = Path("output/eval_results")


def cosine_distance_matrix(embs: np.ndarray) -> np.ndarray:
    """Pairwise cosine distances (1 - cos_sim) for unit-norm rows."""
    cos = embs @ embs.T
    return 1.0 - cos


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation between two 1D arrays."""
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt((a ** 2).sum() * (b ** 2).sum()))
    if denom == 0:
        return 0.0
    return float((a * b).sum() / denom)


def upper_tri_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    idx_i, idx_j = np.triu_indices(n, k=1)
    return mat[idx_i, idx_j]


def compute(pipeline: str) -> dict:
    sample = json.loads(EVAL_SAMPLE.read_text(encoding="utf-8"))
    jobs: list[dict] = sample["jobs"]
    job_by_id = {j["id"]: j for j in jobs}

    emb_path = OUT_BASE / pipeline / "embeddings.npz"
    if not emb_path.exists():
        sys.exit(f"ERROR: {emb_path} not found")
    data = np.load(emb_path)
    ids = [str(x) for x in data["ids"]]
    face_embs: np.ndarray = data["embeddings"]

    # Build matching qwen embedding matrix in same order
    qwen_embs = np.array(
        [job_by_id[i]["embedding"] for i in ids],
        dtype=np.float32,
    )
    # Normalize qwen embeddings for fair cosine comparison
    qwen_embs = qwen_embs / (np.linalg.norm(qwen_embs, axis=1, keepdims=True) + 1e-8)

    # M1, M2: distance matrices
    m1 = cosine_distance_matrix(face_embs)
    m2 = cosine_distance_matrix(qwen_embs)

    # M3: Pearson r between flattened upper triangles
    m3 = pearson_r(upper_tri_values(m1), upper_tri_values(m2))

    # M4: Intra/inter cluster ratio — only cluster_centre category
    centre_ids = [i for i in ids if job_by_id[i]["category"] == "cluster_centre"]
    centre_idx_by_cluster: dict[str, list[int]] = {}
    for local_idx, jid in enumerate(ids):
        if jid not in centre_ids:
            continue
        cluster = job_by_id[jid]["cluster_coarse"]
        centre_idx_by_cluster.setdefault(cluster, []).append(local_idx)

    intra_dists, inter_dists = [], []
    clusters = list(centre_idx_by_cluster.keys())
    for c1 in clusters:
        members = centre_idx_by_cluster[c1]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                intra_dists.append(float(m1[members[i], members[j]]))
        for c2 in clusters:
            if c1 >= c2:
                continue
            for mi in centre_idx_by_cluster[c1]:
                for mj in centre_idx_by_cluster[c2]:
                    inter_dists.append(float(m1[mi, mj]))

    intra_mean = float(np.mean(intra_dists)) if intra_dists else 0.0
    inter_mean = float(np.mean(inter_dists)) if inter_dists else 0.0
    m4_ratio = (inter_mean / intra_mean) if intra_mean > 0 else 0.0

    # M5: Boundary smoothness
    # For each boundary job B with nearest clusters A and C, check that
    # face_dist(B, A_centre) < face_dist(A_centre, C_centre) AND
    # face_dist(B, C_centre) < face_dist(A_centre, C_centre).
    # We use the first cluster_centre face from each cluster as the "centre face".
    cluster_centre_face_idx: dict[str, int] = {
        cluster: members[0] for cluster, members in centre_idx_by_cluster.items()
    }

    boundary_results = []
    for local_idx, jid in enumerate(ids):
        job = job_by_id[jid]
        if job["category"] != "boundary":
            continue
        a_cluster = job.get("nearest_cluster")
        c_cluster = job.get("second_nearest_cluster")
        if a_cluster not in cluster_centre_face_idx or c_cluster not in cluster_centre_face_idx:
            continue
        ai = cluster_centre_face_idx[a_cluster]
        ci = cluster_centre_face_idx[c_cluster]
        d_b_a = float(m1[local_idx, ai])
        d_b_c = float(m1[local_idx, ci])
        d_a_c = float(m1[ai, ci])
        passed = (d_b_a < d_a_c) and (d_b_c < d_a_c)
        boundary_results.append({
            "id": jid, "a": a_cluster, "c": c_cluster,
            "d_b_a": round(d_b_a, 4), "d_b_c": round(d_b_c, 4),
            "d_a_c": round(d_a_c, 4), "pass": passed,
        })

    m5_pass_count = sum(1 for r in boundary_results if r["pass"])
    m5_total      = len(boundary_results)
    m5_pass_rate  = m5_pass_count / m5_total if m5_total else 0.0

    # M6: Sus separation
    low_idxs  = [i for i, jid in enumerate(ids) if job_by_id[jid]["category"] == "sus_low"]
    high_idxs = [i for i, jid in enumerate(ids) if job_by_id[jid]["category"] == "sus_high"]

    low_vs_high = [float(m1[i, j]) for i in low_idxs for j in high_idxs]
    within_low  = [float(m1[low_idxs[i], low_idxs[j]])
                   for i in range(len(low_idxs)) for j in range(i + 1, len(low_idxs))]
    within_high = [float(m1[high_idxs[i], high_idxs[j]])
                   for i in range(len(high_idxs)) for j in range(i + 1, len(high_idxs))]

    mean_cross  = float(np.mean(low_vs_high)) if low_vs_high else 0.0
    mean_within = float(np.mean(within_low + within_high)) if (within_low or within_high) else 0.0
    m6_ratio    = (mean_cross / mean_within) if mean_within > 0 else 0.0

    return {
        "pipeline": pipeline,
        "M3_pearson_r_face_vs_qwen": round(m3, 4),
        "M4_intra_inter_ratio":      round(m4_ratio, 4),
        "M4_intra_mean":             round(intra_mean, 4),
        "M4_inter_mean":             round(inter_mean, 4),
        "M5_boundary_pass_count":    m5_pass_count,
        "M5_boundary_total":         m5_total,
        "M5_boundary_pass_rate":     round(m5_pass_rate, 4),
        "M6_sus_separation_ratio":   round(m6_ratio, 4),
        "M6_cross_band_mean":        round(mean_cross, 4),
        "M6_within_band_mean":       round(mean_within, 4),
        "boundary_details":          boundary_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline", action="append", required=True,
                        choices=["v6", "v7", "v8"])
    args = parser.parse_args()

    for pipeline in args.pipeline:
        print(f"\n── {pipeline} ──────────────────────────────")
        metrics = compute(pipeline)
        out = OUT_BASE / pipeline / "metrics.json"
        out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  M3 Pearson r vs qwen:  {metrics['M3_pearson_r_face_vs_qwen']:.4f}")
        print(f"  M4 intra/inter ratio:  {metrics['M4_intra_inter_ratio']:.4f}")
        print(f"  M5 boundary pass:      {metrics['M5_boundary_pass_count']}/{metrics['M5_boundary_total']}")
        print(f"  M6 sus separation:     {metrics['M6_sus_separation_ratio']:.4f}")
        print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on all three pipelines**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/eval_metrics.py --pipeline v6 --pipeline v7 --pipeline v8`

Expected: prints M3–M6 for each pipeline, saves `metrics.json` in each output dir.

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/eval_metrics.py output/eval_results/v6/metrics.json output/eval_results/v7/metrics.json output/eval_results/v8/metrics.json
git commit -m "$(cat <<'EOF'
feat(phase4): eval metrics M3-M6 for manifold preservation

M3: Pearson r between FaceNet and qwen distance matrices
M4: intra/inter cluster distance ratio (higher = more distinct)
M5: boundary smoothness pass rate (boundary face closer to both its
    nearest clusters than those clusters are to each other)
M6: sus separation ratio (cross-band / within-band distance)

Saves per-pipeline metrics.json under output/eval_results/<pipeline>/.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Comparison report

**Files:**
- Create: `src/eval_report.py`

- [ ] **Step 1: Create the file**

```python
#!/usr/bin/env python3
"""
eval_report.py — Build the three-way v6/v7/v8 comparison report.

Reads output/eval_results/{v6,v7,v8}/metrics.json, prints a comparison table,
applies the Phase 4 gate criteria, saves output/eval_results/comparison.json.

Gate (v8 accepted if):
  - M3 improves by ≥ 0.10 over BOTH v6 and v7
  - M4 improves over both baselines
  - M5 ≥ 7/10
  - M6 does not regress

Usage:
    uv run src/eval_report.py
"""
import json
import sys
from pathlib import Path

OUT_BASE = Path("output/eval_results")


def load_metrics(pipeline: str) -> dict:
    path = OUT_BASE / pipeline / "metrics.json"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found — run eval_metrics.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    m6 = load_metrics("v6")
    m7 = load_metrics("v7")
    m8 = load_metrics("v8")

    print()
    print("Manifold-Preservation Evaluation Report")
    print("=" * 70)
    print()
    print(f"{'Metric':<30s} {'v6':>10s} {'v7':>10s} {'v8':>10s} {'Δ vs best':>12s}")
    print("-" * 70)

    def row(label: str, key: str) -> None:
        v6_val = m6[key]
        v7_val = m7[key]
        v8_val = m8[key]
        best_baseline = max(v6_val, v7_val)
        delta = v8_val - best_baseline
        print(f"{label:<30s} {v6_val:>10.4f} {v7_val:>10.4f} {v8_val:>10.4f} {delta:>+12.4f}")

    row("M3 Pearson r vs qwen",      "M3_pearson_r_face_vs_qwen")
    row("M4 intra/inter ratio",      "M4_intra_inter_ratio")
    row("M5 boundary pass rate",     "M5_boundary_pass_rate")
    row("M6 sus separation ratio",   "M6_sus_separation_ratio")
    print()

    # Gate evaluation
    m3_delta_vs_best = m8["M3_pearson_r_face_vs_qwen"] - max(
        m6["M3_pearson_r_face_vs_qwen"], m7["M3_pearson_r_face_vs_qwen"],
    )
    m4_wins = m8["M4_intra_inter_ratio"] > max(m6["M4_intra_inter_ratio"], m7["M4_intra_inter_ratio"])
    m5_pass = m8["M5_boundary_pass_count"] >= 7
    m6_no_regress = m8["M6_sus_separation_ratio"] >= min(m6["M6_sus_separation_ratio"], m7["M6_sus_separation_ratio"])

    gates = [
        ("M3 +0.10 over best baseline",   m3_delta_vs_best >= 0.10, f"Δ = {m3_delta_vs_best:+.4f}"),
        ("M4 beats both baselines",       m4_wins,                   f"v8 = {m8['M4_intra_inter_ratio']:.4f}"),
        ("M5 ≥ 7/10 boundary",            m5_pass,                   f"{m8['M5_boundary_pass_count']}/10"),
        ("M6 no regression",              m6_no_regress,             f"v8 = {m8['M6_sus_separation_ratio']:.4f}"),
    ]

    print("Gate criteria:")
    all_pass = True
    for label, passed, detail in gates:
        marker = "✓" if passed else "✗"
        print(f"  {marker} {label:<40s} ({detail})")
        if not passed:
            all_pass = False
    print()

    if all_pass:
        print("  ✓ v8 ACCEPTED — scale to full dataset")
    else:
        print("  ✗ v8 REJECTED — iterate or redesign (see spec §8 for next steps)")

    # Save the comparison
    report = {
        "v6_metrics": m6,
        "v7_metrics": m7,
        "v8_metrics": m8,
        "gates": [
            {"label": label, "passed": passed, "detail": detail}
            for label, passed, detail in gates
        ],
        "verdict": "ACCEPTED" if all_pass else "REJECTED",
    }
    out_path = OUT_BASE / "comparison.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Saved: {out_path}")

    if not all_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the report**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/eval_report.py`

Expected: prints comparison table, per-gate pass/fail, and overall verdict. Saves `output/eval_results/comparison.json`. Exits 1 on rejection.

- [ ] **Step 3: Human spot-check the faces**

Open the 50 PNGs from each pipeline side by side:

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
ls output/eval_results/v6/*.png output/eval_results/v7/*.png output/eval_results/v8/*.png
```

Visually confirm: do the v8 faces look more distinct across clusters than v6/v7? Do boundary jobs look visually intermediate? Document observations in the commit message.

- [ ] **Step 4: Commit the report**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/eval_report.py output/eval_results/comparison.json
git commit -m "$(cat <<'EOF'
feat(phase4): three-way eval report with gate evaluation

eval_report.py loads per-pipeline metrics, builds the comparison table,
applies the Phase 4 gate criteria (M3 +0.10, M4 beats baselines, M5 ≥ 7/10,
M6 no regression), prints a verdict, saves comparison.json.

Human spot-check notes: [fill in observations from reviewing the 150 PNGs]

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Gate Decision

**ACCEPTED** (all 4 gates pass + human spot-check confirms visual distinctness):
- Remove the `--count` restriction and run `uv run src/generate_v8.py` for the full 2000-face batch
- Consider scaling further if budget allows (27k = 150h on Flux Dev)
- Update the Map UI to consume `output/dataset_faces_v8/` instead of previous dataset dirs
- Move on to the deferred sus axis recalibration (see spec §9)

**REJECTED** — diagnostic paths:

1. **M3 fails by small margin (<0.05 below target)**: try `--rbf-temperature 0.8` (sharper cluster assignment) and rerun eval_generate + eval_metrics + eval_report for v8 only.

2. **M3 fails badly** (v8 worse than v7 on Pearson): the T5 blending may not actually be interpolating. Inspect the ConditioningAverage chain in the workflow — ComfyUI may be collapsing to one dominant cluster rather than blending. Capture the actual workflow JSON submitted for one job and inspect `conditioning_to_strength` values.

3. **M4 fails** (clusters not distinct enough): prototypes may be too generic. Return to Phase 1, tune the PROTOTYPE_PROMPT to demand more discriminating traits, re-derive, rerun Phase 4.

4. **M5 fails** (boundaries not smooth): increase top-k from 3 to 5, or lower RBF temperature so more clusters contribute.

5. **M6 regresses**: sus is being suppressed by the anchor grounding. Try `--denoise 0.85` or `--denoise 0.90` and rerun eval for v8. If that fails, the sus axis may need its own fix before v8 can ship — defer to spec §9.

6. **All metrics fine but human spot-check reveals identical faces**: FaceNet may not capture the visual differences we care about. Escalate for design review on the metric — perhaps switch to CLIP-image embedding distances, or human ranking.

Document which path was taken in a follow-up commit.
