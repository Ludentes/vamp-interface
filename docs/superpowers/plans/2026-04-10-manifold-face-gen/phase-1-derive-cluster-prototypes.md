# Phase 1 — Derive Cluster Prototypes

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For each of 42 qwen clusters, derive ONE face description from gemma4 grounded in up to 20 central translated job examples from that cluster. Persist as `data/cluster_prototypes.json`.

**Architecture:** For each cluster, (1) load its central jobs by minimising L2 distance to the cluster centroid in qwen space, (2) build a numbered list of up to 20 translated job texts, (3) call gemma4 with a prompt asking for a single face-description sentence that captures the typical worker, (4) store `{cluster_id: description}`. Validate output by sentence-encoding all 42 and checking they don't collapse to a single point.

**Tech Stack:** Python 3.12 + uv, psycopg2 for DB, pandas + torch for parquet/centroids, requests for Ollama, sentence-transformers for validation.

**Working directory:** `/home/newub/w/telejobs/tools/face-pipeline`

**Gate (PASS required to proceed to Phase 3):**
- `data/cluster_prototypes.json` contains exactly 42 entries (all of `C0`–`C40` + `noise`)
- Every entry is a non-empty string ≥ 30 characters
- Pairwise cosine between prototypes shows real variation (std > 0.05) — not all identical

---

## File Structure

| File | Responsibility |
|---|---|
| `src/derive_cluster_prototypes.py` | End-to-end script: load centroids → find central jobs per cluster → gemma4 → JSON |
| `data/cluster_prototypes.json` | 42 entries `{cluster_id: face_description}` |

---

## Task 1: Set up the script skeleton

**Files:**
- Create: `src/derive_cluster_prototypes.py`

- [ ] **Step 1: Create file with imports, constants, and empty main()**

```python
#!/usr/bin/env python3
"""
derive_cluster_prototypes.py — Phase 1: generate one face description per qwen cluster.

For each of 42 coarse clusters (C0–C40 + noise), picks up to 20 central translated
jobs from that cluster and asks gemma4 to write a single face-description sentence
that captures the typical worker for the cluster.

Output: data/cluster_prototypes.json
  {
    "C0":    "A young man in his twenties, slim build, ...",
    "C1":    "A middle-aged man in his forties, ...",
    ...
    "noise": "An unremarkable person of indeterminate occupation, ..."
  }

Usage:
    uv run src/derive_cluster_prototypes.py
    uv run src/derive_cluster_prototypes.py --limit-clusters 5   # smoke test
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

DB_DSN        = "postgresql://USER:PASS@HOST:PORT/DB"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "gemma4:latest"
LAYOUT_PATH   = Path("/home/newub/w/vamp-interface/output/full_layout.parquet")
TRANSLATIONS  = Path("data/job_translations.parquet")
CENTROIDS     = Path("data/cluster_centroids.pt")
OUT_PATH      = Path("data/cluster_prototypes.json")

MAX_EXAMPLES_PER_CLUSTER = 20
MAX_EXAMPLE_CHARS        = 400

PROTOTYPE_PROMPT = """\
Below are {n} job postings that all belong to the same group. Based on these examples,
write ONE sentence describing the face of a typical worker for this group.

Focus on physical appearance that would plausibly fit most of these postings:
  - age range
  - build
  - gender presentation
  - demeanour
  - clothing style
  - facial expression

Output ONLY the single sentence, no preamble, no explanations, no markdown.

Job postings:
{examples}

Face description (single sentence):"""


def main() -> None:
    print("Phase 1 — Derive cluster prototypes")
    # filled in next tasks


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run to verify the file loads cleanly**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_cluster_prototypes.py`

Expected output: `Phase 1 — Derive cluster prototypes`

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/derive_cluster_prototypes.py
git commit -m "$(cat <<'EOF'
chore(phase1): scaffold cluster prototype derivation

Imports, constants, PROTOTYPE_PROMPT template, and empty main() for
the per-cluster gemma4 face description generation.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Load centroids, layout, and translations

**Files:**
- Modify: `src/derive_cluster_prototypes.py`

- [ ] **Step 1: Add `load_data()` function above `main()`**

```python
def load_data() -> tuple[torch.Tensor, list[str], pd.DataFrame]:
    """Load cluster centroids, cluster assignments + translations.

    Returns:
        centroids:    [42, 1024] tensor of cluster centroid embeddings (qwen)
        cluster_keys: list of 42 cluster IDs in centroid order (e.g. ["C0", "C1", ..., "noise"])
        jobs_df:      DataFrame with columns id, cluster_coarse, text_en, embedding
    """
    print("  Loading cluster centroids ...")
    cen_data = torch.load(CENTROIDS, weights_only=True)
    centroids: torch.Tensor = cen_data["centroids"]
    cluster_keys: list[str] = cen_data["cluster_keys"]
    print(f"    {len(cluster_keys)} clusters, centroid shape {tuple(centroids.shape)}")

    print("  Loading layout ...")
    df_layout = pd.read_parquet(LAYOUT_PATH, columns=["id", "cluster_coarse"])

    print("  Loading translations ...")
    df_trans = pd.read_parquet(TRANSLATIONS, columns=["job_id", "text_en"])
    df_trans = df_trans.rename(columns={"job_id": "id"})

    # Load embeddings from DB (only jobs present in both layout and translations)
    import psycopg2  # noqa: PLC0415
    import psycopg2.extras  # noqa: PLC0415
    print("  Fetching embeddings from DB ...")
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id::text, embedding FROM jobs WHERE embedding IS NOT NULL")
        emb_rows = {r["id"]: r["embedding"] for r in cur.fetchall()}
    conn.close()
    print(f"    {len(emb_rows)} embeddings")

    # Join and filter
    merged = df_layout.merge(df_trans, on="id", how="inner")
    merged["embedding"] = merged["id"].map(emb_rows)
    merged = merged[merged["embedding"].notna()]
    merged = merged[merged["text_en"].str.len() > 30]
    print(f"    {len(merged)} jobs with cluster + translation + embedding")

    return centroids, cluster_keys, merged
```

- [ ] **Step 2: Call it from `main()` to verify**

Replace `main()` body:

```python
def main() -> None:
    print("Phase 1 — Derive cluster prototypes")
    print()
    print("── Loading data ──────────────────────────────────")
    centroids, cluster_keys, jobs_df = load_data()
    print(f"\n  Clusters: {cluster_keys[:5]}... ({len(cluster_keys)} total)")
    print(f"  Jobs per cluster (first 5):")
    counts = jobs_df["cluster_coarse"].value_counts()
    for key in cluster_keys[:5]:
        print(f"    {key}: {counts.get(key, 0)}")
```

- [ ] **Step 3: Run and verify counts**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_cluster_prototypes.py`

Expected: prints 42 cluster keys, shows non-zero job counts for the first 5 clusters (each should have at least 20 except possibly very small ones — that's OK, we handle it later).

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/derive_cluster_prototypes.py
git commit -m "$(cat <<'EOF'
feat(phase1): load centroids, layout, translations, embeddings

load_data() joins cluster_centroids.pt, full_layout.parquet,
job_translations.parquet, and DB embeddings on job_id. Drops jobs
with missing translation/embedding or very short text.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Find central jobs per cluster

**Files:**
- Modify: `src/derive_cluster_prototypes.py`

- [ ] **Step 1: Add `parse_embedding()` helper and `central_jobs_for_cluster()` above `main()`**

```python
def parse_embedding(raw) -> list[float]:
    """Convert pgvector string or list into Python list[float]."""
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


def central_jobs_for_cluster(
    cluster_id: str,
    centroid: torch.Tensor,     # [1024]
    jobs_df: pd.DataFrame,
    max_examples: int = MAX_EXAMPLES_PER_CLUSTER,
) -> list[str]:
    """Return up to max_examples translated job texts from this cluster, ordered by
    distance to centroid (closest first). Empty list if cluster has no jobs."""
    cluster_jobs = jobs_df[jobs_df["cluster_coarse"] == cluster_id]
    if len(cluster_jobs) == 0:
        return []

    # Compute squared L2 distance from each job embedding to the centroid
    centroid_np = centroid.numpy()
    distances: list[tuple[float, str]] = []
    for _, row in cluster_jobs.iterrows():
        emb = np.array(parse_embedding(row["embedding"]), dtype=np.float32)
        d = float(np.sum((emb - centroid_np) ** 2))
        distances.append((d, row["text_en"]))

    distances.sort(key=lambda t: t[0])
    return [text for _, text in distances[:max_examples]]
```

- [ ] **Step 2: Extend `main()` to call this for the first 3 clusters as a sanity check**

Append to `main()`:

```python
    print("\n── Sampling central jobs (first 3 clusters) ─────")
    for i, cluster_id in enumerate(cluster_keys[:3]):
        central = central_jobs_for_cluster(cluster_id, centroids[i], jobs_df)
        print(f"  {cluster_id}: {len(central)} central jobs found")
        if central:
            preview = central[0][:80].replace("\n", " ")
            print(f"    first:  {preview!r}")
```

- [ ] **Step 3: Run to verify centrality logic**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_cluster_prototypes.py`

Expected: each of the first 3 clusters prints a count (often 20, sometimes less for small clusters) and a text preview. The previews should look like actual translated job postings.

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/derive_cluster_prototypes.py
git commit -m "$(cat <<'EOF'
feat(phase1): find central jobs per cluster by L2 distance

central_jobs_for_cluster() computes squared L2 distance from each job's
qwen embedding to the cluster centroid and returns the top-max_examples
texts ordered by distance ascending.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Generate one prototype per cluster via gemma4

**Files:**
- Modify: `src/derive_cluster_prototypes.py`

- [ ] **Step 1: Add `generate_prototype()` above `main()`**

```python
def generate_prototype(examples: list[str], session: requests.Session) -> str:
    """Call gemma4 with numbered examples, return single-sentence face description."""
    if not examples:
        return ""

    numbered = "\n".join(f"{i+1}. {e[:MAX_EXAMPLE_CHARS]}" for i, e in enumerate(examples))
    prompt = PROTOTYPE_PROMPT.format(n=len(examples), examples=numbered)

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 150,
            "think": False,
        },
    }
    resp = session.post(OLLAMA_URL, json=payload, timeout=180)
    resp.raise_for_status()
    out = resp.json()["response"].strip()
    return out.split("\n")[0].strip()
```

- [ ] **Step 2: Add `--limit-clusters` arg and main loop**

Replace `main()` entirely:

```python
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-clusters", type=int, default=None,
                        help="Only process the first N clusters (smoke test)")
    args = parser.parse_args()

    print("Phase 1 — Derive cluster prototypes")
    print()
    print("── Loading data ──────────────────────────────────")
    centroids, cluster_keys, jobs_df = load_data()

    if args.limit_clusters:
        cluster_keys = cluster_keys[:args.limit_clusters]
        centroids    = centroids[:args.limit_clusters]
        print(f"\n  Limited to first {len(cluster_keys)} clusters (smoke test)")

    print(f"\n── Generating prototypes via gemma4 ─────────────")
    session = requests.Session()
    prototypes: dict[str, str] = {}

    FALLBACK = (
        "An ordinary person of average build and unremarkable features, "
        "casual clothing, neutral expression, indeterminate demeanour."
    )

    for i, cluster_id in enumerate(tqdm(cluster_keys, desc="Clusters")):
        examples = central_jobs_for_cluster(cluster_id, centroids[i], jobs_df)
        if not examples:
            tqdm.write(f"  {cluster_id}: no translated jobs — using fallback")
            prototypes[cluster_id] = FALLBACK
            continue

        try:
            desc = generate_prototype(examples, session)
        except Exception as e:
            tqdm.write(f"  {cluster_id}: gemma4 error ({e}) — using fallback")
            prototypes[cluster_id] = FALLBACK
            continue

        if len(desc) < 30:
            tqdm.write(f"  {cluster_id}: output too short ({len(desc)} chars) — using fallback")
            prototypes[cluster_id] = FALLBACK
            continue

        prototypes[cluster_id] = desc
        tqdm.write(f"  {cluster_id}: {desc[:80]}")

    print(f"\n  Generated {len(prototypes)} prototypes")
    print()

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(prototypes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved: {OUT_PATH}")
```

- [ ] **Step 3: Smoke test on 3 clusters**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_cluster_prototypes.py --limit-clusters 3`

Expected: 3 prototype sentences printed, `data/cluster_prototypes.json` saved with 3 entries. Each sentence should look like an actual face description. Takes ~1 min.

- [ ] **Step 4: Run full batch (all 42 clusters)**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_cluster_prototypes.py`

Expected: progress bar through 42 clusters, ~5-10 min total. `data/cluster_prototypes.json` has 42 entries. Fallback is used for clusters with no translated jobs or gemma4 errors.

- [ ] **Step 5: Inspect the output**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && python3 -c "import json; p=json.load(open('data/cluster_prototypes.json')); print(f'{len(p)} entries'); [print(f'{k}: {v[:90]}') for k,v in p.items()]"`

Expected: 42 entries printed, each a plausible face description. If any entry is the fallback sentence, note which cluster and why (look for warnings in the earlier run log).

- [ ] **Step 6: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/derive_cluster_prototypes.py data/cluster_prototypes.json
git commit -m "$(cat <<'EOF'
feat(phase1): derive per-cluster face prototypes via gemma4

For each of 42 coarse clusters, picks up to 20 central translated jobs
(by L2 distance in qwen space), calls gemma4 to produce a single
face-description sentence, writes data/cluster_prototypes.json.

Fallback sentence is used when the cluster has no translated jobs,
gemma4 errors, or output is too short (<30 chars).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Validate that prototypes are actually distinct

**Files:**
- Create: `src/validate_cluster_prototypes.py`

Not modifying the derivation script — the validator is a separate one-shot check. Smaller focused files are easier to maintain.

- [ ] **Step 1: Create the validator**

```python
#!/usr/bin/env python3
"""
validate_cluster_prototypes.py — Verify that cluster prototypes are actually distinct.

Encodes all 42 prototype sentences through sentence-transformers/all-MiniLM-L6-v2
and reports pairwise cosine statistics. Gates Phase 3.

Gate criteria (ALL must pass):
  - 42 non-empty entries
  - Each entry >= 30 chars
  - Pairwise cosine std > 0.05 (prototypes are not all identical)
  - Max pairwise cosine < 0.99 (no two prototypes are the same string)

Usage:
    uv run src/validate_cluster_prototypes.py
"""
import json
import sys
from pathlib import Path

import numpy as np

PROTOTYPES = Path("data/cluster_prototypes.json")


def main() -> None:
    if not PROTOTYPES.exists():
        sys.exit(f"ERROR: {PROTOTYPES} not found — run derive_cluster_prototypes.py first")

    prototypes: dict[str, str] = json.loads(PROTOTYPES.read_text(encoding="utf-8"))
    print(f"Loaded {len(prototypes)} prototypes from {PROTOTYPES}")

    # Basic structural checks
    empty   = [k for k, v in prototypes.items() if not v.strip()]
    short   = [k for k, v in prototypes.items() if len(v) < 30]

    if empty:
        print(f"  ✗ {len(empty)} empty entries: {empty}")
    if short:
        print(f"  ✗ {len(short)} entries shorter than 30 chars: {short}")

    # Pairwise cosine via sentence-transformers
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415
    print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    keys = sorted(prototypes.keys())
    texts = [prototypes[k] for k in keys]
    embs = np.asarray(
        model.encode(texts, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    cos = embs @ embs.T
    n = len(keys)
    upper = [cos[i, j] for i in range(n) for j in range(i + 1, n)]

    cos_mean = float(np.mean(upper))
    cos_std  = float(np.std(upper))
    cos_max  = float(np.max(upper))

    print()
    print("── Pairwise cosine statistics ───────────────────")
    print(f"  mean: {cos_mean:.4f}")
    print(f"  std:  {cos_std:.4f}  (need > 0.05)")
    print(f"  max:  {cos_max:.4f}  (need < 0.99)")

    # Find the closest pair for inspection
    closest_i, closest_j, closest_val = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] > closest_val:
                closest_val = float(cos[i, j])
                closest_i, closest_j = i, j
    print(f"\n  Closest pair: {keys[closest_i]} ↔ {keys[closest_j]} (cos={closest_val:.4f})")
    print(f"    {keys[closest_i]}: {prototypes[keys[closest_i]][:80]}")
    print(f"    {keys[closest_j]}: {prototypes[keys[closest_j]][:80]}")

    # Gate
    pass_count  = len(empty) == 0
    pass_short  = len(short) == 0
    pass_std    = cos_std > 0.05
    pass_max    = cos_max < 0.99
    gate_pass   = pass_count and pass_short and pass_std and pass_max

    print()
    if gate_pass:
        print("  ✓ GATE PASS — proceed to Phase 3")
    else:
        print("  ✗ GATE FAIL — do NOT proceed")
        if not pass_count: print(f"    - {len(empty)} empty entries")
        if not pass_short: print(f"    - {len(short)} too-short entries")
        if not pass_std:   print(f"    - std {cos_std:.4f} ≤ 0.05 (prototypes too similar)")
        if not pass_max:   print(f"    - max {cos_max:.4f} ≥ 0.99 (two prototypes identical)")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the validator**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/validate_cluster_prototypes.py`

Expected:
- Loads 42 prototypes
- Pairwise cosine std > 0.05 (ideally 0.10+)
- Max cosine < 0.99 (closest pair is distinct)
- `✓ GATE PASS`

If the validator fails with `std ≤ 0.05`, it means gemma4 produced nearly-identical prototypes — the derivation prompt or temperature needs adjustment. Rerun `derive_cluster_prototypes.py` with adjustments and re-validate.

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/validate_cluster_prototypes.py
git commit -m "$(cat <<'EOF'
feat(phase1): validator for cluster prototype distinctness

Separate validate_cluster_prototypes.py enforces Phase 1 gate:
42 non-empty entries, each ≥30 chars, pairwise cosine std > 0.05,
max < 0.99. Exits 1 on failure so subagent runs surface it.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Gate Decision

**PASS** (all validator checks): proceed to [phase-3-generate-v8.md](phase-3-generate-v8.md).

**FAIL**:
- If `std ≤ 0.05` → prototypes collapsed. Tune `temperature` in `generate_prototype()` (try 0.6–0.8), or sharpen the `PROTOTYPE_PROMPT` (e.g. emphasise "traits that DIFFERENTIATE this group from other job types"). Re-derive and re-validate.
- If many entries are the fallback sentence → translations missing for many clusters. Check `jobs_df["cluster_coarse"].value_counts()` to confirm, and decide whether to generate more translations or accept fallbacks as-is.
- If the closest pair is a genuine duplicate (two clusters really do look the same to the LLM) → this may be signalling that the 42-cluster partition has redundancy. Document in the commit and proceed to Phase 3 anyway; the RBF top-3 blending will collapse them.
