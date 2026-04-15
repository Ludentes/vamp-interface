# Phase 0 — Prototype Clustering Smoke Test

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Validate the core hypothesis: when gemma4 is asked to write face descriptions from job postings, do those descriptions cluster according to the source qwen clusters? If not, the whole Phase 1–4 pipeline is invalid.

**Architecture:** Sample 20 jobs (5 per cluster × 4 clusters), call gemma4 once per job to produce a face description, encode the descriptions through `sentence-transformers/all-MiniLM-L6-v2`, compute cosine similarity matrix, report intra-cluster vs inter-cluster mean cosines.

**Tech Stack:** Python 3.12 + uv, psycopg2 for DB, pandas for parquet, requests for Ollama, sentence-transformers for embedding, numpy.

**Working directory:** `/home/newub/w/telejobs/tools/face-pipeline`

**Gate (PASS required to proceed to Phase 1):**
- Intra-cluster mean cosine **> 0.7**
- Inter-cluster mean cosine **< 0.5**
- Ratio (intra / inter) **> 1.4**

---

## File Structure

Single-file script with deterministic sample and reproducible output.

| File | Responsibility |
|---|---|
| `src/test_prototype_clustering.py` | End-to-end script: sample jobs → gemma4 → embed → report |

---

## Task 1: Set up the script skeleton and constants

**Files:**
- Create: `src/test_prototype_clustering.py`

- [ ] **Step 1: Create the file with imports, constants, and empty main()**

```python
#!/usr/bin/env python3
"""
test_prototype_clustering.py — Phase 0 gate: does gemma4 face description cluster?

Samples 20 jobs (5 per cluster × 4 clusters), calls gemma4 to write a face description
for each, encodes the descriptions through sentence-transformers, and reports whether
the descriptions cluster according to the source qwen clusters.

Gate criteria (must ALL pass to proceed to Phase 1):
  - intra-cluster mean cosine > 0.7
  - inter-cluster mean cosine < 0.5
  - ratio (intra / inter) > 1.4

Usage:
    uv run src/test_prototype_clustering.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
import requests

DB_DSN        = "postgresql://USER:PASS@HOST:PORT/DB"
OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "gemma4:latest"
LAYOUT_PATH   = Path("/home/newub/w/vamp-interface/output/full_layout.parquet")
TRANSLATIONS  = Path("data/job_translations.parquet")
OUT_REPORT    = Path("output/phase0_report.json")

# Four semantically distinct clusters (hand-picked for variety)
TARGET_CLUSTERS: list[tuple[str, str]] = [
    ("C0",  "доставка — delivery/courier"),
    ("C1",  "стройка — construction"),
    ("C5",  "офис — office"),
    ("C29", "торговля — retail/sales"),
]
JOBS_PER_CLUSTER = 5
SAMPLE_SEED      = 42

FACE_PROMPT = """\
Given this job posting, describe the face of a typical worker in ONE sentence.
Focus on physical appearance: age range, build, gender presentation, demeanour, clothing, facial expression.
Output only the single sentence, no preamble, no markdown, no explanations.

Job posting:
{text}

Face description:"""


def main() -> None:
    print("Phase 0 — Prototype Clustering Smoke Test")
    # filled in next tasks


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the file runs (imports succeed)**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/test_prototype_clustering.py`

Expected output: `Phase 0 — Prototype Clustering Smoke Test`

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/test_prototype_clustering.py
git commit -m "$(cat <<'EOF'
chore(phase0): scaffold prototype clustering smoke test

Imports, constants, FACE_PROMPT template, and empty main() for the
Phase 0 GO/NO-GO test. Validates that gemma4 face descriptions cluster
by source qwen cluster before committing to the Phase 1 prototype pipeline.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement the sampling function

**Files:**
- Modify: `src/test_prototype_clustering.py`

- [ ] **Step 1: Add `sample_jobs()` function above `main()`**

Insert this function definition right before `def main()`:

```python
def sample_jobs() -> list[dict]:
    """Deterministic sample: JOBS_PER_CLUSTER jobs from each TARGET_CLUSTERS cluster.

    Joins full_layout.parquet (cluster assignments) with job_translations.parquet
    (English text). Returns list of {id, cluster_coarse, cluster_label, text_en}.
    Uses SAMPLE_SEED for reproducibility.
    """
    rng = np.random.default_rng(SAMPLE_SEED)

    print("  Loading layout ...")
    df_layout = pd.read_parquet(LAYOUT_PATH, columns=["id", "cluster_coarse"])
    print("  Loading translations ...")
    df_trans = pd.read_parquet(TRANSLATIONS, columns=["job_id", "text_en"])
    df_trans = df_trans.rename(columns={"job_id": "id"})

    merged = df_layout.merge(df_trans, on="id", how="inner")
    merged = merged[merged["text_en"].str.len() > 20]  # drop empty/very short translations
    print(f"  {len(merged)} jobs with cluster + translation")

    selected: list[dict] = []
    for cluster_id, label in TARGET_CLUSTERS:
        pool = merged[merged["cluster_coarse"] == cluster_id]
        if len(pool) < JOBS_PER_CLUSTER:
            raise SystemExit(
                f"Cluster {cluster_id} has only {len(pool)} translated jobs "
                f"(need {JOBS_PER_CLUSTER})"
            )
        idxs = rng.choice(len(pool), size=JOBS_PER_CLUSTER, replace=False)
        for i in idxs:
            row = pool.iloc[int(i)]
            selected.append({
                "id":             row["id"],
                "cluster_coarse": cluster_id,
                "cluster_label":  label,
                "text_en":        row["text_en"][:800],  # truncate to stay inside Ollama context
            })
        print(f"  {cluster_id} ({label}): {JOBS_PER_CLUSTER} selected")

    return selected
```

- [ ] **Step 2: Call `sample_jobs()` from `main()` and print summary**

Replace the `main()` body:

```python
def main() -> None:
    print("Phase 0 — Prototype Clustering Smoke Test")
    print()
    print("── Sampling jobs ────────────────────────────────")
    jobs = sample_jobs()
    print(f"\nTotal sampled: {len(jobs)}")
    for j in jobs:
        print(f"  {j['cluster_coarse']:5s} {j['id'][:8]}  {j['text_en'][:60]!r}")
```

- [ ] **Step 3: Run to verify sampling works**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/test_prototype_clustering.py`

Expected: prints "20 sampled", shows 4 clusters × 5 jobs each with truncated text previews. Should exit cleanly with no errors.

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/test_prototype_clustering.py
git commit -m "$(cat <<'EOF'
feat(phase0): deterministic stratified sampling for clustering test

sample_jobs() joins full_layout.parquet + job_translations.parquet,
picks 5 jobs from each of 4 target clusters using seed=42.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Implement gemma4 face description generation

**Files:**
- Modify: `src/test_prototype_clustering.py`

- [ ] **Step 1: Add `generate_face_description()` function above `main()`**

Insert before `def main()`:

```python
def generate_face_description(text_en: str, session: requests.Session) -> str:
    """Call gemma4 via local Ollama to produce a single-sentence face description."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": FACE_PROMPT.format(text=text_en),
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 120,
            "think": False,
        },
    }
    resp = session.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    out = resp.json()["response"].strip()
    # Take only the first line (gemma4 sometimes adds a second sentence)
    return out.split("\n")[0].strip()
```

- [ ] **Step 2: Extend `main()` to generate descriptions for all sampled jobs**

Replace `main()` body entirely:

```python
def main() -> None:
    print("Phase 0 — Prototype Clustering Smoke Test")
    print()
    print("── Sampling jobs ────────────────────────────────")
    jobs = sample_jobs()
    print(f"\nTotal sampled: {len(jobs)}")

    print("\n── Generating face descriptions via gemma4 ──────")
    session = requests.Session()
    for i, job in enumerate(jobs):
        job["face_desc"] = generate_face_description(job["text_en"], session)
        print(f"  [{i+1:2d}/{len(jobs)}] {job['cluster_coarse']}  {job['face_desc'][:70]}")
```

- [ ] **Step 3: Run end-to-end to verify Ollama integration**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/test_prototype_clustering.py`

Expected: after sampling, generates 20 face descriptions (takes ~30-60s). Each printed line shows cluster ID and the first 70 chars of the sentence. Descriptions should sound like face descriptions (mentions "man/woman", "years old", body features, clothing). If gemma4 returns meta-commentary or empty strings, temperature or prompt need adjustment.

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/test_prototype_clustering.py
git commit -m "$(cat <<'EOF'
feat(phase0): gemma4 face description generation

generate_face_description() calls local Ollama with think:false for speed.
Strips to first sentence to handle gemma4 occasionally producing multi-line output.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Embed descriptions with sentence-transformers

**Files:**
- Modify: `src/test_prototype_clustering.py`
- Modify: `pyproject.toml` (add `sentence-transformers` if missing)

- [ ] **Step 1: Check if `sentence-transformers` is installed**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run python -c "import sentence_transformers; print(sentence_transformers.__version__)"`

If it prints a version, skip step 2. If it raises `ModuleNotFoundError`, continue to step 2.

- [ ] **Step 2: Add sentence-transformers dependency**

Run:
```bash
cd /home/newub/w/telejobs/tools/face-pipeline
uv add sentence-transformers
```

Verify: `uv run python -c "import sentence_transformers; print('ok')"` prints `ok`.

- [ ] **Step 3: Add `embed_descriptions()` function above `main()`**

Insert before `def main()`:

```python
def embed_descriptions(descriptions: list[str]) -> np.ndarray:
    """Encode face descriptions with sentence-transformers/all-MiniLM-L6-v2.

    Returns L2-normalized [N, 384] numpy array.
    """
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode(
        descriptions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embs, dtype=np.float32)
```

- [ ] **Step 4: Extend `main()` to embed and print shape**

Append to `main()` (after the description-generation loop):

```python
    print("\n── Embedding descriptions ────────────────────────")
    descs = [j["face_desc"] for j in jobs]
    embs = embed_descriptions(descs)
    print(f"  Embedding matrix: {embs.shape}")
```

- [ ] **Step 5: Run to verify embedding works**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/test_prototype_clustering.py`

Expected: downloads `all-MiniLM-L6-v2` on first run (22 MB), prints `Embedding matrix: (20, 384)`.

- [ ] **Step 6: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add pyproject.toml uv.lock src/test_prototype_clustering.py
git commit -m "$(cat <<'EOF'
feat(phase0): embed face descriptions with sentence-transformers

Uses sentence-transformers/all-MiniLM-L6-v2 (22 MB, English-only, fast)
to produce 384-d L2-normalized vectors for clustering analysis.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Compute clustering metrics and gate

**Files:**
- Modify: `src/test_prototype_clustering.py`

- [ ] **Step 1: Add `compute_metrics()` function above `main()`**

Insert before `def main()`:

```python
def compute_metrics(jobs: list[dict], embs: np.ndarray) -> dict:
    """Compute intra-cluster and inter-cluster mean cosine similarities.

    Returns dict with intra_mean, inter_mean, ratio, per_cluster_intra,
    and a pass flag computed against the Phase 0 gate criteria.
    """
    cos = embs @ embs.T  # [N, N] — pairwise cosine (embs are L2-normalized)
    n = len(jobs)

    intra_pairs: list[float] = []
    inter_pairs: list[float] = []
    per_cluster_intra: dict[str, list[float]] = {}

    for i in range(n):
        for j in range(i + 1, n):
            c_i = jobs[i]["cluster_coarse"]
            c_j = jobs[j]["cluster_coarse"]
            if c_i == c_j:
                intra_pairs.append(float(cos[i, j]))
                per_cluster_intra.setdefault(c_i, []).append(float(cos[i, j]))
            else:
                inter_pairs.append(float(cos[i, j]))

    intra_mean = float(np.mean(intra_pairs)) if intra_pairs else 0.0
    inter_mean = float(np.mean(inter_pairs)) if inter_pairs else 0.0
    ratio      = intra_mean / inter_mean if inter_mean > 0 else float("inf")

    gate_pass = (intra_mean > 0.7) and (inter_mean < 0.5) and (ratio > 1.4)

    return {
        "intra_mean":         round(intra_mean, 4),
        "inter_mean":         round(inter_mean, 4),
        "ratio":              round(ratio, 4),
        "per_cluster_intra":  {k: round(float(np.mean(v)), 4) for k, v in per_cluster_intra.items()},
        "gate_pass":          gate_pass,
        "gate_criteria": {
            "intra_mean > 0.7":  intra_mean > 0.7,
            "inter_mean < 0.5":  inter_mean < 0.5,
            "ratio > 1.4":       ratio > 1.4,
        },
    }
```

- [ ] **Step 2: Extend `main()` to compute metrics, print, and save report**

Append to `main()`:

```python
    print("\n── Clustering metrics ────────────────────────────")
    metrics = compute_metrics(jobs, embs)
    print(f"  intra-cluster mean cosine: {metrics['intra_mean']:.4f}  (need > 0.70)")
    print(f"  inter-cluster mean cosine: {metrics['inter_mean']:.4f}  (need < 0.50)")
    print(f"  ratio (intra / inter):     {metrics['ratio']:.4f}  (need > 1.40)")
    print()
    print("  per-cluster intra:")
    for cluster_id, v in metrics["per_cluster_intra"].items():
        print(f"    {cluster_id:5s} {v:.4f}")

    print()
    if metrics["gate_pass"]:
        print("  ✓ GATE PASS — proceed to Phase 1")
    else:
        print("  ✗ GATE FAIL — do NOT proceed to Phase 1")
        failed = [k for k, v in metrics["gate_criteria"].items() if not v]
        print(f"    Failed criteria: {', '.join(failed)}")

    # Save full report (jobs + descriptions + metrics) for debugging
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "target_clusters":    [{"id": c, "label": l} for c, l in TARGET_CLUSTERS],
        "jobs_per_cluster":   JOBS_PER_CLUSTER,
        "sample_seed":        SAMPLE_SEED,
        "jobs": [
            {"id": j["id"], "cluster": j["cluster_coarse"], "face_desc": j["face_desc"]}
            for j in jobs
        ],
        "metrics": metrics,
    }
    OUT_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  Report saved: {OUT_REPORT}")

    # Exit with non-zero code on gate failure to fail CI / subagent runs
    if not metrics["gate_pass"]:
        sys.exit(1)
```

- [ ] **Step 3: Run end-to-end and capture the result**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/test_prototype_clustering.py`

Expected behaviour depends on gemma4 quality:
- **Ideal**: intra > 0.70, inter < 0.50, ratio > 1.4 → `✓ GATE PASS`
- **Soft fail**: intra ~0.55–0.65, ratio 1.1–1.3 → clusters are somewhat aligned but not strongly. Investigate: (1) sharper prompt, (2) higher temperature, (3) different clusters (some may be semantically ambiguous), (4) the qwen partition itself may be noisy (spec §1 "willing to abandon" clause).
- **Hard fail**: ratio ≤ 1.0 → LLM produces generic descriptions regardless of input. Try a more capable model (gpt-oss) or restructure the prompt.

The script exits with code 1 on failure so the subagent run surfaces the result clearly.

- [ ] **Step 4: Commit the script (regardless of pass/fail)**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/test_prototype_clustering.py output/phase0_report.json
git commit -m "$(cat <<'EOF'
feat(phase0): compute clustering metrics and enforce gate

compute_metrics() calculates intra/inter-cluster mean cosines and ratio.
main() prints the numbers, saves output/phase0_report.json, and exits 1
on gate failure so subagent runs surface the result.

Includes the report from the initial run (pass/fail status depends on
gemma4 output quality — see commit details for the actual numbers).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Gate Decision

After Task 5 runs, `output/phase0_report.json` contains the measured metrics.

**If PASS** (all three criteria met): proceed to [phase-1-derive-cluster-prototypes.md](phase-1-derive-cluster-prototypes.md).

**If SOFT FAIL** (ratio 1.0–1.4, intra 0.55–0.7): document findings in the commit message. Options to try before abandoning:
- Raise prompt specificity (add "focus on unique visual traits that differentiate this worker from others")
- Try a stronger model (gpt-oss:latest) as fallback
- Check whether the 4 TARGET_CLUSTERS are actually semantically distinct — swap in different ones if two look too similar
- Re-run with different SAMPLE_SEED to rule out sampling noise

**If HARD FAIL** (ratio ≤ 1.0): the LLM projection hypothesis is wrong. Stop. Escalate to the user for redesign — do NOT start Phase 1.
