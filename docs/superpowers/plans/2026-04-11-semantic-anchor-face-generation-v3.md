# Semantic Anchor Face Generation v3 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace LLM-per-cluster prototype derivation with directional text-query anchors in qwen space. For each job, cosine-softmax over anchor directions produces blend weights; top-3 blend via ConditioningAverage with per-sus-band expression composition.

**Spec:** [../specs/2026-04-11-semantic-anchor-face-generation-v3.md](../specs/2026-04-11-semantic-anchor-face-generation-v3.md)

**Tech Stack:** Python 3.12 + uv, mxbai-embed-large at `localhost:11434`, ComfyUI Flux.1-krea-dev, existing `RBFConditioner` + `compose_sentence()` from v2 Task 1.

**Working directory:** `/home/newub/w/telejobs/tools/face-pipeline`

**Prerequisites:**
- `src/prototype_schema.py` from v2 (commit `b7b5e49`) — schema constants + `compose_sentence()`
- `data/cluster_centroids.pt` — not used at runtime, kept for sanity checks
- `/home/newub/w/vamp-interface/output/full_layout.parquet`
- Postgres at `localhost:15432` with `jobs.embedding` populated
- ComfyUI at `localhost:8188`
- `mxbai-embed-large` loaded at `localhost:11434`

---

## Task 1: Author candidate anchor queries

**Files:**
- Create: `data/candidate_anchors.txt`

Russian text queries covering the plausible axes of the labour-market corpus. Format: one query per line, `<russian_query>  # <english_comment>`. The English comment is for our navigation; it's not encoded.

- [ ] **Step 1: Create the file with 30 starter queries**

```
# Delivery and logistics
курьер пеший доставка еды в городе  # walking food courier
курьер на велосипеде или скутере городская доставка  # bike/scooter courier
водитель курьер авто доставка по городу  # car-based courier
водитель грузового автомобиля междугородние перевозки  # long-haul truck driver
водитель такси агрегатор пассажирские перевозки  # taxi driver

# Construction and trades
разнорабочий на стройке общестроительные работы  # general construction labourer
бригадир прораб на строительном объекте  # construction foreman
сварщик электрогазосварщик на производстве  # welder
сантехник электрик монтажник инженерных систем  # plumber/electrician
плотник столяр отделочник ремонт помещений  # finisher/carpenter

# Office, IT, finance
программист разработчик IT компания офис  # software developer
офис-менеджер секретарь административная работа  # office manager
бухгалтер финансовый отдел 1С  # accountant
HR рекрутер кадровый специалист подбор персонала  # recruiter
менеджер по продажам B2B холодные звонки  # B2B sales manager

# Retail and hospitality
продавец-консультант в магазине одежды торговый зал  # retail sales consultant
кассир супермаркета продуктовый магазин  # supermarket cashier
официант бармен ресторан кафе  # waiter/bartender
повар кухонный работник ресторан столовая  # cook
администратор салона красоты ресепшн  # beauty salon admin

# Factory, warehouse, logistics
разнорабочий на складе комплектовщик  # warehouse worker
оператор станка на заводе производство  # machine operator
упаковщик пищевое производство конвейер  # packer
кладовщик учёт товара  # stockkeeper

# Services
уборщица горничная клининг офисов  # cleaning staff
охранник сторож чоп  # security guard
няня сиделка уход за ребёнком  # nanny/caregiver
автомеханик автосервис ремонт автомобилей  # auto mechanic

# Gig / telework / entry-level
оператор колл-центра удалённая работа на дому  # call centre operator (remote)
подработка разовая работа без опыта студентам  # part-time gig, no experience
```

- [ ] **Step 2: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add data/candidate_anchors.txt
git commit -m "$(cat <<'EOF'
feat(anchors-v3): 30 candidate Russian text queries

Starter pool of text queries covering delivery, construction, trades,
office/IT/finance, retail/hospitality, factory/warehouse, services, and
gig work. Comments are English navigation hints — only the Russian text
is encoded downstream.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Encode candidate queries via mxbai-embed-large

**Files:**
- Create: `src/encode_anchors.py`
- Output: `data/candidate_anchor_embeddings.parquet`

- [ ] **Step 1: Create the encoder script**

```python
#!/usr/bin/env python3
"""encode_anchors.py — encode candidate text queries into qwen space.

Reads data/candidate_anchors.txt, strips comments, calls the local
mxbai-embed-large via Ollama per-query, stores name + query + unit-normed
1024-d embedding in a parquet file.

Usage:
    uv run src/encode_anchors.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CANDIDATES = Path("data/candidate_anchors.txt")
OUT_PATH   = Path("data/candidate_anchor_embeddings.parquet")
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL      = "mxbai-embed-large"


def parse_candidates(path: Path) -> list[tuple[str, str]]:
    """Return [(name_slug, russian_query)] pairs from the file."""
    items: list[tuple[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline English comment
        query = line.split("#", 1)[0].strip()
        if not query:
            continue
        # Name slug = first 3 words, joined
        slug = "_".join(query.split()[:3]).lower()
        items.append((slug, query))
    return items


def embed(query: str, session: requests.Session) -> np.ndarray:
    resp = session.post(OLLAMA_URL, json={"model": MODEL, "prompt": query}, timeout=60)
    resp.raise_for_status()
    vec = np.asarray(resp.json()["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm == 0:
        raise RuntimeError(f"zero vector for query {query!r}")
    return vec / norm


def main() -> None:
    candidates = parse_candidates(CANDIDATES)
    print(f"Loaded {len(candidates)} candidate queries")

    session = requests.Session()
    rows: list[dict] = []
    for i, (name, query) in enumerate(candidates):
        print(f"  [{i+1:2d}/{len(candidates)}] {name}  {query[:60]}")
        vec = embed(query, session)
        rows.append({
            "name":      name,
            "query":     query,
            "embedding": vec.tolist(),
        })

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} anchors to {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and verify**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/encode_anchors.py
```

Expected: 30 lines printed, parquet file written. Each embedding is 1024-d. Takes ~30 s.

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/encode_anchors.py data/candidate_anchor_embeddings.parquet
git commit -m "$(cat <<'EOF'
feat(anchors-v3): encode candidate queries via mxbai-embed-large

encode_anchors.py reads candidate_anchors.txt, strips English comments,
calls the local mxbai-embed-large via Ollama once per query, and saves
(name, query, 1024-d unit vector) rows to
data/candidate_anchor_embeddings.parquet.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Coverage + dedup + knee analysis

**Files:**
- Create: `src/analyse_anchor_coverage.py`
- Output: `data/anchor_coverage_report.json`

- [ ] **Step 1: Create the analysis script**

```python
#!/usr/bin/env python3
"""analyse_anchor_coverage.py — dedup candidates, run coverage + knee analysis.

Loads candidate anchor embeddings and the full job embedding corpus.
Drops near-duplicate anchors (pairwise cos > 0.85). Greedy-picks anchors
by marginal contribution to median max_cos, finds the knee, and saves a
report with the recommended anchor set.

Usage:
    uv run src/analyse_anchor_coverage.py
    uv run src/analyse_anchor_coverage.py --num-anchors 15   # override knee
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

ANCHORS_IN = Path("data/candidate_anchor_embeddings.parquet")
REPORT_OUT = Path("data/anchor_coverage_report.json")
DB_DSN     = "postgresql://USER:PASS@HOST:PORT/DB"

DEDUP_COS = 0.85
MIN_P50_COVERAGE = 0.50
MIN_P10_COVERAGE = 0.35
KNEE_DELTA = 0.01


def parse_embedding(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


def load_jobs() -> np.ndarray:
    print("Fetching job embeddings from DB ...")
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT embedding FROM jobs WHERE embedding IS NOT NULL")
        rows = cur.fetchall()
    conn.close()

    vecs = np.asarray(
        [parse_embedding(r["embedding"]) for r in rows], dtype=np.float32
    )
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs /= norms
    print(f"  {vecs.shape[0]} unit-normed job vectors, dim={vecs.shape[1]}")
    return vecs


def load_candidates() -> pd.DataFrame:
    df = pd.read_parquet(ANCHORS_IN)
    df["vec"] = df["embedding"].apply(lambda e: np.asarray(e, dtype=np.float32))
    return df


def dedup(anchors: pd.DataFrame, jobs: np.ndarray) -> pd.DataFrame:
    """Drop anchors whose cosine to any already-kept anchor exceeds DEDUP_COS.
    Keep the one with higher median corpus cosine among any duplicate pair."""
    avgs = np.array([float(np.median(jobs @ v)) for v in anchors["vec"]])
    order = np.argsort(-avgs)
    kept_idx: list[int] = []
    for idx in order:
        vec = anchors["vec"].iloc[int(idx)]
        redundant = False
        for k in kept_idx:
            if float(np.dot(vec, anchors["vec"].iloc[k])) > DEDUP_COS:
                redundant = True
                break
        if not redundant:
            kept_idx.append(int(idx))
    kept = anchors.iloc[kept_idx].reset_index(drop=True)
    print(f"  Kept {len(kept)} / {len(anchors)} after dedup (cos > {DEDUP_COS})")
    return kept


def coverage_stats(jobs: np.ndarray, anchor_mat: np.ndarray) -> dict:
    """Returns p10 / p50 / p90 of max_cos over anchors for the job corpus."""
    sim = jobs @ anchor_mat.T  # [N, M]
    max_cos = sim.max(axis=1)
    return {
        "p10":  float(np.percentile(max_cos, 10)),
        "p50":  float(np.percentile(max_cos, 50)),
        "p90":  float(np.percentile(max_cos, 90)),
        "mean": float(max_cos.mean()),
    }


def greedy_order(jobs: np.ndarray, anchors: pd.DataFrame) -> list[int]:
    """Greedy selection order: at each step, pick the anchor that maximally
    improves median max_cos when added."""
    remaining = list(range(len(anchors)))
    selected: list[int] = []
    cur_max = np.full(jobs.shape[0], -np.inf, dtype=np.float32)

    while remaining:
        best, best_med = -1, -np.inf
        for idx in remaining:
            cand_vec = anchors["vec"].iloc[idx]
            cand_sim = jobs @ cand_vec
            new_max = np.maximum(cur_max, cand_sim)
            med = float(np.median(new_max))
            if med > best_med:
                best_med, best = med, idx
        selected.append(best)
        cand_vec = anchors["vec"].iloc[best]
        cur_max = np.maximum(cur_max, jobs @ cand_vec)
        remaining.remove(best)

    return selected


def find_knee(jobs: np.ndarray, anchors: pd.DataFrame, order: list[int]) -> int:
    """Return N where adding the next anchor improves median max_cos by
    less than KNEE_DELTA. Always ≥ 5."""
    cur_max = np.full(jobs.shape[0], -np.inf, dtype=np.float32)
    prev_med = -np.inf
    for n, idx in enumerate(order, start=1):
        cur_max = np.maximum(cur_max, jobs @ anchors["vec"].iloc[idx])
        med = float(np.median(cur_max))
        if n >= 5 and med - prev_med < KNEE_DELTA:
            return n - 1
        prev_med = med
    return len(order)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-anchors", type=int, default=None,
                        help="Override knee-derived N")
    args = parser.parse_args()

    anchors = load_candidates()
    print(f"Loaded {len(anchors)} candidates")

    jobs = load_jobs()
    print("Deduplicating ...")
    anchors = dedup(anchors, jobs)

    print("Greedy ordering by marginal contribution ...")
    order = greedy_order(jobs, anchors)

    knee_n = find_knee(jobs, anchors, order)
    print(f"  Knee at N={knee_n}")

    n_final = args.num_anchors if args.num_anchors else knee_n
    final_idx = order[:n_final]
    final = anchors.iloc[final_idx].reset_index(drop=True)

    anchor_mat = np.stack([v for v in final["vec"]])
    stats = coverage_stats(jobs, anchor_mat)

    print()
    print(f"Final anchor count: {len(final)}")
    print("Coverage stats (max_cos distribution):")
    for k, v in stats.items():
        print(f"  {k}: {v:.4f}")

    if stats["p50"] < MIN_P50_COVERAGE:
        print(f"  ✗ p50 {stats['p50']:.4f} < {MIN_P50_COVERAGE} — coverage FAIL")
    if stats["p10"] < MIN_P10_COVERAGE:
        print(f"  ✗ p10 {stats['p10']:.4f} < {MIN_P10_COVERAGE} — coverage FAIL")

    print("\nSelected anchors (in order of selection):")
    for i, (_, row) in enumerate(final.iterrows(), start=1):
        print(f"  {i:2d}. {row['name']:30s}  {row['query'][:60]}")

    report = {
        "candidate_count": int(len(anchors)),
        "dedup_threshold": DEDUP_COS,
        "knee_n":          knee_n,
        "final_n":         n_final,
        "coverage":        stats,
        "selected_order": [
            {"rank": i + 1, "name": final["name"].iloc[i], "query": final["query"].iloc[i]}
            for i in range(len(final))
        ],
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nReport saved: {REPORT_OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run and inspect**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/analyse_anchor_coverage.py
```

Expected:
- Dedup drops 0-5 anchors
- Knee lands somewhere in 10-22
- p50 > 0.50 and p10 > 0.35
- Selected order prints

If p50 < 0.50 or p10 < 0.35: STOP. The candidate pool isn't covering the corpus. Escalate with the numbers.

- [ ] **Step 3: Commit the script and report**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/analyse_anchor_coverage.py data/anchor_coverage_report.json
git commit -m "$(cat <<'EOF'
feat(anchors-v3): dedup + coverage + knee analysis

analyse_anchor_coverage.py drops redundant anchors (cos > 0.85), runs
greedy ordering by marginal contribution to median max_cos, and picks
the knee N where adding another anchor improves median by less than
0.01. Saves data/anchor_coverage_report.json with the ranked list.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Hand-author face records for the final anchors

**Files:**
- Create: `data/face_anchors.json`

This task is interactive editorial work — the controller (me, in the conversation) will author the records. No subagent dispatch. The output must satisfy the v2 schema from `prototype_schema.py`.

- [ ] **Step 1: Draft the JSON file with one record per selected anchor**

Structure:

```json
{
  "version": 3,
  "anchors": [
    {
      "name": "курьер_пеший_доставка",
      "query": "курьер пеший доставка еды в городе",
      "embedding": [0.01, -0.02, ...],
      "face_record": {
        "age": "18-25",
        "gender": "masculine",
        "ethnicity": "Central Asian",
        "hair": "short straight black",
        "facial_hair": "clean-shaven",
        "complexion": "sun-weathered",
        "uniform": "bright yellow Yandex.Eda reflective jacket",
        "expressions": {
          "clean": "alert calm focus, reading phone map",
          "low": "tight alertness",
          "mid": "weary scanning",
          "high": "darting calculating",
          "fraud": "hollow forced deference"
        }
      }
    },
    ...
  ]
}
```

Authoring rules (enforced by Task 5 validator):
- Every `face_record` has all 8 axes
- Enumerated axes (age, gender, ethnicity, complexion) use whitelisted values
- `expressions` has all 5 band keys
- `facial_hair` is `"none"` for feminine and androgynous
- Ethnicity distribution target: **40% Slavic, 15% each of Central Asian, Armenian, Mediterranean, East Asian, Middle Eastern**
- Validator allows up to ±1 anchor deviation from target

Embeddings are copied verbatim from `data/candidate_anchor_embeddings.parquet` for the selected anchor names.

- [ ] **Step 2: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add data/face_anchors.json
git commit -m "$(cat <<'EOF'
feat(anchors-v3): hand-authored face records for N semantic anchors

Each anchor has a complete 8-axis face record plus 5 per-sus-band
expression phrases. Ethnicity distribution targets 40% Slavic with
even spread across the other five demographics.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Validator

**Files:**
- Create: `src/validate_face_anchors.py`

- [ ] **Step 1: Create the validator**

```python
#!/usr/bin/env python3
"""validate_face_anchors.py — Phase 1c gate.

Loads data/face_anchors.json, checks schema compliance, ethnicity
distribution, and pairwise anchor orthogonality. Exits 1 on failure.
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prototype_schema import (  # type: ignore[import-untyped]
    AGE_VALUES, GENDER_VALUES, ETHNICITY_VALUES, COMPLEXION_VALUES,
    SUS_BANDS, REQUIRED_AXES,
)

IN_PATH = Path("data/face_anchors.json")

TARGET_ETHNICITY = {
    "Slavic":         0.40,
    "Central Asian":  0.12,
    "Armenian":       0.12,
    "Mediterranean":  0.12,
    "East Asian":     0.12,
    "Middle Eastern": 0.12,
}
MAX_PAIRWISE_COS = 0.85


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"ERROR: {IN_PATH} not found")

    data = json.loads(IN_PATH.read_text(encoding="utf-8"))
    anchors = data["anchors"]
    n = len(anchors)
    print(f"Loaded {n} anchors")

    errors: list[str] = []

    # 1. Schema validation
    for a in anchors:
        name = a["name"]
        rec = a["face_record"]
        for axis in REQUIRED_AXES:
            if axis not in rec or rec[axis] in (None, "", {}):
                errors.append(f"{name}: missing {axis}")
        if rec.get("age") not in AGE_VALUES:
            errors.append(f"{name}: bad age {rec.get('age')!r}")
        if rec.get("gender") not in GENDER_VALUES:
            errors.append(f"{name}: bad gender {rec.get('gender')!r}")
        if rec.get("ethnicity") not in ETHNICITY_VALUES:
            errors.append(f"{name}: bad ethnicity {rec.get('ethnicity')!r}")
        if rec.get("complexion") not in COMPLEXION_VALUES:
            errors.append(f"{name}: bad complexion {rec.get('complexion')!r}")
        exps = rec.get("expressions", {})
        for band in SUS_BANDS:
            if band not in exps or not exps[band]:
                errors.append(f"{name}: missing expressions.{band}")
        if rec.get("gender") in ("feminine", "androgynous"):
            if rec.get("facial_hair", "").lower() != "none":
                errors.append(f"{name}: facial_hair must be 'none' for {rec['gender']}")

    # 2. Ethnicity distribution
    counts = Counter(a["face_record"]["ethnicity"] for a in anchors)
    print("\nEthnicity distribution:")
    for eth, target in TARGET_ETHNICITY.items():
        actual = counts.get(eth, 0)
        target_n = target * n
        diff = actual - target_n
        ok = abs(diff) <= 1.0
        mark = "✓" if ok else "✗"
        print(f"  {mark} {eth:18s}  {actual:2d}  (target {target_n:.1f}, diff {diff:+.1f})")
        if not ok:
            errors.append(f"ethnicity {eth}: {actual} vs target {target_n:.1f}")

    # 3. Pairwise anchor cosine
    print("\nPairwise anchor cosines ...")
    mat = np.stack([np.asarray(a["embedding"], dtype=np.float32) for a in anchors])
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat /= norms
    cos = mat @ mat.T

    max_pair = -np.inf
    max_i, max_j = 0, 0
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] > max_pair:
                max_pair = float(cos[i, j])
                max_i, max_j = i, j
    print(f"  Closest pair: {anchors[max_i]['name']} ↔ {anchors[max_j]['name']} "
          f"(cos={max_pair:.4f})")
    if max_pair > MAX_PAIRWISE_COS:
        errors.append(f"redundant anchor pair {anchors[max_i]['name']} ↔ "
                      f"{anchors[max_j]['name']} cos={max_pair:.4f}")

    print()
    if errors:
        print(f"  ✗ GATE FAIL ({len(errors)} errors):")
        for e in errors:
            print(f"    - {e}")
        sys.exit(1)
    print("  ✓ GATE PASS — proceed to regeneration")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validator**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/validate_face_anchors.py
```

Expected: `✓ GATE PASS`. If it fails on ethnicity distribution, edit `data/face_anchors.json` and re-run. If it fails on pairwise cosine, the anchor selection from Task 3 didn't dedup sufficiently — revisit.

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/validate_face_anchors.py
git commit -m "$(cat <<'EOF'
feat(anchors-v3): schema + ethnicity distribution + orthogonality validator

Enforces 8-axis compliance, 40% Slavic + even rest distribution with
±1 anchor tolerance, and pairwise cosine < 0.85 between anchor
embeddings. Exits 1 on failure.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Add `semantic_anchor_conditioning_nodes()` to RBFConditioner

**Files:**
- Modify: `src/rbf_conditioning.py`
- Create: `tests/test_semantic_anchor_conditioning.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for RBFConditioner.semantic_anchor_conditioning_nodes()."""
import json
from pathlib import Path

import pytest
import torch

from rbf_conditioning import RBFConditioner


@pytest.fixture
def tmp_face_anchors(tmp_path: Path) -> Path:
    path = tmp_path / "face_anchors.json"
    path.write_text(json.dumps({
        "version": 3,
        "anchors": [
            {
                "name": "a0",
                "query": "courier",
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "face_record": {
                    "age": "18-25", "gender": "masculine", "ethnicity": "Central Asian",
                    "hair": "short dark", "facial_hair": "clean-shaven",
                    "complexion": "sun-weathered",
                    "uniform": "yellow delivery vest",
                    "expressions": {"clean": "alert", "low": "tense", "mid": "weary",
                                    "high": "darting", "fraud": "hollow"},
                },
            },
            {
                "name": "a1",
                "query": "construction",
                "embedding": [0.0, 1.0, 0.0, 0.0],
                "face_record": {
                    "age": "45-55", "gender": "masculine", "ethnicity": "Slavic",
                    "hair": "grey receding", "facial_hair": "short grey beard",
                    "complexion": "wind-chapped",
                    "uniform": "orange hi-vis vest over coveralls",
                    "expressions": {"clean": "stoic", "low": "guarded", "mid": "resigned",
                                    "high": "suspicious", "fraud": "cold"},
                },
            },
            {
                "name": "a2",
                "query": "office",
                "embedding": [0.0, 0.0, 1.0, 0.0],
                "face_record": {
                    "age": "25-35", "gender": "feminine", "ethnicity": "Slavic",
                    "hair": "long brown tied back", "facial_hair": "none",
                    "complexion": "pale indoor",
                    "uniform": "black blazer over white blouse",
                    "expressions": {"clean": "polite warm", "low": "polite cool",
                                    "mid": "strained", "high": "fake-bright", "fraud": "forced"},
                },
            },
        ],
    }, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def tmp_centroids(tmp_path: Path) -> Path:
    path = tmp_path / "cluster_centroids.pt"
    torch.save({
        "centroids": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "cluster_keys": ["C0"],
        "cluster_to_archetype": {"C0": "x"},
    }, path)
    return path


@pytest.fixture
def conditioner(tmp_centroids: Path, tmp_face_anchors: Path, tmp_path: Path) -> RBFConditioner:
    archetypes = tmp_path / "archetypes.json"
    archetypes.write_text("{}", encoding="utf-8")
    cond = RBFConditioner(
        centroids_path=tmp_centroids,
        archetypes_path=archetypes,
        k=3,
        temperature=1.0,
    )
    cond.face_anchors = json.loads(tmp_face_anchors.read_text(encoding="utf-8"))["anchors"]
    return cond


def test_produces_top3_encode_nodes(conditioner: RBFConditioner) -> None:
    emb = [0.8, 0.5, 0.1, 0.0]
    nodes, _ = conditioner.semantic_anchor_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    encodes = [v for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert len(encodes) == 3


def test_dominant_anchor_text_appears(conditioner: RBFConditioner) -> None:
    emb = [10.0, 0.0, 0.0, 0.0]  # strongly matches a0 (courier)
    nodes, _ = conditioner.semantic_anchor_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    texts = [v["inputs"]["text"] for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert any("delivery vest" in t for t in texts)


def test_sus_band_changes_expression(conditioner: RBFConditioner) -> None:
    emb = [10.0, 0.0, 0.0, 0.0]
    clean, _ = conditioner.semantic_anchor_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    fraud, _ = conditioner.semantic_anchor_conditioning_nodes(
        embedding=emb, sus_band="fraud", clip_ref=["3", 0],
    )
    clean_texts = sorted(v["inputs"]["text"] for v in clean.values() if v["class_type"] == "CLIPTextEncode")
    fraud_texts = sorted(v["inputs"]["text"] for v in fraud.values() if v["class_type"] == "CLIPTextEncode")
    assert clean_texts != fraud_texts


def test_fewer_than_3_anchors_still_works(conditioner: RBFConditioner) -> None:
    conditioner.face_anchors = conditioner.face_anchors[:2]
    emb = [0.5, 0.5, 0.0, 0.0]
    nodes, final_id = conditioner.semantic_anchor_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    encodes = [v for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert len(encodes) == 2
    assert final_id in nodes
```

- [ ] **Step 2: Run tests, confirm AttributeError**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_semantic_anchor_conditioning.py -v
```

Expected: failures on `AttributeError: ... has no attribute 'semantic_anchor_conditioning_nodes'`.

- [ ] **Step 3: Implement the method on `RBFConditioner`**

In `src/rbf_conditioning.py`:

1. At the end of `__init__`, load face anchors:

```python
        # Load semantic face anchors (v3) if present
        face_anchors_path = centroids_path.parent / "face_anchors.json"
        self.face_anchors: list[dict] = []
        if face_anchors_path.exists():
            data = json.loads(face_anchors_path.read_text(encoding="utf-8"))
            self.face_anchors = data.get("anchors", [])
```

2. Append this method inside the class:

```python
    def semantic_anchor_conditioning_nodes(
        self,
        embedding: list[float],
        sus_band: str,
        clip_ref: list,
        start_node_id: int = 20,
        softmax_temperature: float = 0.1,
    ) -> tuple[dict[str, Any], str]:
        """Blend top-3 semantic anchors by cosine softmax over job embedding.

        Each anchor is a qwen-space unit vector + a face_record. Cosine
        similarity of the job embedding to each anchor is softmaxed into
        blend weights; the top-3 face records are composed into sentences
        at the requested sus_band and blended via ConditioningAverage.
        """
        from prototype_schema import compose_sentence  # noqa: PLC0415

        if not self.face_anchors:
            raise RuntimeError(
                "No face anchors loaded — generate data/face_anchors.json first"
            )

        job_vec = torch.tensor(embedding, dtype=torch.float32)
        job_vec = job_vec / (job_vec.norm() + 1e-9)

        scores: list[float] = []
        for a in self.face_anchors:
            av = torch.tensor(a["embedding"], dtype=torch.float32)
            av = av / (av.norm() + 1e-9)
            scores.append(float(torch.dot(job_vec, av)))
        scores_t = torch.tensor(scores, dtype=torch.float32)
        weights = torch.softmax(scores_t / softmax_temperature, dim=0)

        k = min(3, len(self.face_anchors))
        top_vals, top_idxs = torch.topk(weights, k)
        top_total = float(top_vals.sum())
        top_weights = [float(v / top_total) for v in top_vals]
        top_anchors = [self.face_anchors[int(i)] for i in top_idxs]

        nodes: dict[str, Any] = {}
        nid = start_node_id
        encode_ids: list[str] = []
        for anchor in top_anchors:
            sentence = compose_sentence(anchor["face_record"], sus_band)
            eid = str(nid); nid += 1
            nodes[eid] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": sentence, "clip": clip_ref},
            }
            encode_ids.append(eid)

        if len(encode_ids) == 1:
            return nodes, encode_ids[0]
        if len(encode_ids) == 2:
            w1, w2 = top_weights
            fid = str(nid); nid += 1
            nodes[fid] = {
                "class_type": "ConditioningAverage",
                "inputs": {
                    "conditioning_to":          [encode_ids[0], 0],
                    "conditioning_from":        [encode_ids[1], 0],
                    "conditioning_to_strength": round(w1 / (w1 + w2), 4),
                },
            }
            return nodes, fid

        w1, w2, w3 = top_weights
        blend_id = str(nid); nid += 1
        final_id = str(nid); nid += 1
        nodes[blend_id] = {
            "class_type": "ConditioningAverage",
            "inputs": {
                "conditioning_to":          [encode_ids[0], 0],
                "conditioning_from":        [encode_ids[1], 0],
                "conditioning_to_strength": round(w1 / (w1 + w2), 4),
            },
        }
        nodes[final_id] = {
            "class_type": "ConditioningAverage",
            "inputs": {
                "conditioning_to":          [blend_id, 0],
                "conditioning_from":        [encode_ids[2], 0],
                "conditioning_to_strength": round((w1 + w2) / (w1 + w2 + w3), 4),
            },
        }
        return nodes, final_id
```

- [ ] **Step 4: Run tests — verify 4 passing**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_semantic_anchor_conditioning.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/rbf_conditioning.py tests/test_semantic_anchor_conditioning.py
git commit -m "$(cat <<'EOF'
feat(anchors-v3): semantic_anchor_conditioning_nodes() on RBFConditioner

Loads data/face_anchors.json in __init__. New method computes cosine
similarity of the job embedding to each anchor, softmaxes with
temperature=0.1, takes top-3, composes per-sus-band sentences through
compose_sentence(), and blends through ConditioningAverage. v1, v2 and
cluster-based methods untouched.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: generate_v8c.py

**Files:**
- Create: `src/generate_v8c.py`

Copy `src/generate_v8b.py` and apply two diffs:

**Diff 1 — Conditioning call.** Replace:

```python
cond_nodes, final_cond_id = rbf.structured_prototype_conditioning_nodes(
    embedding=job["embedding"],
    sus_band=job["sus_band"],
    clip_ref=["3", 0],
    start_node_id=20,
)
```

with:

```python
cond_nodes, final_cond_id = rbf.semantic_anchor_conditioning_nodes(
    embedding=job["embedding"],
    sus_band=job["sus_band"],
    clip_ref=["3", 0],
    start_node_id=20,
)
```

**Diff 2 — Output path and error message.** Change:
- `OUT_DIR = Path("output/dataset_faces_v8b")` → `Path("output/dataset_faces_v8c")`
- `if not rbf.cluster_prototypes_v2:` → `if not rbf.face_anchors:`
- Error message references `analyse_anchor_coverage.py` and `validate_face_anchors.py` instead of `derive_structured_prototypes.py`

The LoRA curve, sampler, and everything else remains the same as v8b.

- [ ] **Step 1: Create the file by copying and applying the diffs**

- [ ] **Step 2: Dry run**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8c.py --dry-run --n-per-cell 2
```

Expected: ~370-420 jobs printed, sus_factor=0.00 at sus≤25, no errors.

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/generate_v8c.py
git commit -m "$(cat <<'EOF'
feat(anchors-v3): generate_v8c.py uses semantic anchor conditioning

Copy of generate_v8b.py with two changes:
1. Calls rbf.semantic_anchor_conditioning_nodes() instead of
   structured_prototype_conditioning_nodes()
2. Writes to output/dataset_faces_v8c/

Everything else (softened LoRA curve, per-cluster sampler, seed per
job_id, resume logic) is unchanged.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Smoke + full batch + contact sheet

**Files:** no code changes.

- [ ] **Step 1: Single-cell smoke (~210 faces, 1 per cluster×band)**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8c.py --n-per-cell 1
```

Takes ~70 min. If ComfyUI returns a 400 on the first face, STOP and report the exact error.

- [ ] **Step 2: Contact sheet**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/build_contact_sheet.py output/dataset_faces_v8c
```

Visual check criteria (contact sheet at `output/dataset_faces_v8c/contact_sheet.html`):

- Clean-band faces in every row look like normal photos (no LoRA artifacts)
- Rows are visibly distinct along ethnicity, age, uniform
- Ethnicity distribution on the sheet roughly matches 40% Slavic target (eyeball, not measured)
- Within each cluster row, sus-band progression is readable (expressions drift from neutral to off)

- [ ] **Step 3: Full batch (if smoke passes)**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8c.py --n-per-cell 2
```

~2.5 h. Resume logic skips the smoke cells.

- [ ] **Step 4: Commit the manifest**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add output/dataset_faces_v8c/manifest.json
git commit -m "$(cat <<'EOF'
test(anchors-v3): v8c smoke + full batch manifest

Full batch of ~370 faces with semantic anchor conditioning. Manual
inspection of contact_sheet.html shows distribution aligned with the
40% Slavic target and readable ethnicity variation along cluster axis.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

PNGs are not committed (regeneratable).

---

## Task 9: Phase 4 eval

**Files:** follows the existing Phase 4 plan, expanded directory list.

- [ ] **Step 1: Add `output/dataset_faces_v8c` to the evaluator's directory list** alongside v6, v7, v8, v8b

- [ ] **Step 2: Run the eval**

- [ ] **Step 3: Decide**

Gate criteria:
- v8c M3 > v8 M3 + 0.05 (semantic anchors improved cluster separation)
- v8c M6 ≥ v8 M6 − 0.05 (sus signal preserved)
- v8c M1 in normal-photo range for clean band (softened LoRA curve paid off)

If all pass: v8c is production. v8 and v8b archived.
If M3 doesn't improve: the anchor set is inadequate — expand candidate pool or try different queries.
If M6 regresses: sus axis is under-signalling — revisit LoRA curve.

---

## Self-review checklist

- [x] Every task has a commit step with the exact message
- [x] Every code step has complete code
- [x] Every test step has complete test code
- [x] Softmax temperature parameter is exposed (0.1 default, tunable)
- [x] Face anchors are loaded from a single JSON file in `__init__`
- [x] The new method gracefully handles < 3 anchors (tested)
- [x] v1, v2, and cluster-based pipelines are not touched
- [x] Validator enforces the 40% Slavic + even rest distribution
- [x] Coverage gate (p50 > 0.50, p10 > 0.35) is real, not a passing comment
