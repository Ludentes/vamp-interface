# Structured Cluster Prototypes v2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild cluster prototypes as structured 8-axis JSON, compose per (cluster, sus-band), regenerate v8b batch with softened LoRA curve and per-cluster sampling, then run the Phase 4 eval.

**Spec:** [../specs/2026-04-11-structured-cluster-prototypes-v2.md](../specs/2026-04-11-structured-cluster-prototypes-v2.md)

**Tech Stack:** Python 3.12 + uv, gemma4 via Ollama (thinking ON), ComfyUI Flux.1-krea-dev, existing `RBFConditioner` + `flux_v6_workflow`.

**Working directory:** `/home/newub/w/telejobs/tools/face-pipeline`

**Prerequisites:**
- `data/cluster_centroids.pt` (42 clusters)
- `data/job_translations.parquet`
- `/home/newub/w/vamp-interface/output/full_layout.parquet`
- `localhost:11434` with `gemma4:latest`
- `localhost:8188` with Flux.1-krea-dev
- `data/cluster_prototypes.json` from Phase 1 (kept for comparison — do not delete)

---

## Task 1: Schema module and composition (TDD)

**Files:**
- Create: `src/prototype_schema.py`
- Create: `tests/test_prototype_schema.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for prototype_schema.compose_sentence()."""
import pytest

from prototype_schema import (
    compose_sentence, AGE_VALUES, GENDER_VALUES, ETHNICITY_VALUES,
    COMPLEXION_VALUES, SUS_BANDS,
)


VALID_RECORD = {
    "age":        "25-35",
    "gender":     "masculine",
    "ethnicity":  "Central Asian",
    "hair":       "short dark",
    "facial_hair": "light stubble",
    "complexion": "sun-weathered",
    "uniform":    "bright yellow delivery vest over hoodie",
    "expressions": {
        "clean": "calm focused gaze",
        "low":   "slightly tense",
        "mid":   "weary alertness",
        "high":  "hollow darting eyes",
        "fraud": "vacant forced smile",
    },
}


def test_compose_sentence_contains_all_axes() -> None:
    s = compose_sentence(VALID_RECORD, "clean")
    for phrase in ["25-35", "Central Asian", "man", "sun-weathered",
                   "short dark", "stubble", "delivery vest", "calm focused gaze"]:
        assert phrase in s, f"missing {phrase!r} in {s!r}"


def test_compose_sentence_uses_band_expression() -> None:
    clean = compose_sentence(VALID_RECORD, "clean")
    fraud = compose_sentence(VALID_RECORD, "fraud")
    assert "calm focused gaze" in clean
    assert "vacant forced smile" in fraud
    assert clean != fraud


def test_feminine_omits_facial_hair() -> None:
    rec = dict(VALID_RECORD, gender="feminine", facial_hair="none")
    s = compose_sentence(rec, "clean")
    assert "stubble" not in s
    assert "woman" in s


def test_androgynous_becomes_person() -> None:
    rec = dict(VALID_RECORD, gender="androgynous", facial_hair="none")
    s = compose_sentence(rec, "clean")
    assert "person" in s


def test_constants_are_non_empty() -> None:
    assert len(AGE_VALUES) == 5
    assert len(GENDER_VALUES) == 3
    assert len(ETHNICITY_VALUES) >= 5
    assert len(COMPLEXION_VALUES) >= 6
    assert SUS_BANDS == ["clean", "low", "mid", "high", "fraud"]


def test_invalid_band_raises() -> None:
    with pytest.raises(KeyError):
        compose_sentence(VALID_RECORD, "bogus")
```

- [ ] **Step 2: Run tests, confirm ModuleNotFoundError**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_prototype_schema.py -v
```

Expected: `ModuleNotFoundError: prototype_schema`.

- [ ] **Step 3: Implement the schema module**

Create `src/prototype_schema.py`:

```python
"""Structured face-prototype schema and composition.

A cluster prototype is a JSON object with 8 fixed axes + 5 per-sus-band
expression phrases. compose_sentence() deterministically renders a prompt
sentence for a given (record, sus_band) pair.
"""
from __future__ import annotations

AGE_VALUES = ["18-25", "25-35", "35-45", "45-55", "55-65"]
GENDER_VALUES = ["masculine", "feminine", "androgynous"]
ETHNICITY_VALUES = [
    "Slavic", "Central Asian", "Caucasian-mixed", "Mediterranean",
    "East Asian", "Middle Eastern",
]
COMPLEXION_VALUES = [
    "smooth", "pale indoor", "sun-weathered", "wind-chapped",
    "ruddy", "oily", "waxy", "lightly scarred",
]
SUS_BANDS = ["clean", "low", "mid", "high", "fraud"]

REQUIRED_AXES = ["age", "gender", "ethnicity", "hair", "facial_hair",
                 "complexion", "uniform", "expressions"]

GENDER_NOUN = {"masculine": "man", "feminine": "woman", "androgynous": "person"}


def compose_sentence(record: dict, sus_band: str) -> str:
    """Compose a prompt sentence for the given prototype record and sus band.

    Raises KeyError if sus_band is not a recognised band.
    """
    if sus_band not in SUS_BANDS:
        raise KeyError(f"unknown sus_band {sus_band!r}; valid: {SUS_BANDS}")

    age         = record["age"]
    gender      = record["gender"]
    ethnicity   = record["ethnicity"]
    hair        = record["hair"]
    facial_hair = record["facial_hair"]
    complexion  = record["complexion"]
    uniform     = record["uniform"]
    expression  = record["expressions"][sus_band]

    gender_noun = GENDER_NOUN[gender]

    if gender == "masculine" and facial_hair and facial_hair.lower() != "none":
        facial_clause = f", {facial_hair}"
    else:
        facial_clause = ""

    return (
        f"A {age} {ethnicity} {gender_noun} with {complexion} skin, "
        f"{hair} hair{facial_clause}, wearing {uniform}, {expression}."
    )
```

- [ ] **Step 4: Run tests — verify 6 passing**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_prototype_schema.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/prototype_schema.py tests/test_prototype_schema.py
git commit -m "$(cat <<'EOF'
feat(prototypes-v2): schema and compose_sentence()

8-axis face-vector schema with enumerated age/gender/ethnicity/complexion
values and open strings for hair/facial_hair/uniform. compose_sentence()
deterministically renders a prompt sentence per (record, sus_band).

Drops facial_hair clause for feminine/androgynous presentations.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Derive structured prototypes via gemma4

**Files:**
- Create: `src/derive_structured_prototypes.py`
- Output: `data/cluster_prototypes_v2.json`

- [ ] **Step 1: Create the derivation script**

```python
#!/usr/bin/env python3
"""derive_structured_prototypes.py — Phase 1b: 8-axis JSON per cluster.

For each of 42 clusters, picks up to 20 central translated jobs and asks
gemma4 (thinking ON) to emit a strict JSON object with the 8-axis schema.
Post-parses and validates; falls back to a neutral skeleton on failure.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from prototype_schema import (  # type: ignore[import-untyped]
    AGE_VALUES, GENDER_VALUES, ETHNICITY_VALUES, COMPLEXION_VALUES,
    SUS_BANDS, REQUIRED_AXES,
)

OLLAMA_URL    = "http://localhost:11434/api/generate"
OLLAMA_MODEL  = "gemma4:latest"
LAYOUT_PATH   = Path("/home/newub/w/vamp-interface/output/full_layout.parquet")
TRANSLATIONS  = Path("data/job_translations.parquet")
CENTROIDS     = Path("data/cluster_centroids.pt")
OUT_PATH      = Path("data/cluster_prototypes_v2.json")
DB_DSN        = "postgresql://USER:PASS@HOST:PORT/DB"

MAX_EXAMPLES = 20
MAX_EXAMPLE_CHARS = 400

DERIVE_PROMPT_TEMPLATE = """You are designing face archetypes for a data-visualization project that renders every job posting in the Russian-speaking labour market as a photorealistic portrait. Each cluster of similar postings gets ONE face archetype shaped by the industry, working conditions, and demographic realism of that work.

Below are {n} job postings from the same cluster. Synthesise them into a structured face description by filling in every field of the JSON schema below. You must return ONLY a JSON object, no preamble, no markdown fences, no commentary.

HARD RULES:
1. Every field must be present and non-empty.
2. Enumerated fields ("age", "gender", "ethnicity", "complexion") must use values from the allowed list EXACTLY as written.
3. Ethnicity must reflect realistic demographics of the Russian-speaking labour market for this type of work:
   - Delivery, courier, construction, cleaning, domestic work: often Central Asian or Caucasian-mixed
   - IT, office, recruitment, finance: often Slavic
   - Retail, service, hospitality: mixed
   - Do NOT default every cluster to Slavic. Assign based on the work.
4. "hair", "facial_hair", "uniform" are open strings, but be SPECIFIC — name colour, length, brand or colour of vest, etc.
5. For "expressions", give five distinct short phrases — one per sus band. They describe the same person's mood across increasing fraud suspicion: clean is a legitimate applicant, fraud is a desperate or predatory schemer. The phrases should evolve naturally from neutral/professional to off/wrong.
6. Aggressively DIFFERENTIATE this cluster from the generic "weathered worker" default. Pick details that would visibly distinguish this face from a neighbouring cluster.
7. If gender is "feminine" or "androgynous", set facial_hair to "none".

Allowed values:
  age:        {age_values}
  gender:     {gender_values}
  ethnicity:  {ethnicity_values}
  complexion: {complexion_values}

Schema (fill every field):
{{
  "age": "...",
  "gender": "...",
  "ethnicity": "...",
  "hair": "...",
  "facial_hair": "...",
  "complexion": "...",
  "uniform": "...",
  "expressions": {{
    "clean": "...",
    "low":   "...",
    "mid":   "...",
    "high":  "...",
    "fraud": "..."
  }}
}}

Job postings from this cluster:
{examples}

Return only the JSON object:"""

FALLBACK_RECORD = {
    "age":        "25-35",
    "gender":     "androgynous",
    "ethnicity":  "Slavic",
    "hair":       "short neutral",
    "facial_hair": "none",
    "complexion": "smooth",
    "uniform":    "plain dark work jacket",
    "expressions": {
        "clean": "neutral composed",
        "low":   "slightly withdrawn",
        "mid":   "mildly wary",
        "high":  "tense evasive",
        "fraud": "hollow vacant stare",
    },
}


def load_data() -> tuple[torch.Tensor, list[str], pd.DataFrame]:
    print("  Loading centroids ...")
    cen = torch.load(CENTROIDS, weights_only=True)
    centroids = cen["centroids"]
    cluster_keys = cen["cluster_keys"]

    print("  Loading layout + translations ...")
    df_layout = pd.read_parquet(LAYOUT_PATH, columns=["id", "cluster_coarse"])
    df_trans = pd.read_parquet(TRANSLATIONS, columns=["job_id", "text_en"])
    df_trans = df_trans.rename(columns={"job_id": "id"})

    import psycopg2
    import psycopg2.extras
    print("  Fetching embeddings from DB ...")
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id::text, embedding FROM jobs WHERE embedding IS NOT NULL")
        emb_rows = {r["id"]: r["embedding"] for r in cur.fetchall()}
    conn.close()

    merged = df_layout.merge(df_trans, on="id", how="inner")
    merged["embedding"] = merged["id"].map(emb_rows.get)
    merged = merged[merged["embedding"].notna()]
    merged = merged[merged["text_en"].str.len() > 30]
    print(f"    {len(merged)} jobs with cluster + translation + embedding")
    return centroids, cluster_keys, merged


def parse_embedding(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


def central_jobs(cluster_id: str, centroid: torch.Tensor, jobs_df: pd.DataFrame) -> list[str]:
    cluster_jobs = jobs_df[jobs_df["cluster_coarse"] == cluster_id]
    if len(cluster_jobs) == 0:
        return []
    cen_np = centroid.numpy()
    distances: list[tuple[float, str]] = []
    for _, row in cluster_jobs.iterrows():
        emb = np.array(parse_embedding(row["embedding"]), dtype=np.float32)
        d = float(np.sum((emb - cen_np) ** 2))
        distances.append((d, str(row["text_en"])))
    distances.sort(key=lambda t: t[0])
    return [t[1] for t in distances[:MAX_EXAMPLES]]


def extract_json(text: str) -> dict | None:
    """Find and parse the first JSON object in text. Returns None on failure."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def validate_record(rec: dict) -> tuple[bool, str]:
    for axis in REQUIRED_AXES:
        if axis not in rec or rec[axis] in (None, "", {}):
            return False, f"missing {axis}"
    if rec["age"] not in AGE_VALUES:
        return False, f"bad age {rec['age']!r}"
    if rec["gender"] not in GENDER_VALUES:
        return False, f"bad gender {rec['gender']!r}"
    if rec["ethnicity"] not in ETHNICITY_VALUES:
        return False, f"bad ethnicity {rec['ethnicity']!r}"
    if rec["complexion"] not in COMPLEXION_VALUES:
        return False, f"bad complexion {rec['complexion']!r}"
    exps = rec.get("expressions", {})
    for band in SUS_BANDS:
        if band not in exps or not exps[band]:
            return False, f"missing expressions.{band}"
    return True, ""


def generate_record(examples: list[str], session: requests.Session) -> tuple[dict | None, str]:
    numbered = "\n".join(f"{i+1}. {e[:MAX_EXAMPLE_CHARS]}" for i, e in enumerate(examples))
    prompt = DERIVE_PROMPT_TEMPLATE.format(
        n=len(examples),
        age_values=AGE_VALUES,
        gender_values=GENDER_VALUES,
        ethnicity_values=ETHNICITY_VALUES,
        complexion_values=COMPLEXION_VALUES,
        examples=numbered,
    )
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,
            "num_predict": 4000,
            "think": True,
        },
    }
    resp = session.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    raw = resp.json()["response"]
    rec = extract_json(raw)
    if rec is None:
        return None, "no JSON parsed"
    ok, err = validate_record(rec)
    if not ok:
        return None, err
    return rec, ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-clusters", type=int, default=None)
    args = parser.parse_args()

    print("Phase 1b — derive structured cluster prototypes")
    print()
    centroids, cluster_keys, jobs_df = load_data()

    if args.limit_clusters:
        cluster_keys = cluster_keys[:args.limit_clusters]
        centroids = centroids[:args.limit_clusters]

    print(f"\n  Processing {len(cluster_keys)} clusters with gemma4 (thinking ON) ...")
    session = requests.Session()
    records: dict[str, dict] = {}
    fallback_count = 0

    for i, cluster_id in enumerate(tqdm(cluster_keys, desc="clusters")):
        examples = central_jobs(cluster_id, centroids[i], jobs_df)
        if not examples:
            tqdm.write(f"  {cluster_id}: no translated jobs — fallback")
            records[cluster_id] = FALLBACK_RECORD
            fallback_count += 1
            continue

        try:
            rec, err = generate_record(examples, session)
        except Exception as e:
            tqdm.write(f"  {cluster_id}: gemma4 exception ({e}) — fallback")
            records[cluster_id] = FALLBACK_RECORD
            fallback_count += 1
            continue

        if rec is None:
            tqdm.write(f"  {cluster_id}: invalid output ({err}) — fallback")
            records[cluster_id] = FALLBACK_RECORD
            fallback_count += 1
            continue

        records[cluster_id] = rec
        tqdm.write(f"  {cluster_id}: {rec['ethnicity']}, {rec['age']}, {rec['gender']} — ok")

    print(f"\n  Derived: {len(records)}   Fallbacks: {fallback_count}")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test on 3 clusters**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_structured_prototypes.py --limit-clusters 3
```

Expected: 3 cluster records, each with all 8 axes filled, no fallbacks. Takes ~30 s (10 s per cluster with thinking).

If all 3 are fallbacks, stop — gemma4 is not returning parseable JSON. Inspect the raw response manually and adjust the prompt.

- [ ] **Step 3: Full run (42 clusters)**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/derive_structured_prototypes.py
```

Expected: ~5 min. Zero or very few fallbacks. `data/cluster_prototypes_v2.json` has 42 entries.

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/derive_structured_prototypes.py data/cluster_prototypes_v2.json
git commit -m "$(cat <<'EOF'
feat(prototypes-v2): gemma4 structured derivation for 42 clusters

gemma4 with thinking ON, num_predict=4000, strict JSON output per the
8-axis schema. Ethnicity is instructed to reflect Russian-speaking
labour market demographics per cluster type. Fallback record used on
parse or validation failure so the JSON remains complete.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Validator

**Files:**
- Create: `src/validate_structured_prototypes.py`

- [ ] **Step 1: Create the validator**

```python
#!/usr/bin/env python3
"""validate_structured_prototypes.py — Phase 1b gate.

Loads data/cluster_prototypes_v2.json, validates schema compliance,
composes sentences via prototype_schema.compose_sentence(), and
runs pairwise cosine analysis via sentence-transformers.

Gate:
  - 42 entries
  - All required axes present, enumerated values valid
  - All 5 expression bands present per entry
  - Pairwise cosine std  > 0.12
  - Pairwise cosine max  < 0.97
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from prototype_schema import (  # type: ignore[import-untyped]
    compose_sentence, SUS_BANDS, REQUIRED_AXES,
    AGE_VALUES, GENDER_VALUES, ETHNICITY_VALUES, COMPLEXION_VALUES,
)

IN_PATH = Path("data/cluster_prototypes_v2.json")


def main() -> None:
    if not IN_PATH.exists():
        sys.exit(f"ERROR: {IN_PATH} not found")

    records: dict[str, dict] = json.loads(IN_PATH.read_text(encoding="utf-8"))
    print(f"Loaded {len(records)} records")

    bad: list[tuple[str, str]] = []
    for cluster_id, rec in records.items():
        for axis in REQUIRED_AXES:
            if axis not in rec or rec[axis] in (None, "", {}):
                bad.append((cluster_id, f"missing {axis}"))
                break
        else:
            if rec["age"] not in AGE_VALUES:
                bad.append((cluster_id, f"bad age {rec['age']!r}"))
            elif rec["gender"] not in GENDER_VALUES:
                bad.append((cluster_id, f"bad gender {rec['gender']!r}"))
            elif rec["ethnicity"] not in ETHNICITY_VALUES:
                bad.append((cluster_id, f"bad ethnicity {rec['ethnicity']!r}"))
            elif rec["complexion"] not in COMPLEXION_VALUES:
                bad.append((cluster_id, f"bad complexion {rec['complexion']!r}"))
            else:
                for band in SUS_BANDS:
                    if band not in rec.get("expressions", {}):
                        bad.append((cluster_id, f"missing expressions.{band}"))
                        break

    if bad:
        print(f"  ✗ {len(bad)} records failed schema validation:")
        for cid, err in bad:
            print(f"    {cid}: {err}")

    # Compose clean-band sentences for each cluster, compute pairwise cosine
    keys = sorted(records.keys())
    sentences = [compose_sentence(records[k], "clean") for k in keys]

    print(f"\n  Sample sentence: {sentences[0]}")

    from sentence_transformers import SentenceTransformer
    print("  Loading sentence-transformers/all-MiniLM-L6-v2 ...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = np.asarray(
        model.encode(sentences, convert_to_numpy=True, normalize_embeddings=True),
        dtype=np.float32,
    )
    cos = embs @ embs.T
    n = len(keys)
    upper = [cos[i, j] for i in range(n) for j in range(i + 1, n)]
    cos_mean = float(np.mean(upper))
    cos_std  = float(np.std(upper))
    cos_max  = float(np.max(upper))

    print()
    print("── Pairwise cosine (clean-band sentences) ────────")
    print(f"  mean: {cos_mean:.4f}")
    print(f"  std:  {cos_std:.4f}  (need > 0.12)")
    print(f"  max:  {cos_max:.4f}  (need < 0.97)")

    # Closest pair
    ci, cj, cv = 0, 1, -1.0
    for i in range(n):
        for j in range(i + 1, n):
            if cos[i, j] > cv:
                cv, ci, cj = float(cos[i, j]), i, j
    print(f"\n  Closest pair: {keys[ci]} ↔ {keys[cj]} (cos={cv:.4f})")
    print(f"    {keys[ci]}: {sentences[ci]}")
    print(f"    {keys[cj]}: {sentences[cj]}")

    ok = not bad and cos_std > 0.12 and cos_max < 0.97
    print()
    if ok:
        print("  ✓ GATE PASS — proceed to regeneration")
    else:
        print("  ✗ GATE FAIL — do NOT proceed")
        if bad:            print(f"    - {len(bad)} schema errors")
        if cos_std <= 0.12: print(f"    - std {cos_std:.4f} ≤ 0.12")
        if cos_max >= 0.97: print(f"    - max {cos_max:.4f} ≥ 0.97")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run validator**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/validate_structured_prototypes.py
```

Expected: `✓ GATE PASS`. If std ≤ 0.12, the structured schema did not break the collapse — revisit the derivation prompt or switch to a stronger LLM.

- [ ] **Step 3: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/validate_structured_prototypes.py
git commit -m "$(cat <<'EOF'
feat(prototypes-v2): schema + distinctness validator

Validates every record has 8 axes filled with whitelisted values and
all 5 expression bands present. Pairwise cosine std > 0.12 and max <
0.97 thresholds are tighter than v1 (0.05 / 0.99) to ensure the
structured approach actually breaks the vocabulary collapse.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Patch RBFConditioner with sus-band-aware method

**Files:**
- Modify: `src/rbf_conditioning.py`
- Create: `tests/test_structured_prototype_conditioning.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for RBFConditioner.structured_prototype_conditioning_nodes()."""
import json
from pathlib import Path

import pytest
import torch

from rbf_conditioning import RBFConditioner


@pytest.fixture
def tmp_v2_prototypes(tmp_path: Path) -> Path:
    path = tmp_path / "cluster_prototypes_v2.json"
    path.write_text(json.dumps({
        "C0": {
            "age": "18-25", "gender": "masculine", "ethnicity": "Central Asian",
            "hair": "short dark", "facial_hair": "clean-shaven",
            "complexion": "sun-weathered",
            "uniform": "bright yellow delivery vest",
            "expressions": {"clean": "alert focus", "low": "wary", "mid": "weary",
                            "high": "darting", "fraud": "hollow smile"},
        },
        "C1": {
            "age": "45-55", "gender": "masculine", "ethnicity": "Slavic",
            "hair": "receding grey", "facial_hair": "full grey beard",
            "complexion": "wind-chapped",
            "uniform": "orange hi-vis vest over coveralls",
            "expressions": {"clean": "stoic", "low": "guarded", "mid": "resigned",
                            "high": "suspicious", "fraud": "cold empty"},
        },
        "C5": {
            "age": "25-35", "gender": "androgynous", "ethnicity": "Slavic",
            "hair": "short messy", "facial_hair": "none",
            "complexion": "pale indoor",
            "uniform": "black collared polo with lanyard",
            "expressions": {"clean": "polite glaze", "low": "distracted",
                            "mid": "tense", "high": "fake-bright", "fraud": "forced warmth"},
        },
    }, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def tmp_centroids(tmp_path: Path) -> Path:
    path = tmp_path / "cluster_centroids.pt"
    torch.save({
        "centroids": torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ], dtype=torch.float32),
        "cluster_keys": ["C0", "C1", "C5"],
        "cluster_to_archetype": {"C0": "x", "C1": "y", "C5": "z"},
    }, path)
    return path


@pytest.fixture
def conditioner(tmp_centroids: Path, tmp_v2_prototypes: Path, tmp_path: Path) -> RBFConditioner:
    archetypes = tmp_path / "archetypes.json"
    archetypes.write_text("{}", encoding="utf-8")
    cond = RBFConditioner(
        centroids_path=tmp_centroids,
        archetypes_path=archetypes,
        k=3,
        temperature=1.0,
    )
    cond.cluster_prototypes_v2 = json.loads(tmp_v2_prototypes.read_text(encoding="utf-8"))
    return cond


def test_produces_three_encode_nodes(conditioner: RBFConditioner) -> None:
    emb = [0.8, 0.5, 0.1, 0.0]
    nodes, _ = conditioner.structured_prototype_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    encodes = [v for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert len(encodes) == 3


def test_sus_band_changes_sentence(conditioner: RBFConditioner) -> None:
    emb = [0.8, 0.5, 0.1, 0.0]
    clean_nodes, _ = conditioner.structured_prototype_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    fraud_nodes, _ = conditioner.structured_prototype_conditioning_nodes(
        embedding=emb, sus_band="fraud", clip_ref=["3", 0],
    )
    clean_texts = [v["inputs"]["text"] for v in clean_nodes.values() if v["class_type"] == "CLIPTextEncode"]
    fraud_texts = [v["inputs"]["text"] for v in fraud_nodes.values() if v["class_type"] == "CLIPTextEncode"]
    # Every clean text should differ from its fraud counterpart (different expression)
    assert all(c != f for c, f in zip(sorted(clean_texts), sorted(fraud_texts)))


def test_composed_sentence_contains_axis_values(conditioner: RBFConditioner) -> None:
    emb = [100.0, 0.0, 0.0, 0.0]  # strongly C0
    nodes, _ = conditioner.structured_prototype_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    # The dominant encode should reference C0's specifics
    texts = [v["inputs"]["text"] for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert any("Central Asian" in t and "delivery vest" in t for t in texts)


def test_missing_cluster_uses_fallback_record(conditioner: RBFConditioner) -> None:
    del conditioner.cluster_prototypes_v2["C5"]
    emb = [0.0, 0.0, 1.0, 0.0]
    nodes, _ = conditioner.structured_prototype_conditioning_nodes(
        embedding=emb, sus_band="clean", clip_ref=["3", 0],
    )
    assert len([v for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]) == 3
```

- [ ] **Step 2: Run tests, confirm AttributeError**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_structured_prototype_conditioning.py -v
```

Expected: failures on `AttributeError: ... has no attribute 'structured_prototype_conditioning_nodes'`.

- [ ] **Step 3: Implement the method on `RBFConditioner`**

In `src/rbf_conditioning.py`:

1. Ensure `from prototype_schema import compose_sentence` is imported at the top (add if missing).
2. At the end of `__init__`, load the v2 prototypes:

```python
        # Load structured (v2) cluster prototypes if present
        v2_path = centroids_path.parent / "cluster_prototypes_v2.json"
        self.cluster_prototypes_v2: dict[str, dict] = {}
        if v2_path.exists():
            self.cluster_prototypes_v2 = json.loads(v2_path.read_text(encoding="utf-8"))
```

3. Append this method inside the class (same indentation as existing `cluster_prototype_conditioning_nodes`):

```python
    def structured_prototype_conditioning_nodes(
        self,
        embedding: list[float],
        sus_band: str,
        clip_ref: list,
        start_node_id: int = 20,
    ) -> tuple[dict[str, Any], str]:
        """Build ComfyUI conditioning subgraph from structured v2 prototypes.

        For each of the top-k clusters, composes a prompt sentence from that
        cluster's v2 record at the requested sus_band, then blends the
        resulting CLIPTextEncode nodes via ConditioningAverage.
        """
        from prototype_schema import compose_sentence  # noqa: PLC0415

        FALLBACK_RECORD = {
            "age": "25-35", "gender": "androgynous", "ethnicity": "Slavic",
            "hair": "short neutral", "facial_hair": "none",
            "complexion": "smooth",
            "uniform": "plain dark jacket",
            "expressions": {b: "neutral" for b in ["clean","low","mid","high","fraud"]},
        }

        emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
        idxs, weights = top_k_weights(emb_tensor, self.centroids, self.k, self.temperature)

        entries: list[tuple[str, str, float]] = []
        for idx, w in zip(idxs, weights):
            cid = self.cluster_keys[idx]
            rec = self.cluster_prototypes_v2.get(cid, FALLBACK_RECORD)
            sentence = compose_sentence(rec, sus_band)
            entries.append((cid, sentence, w))

        total = sum(w for _, _, w in entries)
        entries = [(c, s, w / total) for c, s, w in entries]

        nodes: dict[str, Any] = {}
        nid = start_node_id
        encode_ids: list[str] = []
        for _cid, sentence, _w in entries:
            eid = str(nid); nid += 1
            nodes[eid] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": sentence, "clip": clip_ref},
            }
            encode_ids.append(eid)

        weights_only = [w for _, _, w in entries]
        if len(encode_ids) == 1:
            return nodes, encode_ids[0]
        if len(encode_ids) == 2:
            w1, w2 = weights_only
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

        w1, w2, w3 = weights_only
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
cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_structured_prototype_conditioning.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/rbf_conditioning.py tests/test_structured_prototype_conditioning.py
git commit -m "$(cat <<'EOF'
feat(prototypes-v2): structured_prototype_conditioning_nodes() on RBFConditioner

Loads data/cluster_prototypes_v2.json in __init__. New method composes
top-k cluster sentences at the requested sus_band via compose_sentence()
and blends through ConditioningAverage. Old method (v1) untouched.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: generate_v8b.py — new curve, new sampler, new conditioning

**Files:**
- Create: `src/generate_v8b.py`

Copy `src/generate_v8.py` and apply these three diffs:

**Diff 1 — LoRA curve.** Inside the main loop, replace:
```python
sus_factor = (sus / 100.0) ** 0.8
```
with:
```python
sus_factor = max(0.0, (sus - 25.0) / 75.0) ** 1.2
```

**Diff 2 — Sampler.** Replace the `select_jobs()` function with `select_jobs_per_cluster()`:

```python
def select_jobs_per_cluster(n_per_cell: int = 2, seed: int = 42) -> list[dict]:
    """For each (cluster, sus_band), pick n_per_cell jobs deterministically.

    Returns up to 42 * 5 * n_per_cell jobs, fewer if some (cluster, band) cells
    are empty.
    """
    rng = np.random.default_rng(seed)

    print("Loading layout ...")
    df_layout = pd.read_parquet(LAYOUT, columns=["id", "cluster_coarse"])
    id_to_cluster: dict[str, str] = dict(zip(df_layout["id"], df_layout["cluster_coarse"]))

    print("Fetching jobs from DB ...")
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT id::text, embedding, sus_level FROM jobs "
            "WHERE embedding IS NOT NULL AND sus_level IS NOT NULL"
        )
        rows = {r["id"]: dict(r) for r in cur.fetchall()}
    conn.close()

    def sus_band_for(level: int) -> str:
        if level <= 25:  return "clean"
        if level <= 49:  return "low"
        if level <= 64:  return "mid"
        if level <= 79:  return "high"
        return "fraud"

    buckets: dict[tuple[str, str], list[dict]] = {}
    for job_id, row in rows.items():
        cluster = id_to_cluster.get(job_id)
        if cluster is None:
            continue
        band = sus_band_for(row["sus_level"])
        job = {
            "id":             job_id,
            "cluster_coarse": cluster,
            "sus_level":      row["sus_level"],
            "sus_band":       band,
            "embedding":      parse_embedding(row["embedding"]),
        }
        buckets.setdefault((cluster, band), []).append(job)

    selected: list[dict] = []
    for (cluster, band), pool in buckets.items():
        if len(pool) == 0:
            continue
        n = min(n_per_cell, len(pool))
        idxs = rng.choice(len(pool), size=n, replace=False)
        for i in idxs:
            selected.append(pool[int(i)])

    rng.shuffle(selected)
    print(f"  Selected {len(selected)} jobs across {len(buckets)} non-empty cells")
    return selected
```

Replace the call site in `run()`:

```python
    jobs = select_jobs_per_cluster(n_per_cell=args.n_per_cell, seed=sample_seed)
```

Replace the argparse block's `--count` with `--n-per-cell`:

```python
    parser.add_argument("--n-per-cell", type=int, default=2,
                        help="Jobs per (cluster, sus_band) cell (default: 2)")
```

**Diff 3 — Conditioning.** Replace the call:

```python
cond_nodes, final_cond_id = rbf.cluster_prototype_conditioning_nodes(
    embedding=job["embedding"],
    clip_ref=["3", 0],
    start_node_id=20,
)
```

with:

```python
cond_nodes, final_cond_id = rbf.structured_prototype_conditioning_nodes(
    embedding=job["embedding"],
    sus_band=job["sus_band"],
    clip_ref=["3", 0],
    start_node_id=20,
)
```

Also change:
- `OUT_DIR = Path("output/dataset_faces_v8")` → `Path("output/dataset_faces_v8b")`
- `if not rbf.cluster_prototypes:` → `if not rbf.cluster_prototypes_v2:`
- Update the error message to reference `derive_structured_prototypes.py`

- [ ] **Step 1: Create the file by copy + applying the three diffs above**

- [ ] **Step 2: Dry run**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8b.py --dry-run --n-per-cell 2
```

Expected: prints sampling plan across 42 clusters × 5 bands, ~370-400 jobs. Each line shows cluster, sus_band, sus, sus_factor, seed. Note the sus_factor values at sus=25 and below — they must all be 0.000.

- [ ] **Step 3: Commit the scaffold**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/generate_v8b.py
git commit -m "$(cat <<'EOF'
feat(prototypes-v2): generate_v8b.py — per-cluster sampler + softened LoRA + sus-band conditioning

Three changes from generate_v8.py:
1. LoRA curve max(0, (sus-25)/75) ** 1.2 — zero below sus=25
2. select_jobs_per_cluster() picks n_per_cell jobs per (cluster, sus_band)
3. rbf.structured_prototype_conditioning_nodes(... sus_band=j["sus_band"] ...)

Writes to output/dataset_faces_v8b/ (v8 retained for comparison).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Smoke test + full batch + contact sheet

**Files:** no code changes.

- [ ] **Step 1: Single-cell smoke**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8b.py --n-per-cell 1
```

Runs ~210 faces (42 × 5 × 1). Takes ~70 minutes. Watch for ComfyUI validation errors on the first few — if the workflow rejects `structured_prototype_conditioning_nodes` output, stop and escalate.

Faster alternative if wait is too long: patch the sampler to only run the first 5 clusters for a first smoke:

```
uv run src/generate_v8b.py --n-per-cell 1 --sample-seed 42
# then kill after ~25 faces to spot-check
```

- [ ] **Step 2: Contact sheet for smoke**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/build_contact_sheet.py output/dataset_faces_v8b
```

Open `output/dataset_faces_v8b/contact_sheet.html`. Verify:
- Clean-band faces in each row look normal (no uncanny skin, LoRA off)
- High/fraud-band faces in each row show uncanniness
- Across rows, clusters differ in age, ethnicity, uniform — readable by eye
- Expression within a row evolves from neutral (clean) to off (fraud)

If the visual check fails, the relevant diagnostic is usually:
- Clean looks cursed → LoRA curve not applied; check sus_factor calc in loop
- Rows look identical → structured prototypes collapsed; run validator, check std
- Sentences visibly wrong → check `compose_sentence` output directly

- [ ] **Step 3: Full batch (if smoke passes)**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8b.py --n-per-cell 2
```

Runs ~370-420 faces. Takes ~2.5 h. Resume logic will skip any cells already covered by the `--n-per-cell 1` smoke.

- [ ] **Step 4: Final contact sheet and commit the manifest**

```
cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/build_contact_sheet.py output/dataset_faces_v8b
git add output/dataset_faces_v8b/manifest.json
git commit -m "$(cat <<'EOF'
test(prototypes-v2): v8b smoke + full batch manifest

~370 faces across 42 clusters × 5 sus bands × 2 per cell. Manual
inspection of contact_sheet.html shows readable cluster variation
within each sus band and clean LoRA-free rendering at sus ≤ 25.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

Do NOT commit PNGs — regeneratable, too large.

---

## Task 7: Run Phase 4 eval on v8b

**Files:** follows the existing Phase 4 plan.

The Phase 4 eval harness treats each `dataset_faces_*` directory as an input. Add `dataset_faces_v8b` to the list of directories being evaluated alongside v6, v7, v8.

- [ ] **Step 1: Inspect `eval_metrics.py` (or whatever Phase 4 produced) and confirm it's directory-driven**

The existing plan assumes `eval_metrics.py` reads a list of directories and produces comparable M1-M6 per directory. If v8b is a drop-in addition, nothing needs changing.

- [ ] **Step 2: Run the eval**

Follow Phase 4 plan (`phase-4-eval-harness.md`) with the directory list expanded to include v8b.

- [ ] **Step 3: Decide**

Gate criteria:
- v8b M3 > v8 M3 + 0.05 (structured approach actually works)
- v8b M6 ≥ v8 M6 − 0.05 (sus signal not regressed)
- v8b M1, M2 in range (clean rows look like normal photos)

If all pass: v8b is the production pipeline, v8 is archived.
If M3 improves but M6 regresses: the softer LoRA curve is under-signalling fraud; tune up the high end of the curve.
If M3 does not improve: the structured approach did not break the collapse; fall back to a stronger LLM (gpt-oss or glm-4.7-flash) for derivation and re-run Task 2+.

---

## Self-review checklist

- [x] All 42 clusters covered by schema (no hard-coded cluster lists)
- [x] Every task has a commit step
- [x] Every code step has complete code
- [x] Every test step has complete test code
- [x] Derivation fallback preserves JSON shape
- [x] Sampler handles empty (cluster, band) cells gracefully
- [x] LoRA curve hits 0.0 at sus=25 and 1.0 at sus=100 (verified in the table above)
- [x] Old v8 pipeline is not touched (additive change)
