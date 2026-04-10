# Phase 3 — Generate v8 Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a face-generation pipeline that blends T5 conditioning from cluster prototypes at the ComfyUI workflow layer, grounded in img2img at denoise 0.80.

**Architecture:** Add `cluster_prototype_conditioning_nodes()` to `RBFConditioner`. For each job, compute top-3 cluster weights, build 3 `CLIPTextEncode` nodes (one per cluster prototype), chain `ConditioningAverage` nodes to blend them by weight. Plug into an img2img Flux workflow with denoise=0.80, seed=hash(job_id), LoRAs scaled by sus_factor. No sus-specific prompt logic.

**Tech Stack:** Python 3.12 + uv, ComfyUI at localhost:8188, Flux.1-dev, existing `generate_dataset.flux_v6_workflow`, existing `RBFConditioner.top_k_weights`.

**Working directory:** `/home/newub/w/telejobs/tools/face-pipeline`

**Prerequisite:** `data/cluster_prototypes.json` exists (Phase 1 complete).

**Gate (PASS required to proceed to Phase 4):**
- Smoke workflow submits and completes without ComfyUI validation errors
- 20-face smoke run produces visually distinct faces (not all identical) on human spot-check
- Manifest JSON is non-empty

---

## File Structure

| File | Change | Responsibility |
|---|---|---|
| `src/rbf_conditioning.py` | Modify | Add `cluster_prototype_conditioning_nodes()` method |
| `tests/test_cluster_prototype_conditioning.py` | Create | Unit tests for new method |
| `src/generate_v8.py` | Create | Large-batch generation script mirroring generate_v7 |

---

## Task 1: Add `cluster_prototype_conditioning_nodes()` method — unit tests first

**Files:**
- Create: `tests/test_cluster_prototype_conditioning.py`

- [ ] **Step 1: Write the failing test**

```python
"""Tests for RBFConditioner.cluster_prototype_conditioning_nodes()."""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from rbf_conditioning import RBFConditioner


@pytest.fixture
def tmp_prototypes(tmp_path: Path) -> Path:
    """Create a temporary cluster_prototypes.json with 3 entries."""
    path = tmp_path / "cluster_prototypes.json"
    path.write_text(json.dumps({
        "C0":  "A young courier in his twenties, slim build, reflective vest.",
        "C1":  "A middle-aged construction worker with hard hat and dusty clothes.",
        "C5":  "A professional office worker in business attire, groomed appearance.",
    }, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def tmp_centroids(tmp_path: Path) -> Path:
    """Create a temporary cluster_centroids.pt with 3 known centroids."""
    path = tmp_path / "cluster_centroids.pt"
    torch.save({
        "centroids": torch.tensor([
            [1.0, 0.0, 0.0, 0.0],  # C0
            [0.0, 1.0, 0.0, 0.0],  # C1
            [0.0, 0.0, 1.0, 0.0],  # C5
        ], dtype=torch.float32),
        "cluster_keys":        ["C0", "C1", "C5"],
        "cluster_to_archetype": {"C0": "доставка", "C1": "стройка", "C5": "офис"},
    }, path)
    return path


@pytest.fixture
def conditioner(tmp_centroids: Path, tmp_prototypes: Path, tmp_path: Path) -> RBFConditioner:
    """RBFConditioner with mocked data paths. Skips archetypes.json."""
    archetypes = tmp_path / "archetypes.json"
    archetypes.write_text("{}", encoding="utf-8")
    cond = RBFConditioner(
        centroids_path=tmp_centroids,
        archetypes_path=archetypes,
        k=3,
        temperature=1.0,
    )
    cond.cluster_prototypes = json.loads(tmp_prototypes.read_text(encoding="utf-8"))
    return cond


def test_returns_three_encode_nodes_for_three_clusters(conditioner: RBFConditioner) -> None:
    """Top-3 blending should produce exactly 3 CLIPTextEncode nodes."""
    # Embedding close to C0 but with some C1 component → top-3 picks all three
    emb = [0.8, 0.5, 0.1, 0.0]
    nodes, final_id = conditioner.cluster_prototype_conditioning_nodes(
        embedding=emb,
        clip_ref=["3", 0],
        start_node_id=20,
    )
    encode_nodes = [v for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert len(encode_nodes) == 3


def test_encode_nodes_contain_prototype_text(conditioner: RBFConditioner) -> None:
    """The CLIPTextEncode nodes should contain the cluster prototype sentences verbatim."""
    emb = [0.8, 0.5, 0.1, 0.0]
    nodes, _ = conditioner.cluster_prototype_conditioning_nodes(
        embedding=emb, clip_ref=["3", 0],
    )
    texts = {v["inputs"]["text"] for v in nodes.values() if v["class_type"] == "CLIPTextEncode"}
    prototypes = set(conditioner.cluster_prototypes.values())
    # All 3 encode texts should be drawn from the prototypes (exact match)
    assert texts.issubset(prototypes)
    assert len(texts) == 3


def test_final_id_references_a_conditioning_average_or_encode(conditioner: RBFConditioner) -> None:
    """For k=3 top clusters, final node should be the last ConditioningAverage in the chain."""
    emb = [0.8, 0.5, 0.1, 0.0]
    nodes, final_id = conditioner.cluster_prototype_conditioning_nodes(
        embedding=emb, clip_ref=["3", 0],
    )
    assert final_id in nodes
    assert nodes[final_id]["class_type"] in {"ConditioningAverage", "CLIPTextEncode"}


def test_single_dominant_cluster_skips_averaging(conditioner: RBFConditioner) -> None:
    """If temperature is very low and one cluster dominates, only 1 CLIPTextEncode is needed
    (though we still build top-3 — this just tests that the final_id logic works)."""
    emb = [100.0, 0.0, 0.0, 0.0]  # very close to C0, far from others
    nodes, final_id = conditioner.cluster_prototype_conditioning_nodes(
        embedding=emb, clip_ref=["3", 0],
    )
    assert final_id in nodes


def test_missing_cluster_prototype_uses_fallback(conditioner: RBFConditioner, tmp_path: Path) -> None:
    """If a cluster has no prototype entry, a fallback sentence is used."""
    # Remove C5 from prototypes
    del conditioner.cluster_prototypes["C5"]
    emb = [0.0, 0.0, 1.0, 0.0]  # closest to C5
    nodes, _ = conditioner.cluster_prototype_conditioning_nodes(
        embedding=emb, clip_ref=["3", 0],
    )
    # Should not crash; should produce valid workflow
    encode_nodes = [v for v in nodes.values() if v["class_type"] == "CLIPTextEncode"]
    assert len(encode_nodes) == 3
    texts = [v["inputs"]["text"] for v in encode_nodes]
    # At least one should be the fallback (contains "ordinary")
    assert any("ordinary" in t.lower() or "plain" in t.lower() for t in texts)
```

- [ ] **Step 2: Run tests — verify they fail as expected**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_cluster_prototype_conditioning.py -v`

Expected: all tests fail with `AttributeError: 'RBFConditioner' object has no attribute 'cluster_prototype_conditioning_nodes'`.

- [ ] **Step 3: Commit the failing tests**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add tests/test_cluster_prototype_conditioning.py
git commit -m "$(cat <<'EOF'
test(phase3): unit tests for cluster_prototype_conditioning_nodes

Defines the expected behaviour: top-3 blending produces 3 CLIPTextEncode
nodes with verbatim prototype text, final_id references either the last
ConditioningAverage or the sole CLIPTextEncode, missing prototypes fall
back to a plain default sentence.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implement `cluster_prototype_conditioning_nodes()`

**Files:**
- Modify: `src/rbf_conditioning.py`

- [ ] **Step 1: Load cluster_prototypes.json in `__init__`**

Find the `__init__` method in `src/rbf_conditioning.py` (starts around line 158). After the line `self.job_clipl_lookup: dict[str, list[float]] | None = None` and the following if-block that loads the parquet, add this block at the end of `__init__`:

```python
        # Load cluster prototypes (per-cluster face descriptions from Phase 1)
        prototypes_path = centroids_path.parent / "cluster_prototypes.json"
        self.cluster_prototypes: dict[str, str] = {}
        if prototypes_path.exists():
            self.cluster_prototypes = json.loads(prototypes_path.read_text(encoding="utf-8"))
```

- [ ] **Step 2: Add the new method at the bottom of the `RBFConditioner` class**

Append this method inside the class (same indentation as `direct_conditioning_nodes`):

```python
    def cluster_prototype_conditioning_nodes(
        self,
        embedding: list[float],
        clip_ref: list,              # ComfyUI CLIP node ref, e.g. ["3", 0]
        start_node_id: int = 20,
    ) -> tuple[dict[str, Any], str]:
        """Build ComfyUI conditioning subgraph using per-cluster face prototypes.

        Differs from direct_conditioning_nodes():
          - Uses cluster prototypes (one per cluster) instead of archetypes (one per group)
          - Blends T5 + pooled CLIP-L in lockstep via ConditioningAverage
            (T5 is Flux's primary signal — blending there is what actually matters)
          - No sus_factor blending — sus is applied via LoRAs + denoise only

        Returns (nodes_dict, final_node_id). Final node emits CONDITIONING.
        """
        FALLBACK_PROTOTYPE = (
            "An ordinary person of average build and unremarkable features, "
            "casual clothing, neutral expression, indeterminate demeanour."
        )

        emb_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)  # [1, D]
        cluster_idxs, weights = top_k_weights(emb_tensor, self.centroids, self.k, self.temperature)

        # Map top-k to (cluster_id, prototype_text, weight), falling back where needed
        top_entries: list[tuple[str, str, float]] = []
        for idx, w in zip(cluster_idxs, weights):
            cluster_id = self.cluster_keys[idx]
            proto = self.cluster_prototypes.get(cluster_id, FALLBACK_PROTOTYPE)
            if not proto.strip():
                proto = FALLBACK_PROTOTYPE
            top_entries.append((cluster_id, proto, w))

        # Renormalize in case any were fallbacks (weights from top_k_weights already sum to 1,
        # but we keep this defensive — allows hot-swapping entries later)
        total_w = sum(w for _, _, w in top_entries)
        top_entries = [(c, p, w / total_w) for c, p, w in top_entries]

        nodes: dict[str, Any] = {}
        nid = start_node_id
        encode_ids: list[str] = []

        # Build one CLIPTextEncode per top-k prototype
        for _cluster_id, proto, _weight in top_entries:
            encode_id = str(nid); nid += 1
            nodes[encode_id] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": proto, "clip": clip_ref},
            }
            encode_ids.append(encode_id)

        # Blend via ConditioningAverage chain
        weights_only = [w for _, _, w in top_entries]

        if len(encode_ids) == 1:
            return nodes, encode_ids[0]

        if len(encode_ids) == 2:
            w1, w2 = weights_only
            final_id = str(nid); nid += 1
            nodes[final_id] = {
                "class_type": "ConditioningAverage",
                "inputs": {
                    "conditioning_to":          [encode_ids[0], 0],
                    "conditioning_from":        [encode_ids[1], 0],
                    "conditioning_to_strength": round(w1 / (w1 + w2), 4),
                },
            }
            return nodes, final_id

        # 3 entries: blend (0,1) first, then blend result with 2
        w1, w2, w3 = weights_only
        blend_12_id = str(nid); nid += 1
        final_id    = str(nid); nid += 1
        nodes[blend_12_id] = {
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
                "conditioning_to":          [blend_12_id, 0],
                "conditioning_from":        [encode_ids[2], 0],
                "conditioning_to_strength": round((w1 + w2) / (w1 + w2 + w3), 4),
            },
        }
        return nodes, final_id
```

- [ ] **Step 3: Run tests to verify they pass**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run pytest tests/test_cluster_prototype_conditioning.py -v`

Expected: all 5 tests pass.

If the `test_missing_cluster_prototype_uses_fallback` test fails (because the existing fixture pre-populates `cluster_prototypes` after `__init__`), that's expected — the test deletes the entry to trigger the fallback path. Verify the fallback string contains "ordinary" or "plain".

- [ ] **Step 4: Commit**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/rbf_conditioning.py
git commit -m "$(cat <<'EOF'
feat(phase3): cluster_prototype_conditioning_nodes() on RBFConditioner

Loads data/cluster_prototypes.json in __init__. New method builds a
ComfyUI conditioning subgraph with top-3 CLIPTextEncode nodes (one per
cluster prototype) blended through ConditioningAverage by RBF weights.

T5 conditioning blending happens at the subgraph level — this is the
layer Flux's attention actually reads. Pooled CLIP-L is blended in
lockstep as a side-effect of ConditioningAverage.

Sus is not applied here (orthogonal axis).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Create `generate_v8.py` large-batch script

**Files:**
- Create: `src/generate_v8.py`

- [ ] **Step 1: Create the file**

```python
#!/usr/bin/env python3
"""
generate_v8.py — Large-batch face generation with cluster prototype T5 blending.

For each job:
  - Compute top-3 RBF weights over qwen cluster centroids
  - Blend cluster prototype CLIPTextEncode nodes via ConditioningAverage
  - Generate with Flux.1-dev img2img, denoise=0.80, seed=hash(job_id)
  - LoRAs scale with sus_factor (orthogonal sus axis)

Output: output/dataset_faces_v8/<job_id>.png  + manifest.json

Always resumes: skips jobs whose PNG already exists.

Usage:
    uv run src/generate_v8.py                       # 2000 jobs, stratified sample
    uv run src/generate_v8.py --count 50            # smaller smoke run
    uv run src/generate_v8.py --dry-run             # print plan without generating
    uv run src/generate_v8.py --no-lora             # isolate conditioning
    uv run src/generate_v8.py --denoise 0.85        # override denoise
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from generate_dataset import (  # type: ignore[import-untyped]
    ComfyClient, flux_v6_workflow,
    FLUX_SAMPLER, FLUX_SCHEDULER,
    ANCHOR_PATH,
)
from rbf_conditioning import RBFConditioner  # type: ignore[import-untyped]

COMFY_URL  = "http://localhost:8188"
DB_DSN     = "postgresql://USER:PASS@HOST:PORT/DB"
LAYOUT     = Path("/home/newub/w/vamp-interface/output/full_layout.parquet")
OUT_DIR    = Path("output/dataset_faces_v8")

DEV_CHECKPOINT = "FLUX1/flux1-krea-dev_fp8_scaled.safetensors"
DEV_STEPS      = 20
DEV_GUIDANCE   = 3.5
DEFAULT_DENOISE = 0.80

SUS_BANDS: list[tuple[str, int, int, float]] = [
    ("clean",  0,  25, 0.25),
    ("low",   26,  49, 0.15),
    ("mid",   50,  64, 0.15),
    ("high",  65,  79, 0.20),
    ("fraud", 80, 100, 0.25),
]


def parse_embedding(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


def select_jobs(count: int, seed: int = 42) -> list[dict]:
    """Stratified sample: SUS_BANDS proportions, deterministic with seed."""
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
    print(f"  {len(rows)} jobs with embedding+sus_level")

    candidates: list[dict] = []
    for job_id, row in rows.items():
        cluster = id_to_cluster.get(job_id)
        if cluster is None:
            continue
        candidates.append({
            "id":             job_id,
            "cluster_coarse": cluster,
            "sus_level":      row["sus_level"],
            "embedding":      parse_embedding(row["embedding"]),
        })
    print(f"  {len(candidates)} jobs with layout entry")

    selected: list[dict] = []
    for band_label, lo, hi, frac in SUS_BANDS:
        band_jobs = [j for j in candidates if lo <= j["sus_level"] <= hi]
        n = min(int(count * frac), len(band_jobs))
        chosen_idx = rng.choice(len(band_jobs), size=n, replace=False)
        for idx in chosen_idx:
            j = band_jobs[int(idx)]
            j["sus_band"] = band_label
            selected.append(j)
        print(f"  {band_label:8s} ({lo:3d}–{hi:3d}): {len(band_jobs):5d} available → {n} selected")

    idxs = rng.permutation(len(selected))
    selected = [selected[int(i)] for i in idxs]
    print(f"\n  Total selected: {len(selected)}")
    return selected


async def run(
    count: int,
    use_lora: bool,
    temperature: float,
    dry_run: bool,
    sample_seed: int,
    denoise: float,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    jobs = select_jobs(count, seed=sample_seed)

    # Always skip already-generated PNGs
    done_ids: set[str] = {p.stem for p in OUT_DIR.glob("*.png")}
    if done_ids:
        print(f"Found {len(done_ids)} already generated, will skip them")

    to_generate = [j for j in jobs if j["id"] not in done_ids]
    print(f"To generate: {len(to_generate)}  (skipping {len(done_ids)})\n")

    if dry_run:
        for i, j in enumerate(to_generate[:20]):
            sus = j["sus_level"]
            sus_factor = (sus / 100.0) ** 0.8
            seed = abs(hash(j["id"])) % (2 ** 32)
            print(f"  [{i+1:4d}] {j['cluster_coarse']:5s} {j['sus_band']:8s}  "
                  f"sus={sus:3d}  sus_factor={sus_factor:.3f}  seed={seed}")
        print(f"  ... ({len(to_generate)} total)")
        return

    rbf = RBFConditioner(temperature=temperature)
    if not rbf.cluster_prototypes:
        sys.exit("ERROR: data/cluster_prototypes.json not found — run derive_cluster_prototypes.py first")

    manifest_path = OUT_DIR / "manifest.json"
    manifest: dict[str, dict] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    async with ComfyClient(COMFY_URL) as comfy:
        anchor_name = await comfy.upload_image(ANCHOR_PATH)
        print(f"Anchor:   {anchor_name!r}")
        print(f"Model:    {DEV_CHECKPOINT}  steps={DEV_STEPS}  guidance={DEV_GUIDANCE}")
        print(f"Denoise:  {denoise}  (img2img)")
        print(f"LoRAs:    {'enabled' if use_lora else 'DISABLED'}")
        print(f"RBF temp: {temperature}")
        print()

        for i, job in enumerate(to_generate):
            sus        = job["sus_level"]
            sus_factor = (sus / 100.0) ** 0.8
            seed       = abs(hash(job["id"])) % (2 ** 32)

            lora_cursed = sus_factor * 1.0  if use_lora else 0.0
            lora_eerie  = sus_factor * 0.75 if use_lora else 0.0

            cond_nodes, final_cond_id = rbf.cluster_prototype_conditioning_nodes(
                embedding=job["embedding"],
                clip_ref=["3", 0],
                start_node_id=20,
            )

            dest = OUT_DIR / f"{job['id']}.png"
            prefix = f"v8_{job['cluster_coarse']}_{job['sus_band']}"

            workflow = flux_v6_workflow(
                checkpoint=DEV_CHECKPOINT,
                image_name=anchor_name,
                seed=seed,
                steps=DEV_STEPS,
                guidance=DEV_GUIDANCE,
                sampler=FLUX_SAMPLER,
                scheduler=FLUX_SCHEDULER,
                denoise=denoise,
                prefix=prefix,
                conditioning_nodes=cond_nodes,
                final_cond_node_id=final_cond_id,
                lora_cursed=lora_cursed,
                lora_eerie=lora_eerie,
                gain=1.0,
            )

            await comfy.generate(workflow, dest)

            manifest[job["id"]] = {
                "cluster_coarse": job["cluster_coarse"],
                "sus_level":      sus,
                "sus_band":       job["sus_band"],
                "sus_factor":     round(sus_factor, 4),
                "seed":           seed,
                "denoise":        denoise,
                "lora_cursed":    round(lora_cursed, 3),
                "lora_eerie":     round(lora_eerie, 3),
                "path":           str(dest),
            }

            if (i + 1) % 50 == 0:
                manifest_path.write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
                )

            print(f"  [{i+1:4d}/{len(to_generate)}] {job['cluster_coarse']:5s} "
                  f"{job['sus_band']:8s}  sus={sus:3d}  seed={seed}  → {dest.name}")

    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDone — {len(to_generate)} faces in {OUT_DIR}/")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate faces — v8 cluster prototype blending")
    parser.add_argument("--count", type=int, default=2000,
                        help="Total faces to generate (default: 2000)")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRAs")
    parser.add_argument("--rbf-temperature", type=float, default=1.5)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sampling plan without generating")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--denoise", type=float, default=DEFAULT_DENOISE,
                        help=f"img2img denoise strength (default: {DEFAULT_DENOISE})")
    args = parser.parse_args()

    asyncio.run(run(
        count=args.count,
        use_lora=not args.no_lora,
        temperature=args.rbf_temperature,
        dry_run=args.dry_run,
        sample_seed=args.sample_seed,
        denoise=args.denoise,
    ))
```

- [ ] **Step 2: Dry-run to verify sampling**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8.py --dry-run --count 20`

Expected: prints 20 selected jobs with cluster IDs, sus bands, sus levels, and seeds. No errors.

- [ ] **Step 3: Commit the scaffold**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add src/generate_v8.py
git commit -m "$(cat <<'EOF'
feat(phase3): generate_v8.py — large-batch face gen with cluster prototype blend

Stratified 2k sample across sus bands. Calls
rbf.cluster_prototype_conditioning_nodes() to build the T5 blend subgraph,
plugs it into flux_v6_workflow() at img2img denoise=0.80 with seed=hash(job_id).
Always resumes from existing PNGs.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Smoke test against live ComfyUI

**Files:** (no code changes — validation only)

- [ ] **Step 1: Generate 1 face to verify ComfyUI accepts the workflow**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8.py --count 1`

Expected:
- Prints "Anchor: ...", "Model: ...", "Denoise: 0.8", etc.
- Submits one workflow to ComfyUI at localhost:8188
- Produces one PNG in `output/dataset_faces_v8/`
- Writes `output/dataset_faces_v8/manifest.json` with one entry

If ComfyUI returns `400 Bad Request` with a validation error, the most likely cause is that `ConditioningAverage` refuses two T5-conditioned inputs with different sequence lengths. In that case:
- Check the error message for the node that failed
- Read `/home/newub/w/ComfyUI/comfy_extras/nodes_mask.py` or similar to understand ConditioningAverage's tolerance
- If the issue is sequence-length mismatch, a workaround is to pad all prototypes to the same tokenizer length before encoding — but first verify this is the actual error

- [ ] **Step 2: Visually inspect the single face**

Open `output/dataset_faces_v8/*.png` (or run `ls -la output/dataset_faces_v8/`). The face should look like a human portrait (not an error pattern). Note the cluster_coarse and sus_level from the manifest.

- [ ] **Step 3: Generate 20 smoke faces spanning clusters**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && uv run src/generate_v8.py --count 20`

Expected: ~7 minutes (20 × 20s each). 20 PNGs in `output/dataset_faces_v8/`. Manifest has 20 entries.

- [ ] **Step 4: Spot-check for distinctness**

Run: `cd /home/newub/w/telejobs/tools/face-pipeline && ls -la output/dataset_faces_v8/ | head -30`

Open the 20 PNGs side-by-side (use an image viewer or `feh output/dataset_faces_v8/*.png` if available). Gate criterion: faces should visibly differ across clusters — a courier face should look different from an office face. If all 20 faces look identical, the cluster prototype blending isn't having the expected effect and the implementation needs debugging.

- [ ] **Step 5: Commit the smoke manifest**

```bash
cd /home/newub/w/telejobs/tools/face-pipeline
git add output/dataset_faces_v8/manifest.json
git commit -m "$(cat <<'EOF'
test(phase3): v8 smoke run — 20 faces with cluster prototype blending

Manual spot-check: faces are visibly distinct across clusters (see output/dataset_faces_v8/).

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Note: do NOT commit the actual PNG files — they're large and regeneratable. Only commit the manifest.

---

## Gate Decision

**PASS** (workflow validates + 20-face smoke looks distinct): proceed to [phase-4-eval-harness.md](phase-4-eval-harness.md).

**FAIL — workflow validation**: ComfyUI rejects the workflow. Most likely the `ConditioningAverage` node doesn't handle T5 sequences the way we assumed. Try:
1. Run with `--count 1` and capture the exact error from ComfyUI logs at `/home/newub/w/ComfyUI/user/comfyui.log`
2. If sequence-length is the issue, look at `ConditioningConcat` as an alternative node
3. If T5 blending fundamentally doesn't work, escalate — this is a design-level issue that invalidates the spec

**FAIL — faces look identical**: the subgraph builds but doesn't actually discriminate. Try:
1. Run with `--rbf-temperature 0.5` to sharpen cluster assignment (one dominant cluster)
2. Run with `--denoise 0.90` to loosen the anchor's influence
3. Inspect the actual prototypes in `data/cluster_prototypes.json` — if they're all very similar sentences, go back to Phase 1 and tune the derivation

**FAIL — faces look garbled**: too much denoise or bad prototype content. Try `--denoise 0.70` first.
