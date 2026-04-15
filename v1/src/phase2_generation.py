#!/usr/bin/env python3
"""
Phase 2: wider denoising range + richer demographic axes.

Changes from Phase 1:
  - Denoising: 0.35 (sus=0) → 0.65 (sus=100). Floor raised so legit faces
    already show meaningful variation from anchor; ceiling raised for stronger
    uncanny effect on fraud.
  - Demographics now driven by PC1/PC2/PC3 with explicit archetypes. Cluster
    membership should be immediately visible before uncanny signal registers.
  - Identity channel (age/gender/hair/build) is orthogonal to expression
    channel (sus/uncanny). Both are additive in the prompt.
  - 8 jobs per cluster (vs 5) for more visual variety in comparison grid.

Usage:
    uv run src/phase2_generation.py
    uv run src/phase2_generation.py --dry-run
    uv run src/phase2_generation.py --jobs-per-cluster 5
"""

import argparse
import asyncio
import hashlib
import json
import math
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
import psycopg2
import psycopg2.extras
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

DB_DSN = "postgresql://USER:PASS@HOST:PORT/DB"
COMFY_URL = "http://localhost:8188"
OUT_DIR = Path("output/phase2")

# Reuse phase1 anchor (same neutral face, same seed = calibration baseline)
ANCHOR_DIR = Path("output/phase1")
ANCHOR_FILENAME = "phase1_anchor.png"

CLUSTERS = [
    {"label": "courier_legit",   "work_type": "доставка", "sus_max": 30},
    {"label": "warehouse_legit", "work_type": "склад",    "sus_max": 30},
    {"label": "office_legit",    "work_type": "офис",     "sus_max": 30},
    {"label": "scam_critical",   "work_type": None,        "sus_min": 80},
]

DEFAULT_JOBS_PER_CLUSTER = 8

CHECKPOINT = "SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors"
IMG_WIDTH  = 832
IMG_HEIGHT = 1216
STEPS      = 20
CFG        = 7.0
SAMPLER    = "dpmpp_2m"
SCHEDULER  = "karras"

ANCHOR_PROMPT = (
    "photorealistic portrait of a nondescript person, neutral expression, "
    "soft studio lighting, plain background, sharp focus, 4k"
)
ANCHOR_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, text, watermark, "
    "deformed, blurry, low quality, nsfw"
)
ANCHOR_SEED = 42


# ── Denoising ────────────────────────────────────────────────────────────────

def sus_to_denoise(sus_level: int) -> float:
    """
    Phase 2: raised floor and ceiling vs Phase 1 (0.20–0.55 → 0.35–0.65).
    Even legit jobs now drift noticeably from anchor so cluster differences
    are visible. High-fraud jobs hit stronger uncanny territory.
    """
    return 0.35 + (sus_level / 100.0) * 0.30


# ── PCA → face descriptor ────────────────────────────────────────────────────

def _age_build(pc1_norm: float) -> str:
    """PC1: physical/manual ↔ knowledge/office work → age + build archetype."""
    if pc1_norm > 0.6:
        return "muscular build, 45-55 years old, weathered skin, calloused hands"
    elif pc1_norm > 0.2:
        return "stocky build, 40-50 years old, practical appearance"
    elif pc1_norm > -0.2:
        return "average build, 35-45 years old, unremarkable features"
    elif pc1_norm > -0.6:
        return "slim build, 28-38 years old, soft features"
    else:
        return "slender, 22-32 years old, youthful professional appearance"


def _gender_presentation(pc2_norm: float) -> str:
    """
    PC2: formal ↔ informal posting → gender presentation and grooming.
    Formal job postings skew toward feminine presentation in Russian market
    (admin, reception, clerical).  Informal skew masculine (manual, trade).
    This is a corpus correlation, not a design choice.
    """
    if pc2_norm > 0.5:
        return "feminine features, well-groomed, light makeup, business attire"
    elif pc2_norm > 0.0:
        return "androgynous features, neat appearance, casual smart dress"
    elif pc2_norm > -0.5:
        return "masculine features, clean-shaven, practical clothing"
    else:
        return "masculine features, stubble, worn workwear"


def _hair_style(pc3_norm: float) -> str:
    """PC3: a secondary style axis → hair description."""
    if pc3_norm > 0.5:
        return "styled hair, neat cut"
    elif pc3_norm > -0.5:
        return "medium-length hair, unstyled"
    else:
        return "close-cropped or buzzcut"


def _uncanny_affect(sus_level: int) -> str:
    """sus_level → expression/uncanny gradient. Independent of identity axes."""
    if sus_level >= 80:
        return (
            "performative smile, slightly too-wide eyes, uncanny valley effect, "
            "wrong facial proportions, forced cheerfulness, hollow affect"
        )
    elif sus_level >= 55:
        return (
            "uncomfortable expression, avoidant gaze, vaguely uneasy, "
            "slight asymmetry, overly composed"
        )
    elif sus_level >= 30:
        return "slightly guarded expression, neutral-to-tense affect"
    else:
        return "natural relaxed expression, open gaze, genuine affect"


def pc_to_face_descriptor(pc1: float, pc2: float, pc3: float, sus: int) -> str:
    """
    Two-channel face encoding:
      Identity channel  (orthogonal to sus): age/build from PC1,
                                              gender presentation from PC2,
                                              hair from PC3.
      Expression channel (driven by sus):    uncanny valley gradient.
    Combined additively in the prompt.
    """
    identity = ", ".join([
        _age_build(pc1),
        _gender_presentation(pc2),
        _hair_style(pc3),
    ])
    affect = _uncanny_affect(sus)
    return f"{identity}, {affect}"


# ── Seed ─────────────────────────────────────────────────────────────────────

def embedding_seed(embedding: list[float]) -> int:
    """Stable seed from first 12 embedding dims. Same job → same face across runs."""
    key = ",".join(f"{v:.4f}" for v in embedding[:12])
    digest = hashlib.md5(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


# ── ComfyUI workflows ─────────────────────────────────────────────────────────

def text2img_workflow(checkpoint, positive, negative, seed, width, height,
                      steps, cfg, sampler, scheduler, filename_prefix) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "CLIPTextEncode",          "inputs": {"text": positive, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",          "inputs": {"text": negative, "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",        "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
            },
        },
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": filename_prefix}},
    }


def img2img_workflow(checkpoint, input_image_name, positive, negative,
                     seed, steps, cfg, sampler, scheduler, denoise, filename_prefix) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage",              "inputs": {"image": input_image_name}},
        "3": {"class_type": "VAEEncode",              "inputs": {"pixels": ["2", 0], "vae": ["1", 2]}},
        "4": {"class_type": "CLIPTextEncode",         "inputs": {"text": positive, "clip": ["1", 1]}},
        "5": {"class_type": "CLIPTextEncode",         "inputs": {"text": negative, "clip": ["1", 1]}},
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0],
                "latent_image": ["3", 0], "seed": seed, "steps": steps, "cfg": cfg,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise,
            },
        },
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix}},
    }


# ── ComfyUI client ────────────────────────────────────────────────────────────

class ComfyClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base, timeout=60.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "ComfyClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def health(self) -> dict:
        r = await self._http.get("/system_stats")
        r.raise_for_status()
        return r.json()

    async def list_checkpoints(self) -> list[str]:
        r = await self._http.get("/models/checkpoints")
        r.raise_for_status()
        return r.json()

    async def upload_image(self, path: Path) -> str:
        with open(path, "rb") as fh:
            r = await self._http.post(
                "/upload/image",
                files={"image": (path.name, fh, "image/png")},
                data={"type": "input", "overwrite": "true"},
            )
            r.raise_for_status()
        return r.json()["name"]

    async def submit(self, workflow: dict) -> str:
        body = {"prompt": workflow, "client_id": str(uuid.uuid4())}
        r = await self._http.post("/prompt", json=body)
        if r.status_code == 400:
            raise RuntimeError(f"Workflow rejected (400): {r.text}")
        r.raise_for_status()
        data = r.json()
        if data.get("error") or data.get("node_errors"):
            raise RuntimeError(f"Workflow validation failed: {data.get('error') or data.get('node_errors')}")
        return data["prompt_id"]

    async def wait(self, prompt_id: str, timeout: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {prompt_id!r} timed out after {timeout}s")
            r = await self._http.get(f"/history/{prompt_id}")
            r.raise_for_status()
            data = r.json()
            if prompt_id in data:
                job = data[prompt_id]
                status = job.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Job {prompt_id!r} failed: {status}")
                if status.get("completed"):
                    return job["outputs"]
            await asyncio.sleep(0.5)

    async def download(self, filename: str, dest: Path) -> None:
        params = {"filename": filename, "subfolder": "", "type": "output"}
        async with self._http.stream("GET", "/view", params=params) as r:
            r.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                async for chunk in r.aiter_bytes():
                    fh.write(chunk)

    def _first_output_filename(self, outputs: dict) -> str:
        for node_out in outputs.values():
            imgs = node_out.get("images", [])
            if imgs:
                return imgs[0]["filename"]
        raise ValueError(f"No images in outputs: {outputs}")

    async def run_workflow(self, workflow: dict, dest: Path) -> None:
        prompt_id = await self.submit(workflow)
        outputs = await self.wait(prompt_id)
        filename = self._first_output_filename(outputs)
        await self.download(filename, dest)


# ── Database ──────────────────────────────────────────────────────────────────

def fetch_cluster_jobs(cluster: dict, n: int) -> list[dict]:
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        conditions = ["embedding IS NOT NULL", "raw_content IS NOT NULL"]
        params: list = []
        if cluster.get("work_type"):
            conditions.append("work_type = %s")
            params.append(cluster["work_type"])
        if "sus_max" in cluster:
            conditions.append("sus_level <= %s")
            params.append(cluster["sus_max"])
        if "sus_min" in cluster:
            conditions.append("sus_level >= %s")
            params.append(cluster["sus_min"])
        params.append(n)
        cur.execute(
            f"SELECT id, raw_content, sus_level, sus_category, work_type, embedding "
            f"FROM jobs WHERE {' AND '.join(conditions)} LIMIT %s",
            params,
        )
        rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def fetch_all_embeddings(limit: int = 10000) -> np.ndarray:
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM jobs WHERE embedding IS NOT NULL LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    conn.close()
    vecs = []
    for (emb,) in rows:
        if isinstance(emb, str):
            vecs.append([float(x) for x in emb.strip("[]").split(",")])
        else:
            vecs.append(list(emb))
    return np.array(vecs, dtype=np.float32)


def parse_embedding(raw) -> list[float]:
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


# ── Utils ─────────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint
    n = args.jobs_per_cluster

    print("=== Phase 2: wider denoising + richer demographics ===\n")
    print(f"  Denoising range: {sus_to_denoise(0):.2f} (sus=0) → {sus_to_denoise(100):.2f} (sus=100)")
    print(f"  Jobs per cluster: {n}")
    print()

    # 1. Fetch jobs
    print("Fetching jobs from DB...")
    all_jobs: list[dict] = []
    cluster_map: dict[str, list[dict]] = {}
    for spec in CLUSTERS:
        jobs = fetch_cluster_jobs(spec, n)
        if len(jobs) < n:
            print(f"  WARNING: {spec['label']!r} has only {len(jobs)} jobs (wanted {n})")
        cluster_map[spec["label"]] = jobs
        all_jobs.extend(jobs)
        sus_vals = [j["sus_level"] for j in jobs]
        print(f"  {spec['label']}: {len(jobs)} jobs | sus {min(sus_vals)}–{max(sus_vals)}")

    print(f"\nTotal: {len(all_jobs)} faces to generate")

    # 2. Fit PCA (3 components now, up from 2)
    print("\nFitting PCA on 10k embeddings (3 components)...")
    all_embs = fetch_all_embeddings(limit=10000)
    print(f"  Loaded {len(all_embs)} embeddings ({all_embs.shape[1]}-d)")

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(all_embs)
    pca = PCA(n_components=10, whiten=True)
    pca.fit(embs_scaled)
    var = pca.explained_variance_ratio_
    print(f"  PC1={var[0]:.3f} PC2={var[1]:.3f} PC3={var[2]:.3f}")

    for job in all_jobs:
        emb = parse_embedding(job["embedding"])
        emb_arr = np.array(emb, dtype=np.float32).reshape(1, -1)
        pcs = pca.transform(scaler.transform(emb_arr))[0]
        job["pc1_norm"] = float(np.clip(pcs[0] / 2.0, -1.0, 1.0))
        job["pc2_norm"] = float(np.clip(pcs[1] / 2.0, -1.0, 1.0))
        job["pc3_norm"] = float(np.clip(pcs[2] / 2.0, -1.0, 1.0))
        job["embed_seed"] = embedding_seed(emb)

    # 3. Preview prompts
    print("\nPrompt preview (2 per cluster):")
    for label, jobs in cluster_map.items():
        print(f"\n  [{label}]")
        for job in jobs[:2]:
            face_attrs = pc_to_face_descriptor(
                job["pc1_norm"], job["pc2_norm"], job["pc3_norm"], job["sus_level"]
            )
            denoise = sus_to_denoise(job["sus_level"])
            print(f"    sus={job['sus_level']:3d} | denoise={denoise:.2f} | {face_attrs[:110]}…")

    if args.dry_run:
        print("\n[DRY RUN] Skipping ComfyUI generation.")
        return

    # 4. Connect to ComfyUI
    print(f"\nConnecting to ComfyUI at {COMFY_URL}...")
    async with ComfyClient(COMFY_URL) as comfy:
        stats = await comfy.health()
        gpu = stats.get("devices", [{}])[0]
        print(f"  OK | {gpu.get('name', 'GPU')} | {gpu.get('vram_free', 0) / 1024**3:.1f} GB free")

        checkpoints = await comfy.list_checkpoints()
        if checkpoint not in checkpoints:
            print(f"\nERROR: {checkpoint!r} not found.")
            print("Available:", checkpoints)
            return

        # Reuse phase1 anchor
        anchor_path = ANCHOR_DIR / ANCHOR_FILENAME
        if not anchor_path.exists():
            print(f"\nERROR: anchor not found at {anchor_path}")
            print("Run phase1_cluster_test.py first to generate the anchor.")
            return
        print(f"\nUsing existing anchor: {anchor_path}")

        anchor_uploaded = await comfy.upload_image(anchor_path)
        print(f"  Uploaded as: {anchor_uploaded!r}")

        # 5. Generate faces
        print(f"\nGenerating {len(all_jobs)} faces…")
        summary_rows = []

        for label, jobs in cluster_map.items():
            print(f"\n  [{label}]")
            for job in jobs:
                job_id = str(job["id"])
                face_attrs = pc_to_face_descriptor(
                    job["pc1_norm"], job["pc2_norm"], job["pc3_norm"], job["sus_level"]
                )
                positive = (
                    f"photorealistic portrait, {face_attrs}, "
                    f"soft studio lighting, plain background, sharp focus, 4k"
                )
                denoise = sus_to_denoise(job["sus_level"])

                dest = OUT_DIR / label / f"{job_id}.png"
                if dest.exists():
                    print(f"    {job_id[:8]}… skip (exists)")
                    continue

                workflow = img2img_workflow(
                    checkpoint=checkpoint,
                    input_image_name=anchor_uploaded,
                    positive=positive,
                    negative=ANCHOR_NEGATIVE,
                    seed=job["embed_seed"],
                    steps=STEPS,
                    cfg=CFG,
                    sampler=SAMPLER,
                    scheduler=SCHEDULER,
                    denoise=denoise,
                    filename_prefix=f"phase2_{label}_{job_id[:8]}",
                )
                await comfy.run_workflow(workflow, dest)
                print(f"    {job_id[:8]}… sus={job['sus_level']:3d} denoise={denoise:.2f} → {dest}")
                summary_rows.append({
                    "cluster": label,
                    "job_id": job_id,
                    "sus_level": job["sus_level"],
                    "pc1": round(job["pc1_norm"], 3),
                    "pc2": round(job["pc2_norm"], 3),
                    "pc3": round(job["pc3_norm"], 3),
                    "denoise": round(denoise, 3),
                    "path": str(dest),
                })

    # 6. Summary
    print(f"\n{'='*60}")
    print(f"Generated {len(summary_rows)} faces → {OUT_DIR}/")
    print("\nPC centroids per cluster:")
    for label, jobs in cluster_map.items():
        if jobs:
            pc1 = sum(j["pc1_norm"] for j in jobs) / len(jobs)
            pc2 = sum(j["pc2_norm"] for j in jobs) / len(jobs)
            pc3 = sum(j["pc3_norm"] for j in jobs) / len(jobs)
            print(f"  {label:25s}  PC1={pc1:+.3f}  PC2={pc2:+.3f}  PC3={pc3:+.3f}")

    print("\nIntra-cluster cosine similarity:")
    for label, jobs in cluster_map.items():
        if len(jobs) >= 2:
            vecs = [parse_embedding(j["embedding"]) for j in jobs]
            pairs = [(i, k) for i in range(len(vecs)) for k in range(i + 1, len(vecs))]
            avg = sum(cosine(vecs[i], vecs[k]) for i, k in pairs) / len(pairs)
            print(f"  {label:25s}: {avg:.3f}")

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump({
            "phase": 2,
            "checkpoint": checkpoint,
            "denoising": {"sus0": sus_to_denoise(0), "sus100": sus_to_denoise(100)},
            "pca_explained_variance": var[:5].tolist(),
            "jobs": summary_rows,
        }, fh, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--jobs-per-cluster", type=int, default=DEFAULT_JOBS_PER_CLUSTER)
    asyncio.run(main(parser.parse_args()))
