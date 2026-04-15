#!/usr/bin/env python3
"""
Phase 1 cluster test: embedding clusters → face grid.

Pulls 5 jobs from each of 4 semantic clusters, fits PCA on all available
embeddings, maps PCA coordinates to face-description text, then generates
one face per job via ComfyUI img2img from a shared neutral anchor.

Visual pass: do jobs from the same cluster look like the same person?
Do scam posts feel wrong in a way legitimate ones don't?

Usage:
    uv run src/phase1_cluster_test.py
    uv run src/phase1_cluster_test.py --checkpoint "SD15/abstractPhoto_abcevereMix.safetensors"
    uv run src/phase1_cluster_test.py --dry-run      # print prompts, skip ComfyUI
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
OUT_DIR = Path("output/phase1")

# 4 clusters: 3 legitimate categories + 1 scam category
CLUSTERS = [
    {"label": "courier_legit",    "work_type": "доставка",  "sus_max": 30},
    {"label": "warehouse_legit",  "work_type": "склад",     "sus_max": 30},
    {"label": "office_legit",     "work_type": "офис",      "sus_max": 30},
    {"label": "scam_critical",    "work_type": None,         "sus_min": 80},  # any work_type
]

JOBS_PER_CLUSTER = 5

# ComfyUI generation params
CHECKPOINT = "SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors"
IMG_WIDTH  = 832
IMG_HEIGHT = 1216
STEPS      = 20
CFG        = 7.0
SAMPLER    = "dpmpp_2m"
SCHEDULER  = "karras"

# Anchor: shared neutral face used as img2img base for all jobs
ANCHOR_PROMPT = (
    "photorealistic portrait of a nondescript person, neutral expression, "
    "soft studio lighting, plain background, sharp focus, 4k"
)
ANCHOR_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, text, watermark, "
    "deformed, blurry, low quality, nsfw"
)
ANCHOR_SEED = 42
ANCHOR_FILENAME = "phase1_anchor.png"

# img2img denoising: how much the anchor is overridden per sus_level bracket
def sus_to_denoise(sus_level: int) -> float:
    """Linear scale: sus 0→0.20 identity-preserving, sus 100→0.55 strong drift."""
    return 0.20 + (sus_level / 100.0) * 0.35


# ── PCA → face attributes ─────────────────────────────────────────────────────

def pc_to_face_descriptor(pc1_norm: float, pc2_norm: float, sus_level: int) -> str:
    """
    Map two normalized PCA coordinates + sus_level to a face description string.

    PC1 empirically tends to separate physical/manual work (high) from
    office/knowledge work (low) in Russian job-post embedding space.
    PC2 tends to separate formal/legitimate postings from informal ones.
    """
    # Physical vs knowledge work axis
    if pc1_norm > 0.5:
        build_desc = "broad shoulders, weathered skin, strong jaw, working-class"
    elif pc1_norm > 0.0:
        build_desc = "average build, unremarkable features"
    elif pc1_norm > -0.5:
        build_desc = "slim build, soft features, tidy appearance"
    else:
        build_desc = "slender, groomed, professional appearance"

    # Formal vs informal posting axis
    if pc2_norm > 0.5:
        style_desc = "well-groomed, clean shave, business casual"
    elif pc2_norm > -0.5:
        style_desc = "casual dress, natural lighting"
    else:
        style_desc = "dishevelled, tired, informal"

    # sus_level → uncanny valley gradient
    if sus_level >= 75:
        affect_desc = (
            "performative smile, slightly too-wide eyes, uncanny valley, "
            "wrong proportions, forced cheerfulness"
        )
    elif sus_level >= 45:
        affect_desc = "vaguely uneasy expression, slightly avoidant gaze"
    else:
        affect_desc = "natural relaxed expression, open gaze"

    return f"{build_desc}, {style_desc}, {affect_desc}"


def embedding_seed(embedding: list[float]) -> int:
    """Stable seed from first 12 embedding dimensions. Similar embeds → similar seeds."""
    key = ",".join(f"{v:.4f}" for v in embedding[:12])
    digest = hashlib.md5(key.encode()).digest()
    return int.from_bytes(digest[:4], "big") % (2**31)


# ── ComfyUI workflows ─────────────────────────────────────────────────────────

def text2img_workflow(checkpoint: str, positive: str, negative: str,
                      seed: int, width: int, height: int,
                      steps: int, cfg: float, sampler: str, scheduler: str,
                      filename_prefix: str) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["2", 0],
                "negative":     ["3", 0],
                "latent_image": ["4", 0],
                "seed":         seed,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": sampler,
                "scheduler":    scheduler,
                "denoise":      1.0,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": filename_prefix},
        },
    }


def img2img_workflow(checkpoint: str, input_image_name: str,
                     positive: str, negative: str,
                     seed: int, steps: int, cfg: float,
                     sampler: str, scheduler: str,
                     denoise: float, filename_prefix: str) -> dict:
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        },
        "2": {
            "class_type": "LoadImage",
            "inputs": {"image": input_image_name},
        },
        "3": {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["2", 0], "vae": ["1", 2]},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive, "clip": ["1", 1]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative, "clip": ["1", 1]},
        },
        "6": {
            "class_type": "KSampler",
            "inputs": {
                "model":        ["1", 0],
                "positive":     ["4", 0],
                "negative":     ["5", 0],
                "latent_image": ["3", 0],
                "seed":         seed,
                "steps":        steps,
                "cfg":          cfg,
                "sampler_name": sampler,
                "scheduler":    scheduler,
                "denoise":      denoise,
            },
        },
        "7": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
        },
        "8": {
            "class_type": "SaveImage",
            "inputs": {"images": ["7", 0], "filename_prefix": filename_prefix},
        },
    }


# ── ComfyUI client (thin, sync-friendly async wrapper) ───────────────────────

class ComfyClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base, timeout=60.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "ComfyClient":
        return self

    async def __aexit__(self, *_: object) -> None:
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
        """Submit, wait, download to dest."""
        prompt_id = await self.submit(workflow)
        outputs = await self.wait(prompt_id)
        filename = self._first_output_filename(outputs)
        await self.download(filename, dest)


# ── Database helpers ──────────────────────────────────────────────────────────

def fetch_cluster_jobs(cluster: dict, n: int) -> list[dict]:
    """Fetch n jobs matching the cluster spec. Returns list of dicts."""
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
    """Load up to `limit` stored embeddings for PCA fitting."""
    conn = psycopg2.connect(DB_DSN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT embedding FROM jobs WHERE embedding IS NOT NULL LIMIT %s",
            (limit,),
        )
        rows = cur.fetchall()
    conn.close()
    # pgvector returns embeddings as '[0.1, 0.2, ...]' strings or lists
    vecs = []
    for (emb,) in rows:
        if isinstance(emb, str):
            vecs.append([float(x) for x in emb.strip("[]").split(",")])
        else:
            vecs.append(list(emb))
    return np.array(vecs, dtype=np.float32)


def parse_embedding(raw) -> list[float]:
    """Parse embedding from psycopg2 (may be string or list)."""
    if isinstance(raw, str):
        return [float(x) for x in raw.strip("[]").split(",")]
    return list(raw)


# ── Main ──────────────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


async def main(args: argparse.Namespace) -> None:
    checkpoint = args.checkpoint

    print("=== Phase 1: embedding clusters → face grid ===\n")

    # 1. Fetch jobs for each cluster
    print("Fetching jobs from DB...")
    all_jobs: list[dict] = []
    cluster_map: dict[str, list[dict]] = {}
    for spec in CLUSTERS:
        jobs = fetch_cluster_jobs(spec, JOBS_PER_CLUSTER)
        if len(jobs) < JOBS_PER_CLUSTER:
            print(f"  WARNING: cluster {spec['label']!r} has only {len(jobs)} jobs (wanted {JOBS_PER_CLUSTER})")
        cluster_map[spec["label"]] = jobs
        all_jobs.extend(jobs)
        print(f"  {spec['label']}: {len(jobs)} jobs (sus_level range: "
              f"{min(j['sus_level'] for j in jobs) if jobs else 'n/a'}"
              f"–{max(j['sus_level'] for j in jobs) if jobs else 'n/a'})")

    print(f"\nTotal jobs to generate: {len(all_jobs)}")

    # 2. Fit PCA on all available embeddings
    print("\nFitting PCA on all available embeddings...")
    all_embs = fetch_all_embeddings(limit=10000)
    print(f"  Loaded {len(all_embs)} embeddings ({all_embs.shape[1]}-d)")

    scaler = StandardScaler()
    embs_scaled = scaler.fit_transform(all_embs)
    pca = PCA(n_components=10, whiten=True)
    pca.fit(embs_scaled)
    explained = pca.explained_variance_ratio_
    print(f"  PC1-PC3 explained variance: {explained[0]:.3f}, {explained[1]:.3f}, {explained[2]:.3f}")

    # Project each job's embedding
    for job in all_jobs:
        emb = parse_embedding(job["embedding"])
        emb_arr = np.array(emb, dtype=np.float32).reshape(1, -1)
        scaled = scaler.transform(emb_arr)
        pcs = pca.transform(scaled)[0]  # shape (10,)
        # Normalize PC1, PC2 to [-1, 1] using 2-sigma range
        job["pc1_norm"] = float(np.clip(pcs[0] / 2.0, -1.0, 1.0))
        job["pc2_norm"] = float(np.clip(pcs[1] / 2.0, -1.0, 1.0))
        job["embed_seed"] = embedding_seed(emb)

    # 3. Build prompts
    print("\nGenerated prompts (dry-run preview):")
    for cluster_label, jobs in cluster_map.items():
        print(f"\n  [{cluster_label}]")
        for job in jobs[:2]:  # show first 2 per cluster
            face_attrs = pc_to_face_descriptor(job["pc1_norm"], job["pc2_norm"], job["sus_level"])
            prompt = (
                f"photorealistic portrait, {face_attrs}, "
                f"soft studio lighting, plain background, sharp focus, 4k"
            )
            denoise = sus_to_denoise(job["sus_level"])
            print(f"    sus={job['sus_level']:3d} | denoise={denoise:.2f} | {prompt[:100]}…")

    if args.dry_run:
        print("\n[DRY RUN] Skipping ComfyUI generation.")
        return

    # 4. ComfyUI: generate anchor face
    print(f"\nConnecting to ComfyUI at {COMFY_URL}...")
    async with ComfyClient(COMFY_URL) as comfy:
        stats = await comfy.health()
        gpu = stats.get("devices", [{}])[0]
        print(f"  ComfyUI OK | {gpu.get('name', 'GPU')} | "
              f"{gpu.get('vram_free', 0) / 1024**3:.1f} GB free")

        # Verify checkpoint exists
        checkpoints = await comfy.list_checkpoints()
        if checkpoint not in checkpoints:
            print(f"\nERROR: checkpoint {checkpoint!r} not found.")
            print("Available checkpoints:")
            for c in checkpoints:
                print(f"  {c}")
            return

        anchor_path = OUT_DIR / ANCHOR_FILENAME
        if anchor_path.exists():
            print(f"\nAnchor face already exists: {anchor_path}")
        else:
            print(f"\nGenerating anchor face ({ANCHOR_PROMPT[:60]}…)")
            workflow = text2img_workflow(
                checkpoint=checkpoint,
                positive=ANCHOR_PROMPT,
                negative=ANCHOR_NEGATIVE,
                seed=ANCHOR_SEED,
                width=IMG_WIDTH,
                height=IMG_HEIGHT,
                steps=STEPS,
                cfg=CFG,
                sampler=SAMPLER,
                scheduler=SCHEDULER,
                filename_prefix="phase1_anchor",
            )
            await comfy.run_workflow(workflow, anchor_path)
            print(f"  Saved: {anchor_path}")

        # Upload anchor to ComfyUI input
        anchor_uploaded_name = await comfy.upload_image(anchor_path)
        print(f"  Anchor uploaded as: {anchor_uploaded_name!r}")

        # 5. Generate one face per job
        print(f"\nGenerating {len(all_jobs)} faces…")
        summary_rows = []

        for cluster_label, jobs in cluster_map.items():
            print(f"\n  [{cluster_label}]")
            for job in jobs:
                job_id = str(job["id"])
                face_attrs = pc_to_face_descriptor(job["pc1_norm"], job["pc2_norm"], job["sus_level"])
                positive = (
                    f"photorealistic portrait, {face_attrs}, "
                    f"soft studio lighting, plain background, sharp focus, 4k"
                )
                negative = ANCHOR_NEGATIVE
                denoise = sus_to_denoise(job["sus_level"])
                seed = job["embed_seed"]

                dest = OUT_DIR / cluster_label / f"{job_id}.png"
                if dest.exists():
                    print(f"    {job_id[:8]}… already exists, skip")
                    continue

                workflow = img2img_workflow(
                    checkpoint=checkpoint,
                    input_image_name=anchor_uploaded_name,
                    positive=positive,
                    negative=negative,
                    seed=seed,
                    steps=STEPS,
                    cfg=CFG,
                    sampler=SAMPLER,
                    scheduler=SCHEDULER,
                    denoise=denoise,
                    filename_prefix=f"phase1_{cluster_label}_{job_id[:8]}",
                )
                await comfy.run_workflow(workflow, dest)
                print(f"    {job_id[:8]}… sus={job['sus_level']:3d} denoise={denoise:.2f} → {dest}")
                summary_rows.append({
                    "cluster": cluster_label,
                    "job_id": job_id,
                    "sus_level": job["sus_level"],
                    "pc1": round(job["pc1_norm"], 3),
                    "pc2": round(job["pc2_norm"], 3),
                    "denoise": round(denoise, 3),
                    "path": str(dest),
                })

    # 6. Summary
    print(f"\n{'='*50}")
    print(f"Generated {len(summary_rows)} faces → {OUT_DIR}/")
    print("\nPC1/PC2 centroids per cluster:")
    for cluster_label, jobs in cluster_map.items():
        if jobs:
            pc1_mean = sum(j["pc1_norm"] for j in jobs) / len(jobs)
            pc2_mean = sum(j["pc2_norm"] for j in jobs) / len(jobs)
            print(f"  {cluster_label:25s}  PC1={pc1_mean:+.3f}  PC2={pc2_mean:+.3f}")

    # Intra-cluster vs cross-cluster cosine similarity
    print("\nEmbedding cosine similarity (sanity check):")
    for cluster_label, jobs in cluster_map.items():
        if len(jobs) >= 2:
            vecs = [parse_embedding(j["embedding"]) for j in jobs]
            pairs = [(i, k) for i in range(len(vecs)) for k in range(i + 1, len(vecs))]
            avg_sim = sum(cosine(vecs[i], vecs[k]) for i, k in pairs) / len(pairs)
            print(f"  intra {cluster_label:25s}: {avg_sim:.3f}")

    # Save manifest
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as fh:
        json.dump({
            "checkpoint": checkpoint,
            "pca_explained_variance": explained[:5].tolist(),
            "jobs": summary_rows,
        }, fh, indent=2)
    print(f"\nManifest saved: {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=CHECKPOINT,
        help="ComfyUI checkpoint path relative to models/checkpoints/",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print prompts and skip ComfyUI generation",
    )
    asyncio.run(main(parser.parse_args()))
