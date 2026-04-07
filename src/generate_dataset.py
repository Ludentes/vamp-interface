#!/usr/bin/env python3
"""
generate_dataset.py — Generate faces for the full test dataset.

Reads data/test_dataset.json (or rescored variant), fits PCA on all embedded
embeddings, maps each job to a face descriptor, generates img2img via ComfyUI.

Output: output/dataset_faces/<cohort>/<job_id>.png     (--face-version 1, default)
        output/dataset_faces_v2/<cohort>/<job_id>.png  (--face-version 2)
        output/dataset_faces/manifest.json

All local: ComfyUI at localhost:8188 (RTX 5090). No remote dependencies.

Face versions:
  1 (default): 3 PCA axes, divisor=2.0 (narrow band spread)
  2:           5 PCA axes, divisor=1.0 (full spread) + complexion + texture axes

Usage:
    uv run src/generate_dataset.py
    uv run src/generate_dataset.py --face-version 2
    uv run src/generate_dataset.py --input data/test_dataset_gemma4.json
    uv run src/generate_dataset.py --sus-source rescore   # use rescored levels
    uv run src/generate_dataset.py --cohorts warehouse_legit courier_scam
    uv run src/generate_dataset.py --dry-run --limit 5
"""

import argparse
import asyncio
import hashlib
import json
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ── Config ────────────────────────────────────────────────────────────────────

COMFY_URL = "http://localhost:8188"
ANCHOR_PATH = Path("output/phase1/phase1_anchor.png")
OUT_DIR = Path("output/dataset_faces")

CHECKPOINT = "SDXL Lightning/juggernautXL_juggXILightningByRD.safetensors"
STEPS = 20
CFG = 7.0
SAMPLER = "dpmpp_2m"
SCHEDULER = "karras"
IMG_W = 832
IMG_H = 1216

# Flux defaults
FLUX_CHECKPOINT = "FLUX1/flux1-krea-dev_fp8_scaled.safetensors"
FLUX_STEPS = 20
FLUX_GUIDANCE = 3.5   # flux-dev guidance scale
FLUX_SAMPLER = "euler"
FLUX_SCHEDULER = "simple"

ANCHOR_NEGATIVE = (
    "cartoon, anime, illustration, painting, drawing, text, watermark, "
    "deformed, blurry, low quality, nsfw"
)


# ── Denoising ─────────────────────────────────────────────────────────────────

def sus_to_denoise(sus: int) -> float:
    """Phase 2 calibration: 0.35 (sus=0) → 0.65 (sus=100)."""
    return 0.35 + (sus / 100.0) * 0.30


# ── Face descriptor (Phase 2 mapping) ─────────────────────────────────────────

def _age_build(pc1: float) -> str:
    if pc1 > 0.6:   return "muscular build, 45-55 years old, weathered skin, calloused hands"
    if pc1 > 0.2:   return "stocky build, 40-50 years old, practical appearance"
    if pc1 > -0.2:  return "average build, 35-45 years old, unremarkable features"
    if pc1 > -0.6:  return "slim build, 28-38 years old, soft features"
    return "slender, 22-32 years old, youthful professional appearance"


def _gender_presentation(pc2: float) -> str:
    if pc2 > 0.5:   return "feminine features, well-groomed, light makeup, business attire"
    if pc2 > 0.0:   return "androgynous features, neat appearance, casual smart dress"
    if pc2 > -0.5:  return "masculine features, clean-shaven, practical clothing"
    return "masculine features, stubble, worn workwear"


def _hair_style(pc3: float) -> str:
    if pc3 > 0.5:   return "styled hair, neat cut"
    if pc3 > -0.5:  return "medium-length hair, unstyled"
    return "close-cropped or buzzcut"


def _complexion(pc4: float) -> str:
    """PC4: fair ↔ dark complexion axis."""
    if pc4 > 0.5:   return "fair complexion, pale skin"
    if pc4 > 0.0:   return "light complexion"
    if pc4 > -0.5:  return "olive skin tone"
    return "dark complexion, tanned skin"


def _texture(pc5: float) -> str:
    """PC5: smooth youthful ↔ weathered textured skin."""
    if pc5 > 0.5:   return "smooth skin, youthful texture"
    if pc5 > -0.5:  return "normal skin texture"
    return "weathered skin, pronounced lines, character marks"


def _uncanny_affect(sus: int) -> str:
    if sus >= 80:
        return (
            "performative smile, slightly too-wide eyes, uncanny valley effect, "
            "wrong facial proportions, forced cheerfulness, hollow affect"
        )
    if sus >= 55:
        return "uncomfortable expression, avoidant gaze, vaguely uneasy, slight asymmetry"
    if sus >= 30:
        return "slightly guarded expression, neutral-to-tense affect"
    return "natural relaxed expression, open gaze, genuine affect"


def face_descriptor(pc1: float, pc2: float, pc3: float, sus: int) -> str:
    """v1: 3 axes, narrow band spread."""
    identity = ", ".join([_age_build(pc1), _gender_presentation(pc2), _hair_style(pc3)])
    affect = _uncanny_affect(sus)
    return f"{identity}, {affect}"


def face_descriptor_v2(pc1: float, pc2: float, pc3: float, pc4: float, pc5: float, sus: int) -> str:
    """v2: 5 axes, full spread — adds complexion + texture."""
    identity = ", ".join([
        _age_build(pc1), _gender_presentation(pc2), _hair_style(pc3),
        _complexion(pc4), _texture(pc5),
    ])
    affect = _uncanny_affect(sus)
    return f"{identity}, {affect}"


# ── v3: work_type as primary identity axis ────────────────────────────────────

# Each archetype: (base_identity, clothing/context)
# PC2 → formality modifier (feminine/androgynous/masculine within type)
# PC3 → age modifier within type
_WORK_TYPE_ARCHETYPE: dict[str, tuple[str, str]] = {
    "склад":    ("stocky build, practical appearance",          "worn workwear, high-vis vest"),
    "стройка":  ("muscular build, weathered skin, calloused hands", "dusty work clothes, hard hat"),
    "уборка":   ("average build, practical appearance",         "cleaning uniform, apron"),
    "офис":     ("slim build, neat grooming",                   "business casual attire"),
    "доставка": ("slim build, youthful appearance",             "casual sportswear, delivery bag"),
    "удалёнка": ("slim build, relaxed posture",                 "casual home wear, soft background"),
    "погрузка": ("heavy build, broad shoulders",                "loading dock wear, gloves"),
    "торговля": ("average build, approachable appearance",      "retail uniform, name badge"),
    "общепит":  ("average build, neat appearance",              "hospitality uniform, apron"),
    "другое":   ("average build, unremarkable features",        "nondescript casual clothing"),
}

_PC2_GENDER: list[tuple[float, str]] = [
    (0.5,  "feminine features, well-groomed"),
    (0.0,  "androgynous features"),
    (-0.5, "masculine features, clean-shaven"),
    (-999, "masculine features, unshaven"),
]

_PC3_AGE: list[tuple[float, str]] = [
    (0.5,  "25-35 years old"),
    (0.0,  "35-45 years old"),
    (-999, "45-55 years old"),
]


def _gender_mod(pc2: float) -> str:
    for threshold, label in _PC2_GENDER:
        if pc2 > threshold:
            return label
    return _PC2_GENDER[-1][1]


def _age_mod(pc3: float) -> str:
    for threshold, label in _PC3_AGE:
        if pc3 > threshold:
            return label
    return _PC3_AGE[-1][1]


def face_descriptor_v3(work_type: str | None, pc2: float, pc3: float, sus: int) -> str:
    """v3: work_type as primary identity axis; PC2=gender mod, PC3=age mod."""
    wt = work_type or "другое"
    base, clothing = _WORK_TYPE_ARCHETYPE.get(wt, _WORK_TYPE_ARCHETYPE["другое"])
    identity = f"{base}, {_age_mod(pc3)}, {_gender_mod(pc2)}, {clothing}"
    affect = _uncanny_affect(sus)
    return f"{identity}, {affect}"


def embedding_seed(emb: list[float]) -> int:
    key = ",".join(f"{v:.4f}" for v in emb[:12])
    return int.from_bytes(hashlib.md5(key.encode()).digest()[:4], "big") % (2**31)


# ── ComfyUI workflows ─────────────────────────────────────────────────────────

def img2img_workflow(checkpoint, image_name, positive, negative,
                     seed, steps, cfg, sampler, scheduler, denoise, prefix) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": checkpoint}},
        "2": {"class_type": "LoadImage",              "inputs": {"image": image_name}},
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
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": prefix}},
    }


FLUX_VAE    = "FLUX1/ae.safetensors"
FLUX_CLIP_L = "clip_l.safetensors"
FLUX_T5     = "t5/t5xxl_fp8_e4m3fn.safetensors"


def flux_img2img_workflow(checkpoint, image_name, positive,
                          seed, steps, guidance, sampler, scheduler, denoise, prefix) -> dict:
    """Flux img2img workflow. fp8 UNet-only checkpoint: separate VAE, CLIP-L, T5.
    Node map:
      1 = UNETLoader       (diffusion model)
      2 = VAELoader        (ae.safetensors)
      3 = DualCLIPLoader   (clip_l + t5xxl_fp8)
      4 = LoadImage
      5 = VAEEncode
      6 = CLIPTextEncode
      7 = FluxGuidance
      8 = KSampler
      9 = VAEDecode
      10= SaveImage
    """
    return {
        "1":  {"class_type": "UNETLoader",     "inputs": {"unet_name": checkpoint.removeprefix("FLUX1/"), "weight_dtype": "fp8_e4m3fn"}},
        "2":  {"class_type": "VAELoader",      "inputs": {"vae_name": FLUX_VAE}},
        "3":  {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4":  {"class_type": "LoadImage",      "inputs": {"image": image_name}},
        "5":  {"class_type": "VAEEncode",      "inputs": {"pixels": ["4", 0], "vae": ["2", 0]}},
        "6":  {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["3", 0]}},
        "7":  {"class_type": "FluxGuidance",   "inputs": {"conditioning": ["6", 0], "guidance": guidance}},
        "8":  {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 0],
                "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise,
            },
        },
        "9":  {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


# ── ComfyUI client ────────────────────────────────────────────────────────────

class ComfyClient:
    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")
        self._http = httpx.AsyncClient(base_url=self._base, timeout=120.0)

    async def close(self): await self._http.aclose()
    async def __aenter__(self): return self
    async def __aexit__(self, *_): await self.close()

    async def health(self) -> dict:
        r = await self._http.get("/system_stats"); r.raise_for_status(); return r.json()

    async def list_checkpoints(self) -> list[str]:
        r = await self._http.get("/models/checkpoints"); r.raise_for_status(); return r.json()

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
        r = await self._http.post("/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())})
        if r.status_code == 400:
            raise RuntimeError(f"Workflow rejected: {r.text}")
        r.raise_for_status()
        data = r.json()
        if data.get("error") or data.get("node_errors"):
            raise RuntimeError(f"Validation failed: {data.get('error') or data.get('node_errors')}")
        return data["prompt_id"]

    async def wait(self, prompt_id: str, timeout: float = 300.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {prompt_id!r} timed out")
            r = await self._http.get(f"/history/{prompt_id}"); r.raise_for_status()
            data = r.json()
            if prompt_id in data:
                job = data[prompt_id]
                status = job.get("status", {})
                if status.get("status_str") == "error":
                    raise RuntimeError(f"Job failed: {status}")
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

    def _first_image(self, outputs: dict) -> str:
        for node_out in outputs.values():
            imgs = node_out.get("images", [])
            if imgs: return imgs[0]["filename"]
        raise ValueError(f"No images in outputs: {outputs}")

    async def generate(self, workflow: dict, dest: Path) -> None:
        pid = await self.submit(workflow)
        outputs = await self.wait(pid)
        fname = self._first_image(outputs)
        await self.download(fname, dest)


# ── PCA projection ────────────────────────────────────────────────────────────

def fit_pca(jobs: list[dict], n_components: int = 10):
    vecs = np.array([j["embedding"] for j in jobs], dtype=np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(vecs)
    pca = PCA(n_components=n_components, whiten=True)
    pca.fit(scaled)
    var = pca.explained_variance_ratio_
    print(f"  PCA fit on {len(vecs)} embeddings | PC1={var[0]:.3f} PC2={var[1]:.3f} PC3={var[2]:.3f}")
    return scaler, pca


def project(job: dict, scaler, pca, n_axes: int = 3, divisor: float = 2.0) -> tuple[float, ...]:
    vec = np.array(job["embedding"], dtype=np.float32).reshape(1, -1)
    pcs = pca.transform(scaler.transform(vec))[0]
    clip = lambda x: float(np.clip(x / divisor, -1.0, 1.0))
    return tuple(clip(pcs[i]) for i in range(n_axes))


# ── Main ──────────────────────────────────────────────────────────────────────

def get_sus(job: dict, source: str) -> int:
    """Get sus_level from original or rescore depending on --sus-source."""
    if source == "rescore":
        rescored = job.get("rescore", {})
        if rescored and rescored.get("sus_level") is not None:
            return rescored["sus_level"]
    return job["sus_level"]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/test_dataset.json"))
    parser.add_argument("--output", type=Path, default=None,
                        help="Output dir (default: output/dataset_faces or output/dataset_faces_v2)")
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--cohorts", nargs="*", help="Limit to these cohorts")
    parser.add_argument("--sus-source", choices=["original", "rescore"], default="original",
                        help="Which sus_level to use for denoising (default: original)")
    parser.add_argument("--face-version", type=int, choices=[1, 2, 3], default=1,
                        help="Face descriptor version: 1=3-axis narrow, 2=5-axis wide, 3=work_type primary")
    parser.add_argument("--flux", action="store_true",
                        help="Use Flux checkpoint + workflow (outputs to dataset_faces_flux)")
    parser.add_argument("--flux-checkpoint", default=FLUX_CHECKPOINT,
                        help=f"Flux checkpoint (default: {FLUX_CHECKPOINT})")
    parser.add_argument("--limit", type=int, help="Max jobs to generate")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Auto output dir
    suffix = f"_v{args.face_version}" if args.face_version > 1 else ""
    flux_prefix = "dataset_faces_flux" if args.flux else "dataset_faces"
    if args.output is None:
        args.output = Path(f"output/{flux_prefix}{suffix}")

    # axis/divisor config per version
    # v3 only needs PC2+PC3 (indices 1,2) but we fit PCA on all and slice
    n_axes   = 5 if args.face_version == 2 else 3
    divisor  = 1.0 if args.face_version == 2 else 2.0

    print(f"=== Dataset face generation ===")
    active_checkpoint = args.flux_checkpoint if args.flux else args.checkpoint
    print(f"  Input:        {args.input}")
    print(f"  Output:       {args.output}")
    print(f"  Sus source:   {args.sus_source}")
    v3_note = ", work_type primary identity" if args.face_version == 3 else ""
    print(f"  Face version: v{args.face_version} ({n_axes} PCA axes, divisor={divisor}{v3_note})")
    print(f"  Checkpoint:   {active_checkpoint}")
    if args.flux:
        print(f"  Backend:      Flux (guidance={FLUX_GUIDANCE}, sampler={FLUX_SAMPLER})")

    if not args.input.exists():
        raise SystemExit(f"Dataset not found: {args.input}")

    with open(args.input, encoding="utf-8") as f:
        dataset = json.load(f)

    jobs: list[dict] = dataset["jobs"]
    if args.cohorts:
        jobs = [j for j in jobs if j["cohort"] in args.cohorts]
    if args.limit:
        jobs = jobs[:args.limit]

    print(f"\n  Jobs:       {len(jobs)}")

    # Count skippable
    skippable = sum(
        1 for j in jobs
        if (args.output / j["cohort"] / f"{j['id']}.png").exists()
    )
    print(f"  Already done: {skippable} (will skip)")
    print(f"  To generate:  {len(jobs) - skippable}")

    # Fit PCA on all embeddings in dataset (use full dataset for stable PCA)
    print("\nFitting PCA on all dataset embeddings...")
    all_jobs_with_emb = [j for j in dataset["jobs"] if j.get("embedding")]
    scaler, pca = fit_pca(all_jobs_with_emb, n_components=max(10, n_axes))

    # Project all jobs
    for job in jobs:
        pcs = project(job, scaler, pca, n_axes=n_axes, divisor=divisor)
        sus = get_sus(job, args.sus_source)
        for i, v in enumerate(pcs):
            job[f"_pc{i+1}"] = v
        job["_sus"] = sus
        job["_denoise"] = sus_to_denoise(sus)
        job["_seed"] = embedding_seed(job["embedding"])
        if args.face_version == 3:
            job["_descriptor"] = face_descriptor_v3(job.get("work_type"), pcs[1], pcs[2], sus)
        elif args.face_version == 2:
            job["_descriptor"] = face_descriptor_v2(pcs[0], pcs[1], pcs[2], pcs[3], pcs[4], sus)
        else:
            job["_descriptor"] = face_descriptor(pcs[0], pcs[1], pcs[2], sus)

    # Preview
    print("\nSample descriptors:")
    from collections import defaultdict
    by_cohort: dict[str, list] = defaultdict(list)
    for j in jobs:
        by_cohort[j["cohort"]].append(j)
    for cohort, cjobs in sorted(by_cohort.items()):
        j = cjobs[0]
        print(f"  [{cohort}] sus={j['_sus']} denoise={j['_denoise']:.2f}")
        print(f"    {j['_descriptor'][:100]}…")

    if args.dry_run:
        print("\n[DRY RUN] Skipping generation.")
        return

    # Connect to ComfyUI
    print(f"\nConnecting to ComfyUI at {COMFY_URL}...")
    async with ComfyClient(COMFY_URL) as comfy:
        stats = await comfy.health()
        gpu = stats.get("devices", [{}])[0]
        free_gb = gpu.get("vram_free", 0) / 1024**3
        print(f"  OK | {gpu.get('name', 'GPU')} | {free_gb:.1f} GB VRAM free")

        checkpoints = await comfy.list_checkpoints()
        if active_checkpoint not in checkpoints:
            print(f"\nERROR: checkpoint not found: {active_checkpoint!r}. Available:")
            for c in checkpoints: print(f"  {c}")
            return

        if not ANCHOR_PATH.exists():
            raise SystemExit(f"Anchor not found: {ANCHOR_PATH}\nRun phase1_cluster_test.py first.")

        anchor_name = await comfy.upload_image(ANCHOR_PATH)
        print(f"  Anchor uploaded as: {anchor_name!r}")

        # Generate
        total = len(jobs)
        done = 0
        skipped = 0
        generated = 0
        errors = 0
        manifest_rows = []
        t0 = time.monotonic()

        print(f"\nGenerating {total} faces…\n")

        for job in jobs:
            job_id = job["id"]
            cohort = job["cohort"]
            dest = args.output / cohort / f"{job_id}.png"

            if dest.exists():
                skipped += 1
                done += 1
                manifest_rows.append({
                    "job_id": job_id, "cohort": cohort,
                    "sus_level": job["_sus"], "denoise": job["_denoise"],
                    **{f"pc{i+1}": round(job[f"_pc{i+1}"], 3) for i in range(n_axes)},
                    "path": str(dest), "status": "skipped",
                })
                continue

            positive = (
                f"photorealistic portrait, {job['_descriptor']}, "
                f"soft studio lighting, plain background, sharp focus, 4k"
            )
            if args.flux:
                workflow = flux_img2img_workflow(
                    checkpoint=active_checkpoint,
                    image_name=anchor_name,
                    positive=positive,
                    seed=job["_seed"],
                    steps=FLUX_STEPS, guidance=FLUX_GUIDANCE,
                    sampler=FLUX_SAMPLER, scheduler=FLUX_SCHEDULER,
                    denoise=job["_denoise"],
                    prefix=f"ds_{cohort[:8]}_{job_id[:8]}",
                )
            else:
                workflow = img2img_workflow(
                    checkpoint=active_checkpoint,
                    image_name=anchor_name,
                    positive=positive,
                    negative=ANCHOR_NEGATIVE,
                    seed=job["_seed"],
                    steps=STEPS, cfg=CFG, sampler=SAMPLER, scheduler=SCHEDULER,
                    denoise=job["_denoise"],
                    prefix=f"ds_{cohort[:8]}_{job_id[:8]}",
                )

            try:
                await comfy.generate(workflow, dest)
                generated += 1
                elapsed = time.monotonic() - t0
                rate = generated / elapsed if elapsed > 0 else 0
                remaining = (total - done - 1) / rate if rate > 0 else 0
                print(f"  [{done+1:4d}/{total}] {cohort:25s} sus={job['_sus']:3d} "
                      f"denoise={job['_denoise']:.2f}  ETA {remaining/60:.1f}min")
                manifest_rows.append({
                    "job_id": job_id, "cohort": cohort,
                    "sus_level": job["_sus"], "denoise": job["_denoise"],
                    **{f"pc{i+1}": round(job[f"_pc{i+1}"], 3) for i in range(n_axes)},
                    "path": str(dest), "status": "generated",
                })
            except Exception as e:
                errors += 1
                print(f"  [{done+1:4d}/{total}] ERROR {job_id[:8]}: {e}")
                manifest_rows.append({
                    "job_id": job_id, "cohort": cohort, "path": str(dest),
                    "status": "error", "error": str(e),
                })

            done += 1

    elapsed = time.monotonic() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min")
    print(f"  Generated: {generated}  Skipped: {skipped}  Errors: {errors}")
    print(f"\nCohort breakdown:")
    for cohort, cjobs in sorted(by_cohort.items()):
        sus_vals = [j["_sus"] for j in cjobs]
        print(f"  {cohort:25s}: {len(cjobs):3d} jobs | sus {min(sus_vals)}-{max(sus_vals)}")

    manifest_path = args.output / "manifest.json"
    args.output.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump({
            "input": str(args.input),
            "sus_source": args.sus_source,
            "checkpoint": args.checkpoint,
            "face_version": args.face_version,
            "n_pca_axes": n_axes,
            "pca_divisor": divisor,
            "total": total, "generated": generated,
            "skipped": skipped, "errors": errors,
            "jobs": manifest_rows,
        }, f, indent=2)
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    asyncio.run(main())
