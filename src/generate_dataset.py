#!/usr/bin/env python3
"""
generate_dataset.py — Generate faces for the full test dataset.

Reads data/test_dataset.json (or rescored variant), fits PCA on all embedded
embeddings, maps each job to a face descriptor, generates img2img via ComfyUI.

Output: output/dataset_faces/<cohort>/<job_id>.png     (--face-version 1, default)
        output/dataset_faces_v2/<cohort>/<job_id>.png  (--face-version 2)
        output/dataset_faces_v4/<cohort>/<job_id>.png  (--face-version 4)
        output/dataset_faces/manifest.json

All local: ComfyUI at localhost:8188 (RTX 5090). No remote dependencies.

Face versions:
  1 (default): 3 PCA axes, divisor=2.0 (narrow band spread)
  2:           5 PCA axes, divisor=1.0 (full spread) + complexion + texture axes
  3:           work_type as primary identity axis; PC2=gender, PC3=age
  4:           PaCMAP (x,y) as continuous identity axis (reads pacmap_layout.json)
               x=0: fraud/office language → slim/professional
               x=1: physical labor language → heavy/weathered
               y=0: heavy/skilled physical → older, stockier
               y=1: light/delivery/remote → younger, slimmer

Usage:
    uv run src/generate_dataset.py
    uv run src/generate_dataset.py --face-version 2
    uv run src/generate_dataset.py --face-version 4          # PaCMAP-driven
    uv run src/generate_dataset.py --face-version 4 --flux   # Flux + PaCMAP
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


# ── v4: PaCMAP (x,y) as continuous identity axis ──────────────────────────────

PACMAP_LAYOUT_PATH = Path("output/pacmap_layout.json")


def load_pacmap_coords(path: Path) -> dict[str, tuple[float, float]]:
    """Load PaCMAP layout → {job_id: (x, y)} with normalised [0,1] coords."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {p["id"]: (p["x"], p["y"]) for p in data["points"]}


def face_descriptor_v4(x: float, y: float, sus: int) -> str:
    """v4: PaCMAP (x,y) → continuous bilinear identity axis.

    x=0: fraud/office language  → slim, professional, smooth skin
    x=1: physical labor language → heavy/muscular, weathered, calloused
    y=0: heavy/skilled physical  → older, construction/loading context
    y=1: light/delivery/remote   → younger, delivery/warehouse/remote context
    """
    # Build (x-primary)
    if x < 0.25:
        build = "slender build, light frame"
    elif x < 0.5:
        build = "slim average build"
    elif x < 0.75:
        build = "stocky build, broad shoulders"
    else:
        build = "heavy muscular build, calloused hands"

    # Age (x adds years, y=low adds years — physical+heavy = older)
    age_score = x * 0.4 + (1.0 - y) * 0.6
    if age_score < 0.25:
        age = "22-32 years old"
    elif age_score < 0.45:
        age = "28-38 years old"
    elif age_score < 0.65:
        age = "35-48 years old"
    else:
        age = "42-58 years old"

    # Skin texture (x-driven: office→smooth, physical→weathered)
    if x < 0.33:
        skin = "smooth skin, indoor complexion"
    elif x < 0.67:
        skin = "normal skin texture"
    else:
        skin = "weathered skin, sun-exposed, work-hardened"

    # Work context / clothing (2D quadrant)
    if x < 0.5 and y > 0.5:
        context = "business casual or remote work attire, neat grooming"
    elif x < 0.5 and y <= 0.5:
        context = "nondescript casual clothing, indoor worker appearance"
    elif x >= 0.5 and y > 0.5:
        context = "practical workwear, warehouse or delivery uniform"
    else:  # x >= 0.5, y <= 0.5
        context = "heavy work clothing, construction or loading dock wear"

    affect = _uncanny_affect(sus)
    return f"{build}, {age}, {skin}, {context}, {affect}"


# ── v5: PaCMAP + amplified gain + geometric uncanny + job_id variety ──────────

def sus_to_denoise_v5(sus: int) -> float:
    """v5 calibration: 0.20 (sus=0) → 0.85 (sus=100). Much wider swing than v1-v4."""
    return 0.20 + (sus / 100.0) * 0.65


def _amp(v: float, gain: float = 1.8) -> float:
    """Stretch [0,1] away from centre. gain>1 pushes extremes further apart.
    Makes faces in the middle of PaCMAP space more distinct from each other."""
    c = v - 0.5
    return max(0.0, min(1.0, 0.5 + c * gain))


# Deterministic per-job variety derived from job_id hash
_EYE_COLORS   = ["blue-grey eyes", "brown eyes", "green eyes", "dark brown eyes", "hazel eyes"]
_HAIR_COLORS  = ["dark hair", "brown hair", "black hair", "greying hair", "salt-and-pepper hair"]


def _job_variety(job_id: str) -> tuple[str, str]:
    """Return (eye_color, hair_color) deterministically from job_id."""
    raw = int(hashlib.md5(job_id.encode()).hexdigest()[:8], 16)
    eye  = _EYE_COLORS[raw % len(_EYE_COLORS)]
    hair = _HAIR_COLORS[(raw >> 8) % len(_HAIR_COLORS)]
    return eye, hair


def _uncanny_affect_v5(sus: int) -> str:
    """Five-tier affect: genuine warmth → geometric horror. Each tier names specific
    structural defects, not moods — Flux responds to geometry, not adjectives."""
    if sus >= 85:
        return (
            "face geometry subtly but unmistakably wrong: eyes fractionally misaligned, "
            "bilateral symmetry violated, teeth too numerous or wrongly spaced, "
            "jaw angle inconsistent with skull, skin texture hyperrealistic yet impossible, "
            "expression does not resolve into any recognisable emotion, "
            "hollow presence behind the eyes, deeply unsettling, liminal horror"
        )
    if sus >= 65:
        return (
            "smile slightly too wide for the face, eyes too still, "
            "micro-expressions firing out of sync with the main expression, "
            "proportions off in ways that are hard to articulate, "
            "performative affect over emptiness, disturbing at second glance"
        )
    if sus >= 40:
        return (
            "avoidant gaze, subtle asymmetry that feels wrong rather than natural, "
            "guarded expression masking something, vaguely uneasy affect"
        )
    if sus >= 15:
        return "slightly guarded, neutral-to-tense, faint unease"
    return "natural open gaze, genuine warm expression, relaxed, trustworthy"


def face_descriptor_v5(x: float, y: float, job_id: str, sus: int) -> str:
    """v5: PaCMAP (x,y) with amplified gain + job_id variety + geometric uncanny.

    Changes from v4:
    - x, y amplified (gain=1.8) → more spread between similar-position faces
    - Exact age (not range) derived continuously from position
    - Eye color and hair color from job_id hash → unique per job
    - Affect prompts describe geometry, not mood
    """
    xa = _amp(x)
    ya = _amp(y)
    eye, hair = _job_variety(job_id)

    # Build (x-primary, 5 buckets now)
    if xa < 0.2:   build = "very slender build, light frame"
    elif xa < 0.4: build = "slim build"
    elif xa < 0.6: build = "average build"
    elif xa < 0.8: build = "stocky build, broad shoulders"
    else:          build = "heavy muscular build, calloused hands"

    # Exact age from continuous score + small hash jitter
    age_score = xa * 0.35 + (1.0 - ya) * 0.55
    h_jitter  = (int(hashlib.md5(job_id.encode()).hexdigest()[8:12], 16) / 65535.0) * 0.10
    age_val   = int(22 + (age_score + h_jitter) * 36)
    age_val   = max(22, min(58, age_val))
    age       = f"{age_val} years old"

    # Skin (x-driven)
    if xa < 0.3:   skin = "pale indoor complexion, smooth skin"
    elif xa < 0.6: skin = "normal skin texture"
    else:          skin = "weathered skin, sun-exposed, work-hardened"

    # Work context quadrant
    if xa < 0.5 and ya > 0.5:   context = "business casual, neat grooming"
    elif xa < 0.5 and ya <= 0.5: context = "nondescript casual clothing"
    elif xa >= 0.5 and ya > 0.5: context = "practical workwear, warehouse or delivery uniform"
    else:                         context = "heavy work clothing, construction wear"

    affect   = _uncanny_affect_v5(sus)
    identity = f"{build}, {age}, {eye}, {hair}, {skin}, {context}"
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
                          seed, steps, guidance, sampler, scheduler, denoise, prefix,
                          lora_cursed: float = 0.0, lora_eerie: float = 0.0) -> dict:
    """Flux img2img workflow with optional sus-scaled LoRA stack.

    Node map (no LoRAs):
      1 = UNETLoader, 2 = VAELoader, 3 = DualCLIPLoader
      4 = LoadImage, 5 = VAEEncode, 6 = CLIPTextEncode
      7 = FluxGuidance, 8 = KSampler, 9 = VAEDecode, 10 = SaveImage

    With LoRAs, LoraLoader nodes are inserted between UNETLoader and KSampler.
    Each LoraLoader passes both model + clip outputs forward.
    Nodes 11, 12 = LoraLoader (Cursed, Eerie). Model ref for KSampler updated.
    """
    unet_name = checkpoint.removeprefix("FLUX1/")
    base = {
        "1":  {"class_type": "UNETLoader",     "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2":  {"class_type": "VAELoader",      "inputs": {"vae_name": FLUX_VAE}},
        "3":  {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4":  {"class_type": "LoadImage",      "inputs": {"image": image_name}},
        "5":  {"class_type": "VAEEncode",      "inputs": {"pixels": ["4", 0], "vae": ["2", 0]}},
    }

    # LoRA chain: UNETLoader → LoraLoader(Cursed) → LoraLoader(Eerie) → KSampler
    # Only add nodes when weight > 0 to keep workflow clean for legit faces
    model_ref = ["1", 0]  # output 0 = model
    clip_ref   = ["3", 0]

    if lora_cursed > 0.0:
        base["11"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref, "clip": clip_ref,
                "lora_name": "Cursed_LoRA_Flux.safetensors",
                "strength_model": round(lora_cursed, 3),
                "strength_clip":  round(lora_cursed * 0.5, 3),  # lighter on CLIP
            },
        }
        model_ref = ["11", 0]
        clip_ref  = ["11", 1]

    if lora_eerie > 0.0:
        base["12"] = {
            "class_type": "LoraLoader",
            "inputs": {
                "model": model_ref, "clip": clip_ref,
                "lora_name": "Eerie_horror_portraits.safetensors",
                "strength_model": round(lora_eerie, 3),
                "strength_clip":  round(lora_eerie * 0.5, 3),
            },
        }
        model_ref = ["12", 0]
        clip_ref  = ["12", 1]

    base["6"]  = {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": clip_ref}}
    base["7"]  = {"class_type": "FluxGuidance",   "inputs": {"conditioning": ["6", 0], "guidance": guidance}}
    base["8"]  = {
        "class_type": "KSampler",
        "inputs": {
            "model": model_ref, "positive": ["7", 0], "negative": ["7", 0],
            "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise,
        },
    }
    base["9"]  = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}}
    base["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}}
    return base


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
    parser.add_argument("--face-version", type=int, choices=[1, 2, 3, 4, 5], default=1,
                        help="Face descriptor version: 1=3-axis narrow, 2=5-axis wide, 3=work_type primary, 4=PaCMAP (x,y) continuous, 5=PaCMAP+gain+geometric uncanny")
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

    # axis/divisor config per version (v4 skips PCA entirely)
    n_axes   = 5 if args.face_version == 2 else 3
    divisor  = 1.0 if args.face_version == 2 else 2.0

    print(f"=== Dataset face generation ===")
    active_checkpoint = args.flux_checkpoint if args.flux else args.checkpoint
    print(f"  Input:        {args.input}")
    print(f"  Output:       {args.output}")
    print(f"  Sus source:   {args.sus_source}")
    if args.face_version == 5:
        print(f"  Face version: v5 (PaCMAP x,y + gain amplification + geometric uncanny)")
        print(f"  Denoise range: 0.20 (sus=0) → 0.85 (sus=100)")
    elif args.face_version == 4:
        print(f"  Face version: v4 (PaCMAP x,y continuous identity axis)")
    elif args.face_version == 3:
        print(f"  Face version: v3 (work_type primary, {n_axes} PCA axes, divisor={divisor})")
    else:
        print(f"  Face version: v{args.face_version} ({n_axes} PCA axes, divisor={divisor})")
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

    # v4/v5: load PaCMAP coordinates
    pacmap_coords: dict[str, tuple[float, float]] = {}
    if args.face_version in (4, 5):
        layout_path = PACMAP_LAYOUT_PATH
        if not layout_path.exists():
            raise SystemExit(
                f"PaCMAP layout not found: {layout_path}\n"
                "Run: uv run src/build_pacmap_layout.py"
            )
        pacmap_coords = load_pacmap_coords(layout_path)
        print(f"\nLoaded PaCMAP layout: {len(pacmap_coords)} points from {layout_path}")
        missing = [j["id"] for j in jobs if j["id"] not in pacmap_coords]
        if missing:
            print(f"  WARNING: {len(missing)} jobs have no PaCMAP coords — will use centroid (0.5, 0.5)")

    # Fit PCA on all embeddings in dataset (skip for v4/v5 — not needed)
    scaler, pca = None, None
    if args.face_version not in (4, 5):
        print("\nFitting PCA on all dataset embeddings...")
        all_jobs_with_emb = [j for j in dataset["jobs"] if j.get("embedding")]
        scaler, pca = fit_pca(all_jobs_with_emb, n_components=max(10, n_axes))

    # Project / prepare all jobs
    for job in jobs:
        sus = get_sus(job, args.sus_source)
        job["_sus"] = sus
        job["_denoise"] = sus_to_denoise(sus)
        job["_seed"] = embedding_seed(job["embedding"])

        if args.face_version in (4, 5):
            x, y = pacmap_coords.get(job["id"], (0.5, 0.5))
            job["_x"] = x
            job["_y"] = y
            if args.face_version == 5:
                job["_descriptor"] = face_descriptor_v5(x, y, job["id"], sus)
                job["_denoise"]    = sus_to_denoise_v5(sus)
            else:
                job["_descriptor"] = face_descriptor_v4(x, y, sus)
        else:
            pcs = project(job, scaler, pca, n_axes=n_axes, divisor=divisor)
            for i, v in enumerate(pcs):
                job[f"_pc{i+1}"] = v
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
                coord_fields = (
                    {"x": round(job["_x"], 4), "y": round(job["_y"], 4)}
                    if args.face_version in (4, 5)
                    else {f"pc{i+1}": round(job[f"_pc{i+1}"], 3) for i in range(n_axes)}
                )
                manifest_rows.append({
                    "job_id": job_id, "cohort": cohort,
                    "sus_level": job["_sus"], "denoise": job["_denoise"],
                    **coord_fields,
                    "path": str(dest), "status": "skipped",
                })
                continue

            sus = job["_sus"]
            lora_prefix = ""
            lora_cursed = 0.0
            lora_eerie  = 0.0
            guidance    = FLUX_GUIDANCE
            if args.face_version == 5:
                t = (sus / 100.0) ** 0.8   # nonlinear — ramps hard at high sus
                lora_cursed = t * 1.0      # 0.0 → 1.0 (full cursed at sus=100)
                lora_eerie  = t * 0.75     # 0.0 → 0.75
                guidance    = 3.5 + t * 2.5  # 3.5 → 6.0
                if sus >= 85:
                    lora_prefix = "cursed, scary looking, sharp teeth, many teeth, eerie, hollow, haunting, "
                elif sus >= 65:
                    lora_prefix = "cursed, eerie, hollow, haunting, "
                elif sus >= 40:
                    lora_prefix = "eerie, haunting, "
            positive = (
                f"photorealistic portrait, {lora_prefix}{job['_descriptor']}, "
                f"soft studio lighting, plain background, sharp focus, 4k"
            )
            if args.flux:
                workflow = flux_img2img_workflow(
                    checkpoint=active_checkpoint,
                    image_name=anchor_name,
                    positive=positive,
                    seed=job["_seed"],
                    steps=FLUX_STEPS, guidance=guidance,
                    sampler=FLUX_SAMPLER, scheduler=FLUX_SCHEDULER,
                    denoise=job["_denoise"],
                    prefix=f"ds_{cohort[:8]}_{job_id[:8]}",
                    lora_cursed=lora_cursed,
                    lora_eerie=lora_eerie,
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
                coord_label = (
                    f"x={job['_x']:.3f} y={job['_y']:.3f}"
                    if args.face_version in (4, 5)
                    else f"pc1={job.get('_pc1', 0):.2f}"
                )
                print(f"  [{done+1:4d}/{total}] {cohort:25s} sus={job['_sus']:3d} "
                      f"denoise={job['_denoise']:.2f} {coord_label}  ETA {remaining/60:.1f}min")
                coord_fields = (
                    {"x": round(job["_x"], 4), "y": round(job["_y"], 4)}
                    if args.face_version in (4, 5)
                    else {f"pc{i+1}": round(job[f"_pc{i+1}"], 3) for i in range(n_axes)}
                )
                manifest_rows.append({
                    "job_id": job_id, "cohort": cohort,
                    "sus_level": job["_sus"], "denoise": job["_denoise"],
                    **coord_fields,
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
