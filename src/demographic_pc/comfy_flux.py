"""Thin Flux Krea ComfyUI client — text2img + img2img workflow builders.

Distilled from v1/src/generate_dataset.py (dropped LoRAs, PCA, DB).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from pathlib import Path

import httpx

import os as _os
FLUX_CHECKPOINT = _os.environ.get("COMFY_UNET", "FLUX1/flux1-krea-dev_fp8_scaled.safetensors")
FLUX_VAE = _os.environ.get("COMFY_VAE", "FLUX1/ae.safetensors")
FLUX_CLIP_L = _os.environ.get("COMFY_CLIP_L", "clip_l.safetensors")
FLUX_T5 = _os.environ.get("COMFY_T5", "t5/t5xxl_fp8_e4m3fn.safetensors")
FLUX_STEPS = 20
FLUX_GUIDANCE = 3.5
FLUX_SAMPLER = "euler"
FLUX_SCHEDULER = "simple"


def flux_txt2img_workflow(
    positive: str, seed: int, width: int, height: int, prefix: str,
    steps: int = FLUX_STEPS, guidance: float = FLUX_GUIDANCE,
    sampler: str = FLUX_SAMPLER, scheduler: str = FLUX_SCHEDULER,
    checkpoint: str = FLUX_CHECKPOINT,
    edit_npz_path: str | None = None, edit_strength: float = 0.0,
) -> dict:
    unet_name = checkpoint if _os.environ.get("COMFY_UNET") else checkpoint.removeprefix("FLUX1/")
    # Optionally insert demographic_pc ApplyConditioningEdit between CLIPTextEncode
    # and FluxGuidance. When edit_strength == 0 we still route through the node so
    # the graph shape is identical across renders (no-op path inside the node).
    text_out_node = "5"
    guidance_input = ["5", 0]
    extra: dict = {}
    if edit_npz_path is not None:
        extra["10"] = {
            "class_type": "ApplyConditioningEdit",
            "inputs": {
                "conditioning": ["5", 0],
                "edit_npz_path": edit_npz_path,
                "strength": edit_strength,
            },
        }
        guidance_input = ["10", 0]
    _ = text_out_node  # debug aid
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": width, "height": height, "batch_size": 1}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["3", 0]}},
        **extra,
        "6": {"class_type": "FluxGuidance", "inputs": {"conditioning": guidance_input, "guidance": guidance}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["6", 0], "negative": ["6", 0],
                "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}},
    }


def flux_img2img_workflow(
    image_name: str, positive: str, seed: int, denoise: float, prefix: str,
    steps: int = FLUX_STEPS, guidance: float = FLUX_GUIDANCE,
    sampler: str = FLUX_SAMPLER, scheduler: str = FLUX_SCHEDULER,
    checkpoint: str = FLUX_CHECKPOINT,
) -> dict:
    unet_name = checkpoint if _os.environ.get("COMFY_UNET") else checkpoint.removeprefix("FLUX1/")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["4", 0], "vae": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive, "clip": ["3", 0]}},
        "7": {"class_type": "FluxGuidance", "inputs": {"conditioning": ["6", 0], "guidance": guidance}},
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0], "positive": ["7", 0], "negative": ["7", 0],
                "latent_image": ["5", 0], "seed": seed, "steps": steps, "cfg": 1.0,
                "sampler_name": sampler, "scheduler": scheduler, "denoise": denoise,
            },
        },
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": prefix}},
    }


class ComfyClient:
    def __init__(self, base_url: str | None = None):
        import os
        base_url = base_url or os.environ.get("COMFY_URL", "http://localhost:8188")
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=180.0)

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "ComfyClient":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

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
        r = await self._http.post(
            "/prompt", json={"prompt": workflow, "client_id": str(uuid.uuid4())}
        )
        r.raise_for_status()
        data = r.json()
        if data.get("error") or data.get("node_errors"):
            raise RuntimeError(f"Validation failed: {data.get('error') or data.get('node_errors')}")
        return data["prompt_id"]

    async def wait(self, prompt_id: str, timeout: float = 600.0) -> dict:
        deadline = time.monotonic() + timeout
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Job {prompt_id} timed out")
            r = await self._http.get(f"/history/{prompt_id}")
            r.raise_for_status()
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
        dest.parent.mkdir(parents=True, exist_ok=True)
        async with self._http.stream(
            "GET", "/view",
            params={"filename": filename, "subfolder": "", "type": "output"},
        ) as r:
            r.raise_for_status()
            with open(dest, "wb") as fh:
                async for chunk in r.aiter_bytes():
                    fh.write(chunk)

    @staticmethod
    def first_image(outputs: dict) -> str:
        for node_out in outputs.values():
            imgs = node_out.get("images", [])
            if imgs:
                return imgs[0]["filename"]
        raise ValueError(f"No images in outputs: {outputs}")

    async def generate(self, workflow: dict, dest: Path) -> None:
        pid = await self.submit(workflow)
        outputs = await self.wait(pid)
        await self.download(self.first_image(outputs), dest)
