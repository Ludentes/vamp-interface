"""Img2img extension of FluxSpace's GenerationPipelineSemantic.

FluxSpace upstream is txt2img only. For Stage 4.5 we want semantic edits
applied on top of a fixed anchor portrait — matching the Ours img2img
regime. This subclass adds an `img2img(...)` method that:

  1. Encodes the input image through the VAE to get the "image latent".
  2. Picks a starting timestep index from `strength` (fraction of the
     schedule to re-noise).
  3. Forms the initial noisy latent as
         latents = (1 - sigma_start) * image_latent + sigma_start * noise
     (rectified flow / Flow Matching formulation).
  4. Runs the same FluxSpace denoising loop from that starting timestep,
     applying edit_prompt / edit_global_scale / edit_content_scale as usual.

No upstream file is modified; we subclass and override nothing — we just
add `img2img`. The txt2img path (`__call__`) is left untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

VENDOR_FS = Path(__file__).resolve().parents[2] / "vendor" / "FluxSpace"
if str(VENDOR_FS) not in sys.path:
    sys.path.insert(0, str(VENDOR_FS))

from flux_semantic_pipeline import (  # type: ignore  # noqa: E402
    GenerationPipelineSemantic,
    PipelineOutput,
    calculate_shift,
    retrieve_timesteps,
)


class SemanticImg2ImgPipeline(GenerationPipelineSemantic):
    """GenerationPipelineSemantic + img2img entry point."""

    def _encode_vae_image(self, image: torch.Tensor) -> torch.Tensor:
        """image: (B, 3, H, W) in [-1, 1] on VAE device/dtype. Returns packed latents."""
        image = image.to(device=self.vae.device, dtype=self.vae.dtype)
        dist = self.vae.encode(image).latent_dist
        latents = dist.sample()  # (B, C, h, w)
        latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def img2img(
        self,
        image: Image.Image,
        prompt: Union[str, List[str]] = None,
        prompt_2: Union[str, List[str]] = None,
        strength: float = 0.9,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 30,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        max_sequence_length: int = 512,
        edit_prompt: str = None,
        edit_start_iter: int = 0,
        edit_stop_iter: Optional[int] = None,
        edit_global_scale: float = 0.0,
        edit_content_scale: float = 0.0,
        attention_threshold: float = 0.5,
    ):
        if not (0.0 < strength <= 1.0):
            raise ValueError(f"strength must be in (0, 1], got {strength}")

        # Resolve spatial size from the input image (must be VAE-divisible).
        if width is None or height is None:
            width, height = image.size
        # Round to vae_scale_factor × 2 (patch size for Flux).
        mult = self.vae_scale_factor * 2
        height = (height // mult) * mult
        width = (width // mult) * mult

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs or {}
        self._interrupt = False
        batch_size = 1
        device = self._execution_device

        # --- Text conditioning (same as __call__) ---
        prompt_embeds, pooled_prompt_embeds, text_ids = self.encode_prompt(
            prompt=prompt, prompt_2=prompt_2,
            prompt_embeds=None, pooled_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt, device=device,
        )
        edit_prompt_embeds, edit_pooled_prompt_embeds, _ = self.encode_prompt(
            prompt=edit_prompt, prompt_2=edit_prompt,
            prompt_embeds=None, pooled_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt, device=device,
        )
        neg_prompt_embeds, _, _ = self.encode_prompt(
            prompt="", prompt_2="",
            prompt_embeds=None, pooled_prompt_embeds=None,
            max_sequence_length=max_sequence_length,
            num_images_per_prompt=num_images_per_prompt, device=device,
        )

        # --- Encode input image through VAE ---
        pixels = self.image_processor.preprocess(image, height=height, width=width)
        pixels = pixels.to(device=device, dtype=prompt_embeds.dtype)
        if pixels.ndim == 3:
            pixels = pixels.unsqueeze(0)
        image_latents = self._encode_vae_image(pixels)  # (1, C, h, w)
        _, num_channels_latents, lat_h, lat_w = image_latents.shape
        image_latents_packed = self.pack_latents(
            image_latents, batch_size, num_channels_latents, lat_h, lat_w,
        )
        latent_image_ids = self.prepare_latent_image_ids(
            batch_size, lat_h // 2, lat_w // 2, device, prompt_embeds.dtype,
        )

        # --- Timesteps (same schedule as __call__) ---
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_sequence_length = image_latents_packed.shape[1]
        mu = calculate_shift(
            image_seq_len=image_sequence_length,
            base_seq_len=self.scheduler.config.base_image_seq_len,
            max_seq_len=self.scheduler.config.max_image_seq_len,
            base_shift=self.scheduler.config.base_shift,
            max_shift=self.scheduler.config.max_shift,
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            scheduler=self.scheduler, num_inference_steps=num_inference_steps,
            device=device, timesteps=None, sigmas=sigmas, mu=mu,
        )

        # Img2img: start at index t_start, skip earlier (lower-sigma) steps.
        t_start = int(round(num_inference_steps * (1.0 - strength)))
        t_start = max(0, min(t_start, num_inference_steps - 1))
        timesteps = timesteps[t_start:]

        # --- Form the noised starting latent ---
        # Flow Matching / rectified flow: x_t = (1 - sigma_t) * x_0 + sigma_t * eps
        # The scheduler stores sigmas; the starting sigma is sigmas[t_start].
        sigma_start = float(self.scheduler.sigmas[t_start])
        noise = torch.randn(
            image_latents_packed.shape, generator=generator,
            device=device, dtype=image_latents_packed.dtype,
        )
        latents = (1.0 - sigma_start) * image_latents_packed + sigma_start * noise

        self._num_timesteps = len(timesteps)

        if self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        else:
            guidance = None

        if edit_stop_iter is None:
            edit_stop_iter = num_inference_steps
        self._joint_attention_kwargs["start_timestep_idx"] = edit_start_iter
        self._joint_attention_kwargs["stop_timestep_idx"] = edit_stop_iter
        self._joint_attention_kwargs["edit_content_scale"] = edit_content_scale
        self._joint_attention_kwargs["attention_threshold"] = attention_threshold

        # Project global scale onto orthogonal complement (same as __call__).
        emb_product = torch.sum(edit_pooled_prompt_embeds * pooled_prompt_embeds, dim=1, keepdim=True)
        emb_norm_squared = torch.sum(pooled_prompt_embeds * pooled_prompt_embeds, dim=1, keepdim=True)
        emb_projection = (emb_product / (emb_norm_squared + 1e-10)) * pooled_prompt_embeds
        ortho_emb = edit_pooled_prompt_embeds - emb_projection
        assert 0.0 <= abs(edit_global_scale) <= 1.0
        edit_pooled_prompt_embeds = (1 - edit_global_scale) * pooled_prompt_embeds + edit_global_scale * ortho_emb

        # --- Denoising loop (identical body to __call__, just starting later) ---
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue
                # current_timestep_idx must be in the SAME FRAME as edit_start_iter
                # (which is expressed against full num_inference_steps). Shift by t_start.
                self._joint_attention_kwargs["current_timestep_idx"] = t_start + i
                self._joint_attention_kwargs["edit_prompt_embeds"] = self.transformer.context_embedder(edit_prompt_embeds)
                self._joint_attention_kwargs["neg_prompt_embeds"] = self.transformer.context_embedder(neg_prompt_embeds)

                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                temb_edit = self.get_temb_for_projections(
                    timestep, pooled_projections=edit_pooled_prompt_embeds,
                    guidance=(guidance * 1000) if guidance is not None else None,
                )
                self._joint_attention_kwargs["temb_edit"] = temb_edit

                noise_pred = self.transformer(
                    hidden_states=latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()

        if output_type == "latent":
            image_out = latents
        else:
            latents_unpacked = self.unpack_latents(latents, height, width, self.vae_scale_factor)
            latents_unpacked = (latents_unpacked / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image_out = self.vae.decode(latents_unpacked.to(self.vae.dtype), return_dict=False)[0]
            image_out = self.image_processor.postprocess(image_out, output_type=output_type)

        if not return_dict:
            return (image_out,)
        return PipelineOutput(images=image_out)
