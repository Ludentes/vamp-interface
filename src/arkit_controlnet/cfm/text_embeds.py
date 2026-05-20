"""One-shot encode the fixed CFM training prompt through FLUX's T5-XXL + CLIP-L.

Run: uv run python -m arkit_controlnet.cfm.text_embeds
Idempotent (skip-if-exists). Writes:
  output/cfm_precompute/text_embeds.pt = {"t5": (1,512,4096), "pooled": (1,768)}
"""
import os
from pathlib import Path

import torch

FIXED_PROMPT = "a portrait photo of a person looking at the camera"
FLUX_HF_ID = "black-forest-labs/FLUX.1-dev"
DEFAULT_OUT = "output/cfm_precompute/text_embeds.pt"


def build_text_embeds(out_path: str = DEFAULT_OUT,
                      flux_hf_id: str = FLUX_HF_ID,
                      device: str = "cuda") -> None:
    out = Path(out_path)
    if out.exists():
        print(f"[text_embeds] {out} exists, skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    from transformers import (AutoTokenizer, CLIPTextModel, CLIPTokenizer,
                              T5EncoderModel)
    dtype = torch.bfloat16

    tok_c = CLIPTokenizer.from_pretrained(flux_hf_id, subfolder="tokenizer")
    enc_c = CLIPTextModel.from_pretrained(
        flux_hf_id, subfolder="text_encoder", torch_dtype=dtype
    ).to(device).eval()
    ids_c = tok_c(FIXED_PROMPT, padding="max_length", max_length=77,
                  truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        pooled = enc_c(ids_c.input_ids, output_hidden_states=False).pooler_output
    del enc_c
    torch.cuda.empty_cache()

    tok_t = AutoTokenizer.from_pretrained(flux_hf_id, subfolder="tokenizer_2")
    enc_t = T5EncoderModel.from_pretrained(
        flux_hf_id, subfolder="text_encoder_2", torch_dtype=dtype
    ).to(device).eval()
    ids_t = tok_t(FIXED_PROMPT, padding="max_length", max_length=512,
                  truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        t5 = enc_t(ids_t.input_ids)[0]
    del enc_t
    torch.cuda.empty_cache()

    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save({"t5": t5.to("cpu", dtype),
                "pooled": pooled.to("cpu", dtype)}, tmp)
    os.replace(tmp, out)
    print(f"[text_embeds] wrote {out}: t5{tuple(t5.shape)} pooled{tuple(pooled.shape)}")


if __name__ == "__main__":
    build_text_embeds()
