"""Stage 2b — capture pooled conditioning vectors for all Stage 2 prompts.

For each of the 1785 prompts, encode through Flux's dual text encoder
(CLIP-L + T5XXL) and produce a 4864-d vector:

    concat[CLIP-L pooled (768), mean-pooled T5 last-hidden (4096)]

Weights loaded directly from ComfyUI model dir:
    /home/newub/w/ComfyUI/models/text_encoders/clip_l.safetensors
    /home/newub/w/ComfyUI/models/text_encoders/t5/t5xxl_fp16.safetensors

The T5 safetensors uses standard `transformers` T5 key naming
(encoder.block.*, shared.weight) so we instantiate a T5EncoderModel from
the public `google/t5-v1_1-xxl` config and load the state dict directly.
CLIP-L is loaded from HF cache (`openai/clip-vit-large-patch14`, already
present locally).

Outputs aligned to full_grid() sample_id order:
    output/demographic_pc/conditioning.npy        shape (N, 4864) float32
    output/demographic_pc/conditioning_ids.json   [sample_id, ...]

Usage:
    uv run python -m src.demographic_pc.stage2b_conditioning
    uv run python -m src.demographic_pc.stage2b_conditioning --limit 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import sentencepiece as spm
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    T5Config,
    T5EncoderModel,
)

from src.demographic_pc.prompts import full_grid

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc"
COMFY_ENC = Path("/home/newub/w/ComfyUI/models/text_encoders")
CLIP_L_SAFETENSORS = COMFY_ENC / "clip_l.safetensors"
T5_SAFETENSORS = COMFY_ENC / "t5" / "t5xxl_fp16.safetensors"

CLIP_HF_ID = "openai/clip-vit-large-patch14"
T5_HF_ID = "google/t5-v1_1-xxl"

CLIP_DIM = 768
T5_DIM = 4096
POOLED_DIM = CLIP_DIM + T5_DIM  # 4864

T5_MAX_LEN = 256  # Flux standard
CLIP_MAX_LEN = 77


def load_clip(device: str, dtype: torch.dtype):
    # HF snapshot already cached; tokenizer + text model come from it.
    tok = CLIPTokenizer.from_pretrained(CLIP_HF_ID)
    model = CLIPTextModel.from_pretrained(CLIP_HF_ID, torch_dtype=dtype).to(device).eval()
    return tok, model


class T5SpmTokenizer:
    """Minimal SentencePiece wrapper matching T5's tokenize-and-pad protocol.

    Uses Google's spiece.model from google/t5-v1_1-xxl directly — transformers
    5.x's CLIP/T5 converter is broken for this spm file.
    """

    EOS = 1
    PAD = 0

    def __init__(self, model_file: str, max_len: int):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.max_len = max_len

    def __call__(self, texts: list[str], device: str) -> dict[str, torch.Tensor]:
        ids_list, mask_list = [], []
        for t in texts:
            ids = self.sp.encode(t, out_type=int)[: self.max_len - 1] + [self.EOS]
            mask = [1] * len(ids)
            pad = self.max_len - len(ids)
            if pad > 0:
                ids += [self.PAD] * pad
                mask += [0] * pad
            ids_list.append(ids)
            mask_list.append(mask)
        return {
            "input_ids": torch.tensor(ids_list, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(mask_list, dtype=torch.long, device=device),
        }


def load_t5(device: str, dtype: torch.dtype):
    spm_path = hf_hub_download(T5_HF_ID, "spiece.model")
    tok = T5SpmTokenizer(spm_path, max_len=T5_MAX_LEN)
    cfg = T5Config.from_pretrained(T5_HF_ID)
    # Instantiate encoder-only then load local fp16 weights.
    model = T5EncoderModel(cfg)
    state = load_file(str(T5_SAFETENSORS))
    missing, unexpected = model.load_state_dict(state, strict=False)
    # `shared.weight` is tied across encoder input embeddings; one of them
    # may appear in "missing" but share memory via tie_weights.
    if unexpected:
        print(f"[stage2b][t5] unexpected keys: {unexpected[:5]}... ({len(unexpected)} total)")
    if missing:
        print(f"[stage2b][t5] missing keys: {missing[:5]}... ({len(missing)} total)")
    model = model.to(dtype=dtype, device=device).eval()
    return tok, model


@torch.inference_mode()
def encode_batch(
    prompts: list[str],
    clip_tok, clip_model,
    t5_tok, t5_model,
    device: str,
) -> np.ndarray:
    # CLIP-L pooled
    c_in = clip_tok(
        prompts, padding="max_length", max_length=CLIP_MAX_LEN,
        truncation=True, return_tensors="pt",
    ).to(device)
    c_out = clip_model(**c_in).pooler_output  # (B, 768)

    # T5 last-hidden mean-pooled over seq (respecting attention mask)
    t_in = t5_tok(prompts, device=device)  # {"input_ids", "attention_mask"}
    t_out = t5_model(**t_in).last_hidden_state  # (B, L, 4096)
    mask = t_in["attention_mask"].unsqueeze(-1).to(t_out.dtype)
    t_mean = (t_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # (B, 4096)
    del t_in, t_out, mask

    vec = torch.cat([c_out.float(), t_mean.float()], dim=-1).cpu().numpy()
    return vec.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    for p in (CLIP_L_SAFETENSORS, T5_SAFETENSORS):
        if not p.exists():
            raise FileNotFoundError(p)

    rows = full_grid()
    if args.limit:
        rows = rows[: args.limit]
    print(f"[stage2b] encoding {len(rows)} prompts (batch={args.batch_size})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print("[stage2b] loading CLIP-L…")
    clip_tok, clip_model = load_clip(device, dtype)
    print("[stage2b] loading T5XXL (fp16 safetensors, ~10GB)…")
    t5_tok, t5_model = load_t5(device, dtype)

    vecs = np.empty((len(rows), POOLED_DIM), dtype=np.float32)
    ids: list[str] = [r.sample_id for r in rows]
    prompts = [r.prompt for r in rows]

    t0 = time.time()
    for i in range(0, len(rows), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        vecs[i : i + len(batch)] = encode_batch(
            batch, clip_tok, clip_model, t5_tok, t5_model, device,
        )
        done = i + len(batch)
        if done % (args.batch_size * 10) == 0 or done == len(rows):
            dt = time.time() - t0
            rate = done / dt
            eta = (len(rows) - done) / rate if rate > 0 else 0
            print(f"  [{done:4d}/{len(rows)}] rate={rate:.1f}/s  eta={eta/60:.1f}min")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "conditioning.npy", vecs)
    with open(OUT_DIR / "conditioning_ids.json", "w") as f:
        json.dump(ids, f)
    print(f"[stage2b] wrote conditioning.npy {vecs.shape} to {OUT_DIR}")


if __name__ == "__main__":
    main()
