"""Inject a Kohya-format SD1.5 LoRA into PersonaLive's UNet (additive merge).

Most civitai SD1.5 LoRAs are saved in Kohya naming convention
(``lora_unet_down_blocks_0_attentions_0_transformer_blocks_0_attn1_to_q``),
which has to be remapped to diffusers convention
(``down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q``) before we
can locate the matching weight in PersonaLive's UNet.

This is the "merge into base" path — for each Kohya pair (lora_down, lora_up,
optional alpha), we write ``W += alpha_eff * (up @ down)`` into the matching
attention weight tensor. Cheaper than wrapping with peft adapters; doesn't
support unloading; fine for batch-render experiments.

Spatial attention layers only — temporal motion-module layers (the AnimateDiff
addition that's not in vanilla SD1.5) have no Kohya counterpart, and we
explicitly want to leave them untouched so the LoRA's effect is on the
appearance branch, not the temporal branch.
"""
from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file


# Kohya prefix → diffusers root.
_KOHYA_PREFIX = "lora_unet_"


def _kohya_to_diffusers(kohya_key: str) -> str | None:
    """Convert a Kohya LoRA key (without the .lora_down/.lora_up suffix)
    into the corresponding diffusers UNet parameter base name (without
    .weight). Returns None if the key isn't a UNet attention key.
    """
    if not kohya_key.startswith(_KOHYA_PREFIX):
        return None
    body = kohya_key[len(_KOHYA_PREFIX):]
    # Replace _ with . selectively. The naive strategy is to split on _
    # and rejoin with . but that breaks down_blocks → down.blocks.
    # The reliable fix: known compound names get protected.
    protect = [
        "down_blocks", "up_blocks", "mid_block",
        "transformer_blocks", "attentions",
        "to_q", "to_k", "to_v", "to_out",
        "proj_in", "proj_out",
        "time_emb_proj", "conv_in", "conv_out", "conv_norm_out",
        "norm_out", "norm_in",
    ]
    # NB: placeholders must contain no '_' — otherwise the second pass
    # `s.replace("_", ".")` shreds them and the third pass can't restore
    # the protected token. (Reviewer C1, 2026-05-06.)
    placeholders = {p: f"\x00{i}\x00" for i, p in enumerate(protect)}
    s = body
    for p, ph in placeholders.items():
        s = s.replace(p, ph)
    s = s.replace("_", ".")
    for p, ph in placeholders.items():
        s = s.replace(ph, p)
    return s


def _list_lora_pairs(state: dict[str, torch.Tensor]) -> dict[str, dict]:
    """Group raw LoRA tensor keys into {kohya_key: {'down', 'up', 'alpha'}}."""
    pairs: dict[str, dict] = {}
    for k, v in state.items():
        if k.endswith(".lora_down.weight"):
            base = k[: -len(".lora_down.weight")]
            pairs.setdefault(base, {})["down"] = v
        elif k.endswith(".lora_up.weight"):
            base = k[: -len(".lora_up.weight")]
            pairs.setdefault(base, {})["up"] = v
        elif k.endswith(".alpha"):
            base = k[: -len(".alpha")]
            pairs.setdefault(base, {})["alpha"] = v
        # Silently ignore TextEncoder LoRA pairs (lora_te_*) — PersonaLive
        # doesn't run a text encoder at inference (CFG=1, distilled).
    return pairs


def diff_lora_against_unet(
    lora_path: str | Path,
    unet_state_dict: dict[str, torch.Tensor],
) -> dict:
    """Probe-only: report match rate of a LoRA's keys against a UNet's
    parameter names. Returns counts and a sample of unmatched keys for
    debugging. Does NOT modify the UNet."""
    state = load_file(str(lora_path))
    pairs = _list_lora_pairs(state)
    unet_keys = set(unet_state_dict.keys())
    matched, unmatched = [], []
    for kohya_base, _parts in pairs.items():
        diff = _kohya_to_diffusers(kohya_base)
        if diff is None:
            unmatched.append((kohya_base, "not lora_unet_*"))
            continue
        target = f"{diff}.weight"
        if target in unet_keys:
            matched.append((kohya_base, target))
        else:
            unmatched.append((kohya_base, f"tried '{target}'"))
    return {
        "n_pairs": len(pairs),
        "n_matched": len(matched),
        "n_unmatched": len(unmatched),
        "matched_sample": matched[:5],
        "unmatched_sample": unmatched[:10],
    }


def apply_kohya_lora_to_unet(
    unet: torch.nn.Module,
    lora_path: str | Path,
    alpha: float = 1.0,
    *,
    verbose: bool = True,
) -> dict:
    """Merge a Kohya SD1.5 LoRA additively into ``unet``.

    For each (lora_down, lora_up, alpha) triple in the LoRA file, we locate
    the matching diffusers parameter ``base.weight`` and add::

        W += alpha * (lora_alpha / rank) * (up @ down)

    Returns a stats dict (n_matched, n_unmatched, sample of skipped keys).
    Mutates ``unet`` in place.
    """
    state = load_file(str(lora_path))
    pairs = _list_lora_pairs(state)
    sd = dict(unet.named_parameters())  # name -> Parameter
    matched, unmatched = 0, []

    for kohya_base, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            unmatched.append((kohya_base, "incomplete pair"))
            continue
        diff = _kohya_to_diffusers(kohya_base)
        if diff is None:
            unmatched.append((kohya_base, "not lora_unet_*"))
            continue
        target_name = f"{diff}.weight"
        if target_name not in sd:
            # Skipped silently if it's a known SD1.5-spatial layer that
            # PersonaLive's reference_unet has but den_unet doesn't, etc.
            unmatched.append((kohya_base, target_name))
            continue
        param = sd[target_name]
        down = parts["down"].to(param.device, dtype=torch.float32)
        up = parts["up"].to(param.device, dtype=torch.float32)
        rank = down.shape[0]
        # Kohya convention: scale = alpha_param / rank if alpha tensor is
        # provided, else 1.0 (rank-dependent default).
        alpha_param = parts.get("alpha")
        if alpha_param is not None:
            scale = float(alpha_param.item()) / rank
        else:
            scale = 1.0
        with torch.no_grad():
            # delta has shape (out, in) for Linear; (out, in, 1, 1) for Conv2d
            # 1×1 kernel. Most attention layers in SD1.5 are Linear, so
            # up @ down works directly. Fall back to general matmul of the
            # 4D-as-2D view if needed.
            if param.dim() == 2:
                delta = up @ down
            elif param.dim() == 4:
                # (out, rank, 1, 1) @ (rank, in, 1, 1)
                delta = torch.einsum("orxy,riab->oixy", up, down)
            else:
                unmatched.append((kohya_base, f"unsupported dim {param.dim()}"))
                continue
            param.add_(alpha * scale * delta.to(param.dtype))
        matched += 1

    if verbose:
        print(f"[lora_inject] matched={matched} unmatched={len(unmatched)} "
              f"alpha={alpha:.2f}")
    if matched == 0 and pairs:
        # Hard-fail: a sweep that silently no-ops would burn the GPU run.
        # (Reviewer C2, 2026-05-06.)
        sample = "\n  ".join(f"{k} → {r}" for k, r in unmatched[:5])
        raise RuntimeError(
            f"LoRA injection matched 0 of {len(pairs)} pairs against UNet — "
            f"key remap is likely broken. First unmatched:\n  {sample}"
        )
    return {
        "n_matched": matched,
        "n_unmatched": len(unmatched),
        "unmatched_sample": unmatched[:10],
    }


if __name__ == "__main__":
    # CLI: probe a LoRA file's match rate against PersonaLive UNets.
    #   python -m arkit_bridge.lora_inject diff /path/to/ghibli.safetensors
    import argparse
    import os
    import sys

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diff"])
    ap.add_argument("lora_path")
    args = ap.parse_args()
    # Resolve before chdir into PersonaLive — relative paths break otherwise.
    args.lora_path = str(Path(args.lora_path).resolve())

    PL = Path(os.path.expanduser("~/w/PersonaLive"))
    sys.path.insert(0, str(PL))
    os.chdir(PL)
    from omegaconf import OmegaConf
    from src.models.unet_2d_condition import UNet2DConditionModel
    from src.models.unet_3d import UNet3DConditionModel

    cfg = OmegaConf.load("configs/prompts/personalive_offline.yaml")
    infer = OmegaConf.load(cfg.inference_config)
    ref = UNet2DConditionModel.from_pretrained(cfg.pretrained_base_model_path, subfolder="unet")
    den = UNet3DConditionModel.from_pretrained_2d(
        cfg.pretrained_base_model_path, "", subfolder="unet",
        unet_additional_kwargs=infer.unet_additional_kwargs,
    )
    print("=== reference_unet (2D) ===")
    print(diff_lora_against_unet(args.lora_path, ref.state_dict()))
    print("=== denoising_unet (3D, spatial layers only) ===")
    print(diff_lora_against_unet(args.lora_path, den.state_dict()))
