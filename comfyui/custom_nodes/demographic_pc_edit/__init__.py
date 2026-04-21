"""ApplyConditioningEdit — inject a precomputed direction into Flux CONDITIONING.

Takes a CONDITIONING (from CLIPTextEncode), an .npz with
    {"pooled_delta": (768,), "seq_delta": (4096,)}
and a lambda scalar. Adds λ·pooled_delta to the CLIP-L pooled vector and
λ·seq_delta broadcast to every T5 token. This matches the 4864-d
concat[CLIP-pool, T5-mean] regression setup used by stage4_regression.py:
adding a uniform shift δ to every T5 token shifts the mean by δ.

Place flow as: CLIPTextEncode → ApplyConditioningEdit → FluxGuidance → KSampler.
"""

from __future__ import annotations

import numpy as np
import torch


class ApplyConditioningEdit:
    CATEGORY = "demographic_pc"
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "edit_npz_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 0.0, "min": -1000.0, "max": 1000.0, "step": 0.001}),
            }
        }

    def apply(self, conditioning, edit_npz_path, strength):
        if not edit_npz_path or strength == 0.0:
            return (conditioning,)
        data = np.load(edit_npz_path)
        pooled_delta = torch.from_numpy(data["pooled_delta"]).float()  # (768,)
        seq_delta = torch.from_numpy(data["seq_delta"]).float()        # (4096,)
        out = []
        for cond, opts in conditioning:
            # cond: (B, L, 4096) T5 seq; opts["pooled_output"]: (B, 768) CLIP-L pool
            sd = seq_delta.to(cond.device, cond.dtype)
            new_cond = cond + strength * sd[None, None, :]
            new_opts = dict(opts)
            if "pooled_output" in new_opts and new_opts["pooled_output"] is not None:
                p = new_opts["pooled_output"]
                pd = pooled_delta.to(p.device, p.dtype)
                new_opts["pooled_output"] = p + strength * pd[None, :]
            out.append([new_cond, new_opts])
        return (out,)


NODE_CLASS_MAPPINGS = {"ApplyConditioningEdit": ApplyConditioningEdit}
NODE_DISPLAY_NAME_MAPPINGS = {"ApplyConditioningEdit": "Apply Conditioning Edit (demographic_pc)"}
