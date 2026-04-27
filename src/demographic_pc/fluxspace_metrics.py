"""FluxSpace metrics pipeline: calibration + measurement + collapse prediction.

Steps:
  1. Calibration — run N diverse base-only Flux renders with FluxSpaceBaseMeasure.
     Per (block_idx, step_idx) we capture attn_base reductions (per-D mean, rms,
     Frobenius, max-abs). Estimate on-manifold distribution from these.
  2. Measurement — run FluxSpaceEditPair at scale=1.0 mix_b=0.5 with measure_path
     on seeds we already have images for (2026, 4242). Capture δ_A, δ_B, δ_mean,
     steered attn, pair-wise cosines.
  3. Analysis — compute a family of predictors for collapse; compare against
     known sweep labels (s ∈ {-2,-1,-0.5,0.5,1,1.5,2,3} → {collapse/safe}).

Predictors we try:
  * Cheap heuristic:   max|attn_s| = max over (block,step) of |attn_base + s·δ|
  * Frobenius curve:   ‖attn_s‖_F = quadratic in s
  * Per-D Mahalanobis: d_M(attn_s) vs diagonal-Σ from calibration (quadratic in s)
  * Per-D z-score max: max over (block,step,d) of |attn_s_d − μ_d|/σ_d

All four are analytic quadratics in s per (block, step, d) — one measurement
pass at scale=1.0 gives us the full curves for free.

Usage:
    uv run python -m src.demographic_pc.fluxspace_metrics --calibrate
    uv run python -m src.demographic_pc.fluxspace_metrics --measure
    uv run python -m src.demographic_pc.fluxspace_metrics --analyze
"""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
from pathlib import Path

import numpy as np

from src.demographic_pc.comfy_flux import (
    ComfyClient, FLUX_CHECKPOINT, FLUX_VAE, FLUX_CLIP_L, FLUX_T5,
    FLUX_STEPS, FLUX_GUIDANCE, FLUX_SAMPLER, FLUX_SCHEDULER,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "demographic_pc" / "fluxspace_metrics"
CAL_DIR = OUT / "calibration"
MEAS_DIR = OUT / "measurement"
ANAL_DIR = OUT / "analysis"

W, H = 512, 512
SCALE_SWEEP_LABELS = {
    -2.0: "collapse", -1.5: "collapse",
    -1.0: "collapse", -0.9: "collapse", -0.8: "collapse", -0.7: "collapse",
    -0.6: "safe", -0.5: "safe", -0.3: "safe",
    0.0: "safe", 0.3: "safe", 0.5: "safe",
    0.7: "safe", 0.8: "safe", 0.9: "safe",
    1.0: "safe", 1.1: "safe", 1.2: "safe", 1.3: "safe",
    1.4: "collapse", 1.5: "collapse", 1.6: "collapse",
    1.7: "collapse", 1.8: "collapse", 1.9: "collapse",
    2.0: "collapse", 2.5: "collapse", 3.0: "collapse",
}
MEASURE_SEEDS = [2026, 4242]

# Calibration prompts — diverse demographics, consistent photographic style.
CALIBRATION_PROMPTS = [
    ("adult_latin_f",    "A photorealistic portrait photograph of an adult Latin American woman, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("adult_asian_m",    "A photorealistic portrait photograph of an adult East Asian man, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("adult_black_f",    "A photorealistic portrait photograph of an adult Black woman, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("adult_european_m", "A photorealistic portrait photograph of an adult European man, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("adult_middle_f",   "A photorealistic portrait photograph of an adult Middle Eastern woman, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("elderly_latin_m",  "A photorealistic portrait photograph of an elderly Latin American man, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("elderly_asian_f",  "A photorealistic portrait photograph of an elderly East Asian woman, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("young_black_m",    "A photorealistic portrait photograph of a young Black man, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("young_european_f", "A photorealistic portrait photograph of a young European woman, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("adult_southasian_f","A photorealistic portrait photograph of an adult South Asian woman, neutral expression, plain grey background, studio lighting, sharp focus."),
]
CAL_SEED_BASE = 10000

PAIR_BASE_PROMPT = (
    "A photorealistic portrait photograph of an adult Latin American woman, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)
PAIR_EDIT_A = "A person wearing thick-rimmed eyeglasses."
PAIR_EDIT_B = (
    "A photorealistic portrait photograph of an adult Latin American woman "
    "wearing thick-rimmed eyeglasses, neutral expression, plain grey "
    "background, studio lighting, sharp focus."
)

# Cross-demographic confirmation (glasses axis, same A, per-base spliced B).
# Each entry: (short_name, base_prompt, splice_prompt).
CROSS_BASES = [
    ("asian_m", CALIBRATION_PROMPTS[1][1],
     "A photorealistic portrait photograph of an adult East Asian man wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("black_f", CALIBRATION_PROMPTS[2][1],
     "A photorealistic portrait photograph of an adult Black woman wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("european_m", CALIBRATION_PROMPTS[3][1],
     "A photorealistic portrait photograph of an adult European man wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1],
     "A photorealistic portrait photograph of an elderly Latin American man wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("young_european_f", CALIBRATION_PROMPTS[8][1],
     "A photorealistic portrait photograph of a young European woman wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
    ("southasian_f", CALIBRATION_PROMPTS[9][1],
     "A photorealistic portrait photograph of an adult South Asian woman wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
]
CROSS_SEED = 2026
CROSS_DIR = OUT / "crossdemo"

# Smile axis: analogous A=bare / B=splice pattern.
# Base prompts include "neutral expression" — splice replaces with "smiling warmly".
SMILE_EDIT_A = "A person smiling warmly."
SMILE_BASE_PROMPT = (
    "A photorealistic portrait photograph of an adult Latin American woman, "
    "neutral expression, plain grey background, studio lighting, sharp focus."
)
SMILE_EDIT_B_LATIN = (
    "A photorealistic portrait photograph of an adult Latin American woman "
    "smiling warmly, plain grey background, studio lighting, sharp focus."
)

def _smile_splice(demo_phrase: str) -> str:
    return (f"A photorealistic portrait photograph of {demo_phrase} smiling warmly, "
            f"plain grey background, studio lighting, sharp focus.")

SMILE_CROSS_BASES = [
    ("asian_m", CALIBRATION_PROMPTS[1][1], _smile_splice("an adult East Asian man")),
    ("black_f", CALIBRATION_PROMPTS[2][1], _smile_splice("an adult Black woman")),
    ("european_m", CALIBRATION_PROMPTS[3][1], _smile_splice("an adult European man")),
    ("elderly_latin_m", CALIBRATION_PROMPTS[5][1], _smile_splice("an elderly Latin American man")),
    ("young_european_f", CALIBRATION_PROMPTS[8][1], _smile_splice("a young European woman")),
    ("southasian_f", CALIBRATION_PROMPTS[9][1], _smile_splice("an adult South Asian woman")),
]

AXES = {
    "glasses": {
        "edit_a": PAIR_EDIT_A,
        "bases": CROSS_BASES,
        "latin_base": PAIR_BASE_PROMPT,
        "latin_splice": PAIR_EDIT_B,
        "clip_pos": "a photo of a person wearing glasses",
        "clip_neg": "a photo of a person not wearing glasses",
    },
    "smile": {
        "edit_a": SMILE_EDIT_A,
        "bases": SMILE_CROSS_BASES,
        "latin_base": SMILE_BASE_PROMPT,
        "latin_splice": SMILE_EDIT_B_LATIN,
        "clip_pos": "a photo of a person smiling",
        "clip_neg": "a photo of a person with a neutral expression",
    },
}


def _axis_dir(axis: str) -> Path:
    return CROSS_DIR if axis == "glasses" else CROSS_DIR / axis


# --- workflow builders -------------------------------------------------------

def _base_workflow() -> dict:
    import os as _os
    unet_name = FLUX_CHECKPOINT if _os.environ.get("COMFY_UNET") else FLUX_CHECKPOINT.removeprefix("FLUX1/")
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "fp8_e4m3fn"}},
        "2": {"class_type": "VAELoader", "inputs": {"vae_name": FLUX_VAE}},
        "3": {"class_type": "DualCLIPLoader", "inputs": {"clip_name1": FLUX_CLIP_L, "clip_name2": FLUX_T5, "type": "flux"}},
        "4": {"class_type": "EmptySD3LatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
    }


def calibration_workflow(prompt: str, seed: int, measure_path: str, prefix: str) -> dict:
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["10"] = {
        "class_type": "FluxSpaceBaseMeasure",
        "inputs": {"model": ["1", 0], "measure_path": measure_path},
    }
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["10", 0], "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


def pair_compose_workflow(seed: int, prefix: str,
                          base_prompt: str,
                          pairs: list[dict]) -> dict:
    """Chain N FluxSpaceEditPair nodes on the same base render.

    Each `pairs` entry: {edit_a, edit_b, scale, start_percent, end_percent,
    measure_path (optional)}. The node's .patch() clones the model and adds
    an attn-output patch; chaining two nodes stacks the patches sequentially.
    Open question answered by running this: whether ComfyUI's patch stacking
    preserves the per-node edit caches distinctly (so each pair steers
    independently) or collapses them.
    """
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    prev_model_ref = ["1", 0]
    next_node_id = 20
    for i, pair in enumerate(pairs):
        ea_id = str(next_node_id);      next_node_id += 1
        eb_id = str(next_node_id);      next_node_id += 1
        patch_id = str(next_node_id);   next_node_id += 1
        wf[ea_id] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": pair["edit_a"], "clip": ["3", 0]}}
        wf[eb_id] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": pair["edit_b"], "clip": ["3", 0]}}
        pair_inputs = {
            "model": prev_model_ref,
            "edit_conditioning_a": [ea_id, 0],
            "edit_conditioning_b": [eb_id, 0],
            "scale": float(pair["scale"]), "mix_b": 0.5,
            "start_percent": float(pair.get("start_percent", 0.15)),
            "end_percent": float(pair.get("end_percent", 1.0)),
            "double_blocks_only": False, "verbose": False,
        }
        if pair.get("measure_path"):
            pair_inputs["measure_path"] = pair["measure_path"]
        wf[patch_id] = {"class_type": "FluxSpaceEditPair", "inputs": pair_inputs}
        prev_model_ref = [patch_id, 0]
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": prev_model_ref, "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


def pair_multi_measure_workflow(seed: int, measure_path: str | None, prefix: str,
                                base_prompt: str,
                                pairs: list[dict],
                                start_percent: float = 0.15,
                                end_percent: float = 1.0) -> dict:
    """Additive N-pair composition via the FluxSpaceEditPairMulti node.

    Each entry in `pairs`: {edit_a, edit_b, scale, mix_b (optional, default 0.5)}.
    Up to 4 pairs. Patches the model in one attention hook that sums per-slot
    deltas in the base pass — which chaining separate FluxSpaceEditPair nodes
    cannot do (single-slot attn1 patch overwrites).
    """
    if not pairs:
        raise ValueError("pairs must be non-empty")
    if len(pairs) > 4:
        raise ValueError("FluxSpaceEditPairMulti supports at most 4 slots")
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}

    multi_inputs: dict = {
        "model": ["1", 0],
        "start_percent": float(start_percent),
        "end_percent": float(end_percent),
        "double_blocks_only": False,
        "verbose": False,
    }
    next_id = 30
    for slot_i, pair in enumerate(pairs, start=1):
        ea_id = str(next_id); next_id += 1
        eb_id = str(next_id); next_id += 1
        wf[ea_id] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": pair["edit_a"], "clip": ["3", 0]}}
        wf[eb_id] = {"class_type": "CLIPTextEncode",
                     "inputs": {"text": pair["edit_b"], "clip": ["3", 0]}}
        multi_inputs[f"edit_cond_a_{slot_i}"] = [ea_id, 0]
        multi_inputs[f"edit_cond_b_{slot_i}"] = [eb_id, 0]
        multi_inputs[f"scale_{slot_i}"] = float(pair["scale"])
        multi_inputs[f"mix_b_{slot_i}"] = float(pair.get("mix_b", 0.5))
        if pair.get("label"):
            multi_inputs[f"label_{slot_i}"] = str(pair["label"])
    if measure_path:
        multi_inputs["measure_path"] = measure_path
    wf["29"] = {"class_type": "FluxSpaceEditPairMulti", "inputs": multi_inputs}

    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["29", 0], "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


def pair_measure_workflow(seed: int, measure_path: str | None, prefix: str,
                          base_prompt: str = PAIR_BASE_PROMPT,
                          edit_a: str = PAIR_EDIT_A,
                          edit_b: str = PAIR_EDIT_B,
                          scale: float = 1.0,
                          start_percent: float = 0.15,
                          end_percent: float = 1.0) -> dict:
    wf = _base_workflow()
    wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": base_prompt, "clip": ["3", 0]}}
    wf["6"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": ["5", 0], "guidance": FLUX_GUIDANCE}}
    wf["20"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit_a, "clip": ["3", 0]}}
    wf["21"] = {"class_type": "CLIPTextEncode", "inputs": {"text": edit_b, "clip": ["3", 0]}}
    pair_inputs = {
        "model": ["1", 0],
        "edit_conditioning_a": ["20", 0],
        "edit_conditioning_b": ["21", 0],
        "scale": float(scale), "mix_b": 0.5,
        "start_percent": float(start_percent), "end_percent": float(end_percent),
        "double_blocks_only": False, "verbose": False,
    }
    if measure_path:
        pair_inputs["measure_path"] = measure_path
    wf["22"] = {"class_type": "FluxSpaceEditPair", "inputs": pair_inputs}
    wf["7"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["22", 0], "positive": ["6", 0], "negative": ["6", 0],
            "latent_image": ["4", 0], "seed": seed, "steps": FLUX_STEPS, "cfg": 1.0,
            "sampler_name": FLUX_SAMPLER, "scheduler": FLUX_SCHEDULER, "denoise": 1.0,
        },
    }
    wf["8"] = {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["2", 0]}}
    wf["9"] = {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": prefix}}
    return wf


# --- runners -----------------------------------------------------------------

async def do_calibrate() -> None:
    CAL_DIR.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for i, (name, prompt) in enumerate(CALIBRATION_PROMPTS):
            seed = CAL_SEED_BASE + i
            img_dest = CAL_DIR / f"cal_{i:02d}_{name}.png"
            meas_path = str(CAL_DIR / f"cal_{i:02d}_{name}.pkl")
            if img_dest.exists() and Path(meas_path).exists():
                continue
            print(f"[cal {i:02d}/{len(CALIBRATION_PROMPTS)}] {name}")
            await client.generate(
                calibration_workflow(prompt, seed, meas_path, f"cal_{i:02d}_{name}"),
                img_dest,
            )


async def do_crossdemo_measure(axis: str = "glasses") -> None:
    """One pair-measurement pass per cross-demo base (s=1, mix_b=0.5)."""
    cfg = AXES[axis]
    root = _axis_dir(axis)
    (root / "measurement").mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for name, base, splice in cfg["bases"]:
            img_dest = root / "measurement" / f"{name}_meas.png"
            meas_path = str(root / "measurement" / f"{name}_meas.pkl")
            if img_dest.exists() and Path(meas_path).exists():
                print(f"[cross-meas:{axis}] skip {name}")
                continue
            print(f"[cross-meas:{axis}] {name}")
            await client.generate(
                pair_measure_workflow(CROSS_SEED, meas_path, f"fscross_{axis}_{name}_meas",
                                      base_prompt=base, edit_a=cfg["edit_a"],
                                      edit_b=splice, scale=1.0),
                img_dest,
            )


def _predict_edges_ratio(meas_pkl: Path, T_ratio: float) -> tuple[float | None, float | None, float]:
    """Ratio metric: max_env(s) / max_env(s=0) — base-prompt-invariant."""
    cal_runs = [_load_pkl(p) for p in sorted(CAL_DIR.glob("cal_*.pkl"))]
    cal_mean = _stack_reduction(cal_runs, ["attn_base", "mean_d"])
    mu_d_local = {k: v.mean(axis=0) for k, v in cal_mean.items()}

    meas = _load_pkl(meas_pkl)
    ab_list, d_list = [], []
    for step, blocks in meas["steps"].items():
        for bk, e in blocks.items():
            if (bk, step) not in mu_d_local:
                continue
            ab_list.append(e["attn_base"]["mean_d"].numpy())
            d_list.append(e["delta_mix"]["mean_d"].numpy())
    base = max(float(np.abs(ab).max()) for ab in ab_list)
    dense_s = np.round(np.arange(-3.0, 3.0001, 0.05), 2)
    safe = []
    for s in dense_s:
        me = max(float(np.abs(ab + s*d).max()) for ab, d in zip(ab_list, d_list))
        if me / base <= T_ratio:
            safe.append(float(s))
    s_lo = min(safe) if safe else None
    s_hi = max(safe) if safe else None
    return s_lo, s_hi, base


def _predict_edges(meas_pkl: Path, T: float,
                   metric: str = "max_env") -> tuple[float | None, float | None, list]:
    """Given a measurement pkl and a fitted threshold, return (s_lo, s_hi,
    full s,metric curve) via dense scan. Shares μ/σ with calibration corpus."""
    cal_runs = [_load_pkl(p) for p in sorted(CAL_DIR.glob("cal_*.pkl"))]
    cal_mean = _stack_reduction(cal_runs, ["attn_base", "mean_d"])
    mu_d = {k: v.mean(axis=0) for k, v in cal_mean.items()}
    sigma_d = {k: v.std(axis=0) + 1e-6 for k, v in cal_mean.items()}

    meas = _load_pkl(meas_pkl)
    ab_list, d_list, mu_list, sig_list = [], [], [], []
    for step, blocks in meas["steps"].items():
        for bk, e in blocks.items():
            key = (bk, step)
            if key not in mu_d:
                continue
            ab_list.append(e["attn_base"]["mean_d"].numpy())
            d_list.append(e["delta_mix"]["mean_d"].numpy())
            mu_list.append(mu_d[key])
            sig_list.append(sigma_d[key])

    dense_s = np.round(np.arange(-3.0, 3.0001, 0.05), 2)
    curve = []
    for s in dense_s:
        if metric == "max_env":
            m = max(float(np.abs(ab + s * d).max())
                    for ab, d in zip(ab_list, d_list))
        else:
            m = max(float(np.sqrt((((ab + s*d - mu)/sig)**2).sum()))
                    for ab, d, mu, sig in zip(ab_list, d_list, mu_list, sig_list))
        curve.append((float(s), m))
    safe = [s for s, m in curve if m <= T]
    s_lo = min(safe) if safe else None
    s_hi = max(safe) if safe else None
    return s_lo, s_hi, curve


def _verification_scales(s_lo: float | None, s_hi: float | None) -> list[float]:
    """Scales straddling the predicted edges, plus a couple of controls."""
    scales = set()
    # Edge straddles at 0.1 either side.
    for edge in (s_lo, s_hi):
        if edge is None:
            continue
        for off in (-0.2, -0.1, 0.0, 0.1, 0.2):
            scales.add(round(edge + off, 2))
    scales.add(0.0)   # baseline
    scales.add(1.0)   # known safe
    return sorted(scales)


async def do_crossdemo_verify(axis: str = "glasses") -> None:
    """For each cross-demo base, read predicted edges, then render the
    verification sweep."""
    cfg = AXES[axis]
    root = _axis_dir(axis)
    T_ratio = 1.275  # from glasses latin_f fit, axis-universal hypothesis
    (root / "verify").mkdir(parents=True, exist_ok=True)
    summary = {}
    async with ComfyClient() as client:
        for name, base, splice in cfg["bases"]:
            meas_pkl = root / "measurement" / f"{name}_meas.pkl"
            if not meas_pkl.exists():
                print(f"[cross-verify:{axis}] missing measurement for {name}; skip")
                continue
            s_lo, s_hi, base_env = _predict_edges_ratio(meas_pkl, T_ratio)
            scales = _verification_scales(s_lo, s_hi)
            summary[name] = {"s_lo": s_lo, "s_hi": s_hi,
                             "base_max_env": base_env, "T_ratio": T_ratio,
                             "scales": scales}
            print(f"[cross-verify:{axis}] {name}: safe∈[{s_lo},{s_hi}] scales={scales}")
            out_dir = root / "verify" / name
            out_dir.mkdir(parents=True, exist_ok=True)
            for s in scales:
                dest = out_dir / f"s{s:+.2f}.png"
                if dest.exists():
                    continue
                print(f"  render s={s:+.2f}")
                await client.generate(
                    pair_measure_workflow(CROSS_SEED, None,
                                          f"fscross_{axis}_{name}_s{s:+.2f}",
                                          base_prompt=base, edit_a=cfg["edit_a"],
                                          edit_b=splice, scale=s),
                    dest,
                )
    with (root / "predictions.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[cross-verify:{axis}] wrote {root}/predictions.json")


async def do_startpct_sweep() -> None:
    """Later start_percent sweep — does delaying edit onset preserve identity
    while keeping attribute?  Fixed s=1, mix_b=0.5, two bases + latin_f."""
    out_root = CROSS_DIR / "startpct"
    out_root.mkdir(parents=True, exist_ok=True)
    starts = [0.15, 0.20, 0.30, 0.40]
    targets = [
        ("elderly_latin_m", CALIBRATION_PROMPTS[5][1],
         "A photorealistic portrait photograph of an elderly Latin American man wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
        ("southasian_f", CALIBRATION_PROMPTS[9][1],
         "A photorealistic portrait photograph of an adult South Asian woman wearing thick-rimmed eyeglasses, neutral expression, plain grey background, studio lighting, sharp focus."),
        ("latin_f", PAIR_BASE_PROMPT, PAIR_EDIT_B),
    ]
    async with ComfyClient() as client:
        for name, base, splice in targets:
            bdir = out_root / name
            bdir.mkdir(parents=True, exist_ok=True)
            for sp in starts:
                dest = bdir / f"start{sp:.2f}.png"
                if dest.exists():
                    continue
                print(f"[startpct] {name} start={sp:.2f}")
                await client.generate(
                    pair_measure_workflow(CROSS_SEED, None,
                                          f"fssp_{name}_start{sp:.2f}",
                                          base_prompt=base, edit_a=PAIR_EDIT_A,
                                          edit_b=splice, scale=1.0,
                                          start_percent=sp, end_percent=1.0),
                    dest,
                )
    print(f"[startpct] done → {out_root}")


async def do_measure() -> None:
    MEAS_DIR.mkdir(parents=True, exist_ok=True)
    async with ComfyClient() as client:
        for seed in MEASURE_SEEDS:
            img_dest = MEAS_DIR / f"pair_s{seed}.png"
            meas_path = str(MEAS_DIR / f"pair_s{seed}.pkl")
            if img_dest.exists() and Path(meas_path).exists():
                continue
            print(f"[meas seed={seed}]")
            await client.generate(
                pair_measure_workflow(seed, meas_path, f"fsmet_pair_s{seed}"),
                img_dest,
            )


# --- analysis ----------------------------------------------------------------

def _load_pkl(p: Path) -> dict:
    with p.open("rb") as f:
        return pickle.load(f)


def _stack_reduction(metrics_runs: list[dict], field_path: list[str]) -> dict:
    """Given a list of parsed measure dicts (one per render), return a
    dict (block_key, step) → np.ndarray of shape (n_runs, D).

    field_path is e.g. ["attn_base", "mean_d"] meaning
    run["steps"][step][block_key]["attn_base"]["mean_d"].
    """
    out: dict[tuple, list[np.ndarray]] = {}
    for run in metrics_runs:
        for step, blocks in run.get("steps", {}).items():
            for bk, entries in blocks.items():
                node = entries
                try:
                    for k in field_path:
                        node = node[k]
                    v = node.numpy() if hasattr(node, "numpy") else np.asarray(node)
                except (KeyError, TypeError):
                    continue
                out.setdefault((bk, step), []).append(v)
    return {k: np.stack(v) for k, v in out.items()}


def analyze() -> None:
    ANAL_DIR.mkdir(parents=True, exist_ok=True)

    # --- load calibration ---
    cal_runs = [_load_pkl(p) for p in sorted(CAL_DIR.glob("cal_*.pkl"))]
    print(f"[analyze] loaded {len(cal_runs)} calibration runs")

    # Per (block, step) compute per-D μ and σ from mean_d across calibration.
    # (We use attn_base.mean_d as the per-D location. For σ we use spread of
    # mean_d across the calibration corpus — this captures prompt-to-prompt
    # variance at the per-dim level, which is exactly what we want for
    # "is this activation on-manifold".)
    cal_mean = _stack_reduction(cal_runs, ["attn_base", "mean_d"])
    cal_fro  = {}
    for run in cal_runs:
        for step, blocks in run.get("steps", {}).items():
            for bk, e in blocks.items():
                cal_fro.setdefault((bk, step), []).append(e["attn_base"]["fro"])
    cal_fro = {k: np.asarray(v) for k, v in cal_fro.items()}

    mu_d = {k: v.mean(axis=0) for k, v in cal_mean.items()}
    sigma_d = {k: v.std(axis=0) + 1e-6 for k, v in cal_mean.items()}
    fro_mean = {k: float(v.mean()) for k, v in cal_fro.items()}

    # --- load measurement (A+B pair) ---
    meas_paths = sorted(MEAS_DIR.glob("pair_s*.pkl"))
    if not meas_paths:
        print("[analyze] no measurement files found — run --measure first")
        return

    scales = sorted(SCALE_SWEEP_LABELS.keys())
    predictions = {}

    for mp in meas_paths:
        meas = _load_pkl(mp)
        seed_tag = mp.stem
        print(f"[analyze] processing {seed_tag}")

        # For each (block, step) we have attn_base.mean_d and delta_mix.mean_d.
        # steered_at_scale.mean_d is attn_base.mean_d + scale·δ — ignore, we
        # recompute for arbitrary s.
        rows = []  # per (block, step) scalars for every s
        for step, blocks in meas["steps"].items():
            for bk, e in blocks.items():
                key = (bk, step)
                if key not in mu_d:
                    continue
                ab = e["attn_base"]["mean_d"].numpy()
                d  = e["delta_mix"]["mean_d"].numpy()
                mu = mu_d[key]
                sig = sigma_d[key]

                # Mahalanobis (diagonal Σ_prompt) as quadratic in s: at²+bt+c
                v0 = (ab - mu) / sig
                v1 = d / sig
                a_coef = float((v1 * v1).sum())
                b_coef = float(2.0 * (v0 * v1).sum())
                c_coef = float((v0 * v0).sum())

                for s in scales:
                    d_m2 = max(0.0, a_coef * s * s + b_coef * s + c_coef)
                    d_m = float(np.sqrt(d_m2))
                    zmax = float(np.abs(v0 + s * v1).max())
                    # Envelope on max|attn_s_d| across d — bounds max excursion
                    max_env = float(np.abs(ab + s * d).max())
                    # Frobenius of steered (exact: ‖ab + s·d‖_F for per-D proxies)
                    steered_fro_proxy = float(np.sqrt(((ab + s * d) ** 2).sum()))
                    # Normalise by calibration Frobenius
                    fro_ratio = steered_fro_proxy / max(1e-6, fro_mean.get(key, 1.0) / np.sqrt(ab.size))
                    rows.append({
                        "block": bk, "step": step, "s": s,
                        "d_mahal": d_m, "zmax": zmax,
                        "max_env": max_env, "fro_ratio": fro_ratio,
                    })

        # Aggregate per scale: max over (block, step) — "worst excursion wins"
        agg = {}
        for s in scales:
            sr = [r for r in rows if r["s"] == s]
            agg[s] = {
                "d_mahal_max":  max(r["d_mahal"] for r in sr),
                "d_mahal_mean": float(np.mean([r["d_mahal"] for r in sr])),
                "zmax_max":     max(r["zmax"] for r in sr),
                "max_env_max":  max(r["max_env"] for r in sr),
                "fro_ratio_max": max(r["fro_ratio"] for r in sr),
                "label": SCALE_SWEEP_LABELS[s],
            }
        predictions[seed_tag] = agg

    # --- print table + save ---
    ANAL_DIR.mkdir(parents=True, exist_ok=True)
    out_json = ANAL_DIR / "predictions.json"
    with out_json.open("w") as f:
        json.dump(predictions, f, indent=2)

    # Simple report
    for seed_tag, agg in predictions.items():
        print(f"\n=== {seed_tag} ===")
        hdr = f"  {'s':>5} {'label':>8} | {'d_M_max':>9} {'d_M_mean':>9} {'zmax':>8} {'max_env':>8} {'fro_r':>8}"
        print(hdr)
        for s in scales:
            a = agg[s]
            print(f"  {s:>+5.1f} {a['label']:>8} | "
                  f"{a['d_mahal_max']:>9.2f} {a['d_mahal_mean']:>9.2f} "
                  f"{a['zmax_max']:>8.2f} {a['max_env_max']:>8.2f} "
                  f"{a['fro_ratio_max']:>8.2f}")

    # --- threshold fitting ---------------------------------------------------
    # For each metric: fit threshold T on seed_train (maximise accuracy) and
    # evaluate on seed_test. Report leave-one-out cross-seed accuracy.
    def _fit_T(agg, metric):
        best_acc, best_T = 0.0, None
        # Candidate thresholds = midpoints between consecutive sorted values
        vals = sorted(set(agg[s][metric] for s in scales))
        cand = [v for v in vals] + [
            0.5 * (vals[i] + vals[i+1]) for i in range(len(vals)-1)
        ]
        for T in cand:
            correct = sum(1 for s in scales
                          if (agg[s][metric] > T) == (agg[s]["label"] == "collapse"))
            acc = correct / len(scales)
            if acc > best_acc:
                best_acc, best_T = acc, T
        return best_T, best_acc

    def _eval(agg, metric, T):
        correct = sum(1 for s in scales
                      if (agg[s][metric] > T) == (agg[s]["label"] == "collapse"))
        return correct / len(scales)

    seed_tags = list(predictions.keys())
    print("\n=== per-seed in-sample thresholds ===")
    fitted = {}
    for seed_tag, agg in predictions.items():
        fitted[seed_tag] = {}
        print(f"\n{seed_tag}:")
        for metric in ("d_mahal_max", "zmax_max", "max_env_max", "fro_ratio_max"):
            T, acc = _fit_T(agg, metric)
            fitted[seed_tag][metric] = T
            print(f"  {metric:>15}: best_T={T:.2f}  acc={acc:.2%}")

    if len(seed_tags) >= 2:
        print("\n=== cross-seed evaluation (fit on A, test on B) ===")
        for train in seed_tags:
            for test in seed_tags:
                if train == test:
                    continue
                print(f"\nfit={train} → test={test}:")
                for metric in ("d_mahal_max", "zmax_max", "max_env_max", "fro_ratio_max"):
                    T = fitted[train][metric]
                    acc = _eval(predictions[test], metric, T)
                    print(f"  {metric:>15}: T={T:.2f}  acc={acc:.2%}")

    # --- manifold edge prediction -------------------------------------------
    # Using the best metric (max_env_max) + threshold fit, find s_lo, s_hi
    # (smallest/largest s with metric ≤ T) over a dense grid. This is the
    # "edge of the manifold" predicted from a single measurement pass.
    print("\n=== manifold edges from dense s-scan ===")
    dense_s = np.round(np.arange(-3.0, 3.0001, 0.05), 2)
    for seed_tag in seed_tags:
        meas = _load_pkl(MEAS_DIR / f"{seed_tag}.pkl")
        ab_list, d_list, mu_list, sig_list = [], [], [], []
        for step, blocks in meas["steps"].items():
            for bk, e in blocks.items():
                key = (bk, step)
                if key not in mu_d:
                    continue
                ab_list.append(e["attn_base"]["mean_d"].numpy())
                d_list.append(e["delta_mix"]["mean_d"].numpy())
                mu_list.append(mu_d[key])
                sig_list.append(sigma_d[key])

        print(f"\n{seed_tag}:")
        for metric_name in ("max_env", "d_mahal"):
            T = fitted[seed_tag][f"{metric_name}_max"]
            edges = []
            for s in dense_s:
                if metric_name == "max_env":
                    m = max(float(np.abs(ab + s * d).max())
                            for ab, d in zip(ab_list, d_list))
                else:
                    m = max(float(np.sqrt(((ab + s*d - mu)/sig * (ab + s*d - mu)/sig).sum()))
                            for ab, d, mu, sig in zip(ab_list, d_list, mu_list, sig_list))
                if m <= T:
                    edges.append(s)
            s_lo = min(edges) if edges else None
            s_hi = max(edges) if edges else None
            print(f"  {metric_name:>8}  T={T:.3f}  predicted safe s ∈ [{s_lo}, {s_hi}]")

    print(f"\n[analyze] wrote {out_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    ap.add_argument("--crossdemo-measure", action="store_true")
    ap.add_argument("--crossdemo-verify", action="store_true")
    ap.add_argument("--startpct-sweep", action="store_true")
    ap.add_argument("--axis", default="glasses", choices=list(AXES))
    args = ap.parse_args()
    if args.calibrate:
        asyncio.run(do_calibrate())
    if args.measure:
        asyncio.run(do_measure())
    if args.analyze:
        analyze()
    if args.crossdemo_measure:
        asyncio.run(do_crossdemo_measure(args.axis))
    if args.crossdemo_verify:
        asyncio.run(do_crossdemo_verify(args.axis))
    if args.startpct_sweep:
        asyncio.run(do_startpct_sweep())
    if not any((args.calibrate, args.measure, args.analyze,
                args.crossdemo_measure, args.crossdemo_verify, args.startpct_sweep)):
        ap.print_help()


if __name__ == "__main__":
    main()
