"""Slider quality measurement: render grid, score, write parquet.

Implements the procedure in
`docs/research/2026-04-26-slider-quality-measurement.md`. Two phases:

1. ``render`` — for each (prompt, strength, seed) triple, generate a PNG
   under ``models/sliders/<name>/<ckpt_tag>/renders/<prompt_id>/seed{S}_str{V}.png``.
   Skip-if-exists for resumability.
2. ``score`` — score every PNG (intended metric, ArcFace identity vs the
   same prompt/seed at strength 0, SigLIP drift channels, MediaPipe 52-d
   blendshapes, simple lighting stats). Write
   ``models/sliders/<name>/<ckpt_tag>/eval.parquet``.

Default ``--phase both`` runs render then score. Render-only and score-only
exist for re-running just the metrics layer when probes change.

Usage:
    uv run python -m src.demographic_pc.measure_slider \\
        --slider-name glasses_v4 \\
        --checkpoint /home/newub/w/vamp-interface/output/ai_toolkit_runs/glasses_slider_v4/glasses_slider_v4_000000600.safetensors \\
        --intended-axis glasses
"""
from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
SLIDERS_ROOT = ROOT / "models" / "sliders"

FLUX_HF_ID = "black-forest-labs/FLUX.1-Krea-dev"
RES = 512
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5

STRENGTHS = [-2.5, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.5]
SEEDS = [1337, 2026, 4242]


# ── prompts ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PromptCell:
    pid: str
    pool: str  # "in_distribution" | "held_out"
    text: str


# 9 in-distribution prompts (the ai-toolkit v4 trainer's sample prompts)
IN_DIST = [
    PromptCell("in_asian_m_studio",
               "in_distribution",
               "a photorealistic portrait photograph of an east asian man, plain grey background, studio lighting"),
    PromptCell("in_black_f_studio",
               "in_distribution",
               "a photorealistic portrait photograph of a black woman, plain grey background, studio lighting"),
    PromptCell("in_european_m_studio",
               "in_distribution",
               "a photorealistic portrait photograph of a european man, plain grey background, studio lighting"),
]

# Held-out prompts: never seen during training
HELD_OUT = [
    PromptCell("ho_southasian_f_studio",
               "held_out",
               "a portrait photograph of a south asian woman, plain grey background, studio lighting"),
    PromptCell("ho_latino_m_cafe",
               "held_out",
               "a portrait photograph of a latino man, soft window light, cafe background"),
    PromptCell("ho_elderly_white_f_natural",
               "held_out",
               "a portrait photograph of an elderly white woman, natural daylight"),
    PromptCell("ho_young_black_m_harsh",
               "held_out",
               "a portrait photograph of a young black man, harsh side lighting, plain wall"),
    PromptCell("ho_middleeast_neutral",
               "held_out",
               "a portrait photograph of a middle eastern person, neutral light, indoor"),
    PromptCell("ho_eastasian_f_studio",
               "held_out",
               "a portrait photograph of an east asian woman, plain grey background, studio lighting"),
]

PROMPTS = IN_DIST + HELD_OUT


# ── intent table ────────────────────────────────────────────────────────────
# Maps each prompt_id to the demographic the prompt is *trying* to depict
# and the closest BASE_META key (for atom-projection mu/sigma lookup). When
# `base` is None, atom projection falls back to a global mu/sigma. Schema
# matches sample_index.parquet (ethnicity ∈ {asian, black, european,
# southasian, latin, middleeast}; age ∈ {young, adult, elderly}).

INTENT_TABLE: dict[str, dict] = {
    "in_asian_m_studio":          {"base": "asian_m",     "ethnicity": "asian",     "gender": "m", "age": "adult"},
    "in_black_f_studio":          {"base": "black_f",     "ethnicity": "black",     "gender": "f", "age": "adult"},
    "in_european_m_studio":       {"base": "european_m",  "ethnicity": "european",  "gender": "m", "age": "adult"},
    "ho_southasian_f_studio":     {"base": "southasian_f","ethnicity": "southasian","gender": "f", "age": "adult"},
    "ho_latino_m_cafe":           {"base": None,          "ethnicity": "latin",     "gender": "m", "age": "adult"},
    "ho_elderly_white_f_natural": {"base": None,          "ethnicity": "european",  "gender": "f", "age": "elderly"},
    "ho_young_black_m_harsh":     {"base": None,          "ethnicity": "black",     "gender": "m", "age": "young"},
    "ho_middleeast_neutral":      {"base": None,          "ethnicity": "middleeast","gender": "u", "age": "adult"},
    "ho_eastasian_f_studio":      {"base": None,          "ethnicity": "asian",     "gender": "f", "age": "adult"},
}


# ── SigLIP probes ────────────────────────────────────────────────────────────
# (probe_name, positive, negative). Always measured regardless of axis;
# the axis-relevant probe doubles as `intended_metric` for non-blendshape axes.

SIGLIP_PROBES = [
    ("glasses",
     "a photo of a person wearing eyeglasses, eyewear visible on face",
     "a photo of a person without glasses, bare face, no eyewear"),
    # Hair styles (4-way one-vs-rest collapsed to bipolar margins)
    ("hair_long",
     "a photo of a person with long flowing hair past the shoulders",
     "a photo of a person with very short cropped hair"),
    ("hair_curly",
     "a photo of a person with tight curly hair, afro texture",
     "a photo of a person with straight smooth hair"),
    # Accessories
    ("earrings",
     "a photo of a person wearing earrings, jewelry visible at the ears",
     "a photo of a person without earrings, bare ears"),
    # Clothing / framing
    ("formal_clothing",
     "a photo of a person wearing a collared shirt or polo, well-groomed studio portrait",
     "a photo of a person wearing a casual t-shirt, bare shoulders, informal"),
    # Beard (catches the v4 'studio masculinity' bundle)
    ("beard",
     "a photo of a person with a beard or stubble",
     "a photo of a clean-shaven person with no facial hair"),
]


# ── prompt-embed precompute ──────────────────────────────────────────────────

def build_pipeline(load_lora_path: Path | None):
    from diffusers import FluxPipeline
    print(f"[pipeline] loading {FLUX_HF_ID}")
    pipe = FluxPipeline.from_pretrained(FLUX_HF_ID, torch_dtype=torch.bfloat16)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    if load_lora_path is not None:
        print(f"[pipeline] loading LoRA: {load_lora_path}")
        pipe.load_lora_weights(str(load_lora_path), adapter_name="slider")
    return pipe


def precompute_embeds(pipe, prompts: list[PromptCell], device: torch.device):
    encoded = {}
    with torch.no_grad():
        for pc in prompts:
            pe, pp, _ = pipe.encode_prompt(
                prompt=pc.text, prompt_2=pc.text,
                device=device, num_images_per_prompt=1,
                max_sequence_length=512,
            )
            encoded[pc.pid] = (pe, pp)
            print(f"  [embed] {pc.pid}")
    return encoded


# ── render phase ─────────────────────────────────────────────────────────────

def render_path(out_root: Path, pid: str, seed: int, strength: float) -> Path:
    return out_root / pid / f"seed{seed}_str{strength:+.2f}.png"


def render_grid(checkpoint: Path, ckpt_root: Path,
                strengths: list[float] | None = None,
                seeds: list[int] | None = None) -> None:
    out_root = ckpt_root / "renders"
    out_root.mkdir(parents=True, exist_ok=True)
    strengths = list(strengths) if strengths else STRENGTHS
    seeds = list(seeds) if seeds else SEEDS

    # First pass: which cells are missing?
    todo: list[tuple[PromptCell, int, float]] = []
    for pc in PROMPTS:
        for seed in seeds:
            for s in strengths:
                if not render_path(out_root, pc.pid, seed, s).exists():
                    todo.append((pc, seed, s))
    if not todo:
        print(f"[render] all {len(PROMPTS) * len(seeds) * len(strengths)} cells exist; skipping")
        return
    print(f"[render] {len(todo)} cells to render (of "
          f"{len(PROMPTS) * len(seeds) * len(strengths)} total)")

    pipe = build_pipeline(checkpoint)
    device = torch.device("cuda")
    encoded = precompute_embeds(pipe, PROMPTS, device)

    t0 = time.time()
    for i, (pc, seed, s) in enumerate(todo):
        target = render_path(out_root, pc.pid, seed, s)
        target.parent.mkdir(parents=True, exist_ok=True)

        pipe.set_adapters(["slider"], adapter_weights=[float(s)])
        gen = torch.Generator(device=device).manual_seed(seed)
        pe, pp = encoded[pc.pid]
        with torch.no_grad():
            img = pipe(
                prompt_embeds=pe, pooled_prompt_embeds=pp,
                height=RES, width=RES,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=gen,
            ).images[0]
        img.save(target)
        if (i + 1) % 10 == 0 or i + 1 == len(todo):
            dt = time.time() - t0
            eta = dt / (i + 1) * (len(todo) - i - 1)
            print(f"  [{i+1}/{len(todo)}] {pc.pid} seed={seed} s={s:+.2f} "
                  f"({(i+1)/dt:.2f} img/s, eta {eta/60:.1f}m)")

    del pipe
    gc.collect()
    torch.cuda.empty_cache()


# ── score phase ──────────────────────────────────────────────────────────────

def _project_atoms(scores: dict, intent_base: str | None,
                   nmf_artifacts: tuple) -> np.ndarray:
    """Project blendshape scores onto NMF atoms.

    Uses base-specific mu/sigma when `intent_base` is in BASE_META; falls
    back to a per-channel mean across bases for prompts outside the
    training set (held-out demographics). Same pipeline as
    `build_sample_index.project_sample`.
    """
    (W, W_pinv, _, _, mu, sigma, _, channels_full,
     prune_mask, base_idx) = nmf_artifacts
    if intent_base is not None and intent_base in base_idx:
        mu_use = mu[base_idx[intent_base]]
        sigma_use = sigma[base_idx[intent_base]]
    else:
        mu_use = mu.mean(axis=0)
        sigma_use = sigma.mean(axis=0)
    x = np.array([scores.get(c, 0.0) for c in channels_full])
    sigma_safe = np.where(sigma_use < 1e-4, 1.0, sigma_use)
    x_res = (x - mu_use) / sigma_safe
    x_res = x_res[prune_mask]
    x_pos = np.clip(x_res, 0.0, None)
    x_neg = np.clip(-x_res, 0.0, None)
    stacked = np.concatenate([x_pos, x_neg])
    return np.clip(stacked @ W_pinv, 0.0, None)


def score_grid(intended_axis: str, ckpt_root: Path, slider_name: str, ckpt_tag: str) -> None:
    out_root = ckpt_root / "renders"
    parquet_path = ckpt_root / "eval.parquet"

    # Discover all rendered cells from disk (not the module-level
    # STRENGTHS/SEEDS, so zoom-render extensions are picked up).
    pid_to_pc = {pc.pid: pc for pc in PROMPTS}
    cells = []
    for pid_dir in sorted(out_root.iterdir()) if out_root.exists() else []:
        if not pid_dir.is_dir() or pid_dir.name not in pid_to_pc:
            continue
        pc = pid_to_pc[pid_dir.name]
        for png in sorted(pid_dir.glob("seed*_str*.png")):
            stem = png.stem  # seed{S}_str{V}
            try:
                seed_part, str_part = stem.split("_")
                seed = int(seed_part.removeprefix("seed"))
                s_val = float(str_part.removeprefix("str"))
            except (ValueError, AttributeError):
                continue
            cells.append((pc, seed, s_val, png))
    if not cells:
        print(f"[score] no renders under {out_root}, aborting")
        return
    cells.sort(key=lambda c: (c[0].pid, c[1], c[2]))
    print(f"[score] {len(cells)} cells to score "
          f"(strengths={sorted({c[2] for c in cells})}, "
          f"seeds={sorted({c[1] for c in cells})})")

    # ── load measurement models ──
    from src.demographic_pc.build_sample_index import load_nmf
    from src.demographic_pc.classifiers import (
        FairFaceClassifier, InsightFaceClassifier, MiVOLOClassifier,
    )
    from src.demographic_pc.score_blendshapes import make_landmarker
    from src.demographic_pc.score_blendshapes import score_png as bs_score
    from src.demographic_pc.score_clip_probes import Siglip2Backend

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[score] loading NMF basis (canonical-schema atom projection)")
    nmf_artifacts = load_nmf()
    W = nmf_artifacts[0]
    channels_full = nmf_artifacts[7]
    n_atoms = W.shape[0]

    print("[score] loading InsightFace (detection + genderage + ArcFace)")
    ins = InsightFaceClassifier(with_embedding=True)
    print("[score] loading MiVOLO (age + gender)")
    mivolo = MiVOLOClassifier()
    print("[score] loading FairFace (ethnicity + age-bin + gender)")
    fairface = FairFaceClassifier()

    print("[score] loading SigLIP-2")
    siglip = Siglip2Backend(device)
    probe_feats = {}
    with torch.no_grad():
        for name, pos, neg in SIGLIP_PROBES:
            tok = siglip.processor(text=[pos, neg], return_tensors="pt",
                                    padding="max_length", truncation=True).to(device)
            tf = siglip._as_tensor(siglip.model.get_text_features(**tok))
            tf = tf / tf.norm(dim=-1, keepdim=True)
            probe_feats[name] = tf

    print("[score] loading MediaPipe FaceLandmarker")
    landmarker = make_landmarker()

    # ── pass 1: per-image embeddings + probes + blendshapes + classifiers ──
    rows = []
    arcface_emb: dict[tuple[str, int, float], np.ndarray | None] = {}
    t0 = time.time()
    import cv2
    for i, (pc, seed, s, path) in enumerate(cells):
        intent = INTENT_TABLE[pc.pid]
        bgr = cv2.imread(str(path))

        # InsightFace: detection + ArcFace embedding + age/gender obs
        ins_out = (ins.predict(bgr) if bgr is not None
                   else {"detected": False, "embedding": None,
                         "age": None, "gender": None})
        arcface_emb[(pc.pid, seed, s)] = ins_out.get("embedding")

        # MiVOLO: age + gender
        mv_out = (mivolo.predict(bgr) if bgr is not None
                  else {"age": None, "gender": None, "gender_conf": None})
        # FairFace: ethnicity + age-bin + gender (with dlib alignment)
        ff_out = (fairface.predict(bgr) if bgr is not None
                  else {"detected": False, "age_bin": None, "gender": None,
                        "race": None})

        # SigLIP probes
        feat = siglip.encode_image(path)
        sig_scores = {}
        for name, pf in probe_feats.items():
            sims = (feat @ pf.T).squeeze(0)
            sig_scores[f"siglip_{name}_margin"] = float(sims[0] - sims[1])

        # MediaPipe blendshapes (52-d ARKit)
        bs = bs_score(landmarker, path) or {}
        bs_cols = {f"bs_{k}": v for k, v in bs.items()}

        # NMF atom projection (canonical basis used by sample_index.parquet)
        atoms = _project_atoms(bs, intent["base"], nmf_artifacts)
        atom_cols = {f"atom_{k:02d}": float(atoms[k]) for k in range(n_atoms)}

        # Lighting stats
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        L = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
        gx = np.gradient(L, axis=1); gy = np.gradient(L, axis=0)
        lighting_contrast = float(L.std())
        lighting_gradient = float(np.mean(np.sqrt(gx ** 2 + gy ** 2)))

        row = {
            # ── provenance (matches sample_index.parquet shape) ──
            "source": f"{slider_name}/{ckpt_tag}",
            "rel": f"{pc.pid}/seed{seed}_str{s:+.2f}.png",
            "base": intent["base"],
            "ethnicity": intent["ethnicity"],
            "gender": intent["gender"],
            "age": intent["age"],
            "axis": intended_axis,
            "scale": float(s),
            "seed": int(seed),
            "start_pct": float("nan"),
            "has_attn": False,
            "attn_tag": None,
            "attn_row": -1,
            "atom_source": "pinv_lora",
            "img_path": str(path.relative_to(ROOT)),
            "corpus_version": "lora_eval_v1",
            "alpha": float("nan"),
            # ── slider-eval-specific provenance ──
            "slider_name": slider_name,
            "checkpoint": ckpt_tag,
            "prompt_id": pc.pid,
            "prompt_pool": pc.pool,
            # ── per-image observations ──
            "face_detected": bool(ins_out.get("detected", False)),
            "fairface_detected": bool(ff_out.get("detected", False)),
            # demographic predictions (observed)
            "observed_age_mivolo": mv_out.get("age"),
            "observed_age_insightface": ins_out.get("age"),
            "observed_gender_mivolo": mv_out.get("gender"),
            "observed_gender_mivolo_conf": mv_out.get("gender_conf"),
            "observed_gender_insightface": ins_out.get("gender"),
            "observed_gender_fairface": ff_out.get("gender"),
            "observed_age_bin_fairface": ff_out.get("age_bin"),
            "observed_ethnicity_fairface": ff_out.get("race"),
            # SigLIP, blendshapes, atoms
            **sig_scores,
            **bs_cols,
            **atom_cols,
            "lighting_contrast": lighting_contrast,
            "lighting_gradient": lighting_gradient,
        }
        rows.append(row)
        if (i + 1) % 25 == 0 or i + 1 == len(cells):
            dt = time.time() - t0
            print(f"  [{i+1}/{len(cells)}] {(i+1)/dt:.1f} img/s")

    landmarker.close()
    del siglip
    gc.collect()
    torch.cuda.empty_cache()

    df = pd.DataFrame(rows)

    # ── pass 2: ArcFace identity cosines (same-prompt, same-seed, s=0 baseline) ──
    cos_to_baseline = []
    for r in rows:
        key_self = (r["prompt_id"], r["seed"], r["scale"])
        key_base = (r["prompt_id"], r["seed"], 0.0)
        e_self = arcface_emb.get(key_self)
        e_base = arcface_emb.get(key_base)
        if e_self is None or e_base is None:
            cos_to_baseline.append(np.nan)
        else:
            cos_to_baseline.append(float(np.dot(e_self, e_base)))
    df["identity_cos_to_base"] = cos_to_baseline
    df["identity_pass_075"] = (df["identity_cos_to_base"] >= 0.75).astype("int8")

    # ── intended_metric (per axis) ──
    if intended_axis == "glasses":
        df["intended_metric"] = df["siglip_glasses_margin"]
    elif intended_axis == "smile":
        df["intended_metric"] = df.get("bs_mouthSmileLeft", 0.0) + df.get("bs_mouthSmileRight", 0.0)
    elif intended_axis == "eye_squint":
        df["intended_metric"] = df.get("bs_eyeSquintLeft", 0.0) + df.get("bs_eyeSquintRight", 0.0)
    else:
        print(f"[score] WARNING: unknown intended-axis {intended_axis!r}, "
              f"intended_metric not assigned")
        df["intended_metric"] = np.nan

    df.to_parquet(parquet_path, index=False, compression="zstd")
    print(f"[save] {parquet_path} rows={len(df)} cols={df.shape[1]}")

    # ── pass criteria summary ──
    summarize(df, intended_axis)

    # ── visual collage ──
    collage_path = ckpt_root / f"{slider_name}_{ckpt_tag}_eval_collage.png"
    make_collage(df, out_root, collage_path)


# Axes whose negative direction has no semantic content. Adding glasses to
# a face is a concept; "removing glasses" from a baseline that has none
# isn't — the slider just drifts away in the only direction it can.
ONE_SIDED_AXES = {"glasses"}


def summarize(df: pd.DataFrame, intended_axis: str) -> None:
    if intended_axis in ONE_SIDED_AXES:
        _summarize_one_sided(df, intended_axis)
    else:
        _summarize_bidirectional(df, intended_axis)


def _summarize_bidirectional(df: pd.DataFrame, intended_axis: str) -> None:
    from scipy.stats import spearmanr
    print(f"\n──── pass criteria for axis={intended_axis} (bidirectional) ────")
    valid = df.dropna(subset=["intended_metric"])
    for pool in ["in_distribution", "held_out"]:
        sub = valid[valid["prompt_pool"] == pool]
        if len(sub) < 5:
            print(f"  [{pool}] insufficient data")
            continue
        rho, _ = spearmanr(sub["scale"], sub["intended_metric"])
        m_neg = sub[sub["scale"] == -1.5]["intended_metric"].mean()
        m_zero = sub[sub["scale"] == 0.0]["intended_metric"].mean()
        m_pos = sub[sub["scale"] == 1.5]["intended_metric"].mean()
        print(f"  [{pool}] Spearman ρ(s, intended) = {rho:+.3f}  (want ≥ 0.9)")
        print(f"           mean@-1.5={m_neg:+.3f}  mean@0={m_zero:+.3f}  mean@+1.5={m_pos:+.3f}  "
              f"separation={m_pos - m_neg:+.3f} (want ≥ 0.3)")
    usable = valid[valid["scale"].abs() <= 1.5]
    af = usable["identity_cos_to_base"].dropna()
    if len(af):
        n_bad = (af < 0.4).sum()
        print(f"  identity: ArcFace cos in |s|≤1.5 — min={af.min():.3f} "
              f"mean={af.mean():.3f} ({n_bad} cells <0.4 of {len(af)})  (want all ≥0.4)")
    print("──────────────────────────────────────────────")


def _summarize_one_sided(df: pd.DataFrame, intended_axis: str) -> None:
    """Spearman + separation computed on the positive half only.

    Identity preservation is also reported only on the positive usable
    window. The negative half is dropped from pass criteria but still
    rendered and stored for later inspection.
    """
    from scipy.stats import spearmanr
    print(f"\n──── pass criteria for axis={intended_axis} (one-sided positive) ────")
    valid = df.dropna(subset=["intended_metric"])
    pos = valid[valid["scale"] >= 0]
    s_max = float(pos["scale"].max())

    for pool in ["in_distribution", "held_out"]:
        sub = pos[pos["prompt_pool"] == pool]
        if len(sub) < 5:
            print(f"  [{pool}] insufficient data")
            continue
        rho, _ = spearmanr(sub["scale"], sub["intended_metric"])
        means = sub.groupby("scale")["intended_metric"].mean()
        cells = "  ".join(f"s=+{s:.2f}:{means.get(s, float('nan')):+.3f}"
                          for s in sorted(means.index))
        sep = means.get(s_max, float("nan")) - means.get(0.0, float("nan"))
        print(f"  [{pool}] Spearman ρ over s∈[0,+{s_max:.2f}] = {rho:+.3f}  (want ≥ 0.9)")
        print(f"           {cells}")
        print(f"           separation @s_max−@0 = {sep:+.3f}  (calibrate per-axis; "
              f"glasses Δ≈+0.05 visible-on-most-cells)")

    # Identity preservation across the positive usable window (≤ +1.5)
    pos_usable = pos[pos["scale"] <= 1.5]
    af = pos_usable["identity_cos_to_base"].dropna()
    if len(af):
        n_bad = (af < 0.4).sum()
        print(f"  identity: ArcFace cos in s∈[0,+1.5] — min={af.min():.3f} "
              f"mean={af.mean():.3f} ({n_bad} cells <0.4 of {len(af)})  (want all ≥0.4)")

    # Fraction-with-feature: SigLIP margin > 0 as a binary proxy. Threshold
    # 0 is the natural pos vs neg decision boundary for the bipolar probe.
    feature_col = {"glasses": "siglip_glasses_margin"}.get(intended_axis)
    if feature_col and feature_col in df.columns:
        print(f"  fraction with {intended_axis} present (siglip > 0):")
        for pool in ["in_distribution", "held_out"]:
            sub = pos[pos["prompt_pool"] == pool]
            if len(sub) < 5:
                continue
            frac = sub.groupby("scale").apply(
                lambda g: (g[feature_col] > 0).mean()
            )
            cells = "  ".join(f"+{s:.2f}:{frac.get(s, 0.0):.2f}"
                              for s in sorted(frac.index))
            print(f"    [{pool}] {cells}")
    print("──────────────────────────────────────────────")


def make_collage(df: pd.DataFrame, render_root: Path, out_path: Path) -> None:
    """One row per prompt (held-out first), columns over strengths, single seed."""
    seed = SEEDS[0]
    rows = sorted(df["prompt_id"].unique(),
                  key=lambda pid: (0 if pid.startswith("ho_") else 1, pid))
    cols = sorted(df["scale"].unique())

    thumb = 192
    pad = 4
    label_h = 28
    row_lab_w = 220
    W = row_lab_w + len(cols) * thumb + (len(cols) + 1) * pad
    H = label_h + len(rows) * thumb + (len(rows) + 1) * pad

    canvas = Image.new("RGB", (W, H), (22, 22, 22))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()

    for ci, s in enumerate(cols):
        x = row_lab_w + ci * (thumb + pad) + pad
        draw.text((x, 8), f"{s:+.2f}", fill=(220, 220, 220), font=font)

    for ri, pid in enumerate(rows):
        y = label_h + ri * (thumb + pad) + pad
        draw.text((8, y + thumb // 2 - 8), pid, fill=(200, 200, 200), font=font)
        for ci, s in enumerate(cols):
            x = row_lab_w + ci * (thumb + pad) + pad
            p = render_path(render_root, pid, seed, s)
            if p.exists():
                im = Image.open(p).convert("RGB").resize((thumb, thumb))
                canvas.paste(im, (x, y))
    canvas.save(out_path)
    print(f"[collage] {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slider-name", required=True,
                    help="logical name, e.g. glasses_v4")
    ap.add_argument("--checkpoint", required=True, type=Path,
                    help="path to .safetensors")
    ap.add_argument("--intended-axis", required=True,
                    choices=["glasses", "smile", "eye_squint"])
    ap.add_argument("--phase", default="both",
                    choices=["render", "score", "both"])
    ap.add_argument("--ckpt-tag", default=None,
                    help="subdir name; defaults to checkpoint stem")
    ap.add_argument("--strengths", type=float, nargs="*", default=None,
                    help="override default STRENGTHS list (zoom-render mode)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override default SEEDS list (zoom-render mode)")
    args = ap.parse_args()

    ckpt_tag = args.ckpt_tag or args.checkpoint.stem
    ckpt_root = SLIDERS_ROOT / args.slider_name / ckpt_tag
    ckpt_root.mkdir(parents=True, exist_ok=True)

    if args.phase in ("render", "both"):
        render_grid(args.checkpoint, ckpt_root,
                    strengths=args.strengths, seeds=args.seeds)
    if args.phase in ("score", "both"):
        score_grid(args.intended_axis, ckpt_root, args.slider_name, ckpt_tag)


if __name__ == "__main__":
    main()
