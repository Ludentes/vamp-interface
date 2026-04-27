"""Iterate on prompt-pair variants, measure effect in vocabulary space, rank.

Single-axis iteration tool. Given 2–3 candidate prompt-pair variants,
render each on a small fixed eval grid, score with MediaPipe + NMF atom
projection + SigLIP-2 probes + MiVOLO/FairFace/InsightFace + ArcFace
identity cosine, fit slopes, rank by a composite score, and write a
markdown recommendation.

Usage:
    uv run python -m src.demographic_pc.promptpair_iterate \
        --axis smile --iter 01 --spec specs/smile_iter01.yaml --run

Spec YAML:
    axis: smile
    target_probe: siglip_smiling_margin     # optional
    target_atom: 16                         # optional
    variants:
      v1: {pos: "...", neg: "..."}
      v2: {pos: "...", neg: "..."}
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "output/demographic_pc/promptpair_iterate"
DICTIONARY_PARQUET = ROOT / "output/demographic_pc/effect_matrix_dictionary.parquet"

EVAL_BASES = [
    ("young_european_f",
     "A photorealistic portrait photograph of a young European woman, "
     "neutral expression, plain grey background, studio lighting, sharp focus."),
    ("european_m",
     "A photorealistic portrait photograph of an adult European man, "
     "neutral expression, plain grey background, studio lighting, sharp focus."),
    ("elderly_latin_m",
     "A photorealistic portrait photograph of an elderly Latin American man, "
     "neutral expression, plain grey background, studio lighting, sharp focus."),
]
EVAL_SEEDS = [2026, 4242]
EVAL_SCALES = [0.0, 0.5, 1.0]


# ── rendering ────────────────────────────────────────────────────────────────


BASE_AGE_WORDS = {
    "young_european_f": "young",
    "european_m":       "adult",
    "elderly_latin_m":  "elderly",
}

BASE_ETHNICITY_WORDS = {
    "young_european_f": "European",
    "european_m":       "European",
    "elderly_latin_m":  "Latin American",
}

BASE_GENDER_WORDS = {
    "young_european_f": "woman",
    "european_m":       "man",
    "elderly_latin_m":  "man",
}


def _resolve_prompt(spec_val, base_name: str) -> str:
    """pos/neg can be a string (shared) or a dict keyed by base name. Also
    supports `{age}` / `{ethnicity}` / `{gender}` placeholders."""
    if isinstance(spec_val, dict):
        s = spec_val.get(base_name, spec_val.get("default", ""))
    else:
        s = spec_val
    return s.format(
        age=BASE_AGE_WORDS.get(base_name, "adult"),
        ethnicity=BASE_ETHNICITY_WORDS.get(base_name, ""),
        gender=BASE_GENDER_WORDS.get(base_name, "person"),
    )


async def render_variant(variant_key: str, pos_spec, neg_spec, out_dir: Path,
                          scales: list[float] | None = None) -> None:
    """Render the 18-image eval grid for one variant. Resumable.

    pos_spec / neg_spec: str with optional {age} placeholder, OR dict keyed
    by base name for per-base prompts.
    scales: optional override of EVAL_SCALES for this iteration.
    """
    from src.demographic_pc.comfy_flux import ComfyClient
    from src.demographic_pc.fluxspace_metrics import pair_measure_workflow

    use_scales = scales if scales is not None else EVAL_SCALES

    async with ComfyClient() as client:
        for base_name, base_prompt in EVAL_BASES:
            edit_a = _resolve_prompt(pos_spec, base_name)
            edit_b = _resolve_prompt(neg_spec, base_name)
            (out_dir / variant_key / base_name).mkdir(parents=True, exist_ok=True)
            for seed in EVAL_SEEDS:
                for s in use_scales:
                    dest = out_dir / variant_key / base_name / f"seed{seed}_s{s:+.2f}.png"
                    meas_path = dest.with_suffix(".pkl")
                    if dest.exists() and dest.stat().st_size > 1024 and meas_path.exists():
                        continue
                    # Save the FluxSpaceEditPair measurement (attn_base + delta_mix
                    # tensors) alongside the render. Required for cache-δ axis
                    # extraction (see framework-procedure milestone 8).
                    wf = pair_measure_workflow(
                        seed, str(meas_path), f"ppi_{variant_key}_{base_name}_s{seed}_{s:+.2f}",
                        base_prompt=base_prompt, edit_a=edit_a, edit_b=edit_b,
                        scale=s, start_percent=0.15, end_percent=1.0)
                    await client.generate(wf, dest)
            print(f"  [render] {variant_key}/{base_name} done  ('{edit_a[:50]}')")


# ── scoring ──────────────────────────────────────────────────────────────────


def _load_scorers():
    import torch
    from src.demographic_pc.classifiers import (
        MiVOLOClassifier, FairFaceClassifier, InsightFaceClassifier,
    )
    from src.demographic_pc.score_clip_probes import Siglip2Backend
    from src.demographic_pc.build_sample_index import load_nmf

    nmf = load_nmf()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("  [load] MiVOLO…"); mv = MiVOLOClassifier()
    print("  [load] FairFace…"); ff = FairFaceClassifier()
    print("  [load] InsightFace…"); ins = InsightFaceClassifier(with_embedding=True)
    print("  [load] SigLIP-2…"); sig_backend = Siglip2Backend(device)
    probe_feats = sig_backend.encode_probes()
    return nmf, mv, ff, ins, sig_backend, probe_feats


def score_variant(variant_key: str, out_dir: Path, scorers) -> pd.DataFrame:
    """Score every PNG for this variant: MediaPipe atoms + SigLIP probes +
    classifiers + ArcFace identity cosine to seed-matched scale=0 anchor."""
    import cv2
    from src.demographic_pc.score_blendshapes import make_landmarker, score_png
    from src.demographic_pc.build_sample_index import project_sample

    (nmf, mv, ff, ins, sig_backend, probe_feats) = scorers
    (W, W_pinv, H_fit, h_lookup, mu, sigma, channels_raw, channels_full,
     prune_mask, base_idx) = nmf
    atom_count = W.shape[0]

    rows = []
    pngs = sorted((out_dir / variant_key).rglob("*.png"))
    print(f"  [score] {variant_key}: {len(pngs)} images")

    with make_landmarker() as lm:
        for p in pngs:
            parts = p.relative_to(out_dir / variant_key).parts
            base = parts[0]
            fname = parts[1]
            seed = int(fname.split("_")[0].removeprefix("seed"))
            scale = float(fname.split("_")[1].removeprefix("s").removesuffix(".png"))

            bgr = cv2.imread(str(p))
            bs = score_png(lm, p) if bgr is not None else None

            row: dict = {
                "variant": variant_key, "base": base, "seed": seed, "scale": scale,
                "img_path": str(p.relative_to(ROOT)),
            }

            # blendshape + atom projection
            if bs is not None and base in base_idx:
                atoms = project_sample(bs, channels_full, prune_mask, mu, sigma,
                                       base_idx[base], W_pinv)
                for c in channels_full:
                    row[f"bs_{c}"] = float(bs.get(c, 0.0))
                for k in range(atom_count):
                    row[f"atom_{k:02d}"] = float(atoms[k])
            else:
                for c in channels_full:
                    row[f"bs_{c}"] = np.nan
                for k in range(atom_count):
                    row[f"atom_{k:02d}"] = np.nan

            # classifiers
            if bgr is not None:
                m = mv.predict(bgr)
                row["mv_age"] = m["age"]; row["mv_gender"] = m["gender"]; row["mv_gender_conf"] = m["gender_conf"]
                f = ff.predict(bgr)
                row["ff_detected"] = f["detected"]
                row["ff_age_bin"] = f["age_bin"]; row["ff_gender"] = f["gender"]; row["ff_race"] = f["race"]
                i = ins.predict(bgr)
                row["ins_detected"] = i["detected"]
                row["ins_age"] = i["age"]; row["ins_gender"] = i["gender"]
                row["ins_embedding"] = i["embedding"].tolist() if i["embedding"] is not None else None
                # SigLIP probes
                feat = sig_backend.encode_image(p)
                for name, pf in probe_feats.items():
                    sims = (feat @ pf.T).squeeze(0)
                    row[f"siglip_{name}_margin"] = float(sims[0] - sims[1])
                feat_np = feat.squeeze(0).cpu().numpy().astype(np.float32)
                row["siglip_img_feat"] = feat_np.tolist()

            rows.append(row)

    df = pd.DataFrame(rows)

    # identity + siglip-img cosines: seed-matched scale=0 anchor per base
    def _arr(col, dim):
        out = np.zeros((len(df), dim), dtype=np.float32)
        for ix, v in enumerate(df[col]):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            out[ix] = np.asarray(v, dtype=np.float32)
        return out

    arc = _arr("ins_embedding", 512)
    sig = _arr("siglip_img_feat", 1152)
    arc_has = df["ins_embedding"].apply(lambda v: v is not None).to_numpy()
    mask0 = np.isclose(df["scale"].to_numpy(), 0.0)

    anchor_arc: dict[tuple[str, int], np.ndarray] = {}
    anchor_sig: dict[tuple[str, int], np.ndarray] = {}
    for (base, seed), grp in df[mask0].groupby(["base", "seed"]):
        ix = grp.index.to_numpy()
        if arc_has[ix].any():
            anchor_arc[(base, seed)] = arc[ix][arc_has[ix]].mean(axis=0)
        anchor_sig[(base, seed)] = sig[ix].mean(axis=0)

    id_cos = np.full(len(df), np.nan, dtype=np.float32)
    sig_cos = np.full(len(df), np.nan, dtype=np.float32)
    for ix, r in df.iterrows():
        key = (r["base"], int(r["seed"]))
        if key in anchor_arc and arc_has[ix]:
            a = anchor_arc[key]
            na = np.linalg.norm(a); nv = np.linalg.norm(arc[ix])
            if na * nv > 1e-9:
                id_cos[ix] = float((a * arc[ix]).sum() / (na * nv))
        if key in anchor_sig:
            a = anchor_sig[key]
            na = np.linalg.norm(a); nv = np.linalg.norm(sig[ix])
            if na * nv > 1e-9:
                sig_cos[ix] = float((a * sig[ix]).sum() / (na * nv))
    df["identity_cos_to_base"] = id_cos
    df["siglip_img_cos_to_base"] = sig_cos

    # drop heavy list columns from the saved parquet
    for c in ("ins_embedding", "siglip_img_feat"):
        if c in df.columns:
            df = df.drop(columns=c)
    return df


# ── slope fitting / ranking ──────────────────────────────────────────────────


def _slope(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(pd.to_numeric(pd.Series(y), errors="coerce"), dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 2:
        return float("nan"), float("nan")
    b, a = np.polyfit(x[m], y[m], 1)
    yhat = a + b * x[m]
    ss_res = float(((y[m] - yhat) ** 2).sum())
    ss_tot = float(((y[m] - y[m].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else float("nan")
    return float(b), float(r2)


def summarize(df: pd.DataFrame, target_probe: str | None, target_atom: int | None) -> pd.DataFrame:
    """One row per (variant, base); mean/slope of readouts across seeds × scales."""
    atom_cols = [c for c in df.columns if c.startswith("atom_")]
    rows = []
    for (variant, base), cell in df.groupby(["variant", "base"]):
        x = cell["scale"].to_numpy()
        rec: dict = {"variant": variant, "base": base, "n": len(cell)}

        # race & gender flip detection: compare mode at min scale vs mode at max scale
        sorted_x = sorted(set(x))
        if len(sorted_x) >= 2:
            mask_lo = cell["scale"] == sorted_x[0]
            mask_hi = cell["scale"] == sorted_x[-1]
            if mask_lo.any() and mask_hi.any():
                race_lo = cell.loc[mask_lo, "ff_race"].mode()
                race_hi = cell.loc[mask_hi, "ff_race"].mode()
                gen_lo = cell.loc[mask_lo, "ff_gender"].mode()
                gen_hi = cell.loc[mask_hi, "ff_gender"].mode()
                rec["race_lo"] = race_lo.iloc[0] if len(race_lo) else None
                rec["race_hi"] = race_hi.iloc[0] if len(race_hi) else None
                rec["race_flipped"] = (rec["race_lo"] != rec["race_hi"]) if rec["race_lo"] and rec["race_hi"] else False
                rec["gender_lo"] = gen_lo.iloc[0] if len(gen_lo) else None
                rec["gender_hi"] = gen_hi.iloc[0] if len(gen_hi) else None
                rec["gender_flipped"] = (rec["gender_lo"] != rec["gender_hi"]) if rec["gender_lo"] and rec["gender_hi"] else False

        if target_probe and target_probe in cell.columns:
            b, r2 = _slope(x, cell[target_probe].to_numpy())
            rec["target_slope"] = b; rec["target_r2"] = r2
        else:
            rec["target_slope"] = np.nan; rec["target_r2"] = np.nan

        # primary atom: max |slope| if not specified
        if target_atom is not None:
            k = target_atom
            b, r2 = _slope(x, cell[f"atom_{k:02d}"].to_numpy())
            rec["target_atom"] = f"atom_{k:02d}"
            rec["target_atom_slope"] = b; rec["target_atom_r2"] = r2
        else:
            best, bval, br2 = None, 0.0, 0.0
            for a in atom_cols:
                b, r2 = _slope(x, cell[a].to_numpy())
                if np.isfinite(b) and abs(b) > abs(bval):
                    best, bval, br2 = a, b, r2
            rec["target_atom"] = best
            rec["target_atom_slope"] = bval; rec["target_atom_r2"] = br2

        # atom mass (sum |slope| across all atoms)
        mass = 0.0
        for a in atom_cols:
            b, _ = _slope(x, cell[a].to_numpy())
            if np.isfinite(b):
                mass += abs(b)
        rec["atom_total_mass"] = mass
        rec["atom_purity"] = abs(rec["target_atom_slope"]) / mass if mass > 1e-9 else np.nan

        b, _ = _slope(x, cell["mv_age"].to_numpy().astype(float))
        rec["mv_age_slope"] = b
        b, _ = _slope(x, cell["ins_age"].to_numpy().astype(float))
        rec["ins_age_slope"] = b
        b, _ = _slope(x, 1.0 - cell["identity_cos_to_base"].to_numpy().astype(float))
        rec["identity_drift_slope"] = b
        b, _ = _slope(x, 1.0 - cell["siglip_img_cos_to_base"].to_numpy().astype(float))
        rec["total_drift_slope"] = b

        rows.append(rec)
    return pd.DataFrame(rows)


def append_to_dictionary(spec: dict, iter_dir: Path, full: pd.DataFrame) -> None:
    """Emit per-(variant, base) slope rows to the canonical dictionary parquet.

    One row per (iter, variant, base) cell. Readout slopes are computed here
    (not reusing summarize's primary-only output) so we capture ALL atom +
    blendshape + classifier + probe slopes for the solver.
    """
    from datetime import datetime
    iteration_id = f"{spec['axis']}/{iter_dir.name}"
    ts = datetime.utcnow().isoformat(timespec="seconds")

    # every numeric, scale-dependent readout except scale itself and obvious keys
    exclude = {"variant", "base", "seed", "scale", "img_path",
               "mv_gender", "ff_age_bin", "ff_gender", "ff_race",
               "ff_detected", "ins_gender", "ins_detected"}
    atom_cols = [c for c in full.columns if c.startswith("atom_")]
    bs_cols   = [c for c in full.columns if c.startswith("bs_")]
    sig_cols  = [c for c in full.columns if c.startswith("siglip_") and c.endswith("_margin")]
    num_cols = atom_cols + bs_cols + sig_cols + [
        "mv_age", "mv_gender_conf", "ins_age",
        "identity_cos_to_base", "siglip_img_cos_to_base",
    ]
    num_cols = [c for c in num_cols if c in full.columns and c not in exclude]

    rows = []
    for (variant, base), cell in full.groupby(["variant", "base"]):
        x = cell["scale"].to_numpy()
        rec: dict = {
            "iteration_id": iteration_id,
            "axis": spec["axis"],
            "variant": variant,
            "base": base,
            "prompt_pos": spec["variants"][variant]["pos"] if isinstance(spec["variants"][variant]["pos"], str) else json.dumps(spec["variants"][variant]["pos"]),
            "prompt_neg": spec["variants"][variant]["neg"] if isinstance(spec["variants"][variant]["neg"], str) else json.dumps(spec["variants"][variant]["neg"]),
            "n_seeds": int(cell["seed"].nunique()),
            "n_scales": int(cell["scale"].nunique()),
            "scale_max": float(cell["scale"].max()),
            "rendered_at": ts,
        }
        for col in num_cols:
            b, r2 = _slope(x, cell[col].to_numpy())
            rec[f"slope_{col}"] = b
            if col in ("mv_age", "identity_cos_to_base", "siglip_img_cos_to_base"):
                rec[f"r2_{col}"] = r2
        rows.append(rec)
    new_df = pd.DataFrame(rows)

    # accumulate, replacing any prior row for the same (iteration_id, variant, base)
    if DICTIONARY_PARQUET.exists():
        old = pd.read_parquet(DICTIONARY_PARQUET)
        key_cols = ["iteration_id", "variant", "base"]
        old = old[~old.set_index(key_cols).index.isin(new_df.set_index(key_cols).index)]
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
    DICTIONARY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(DICTIONARY_PARQUET, index=False, compression="zstd")
    print(f"[dict] → {DICTIONARY_PARQUET}  rows={len(combined)} (+{len(new_df)} new)")


def compose_score(agg: pd.DataFrame) -> pd.DataFrame:
    """One row per variant (means across bases); composite score."""
    out = agg.groupby("variant").agg(
        target_slope=("target_slope", "mean"),
        target_r2=("target_r2", "mean"),
        target_atom_slope=("target_atom_slope", "mean"),
        atom_purity=("atom_purity", "mean"),
        mv_age_slope=("mv_age_slope", "mean"),
        ins_age_slope=("ins_age_slope", "mean"),
        identity_drift_slope=("identity_drift_slope", "mean"),
        total_drift_slope=("total_drift_slope", "mean"),
    ).reset_index()
    # score: reward target, penalize age and identity drift
    out["score"] = out["target_slope"] / (
        1.0 + 0.05 * out["mv_age_slope"].abs() + 1.5 * out["identity_drift_slope"].abs()
    )
    return out.sort_values("score", ascending=False)


# ── report ───────────────────────────────────────────────────────────────────


def write_report(spec: dict, per_base: pd.DataFrame, per_variant: pd.DataFrame,
                 iter_dir: Path) -> None:
    axis = spec["axis"]
    lines = [
        "---", "status: live", "topic: metrics-and-direction-quality", "---", "",
        f"# Prompt-pair iterate — axis=`{axis}` — iter `{iter_dir.name}`", "",
        f"**Target probe:** `{spec.get('target_probe')}` · **target atom:** `atom_{spec.get('target_atom'):02d}`" if spec.get("target_atom") is not None else f"**Target probe:** `{spec.get('target_probe')}`",
        "",
        "## Variants", "",
    ]
    for vkey, v in spec["variants"].items():
        lines.append(f"- **{vkey}**  pos=`{v['pos']}`  neg=`{v['neg']}`")
    lines += ["", "## Per-variant summary (means across 3 bases × 2 seeds × 3 scales)", ""]
    lines += ["| variant | target | R² | target_atom | atom_slope | purity | mv_age | ins_age | id_drift | total_drift | score |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in per_variant.iterrows():
        lines.append(
            f"| **{r['variant']}** | {r['target_slope']:+.3f} | {r['target_r2']:.2f} | "
            f"atom_{int(spec.get('target_atom', 16)):02d} | {r['target_atom_slope']:+.3f} | "
            f"{r['atom_purity']:.2f} | {r['mv_age_slope']:+.2f} | {r['ins_age_slope']:+.2f} | "
            f"{r['identity_drift_slope']:+.3f} | {r['total_drift_slope']:+.3f} | **{r['score']:.3f}** |"
        )
    lines += ["", "## Per-(variant, base) detail", ""]
    lines += ["| variant | base | target | atom | mv_age | id_drift | race (lo → hi) | gender |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in per_base.sort_values(["variant", "base"]).iterrows():
        race_txt = f"{r.get('race_lo', '?')} → {r.get('race_hi', '?')}"
        if r.get("race_flipped"):
            race_txt = f"**{race_txt}**"
        gen_txt = f"{r.get('gender_lo', '?')} → {r.get('gender_hi', '?')}"
        if r.get("gender_flipped"):
            gen_txt = f"**{gen_txt}**"
        lines.append(
            f"| {r['variant']} | {r['base']} | {r['target_slope']:+.3f} | "
            f"{r['target_atom_slope']:+.2f} | "
            f"{r['mv_age_slope']:+.2f} | {r['identity_drift_slope']:+.3f} | "
            f"{race_txt} | {gen_txt} |"
        )

    winner = per_variant.iloc[0]
    lines += ["", "## Recommendation", "",
              f"**Winner: `{winner['variant']}`** with composite score {winner['score']:.3f}.",
              "",
              f"- Target slope {winner['target_slope']:+.3f} (R²={winner['target_r2']:.2f})",
              f"- Age drift (MiVOLO) {winner['mv_age_slope']:+.2f} y/scale (InsightFace {winner['ins_age_slope']:+.2f})",
              f"- Identity drift {winner['identity_drift_slope']:+.3f}/scale",
              f"- Atom-purity {winner['atom_purity']:.2f}",
              "",
              "Scoring: `target / (1 + 0.05·|age| + 1.5·|id_drift|)`. Rewards target, "
              "penalizes age + identity drift.",
              ""]
    (iter_dir / "report.md").write_text("\n".join(lines) + "\n")


# ── main orchestrator ────────────────────────────────────────────────────────


async def run(spec: dict, iter_dir: Path, do_render: bool, do_score: bool) -> None:
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "spec.yaml").write_text(yaml.safe_dump(spec))

    if do_render:
        scales_override = spec.get("scales")
        for vkey, v in spec["variants"].items():
            print(f"[render] {vkey}")
            await render_variant(vkey, v["pos"], v["neg"], iter_dir, scales=scales_override)

    if do_score:
        scorers = _load_scorers()
        frames = []
        for vkey in spec["variants"].keys():
            frames.append(score_variant(vkey, iter_dir, scorers))
        full = pd.concat(frames, ignore_index=True)
        full.to_parquet(iter_dir / "results.parquet", index=False, compression="zstd")
        print(f"[save] → {iter_dir / 'results.parquet'}  rows={len(full)}")

        per_base = summarize(full, spec.get("target_probe"), spec.get("target_atom"))
        per_variant = compose_score(per_base)
        per_base.to_parquet(iter_dir / "per_base.parquet", index=False, compression="zstd")
        per_variant.to_parquet(iter_dir / "per_variant.parquet", index=False, compression="zstd")
        write_report(spec, per_base, per_variant, iter_dir)
        print(f"[save] → {iter_dir / 'report.md'}")

        append_to_dictionary(spec, iter_dir, full)
        print()
        print(per_variant.round(3).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axis", required=True)
    ap.add_argument("--iter", required=True, help="iteration label, e.g. '01'")
    ap.add_argument("--spec", type=Path, required=True, help="YAML spec path")
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--run", action="store_true", help="equivalent to --render --score")
    args = ap.parse_args()
    spec = yaml.safe_load(args.spec.read_text())
    iter_dir = OUT_ROOT / args.axis / f"iter_{args.iter}"
    do_render = args.render or args.run
    do_score = args.score or args.run
    asyncio.run(run(spec, iter_dir, do_render, do_score))


if __name__ == "__main__":
    main()
