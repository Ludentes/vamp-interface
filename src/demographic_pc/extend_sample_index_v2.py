"""Extend the canonical sample_index parquet with:

  - overnight_new_axes renders (2026-04-24 — eye_squint, brow_lift,
    brow_furrow, gaze_horizontal, mouth_stretch, age, gender, hair_*, etc.)
  - reprompt_v2 renders (mouth_stretch_v2, brow_furrow_v2)
  - crossdemo_v2_expand renders (Strategy A expanded corpus)

Adds new derived columns to **all rows**:

  - `identity_cos_to_base`: cos(ins_embedding, anchor_embedding) where
    anchor = the α=0 row in the same (source, axis, base, seed) group
  - `identity_pass_075`: bool, cos ≥ 0.75

Also extends classifier_scores.parquet with InsightFace embeddings for
new rows.

Run:
    uv run python src/demographic_pc/extend_sample_index_v2.py

Creates dated backups: sample_index.parquet.bak.<timestamp> before overwriting.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.demographic_pc.build_sample_index import (        # noqa: E402
    BASE_META, load_nmf, project_sample, parse_filename_meta,
    OUT_PARQUET as SAMPLE_INDEX_PATH,
)
from src.demographic_pc.classifiers import InsightFaceClassifier  # noqa: E402

METRICS = ROOT / "output/demographic_pc/fluxspace_metrics"
CROSSDEMO = METRICS / "crossdemo"
CROSSDEMO_V2 = METRICS / "crossdemo_v2"
CROSSDEMO_V3 = METRICS / "crossdemo_v3"
CROSSDEMO_V3_1 = METRICS / "crossdemo_v3_1"
OVERNIGHT_BS_JSON = ROOT / "output/demographic_pc/overnight_blendshapes.json"
CLASSIFIER_PARQUET = ROOT / "output/demographic_pc/classifier_scores.parquet"

# Extended base metadata — full CALIBRATION_PROMPTS set used by expand_corpus_v2
EXTENDED_BASE_META = {
    **BASE_META,
    "adult_latin_f":     ("latin",      "f", "adult"),
    "adult_asian_m":     ("asian",      "m", "adult"),
    "adult_black_f":     ("black",      "f", "adult"),
    "adult_european_m":  ("european",   "m", "adult"),
    "adult_middle_f":    ("middle",     "f", "adult"),
    "elderly_asian_f":   ("asian",      "f", "elderly"),
    "young_black_m":     ("black",      "m", "young"),
    "adult_southasian_f":("southasian", "f", "adult"),
}


def discover_sources() -> list[dict]:
    """Walk crossdemo/ and crossdemo_v2/ for axis subdirs with rendered
    PNGs, returning a list of source descriptors."""
    sources = []
    for root, prefix in [
        (CROSSDEMO, ""),
        (CROSSDEMO_V2, "v2_"),
        (CROSSDEMO_V3, "v3_"),
        (CROSSDEMO_V3_1, "v3_1_"),
    ]:
        if not root.exists():
            continue
        for axis_dir in sorted(root.iterdir()):
            if not axis_dir.is_dir():
                continue
            # Find <axis>_inphase subdir
            inphase = axis_dir / f"{axis_dir.name}_inphase"
            if not inphase.exists():
                continue
            # Any PNGs under bases?
            n_png = sum(1 for _ in inphase.glob("*/*.png"))
            if n_png == 0:
                continue
            tag = f"{prefix}{axis_dir.name}"
            sources.append({
                "tag": tag,
                "axis": axis_dir.name,
                "img_dir": inphase,
                "blendshapes_json": inphase / "blendshapes.json",
                "n_png": n_png,
                "corpus_version": (
                    "v3_1_multipair_clean" if prefix == "v3_1_"
                    else "v3_multipair" if prefix == "v3_"
                    else "v2_expand" if prefix == "v2_"
                    else "v1"
                ),
            })
    return sources


def existing_sources(df: pd.DataFrame) -> set[str]:
    return set(df["source"].unique())


# ---------- MediaPipe scoring ----------

def ensure_blendshapes(src: dict) -> dict:
    """Load or compute blendshapes.json for a source. Keys are paths
    relative to img_dir."""
    bs_path = src["blendshapes_json"]
    if bs_path.exists():
        return json.loads(bs_path.read_text())

    # Fall back: check overnight_blendshapes.json for overnight axes
    if OVERNIGHT_BS_JSON.exists():
        all_bs = json.loads(OVERNIGHT_BS_JSON.read_text())
        # overnight keys are like "crossdemo/<axis>/<axis>_inphase/<base>/s...png"
        prefix = str(src["img_dir"].relative_to(ROOT)) + "/"
        subset = {
            k.removeprefix(prefix): v
            for k, v in all_bs.items()
            if k.startswith(prefix)
        }
        if subset:
            # Cache the subset alongside the images
            bs_path.write_text(json.dumps(subset))
            print(f"    [bs] extracted {len(subset)} from overnight_blendshapes.json "
                  f"→ {bs_path.name}")
            return subset

    # Score from scratch with MediaPipe
    print(f"    [bs] scoring {src['n_png']} PNGs with MediaPipe …")
    from src.demographic_pc.score_blendshapes import make_landmarker, score_png
    lm = make_landmarker()
    out: dict[str, dict] = {}
    for png in sorted(src["img_dir"].glob("*/*.png")):
        rel = str(png.relative_to(src["img_dir"]))
        try:
            scores = score_png(lm, png)
            if scores is not None:
                out[rel] = scores
        except Exception as e:
            print(f"      [skip] {rel}: {e}")
    bs_path.write_text(json.dumps(out))
    print(f"    [bs] wrote {len(out)} → {bs_path.name}")
    return out


# ---------- InsightFace scoring ----------

_IF_CLIENT = None

def get_insightface():
    global _IF_CLIENT
    if _IF_CLIENT is None:
        _IF_CLIENT = InsightFaceClassifier(ctx_id=0, with_embedding=True)
    return _IF_CLIENT


def score_identity(src: dict, existing_embeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute ArcFace embedding per PNG under src['img_dir']. Returns
    dict keyed by `img_path_rel_to_root` (matching classifier_scores convention)."""
    out = {}
    clf = get_insightface()
    pngs = sorted(src["img_dir"].glob("*/*.png"))
    skipped = 0
    for i, png in enumerate(pngs):
        key = str(png.relative_to(ROOT))
        if key in existing_embeds:
            out[key] = existing_embeds[key]
            skipped += 1
            continue
        img = cv2.imread(str(png))
        if img is None:
            continue
        pred = clf.predict(img)
        if pred["detected"] and pred["embedding"] is not None:
            out[key] = np.asarray(pred["embedding"], dtype=np.float32)
        else:
            out[key] = None  # no face
    if skipped:
        print(f"    [if] scored {len(pngs)-skipped} new, {skipped} cached")
    else:
        print(f"    [if] scored {len(pngs)} images")
    return out


# ---------- Row builder ----------

def parse_base_extended(rel: str) -> str | None:
    top = rel.split("/", 1)[0]
    if top in EXTENDED_BASE_META:
        return top
    return None


def build_rows(src: dict, bs_data: dict, embeds: dict[str, np.ndarray],
               nmf_ctx) -> list[dict]:
    (W, W_pinv, H_fit, h_lookup, mu, sigma, channels_raw, channels_full,
     prune_mask, base_idx) = nmf_ctx

    rows = []
    for rel, scores in bs_data.items():
        base = parse_base_extended(rel)
        if base is None:
            continue
        eth, gen, age = EXTENDED_BASE_META[base]
        meta = parse_filename_meta(rel)
        # Atoms via pinv (these are not in the NMF fit)
        # base_idx may not have the new bases → fall back to mean of known
        bi = base_idx.get(base)
        if bi is None:
            # Compute residual using global mu/sigma mean (stand-in for new bases)
            bi_mu = mu.mean(axis=0)
            bi_sigma = sigma.mean(axis=0)
            x = np.array([scores.get(c, 0.0) for c in channels_full])
            sigma_safe = np.where(bi_sigma < 1e-4, 1.0, bi_sigma)
            x_res = (x - bi_mu) / sigma_safe
            x_res = x_res[prune_mask]
            x_pos = np.clip(x_res, 0.0, None)
            x_neg = np.clip(-x_res, 0.0, None)
            atoms = np.clip(np.concatenate([x_pos, x_neg]) @ W_pinv, 0.0, None)
        else:
            atoms = project_sample(scores, channels_full, prune_mask, mu, sigma,
                                   bi, W_pinv)

        img_path = str((src["img_dir"] / rel).relative_to(ROOT))
        emb = embeds.get(img_path)
        row = {
            "source": src["tag"], "rel": rel, "base": base,
            "ethnicity": eth, "gender": gen, "age": age,
            "axis": src["axis"], "scale": meta["scale"], "seed": meta["seed"],
            "start_pct": meta["start_pct"], "alpha": meta["alpha"],
            "has_attn": False, "attn_tag": None, "attn_row": -1,
            "atom_source": "pinv",
            "corpus_version": src["corpus_version"],
            "img_path": img_path,
            "_ins_embedding": emb,  # temp column, will split into classifier_scores
        }
        for c in channels_full:
            row[f"bs_{c}"] = float(scores.get(c, 0.0))
        for k in range(len(atoms)):
            row[f"atom_{k:02d}"] = float(atoms[k])
        rows.append(row)
    return rows


# ---------- identity_cos_to_base ----------

def compute_identity_cos(df: pd.DataFrame, embed_col: str) -> pd.Series:
    """For each row, compute cos with the α=0 (or scale=0) anchor in the
    same (source, axis, base, seed) group. Returns a Series aligned to df."""
    cos = pd.Series(np.nan, index=df.index, dtype="float32")

    # Group by (source, axis, base, seed)
    group_keys = ["source", "axis", "base", "seed"]
    for keys, gdf in df.groupby(group_keys, dropna=False):
        # Pick anchor: prefer alpha==0.0, else scale==0.0
        anchors = gdf[gdf["alpha"] == 0.0]
        if len(anchors) == 0:
            anchors = gdf[gdf["scale"] == 0.0]
        if len(anchors) == 0:
            continue
        anchor_emb = None
        for _, r in anchors.iterrows():
            if r[embed_col] is not None:
                anchor_emb = np.asarray(r[embed_col], dtype=np.float32)
                break
        if anchor_emb is None:
            continue
        anchor_emb = anchor_emb / (np.linalg.norm(anchor_emb) + 1e-12)
        for idx, r in gdf.iterrows():
            emb = r[embed_col]
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-12)
            cos.loc[idx] = float(np.dot(anchor_emb, emb))
    return cos


# ---------- main ----------

def main():
    print(f"[load] sample_index from {SAMPLE_INDEX_PATH}")
    df_existing = pd.read_parquet(SAMPLE_INDEX_PATH)
    print(f"  existing shape: {df_existing.shape}")
    exist_srcs = existing_sources(df_existing)
    print(f"  sources: {sorted(exist_srcs)}")

    # Load NMF once
    print("[load] NMF basis")
    nmf_ctx = load_nmf()
    channels_full = nmf_ctx[7]
    bs_cols = [f"bs_{c}" for c in channels_full]
    atom_cols = [f"atom_{k:02d}" for k in range(nmf_ctx[0].shape[0])]

    # Load existing classifier embeddings
    print(f"[load] classifier_scores from {CLASSIFIER_PARQUET}")
    if CLASSIFIER_PARQUET.exists():
        df_cls = pd.read_parquet(CLASSIFIER_PARQUET)
        print(f"  existing shape: {df_cls.shape}")
        existing_embeds = {
            r["img_path"]: np.asarray(r["ins_embedding"], dtype=np.float32)
            if r["ins_embedding"] is not None else None
            for _, r in df_cls.iterrows()
        }
    else:
        df_cls = pd.DataFrame()
        existing_embeds = {}

    # Discover sources to add OR rescore (source in index but identity_cos
    # missing for its rows — likely means a previous run failed to persist
    # embeddings to classifier_scores.parquet).
    sources = discover_sources()
    sources_needing_embeds: set[str] = set()
    if "identity_cos_to_base" in df_existing.columns:
        # Sources where >30% of rows lack identity_cos AND have an img_path
        # whose embedding isn't in existing_embeds → need rescoring.
        for s in sources:
            rows = df_existing[df_existing["source"] == s["tag"]]
            if len(rows) == 0:
                continue
            n_no_emb = sum(
                1 for p in rows["img_path"]
                if p not in existing_embeds or existing_embeds[p] is None
            )
            if n_no_emb / len(rows) > 0.3:
                sources_needing_embeds.add(s["tag"])

    missing = [s for s in sources
               if (s["tag"] not in exist_srcs) or (s["tag"] in sources_needing_embeds)]
    print(f"\n[discover] {len(sources)} crossdemo sources on disk, "
          f"{len([s for s in sources if s['tag'] not in exist_srcs])} new, "
          f"{len(sources_needing_embeds)} need embedding rescore")
    for s in missing:
        print(f"  + {s['tag']:<30s} axis={s['axis']:<20s} n_png={s['n_png']}")

    if not missing:
        print("\nNo new sources to add. Will only (re)compute identity_cos_to_base.")

    new_rows = []
    new_embeds: dict[str, np.ndarray] = {}
    for src in missing:
        is_rescore = src["tag"] in exist_srcs  # rescore path, not add
        print(f"\n[src] {src['tag']} ({'rescore-embeds' if is_rescore else 'new'})")
        bs = ensure_blendshapes(src)
        embeds = score_identity(src, existing_embeds)
        new_embeds.update(embeds)
        if not is_rescore:
            rows = build_rows(src, bs, embeds, nmf_ctx)
            new_rows.extend(rows)
            print(f"  → {len(rows)} rows built")

    # Build new DataFrame chunk + merge
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        # Ensure all channel cols exist
        for c in bs_cols + atom_cols:
            if c not in df_new.columns:
                df_new[c] = 0.0
        meta_cols = ["source", "rel", "base", "ethnicity", "gender", "age",
                     "axis", "scale", "seed", "start_pct", "has_attn",
                     "attn_tag", "attn_row", "atom_source", "img_path"]
        # New extra cols
        extras = ["corpus_version", "alpha"]
        all_cols = meta_cols + extras + bs_cols + atom_cols
        # Align existing df to these cols (fill missing with NaN)
        for c in extras:
            if c not in df_existing.columns:
                df_existing[c] = np.nan
        df_new = df_new[all_cols + ["_ins_embedding"]]
        df_existing = df_existing[[c for c in all_cols if c in df_existing.columns]]
        # Concat
        df_new_no_emb = df_new.drop(columns=["_ins_embedding"])
        df_combined = pd.concat([df_existing, df_new_no_emb], ignore_index=True)
        print(f"\n[combine] {len(df_existing)} + {len(df_new_no_emb)} = {len(df_combined)}")
    else:
        df_combined = df_existing
        if "corpus_version" not in df_combined.columns:
            df_combined["corpus_version"] = "v1"
        if "alpha" not in df_combined.columns:
            df_combined["alpha"] = np.nan

    # Build full embedding lookup and add to combined df for cosine compute
    print("\n[identity] computing cosine similarity to α=0 anchors")
    all_embeds = {**existing_embeds, **new_embeds}
    df_combined["_embed"] = df_combined["img_path"].map(all_embeds)
    df_combined["identity_cos_to_base"] = compute_identity_cos(df_combined, "_embed")
    df_combined["identity_pass_075"] = (df_combined["identity_cos_to_base"] >= 0.75)
    df_combined = df_combined.drop(columns=["_embed"])

    n_with_cos = df_combined["identity_cos_to_base"].notna().sum()
    n_pass = df_combined["identity_pass_075"].sum()
    print(f"  {n_with_cos} rows have identity_cos, {n_pass} pass τ=0.75")

    # Backup + write sample_index
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup = SAMPLE_INDEX_PATH.with_suffix(f".parquet.bak.{ts}")
    shutil.copy2(SAMPLE_INDEX_PATH, backup)
    print(f"\n[backup] {backup.name}")
    df_combined.to_parquet(SAMPLE_INDEX_PATH, index=False, compression="zstd")
    print(f"[save] → {SAMPLE_INDEX_PATH} "
          f"({SAMPLE_INDEX_PATH.stat().st_size / 1024 / 1024:.2f} MB, "
          f"{df_combined.shape[0]} rows × {df_combined.shape[1]} cols)")

    # Update classifier_scores with new embeddings. Cast all to float32
    # to avoid pyarrow dtype-mix errors when mixing with existing rows.
    if new_embeds:
        new_cls_rows = [
            {"img_path": k,
             "ins_embedding": v.astype(np.float32) if v is not None else None,
             "ins_detected": v is not None}
            for k, v in new_embeds.items()
        ]
        df_new_cls = pd.DataFrame(new_cls_rows)
        # Coerce existing embeddings to float32 too
        if not df_cls.empty and "ins_embedding" in df_cls.columns:
            df_cls["ins_embedding"] = df_cls["ins_embedding"].apply(
                lambda v: np.asarray(v, dtype=np.float32) if v is not None else None
            )
        if not df_cls.empty:
            keep_existing = df_cls[~df_cls["img_path"].isin(df_new_cls["img_path"])]
            df_cls_combined = pd.concat([keep_existing, df_new_cls], ignore_index=True)
        else:
            df_cls_combined = df_new_cls
        backup_cls = CLASSIFIER_PARQUET.with_suffix(f".parquet.bak.{ts}")
        shutil.copy2(CLASSIFIER_PARQUET, backup_cls) if CLASSIFIER_PARQUET.exists() else None
        df_cls_combined.to_parquet(CLASSIFIER_PARQUET, index=False, compression="zstd")
        print(f"[save] classifier_scores: {len(df_cls_combined)} rows")

    # Summary
    print("\n[summary] identity pass rate per (source, alpha):")
    if n_with_cos > 0:
        pivot = df_combined.dropna(subset=["identity_cos_to_base"]).groupby(
            ["source", "alpha"])["identity_pass_075"].agg(["mean", "count"])
        print(pivot.to_string())


if __name__ == "__main__":
    main()
