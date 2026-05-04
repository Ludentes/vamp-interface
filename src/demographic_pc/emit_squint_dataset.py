"""Flatten pair_manifest.parquet into an ai-toolkit Path B dataset.

Both sha_pos and sha_neg of every kept pair are emitted as anchor
images. Each <sha>.png gets:
  - <sha>.txt : caption with demographics
  - <sha>.json : sidecar with {"source": ..., "role": "pos|neg",
                  "sha": ..., "ff_race", "ff_gender", "ff_age_bin"}

The trainer reads the eye mask separately from output/squint_path_b/eye_masks/<sha>.npy
(see ConceptSliderTrainer._load_eye_mask). The sidecar JSON drives the
per-source loss weighting.

If a sha appears as both pos in one pair and neg in another (rare after
cluster_cap=1), we keep the first role we see and emit it once.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "output/squint_path_b/pair_manifest.parquet"
IMG_DIR = REPO / "output/ffhq_images"
DATASET_DIR = REPO / "datasets/ai_toolkit_squint_v0/train"


_AGE_WORD = {"young": "young", "adult": "adult", "elderly": "elderly"}
_GENDER_WORD = {"m": "man", "f": "woman"}
_RACE_WORD = {
    "white": "white",
    "black": "black",
    "east_asian": "East Asian",
    "south_asian": "South Asian",
    "latino": "Latino",
    "middle_eastern": "Middle Eastern",
    "southeast_asian": "Southeast Asian",
}


def make_caption(race: str | None, gender: str | None, age: str | None) -> str:
    age_w = _AGE_WORD.get(str(age), "adult")
    gender_w = _GENDER_WORD.get(str(gender), "person")
    race_w = _RACE_WORD.get(str(race), "")
    article = "an" if race_w.lower().startswith(("a", "e", "i", "o", "u")) else "a"
    if race_w:
        subj = f"{article} {age_w} {race_w} {gender_w}"
    else:
        subj = f"a {age_w} {gender_w}"
    return f"a photorealistic portrait photograph of {subj}, plain grey background, studio lighting"


def main() -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MANIFEST)
    kept = m[m["kept"]].copy()
    print(f"[manifest] {len(kept)} kept pairs ({kept['source'].value_counts().to_dict()})")

    # flatten: emit each row twice (pos then neg)
    rows: list[dict] = []
    for _, r in kept.iterrows():
        for role, sha in (("pos", r["sha_pos"]), ("neg", r["sha_neg"])):
            rows.append({
                "sha": sha,
                "role": role,
                "source": r["source"],
                "ff_race": r.get("ff_race"),
                "ff_gender": r.get("ff_gender"),
                "ff_age_bin": r.get("ff_age_bin"),
            })
    flat = pd.DataFrame(rows)
    # dedup: same sha may appear in multiple pairs; keep first occurrence
    flat = flat.drop_duplicates(subset=["sha"]).reset_index(drop=True)
    print(f"[flatten] {len(flat)} unique anchor shas "
          f"(roles: {flat['role'].value_counts().to_dict()}, "
          f"sources: {flat['source'].value_counts().to_dict()})")

    n_emitted = n_skipped = n_missing = 0
    for _, r in flat.iterrows():
        sha = r["sha"]
        src_png = IMG_DIR / f"{sha}.png"
        if not src_png.exists():
            n_missing += 1
            continue
        dst_png = DATASET_DIR / f"{sha}.png"
        dst_txt = DATASET_DIR / f"{sha}.txt"
        dst_json = DATASET_DIR / f"{sha}.json"

        if not dst_png.exists():
            shutil.copy2(src_png, dst_png)
        else:
            n_skipped += 1
            continue

        caption = make_caption(r["ff_race"], r["ff_gender"], r["ff_age_bin"])
        dst_txt.write_text(caption + "\n")

        sidecar = {
            "sha": sha,
            "role": r["role"],
            "source": r["source"],
            "ff_race": str(r["ff_race"]) if pd.notna(r["ff_race"]) else None,
            "ff_gender": str(r["ff_gender"]) if pd.notna(r["ff_gender"]) else None,
            "ff_age_bin": str(r["ff_age_bin"]) if pd.notna(r["ff_age_bin"]) else None,
        }
        dst_json.write_text(json.dumps(sidecar))
        n_emitted += 1

    print(f"[emit] {n_emitted} new, {n_skipped} cached, {n_missing} missing-png "
          f"-> {DATASET_DIR}")
    total = len(list(DATASET_DIR.glob("*.png")))
    print(f"[dataset] {total} pngs total in {DATASET_DIR}")


if __name__ == "__main__":
    main()
