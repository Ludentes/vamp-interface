"""Curate the ai_toolkit_glasses_v2 paired dataset for v5 slider training.

Pairs (with-glasses, without-glasses) for ai-toolkit's concept_slider.
Sources:
  - v3_1_glasses (main thin/horn variant): α=0 → reg, α=0.8 → train
  - v3_1_glasses_acetate (anti-zebra thick acetate): α=0.6 → train,
    paired with main α=0 reg (same base+seed; α=0 is no-edit baseline
    so it's invariant across pair-prompt sets)

Dropped pairs: identity_cos_to_base < 0.4 between α=0 anchor and the
positive sample (broken pairs where the base demographic shifted too
much during pair-edit).

Captions are demographic-conditioned, replacing v1's identical
"a person..." string. Trainer's slider:positive_prompt /
negative_prompt do the actual axis supervision; the dataset captions
just keep the LoRA conditioned on the demographic intent.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"
OUT_ROOT = ROOT / "datasets/ai_toolkit_glasses_v2"

DEMO_PHRASE = {
    "adult_latin_f":     "an adult Latin American woman",
    "adult_asian_m":     "an adult East Asian man",
    "adult_black_f":     "an adult Black woman",
    "adult_european_m":  "an adult European man",
    "adult_middle_f":    "an adult Middle Eastern woman",
    "adult_southasian_f":"an adult South Asian woman",
    "elderly_latin_m":   "an elderly Latin American man",
    "elderly_asian_f":   "an elderly East Asian woman",
    "young_black_m":     "a young Black man",
    "young_european_f":  "a young European woman",
}

POS_TEMPLATE = ("a photorealistic portrait photograph of {phrase} wearing "
                "eyeglasses, plain grey background, studio lighting")
NEG_TEMPLATE = ("a photorealistic portrait photograph of {phrase}, "
                "plain grey background, studio lighting")

IDENTITY_FLOOR = 0.4


def main() -> None:
    df = pd.read_parquet(SAMPLE_INDEX)
    train_dir = OUT_ROOT / "train"
    reg_dir = OUT_ROOT / "reg"
    train_dir.mkdir(parents=True, exist_ok=True)
    reg_dir.mkdir(parents=True, exist_ok=True)

    # α=0 anchors from main glasses (same images function as negatives for
    # both main and acetate positives — α=0 is the no-edit base render).
    main = df[df.source == "v3_1_glasses"].copy()
    anchors = main[main.alpha == 0.0].set_index(["base", "seed"])
    pos_main = main[main.alpha == 0.8]
    pos_acetate = df[df.source == "v3_1_glasses_acetate"]
    pos_acetate = pos_acetate[pos_acetate.alpha == 0.6]

    pair_idx = 0
    n_dropped = 0

    def emit(positive_row, label: str) -> None:
        nonlocal pair_idx, n_dropped
        base = positive_row["base"]
        seed = positive_row["seed"]
        if (base, seed) not in anchors.index:
            return
        anchor_row = anchors.loc[(base, seed)]
        if isinstance(anchor_row, pd.DataFrame):
            anchor_row = anchor_row.iloc[0]
        if positive_row["identity_cos_to_base"] < IDENTITY_FLOOR:
            n_dropped += 1
            print(f"  [drop] {label} {base} s={seed} "
                  f"id_cos={positive_row['identity_cos_to_base']:.2f}")
            return
        if base not in DEMO_PHRASE:
            return
        phrase = DEMO_PHRASE[base]
        stem = f"pair_{pair_idx:03d}_{label}_{base}_s{seed}"
        train_png = train_dir / f"{stem}.png"
        reg_png = reg_dir / f"{stem}.png"
        train_txt = train_dir / f"{stem}.txt"
        reg_txt = reg_dir / f"{stem}.txt"
        shutil.copy(ROOT / positive_row["img_path"], train_png)
        shutil.copy(ROOT / anchor_row["img_path"], reg_png)
        train_txt.write_text(POS_TEMPLATE.format(phrase=phrase))
        reg_txt.write_text(NEG_TEMPLATE.format(phrase=phrase))
        pair_idx += 1

    print(f"[pairs] main candidates:    {len(pos_main)}")
    for _, r in pos_main.iterrows():
        emit(r, "main")
    print(f"[pairs] acetate candidates: {len(pos_acetate)}")
    for _, r in pos_acetate.iterrows():
        emit(r, "acetate")

    print(f"\n[done] {pair_idx} pairs written  ({n_dropped} dropped on identity floor)")
    print(f"  → {train_dir} ({len(list(train_dir.glob('*.png')))} pngs)")
    print(f"  → {reg_dir} ({len(list(reg_dir.glob('*.png')))} pngs)")


if __name__ == "__main__":
    main()
