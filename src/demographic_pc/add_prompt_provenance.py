"""Add prompt-provenance columns to sample_index.parquet.

Scope: overnight_* sources (smile/beard/beard_rebalance/rebalance) —
1,536 rows. Pre-overnight sources stay NaN; those experiments are
analyzed at the FluxSpace-parameter level and prompt text is not
load-bearing for downstream effect-matrix work.

Columns added:
    prompt_base            full base prompt string
    prompt_edit_pos        positive edit prompt ("edit_a")
    prompt_edit_neg        negative edit prompt ("edit_b" — demo+clause splice)
    prompt_intent_age      raw age word ("adult", "elderly", "young")
    prompt_intent_ethnicity  raw ethnicity word from prompt
    prompt_intent_gender   raw gender word ("man"/"woman")
    prompt_extras          non-demographic base modifiers ("bearded" or None)

Source of truth is overnight_render_batch.py constants (BASES_ALL,
BEARD_REBAL_BASES, SMILE_LADDER, BEARD_ADD/REMOVE, REBALANCE_AXES,
_splice). We mirror those here rather than importing, so this script
doesn't re-trigger a ComfyClient import.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "models/blendshape_nmf/sample_index.parquet"

# ── base prompts & their intent parse ────────────────────────────────────────

BASE_PROMPTS: dict[str, str] = {
    "asian_m":
        "A photorealistic portrait photograph of an adult East Asian man, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "black_f":
        "A photorealistic portrait photograph of an adult Black woman, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "european_m":
        "A photorealistic portrait photograph of an adult European man, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "elderly_latin_m":
        "A photorealistic portrait photograph of an elderly Latin American man, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "young_european_f":
        "A photorealistic portrait photograph of a young European woman, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "southasian_f":
        "A photorealistic portrait photograph of an adult South Asian woman, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "asian_m_bearded":
        "A photorealistic portrait photograph of an adult East Asian man with a thick full beard, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "european_m_bearded":
        "A photorealistic portrait photograph of an adult European man with a thick full beard, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
    "elderly_latin_m_bearded":
        "A photorealistic portrait photograph of an elderly Latin American man with a thick full beard, "
        "neutral expression, plain grey background, studio lighting, sharp focus.",
}

# (age_word, ethnicity_word, gender_word, extras)
BASE_INTENT: dict[str, tuple[str, str, str, str | None]] = {
    "asian_m":                 ("adult",   "East Asian",        "man",   None),
    "black_f":                 ("adult",   "Black",             "woman", None),
    "european_m":              ("adult",   "European",          "man",   None),
    "elderly_latin_m":         ("elderly", "Latin American",    "man",   None),
    "young_european_f":        ("young",   "European",          "woman", None),
    "southasian_f":            ("adult",   "South Asian",       "woman", None),
    "asian_m_bearded":         ("adult",   "East Asian",        "man",   "bearded"),
    "european_m_bearded":      ("adult",   "European",          "man",   "bearded"),
    "elderly_latin_m_bearded": ("elderly", "Latin American",    "man",   "bearded"),
}

# demographic clause used by _splice — part before the edit clause.
BASE_DEMO_CLAUSE: dict[str, str] = {
    "asian_m":                 "an adult East Asian man",
    "black_f":                 "an adult Black woman",
    "european_m":              "an adult European man",
    "elderly_latin_m":         "an elderly Latin American man",
    "young_european_f":        "a young European woman",
    "southasian_f":            "an adult South Asian woman",
    "asian_m_bearded":         "an adult East Asian man with a thick full beard",
    "european_m_bearded":      "an adult European man with a thick full beard",
    "elderly_latin_m_bearded": "an elderly Latin American man with a thick full beard",
}


# ── edit prompts (edit_a) by (axis, subtag) ──────────────────────────────────

SMILE_EDIT_A: dict[str, tuple[str, str]] = {
    "faint": ("A person with a faint closed-mouth smile.",
              "with a faint closed-mouth smile"),
    "warm":  ("A person smiling warmly.",
              "smiling warmly"),
    "broad": ("A person grinning broadly with teeth showing.",
              "grinning broadly with teeth showing"),
    "manic": ("A person with a manic wide-open grin, teeth bared.",
              "with a manic wide-open grin, teeth bared"),
}

BEARD_EDIT_A = {
    "add":    ("A man with a thick full beard.",    "with a thick full beard"),
    "remove": ("A clean-shaven man, no beard.",     "clean-shaven, no facial hair"),
}

REBAL_EDIT_A: dict[str, tuple[str, str]] = {
    "anger":    ("A person with angry brows lowered and jaw tensed.",
                 "with angry brows lowered and jaw tensed"),
    "surprise": ("A person with raised brows and mouth agape.",
                 "with raised brows and mouth agape"),
    "pucker":   ("A person with lips puckered forward.",
                 "with lips puckered forward"),
}


def _splice(demo: str, clause: str) -> str:
    return (f"A photorealistic portrait photograph of {demo} {clause}, "
            f"plain grey background, studio lighting, sharp focus.")


def _resolve(row: pd.Series) -> dict:
    """Return prompt-provenance dict for an overnight_* row, else empty dict."""
    src = row["source"]
    base = row["base"]
    subtag = row["subtag"]

    if base not in BASE_PROMPTS:
        return {}
    base_prompt = BASE_PROMPTS[base]
    age, eth, gen, extras = BASE_INTENT[base]
    demo = BASE_DEMO_CLAUSE[base]

    edit_pos = None
    clause = None
    if src == "overnight_smile" and subtag in SMILE_EDIT_A:
        edit_pos, clause = SMILE_EDIT_A[subtag]
    elif src == "overnight_beard" and subtag in BEARD_EDIT_A:
        edit_pos, clause = BEARD_EDIT_A[subtag]
    elif src == "overnight_beard_rebalance" and subtag == "remove":
        edit_pos, clause = BEARD_EDIT_A["remove"]
    elif src == "overnight_rebalance" and subtag in REBAL_EDIT_A:
        edit_pos, clause = REBAL_EDIT_A[subtag]

    edit_neg = _splice(demo, clause) if clause else None

    return {
        "prompt_base": base_prompt,
        "prompt_edit_pos": edit_pos,
        "prompt_edit_neg": edit_neg,
        "prompt_intent_age": age,
        "prompt_intent_ethnicity": eth,
        "prompt_intent_gender": gen,
        "prompt_extras": extras,
    }


def main() -> None:
    idx = pd.read_parquet(INDEX)
    cols = ["prompt_base", "prompt_edit_pos", "prompt_edit_neg",
            "prompt_intent_age", "prompt_intent_ethnicity", "prompt_intent_gender",
            "prompt_extras"]
    for c in cols:
        idx[c] = None

    n_filled = 0
    for i, row in idx.iterrows():
        res = _resolve(row)
        if res:
            for k, v in res.items():
                idx.at[i, k] = v
            n_filled += 1

    idx.to_parquet(INDEX, index=False, compression="zstd")
    print(f"[save] → {INDEX}  prompt-rows filled={n_filled}/{len(idx)} "
          f"({100 * n_filled / len(idx):.1f}%)")
    print(f"[stats] filled by source:")
    print(idx[idx["prompt_base"].notna()].groupby("source").size().to_string())
    # sanity: sample a row with edit prompt
    sample = idx[idx["prompt_edit_pos"].notna()].sample(1, random_state=0).iloc[0]
    print(f"\n[sample] {sample['source']}/{sample['subtag']}/{sample['base']}")
    print(f"  base:      {sample['prompt_base']}")
    print(f"  edit_pos:  {sample['prompt_edit_pos']}")
    print(f"  edit_neg:  {sample['prompt_edit_neg']}")
    print(f"  intent:    age={sample['prompt_intent_age']!r}  eth={sample['prompt_intent_ethnicity']!r}  "
          f"gen={sample['prompt_intent_gender']!r}  extras={sample['prompt_extras']!r}")


if __name__ == "__main__":
    main()
