"""Stage 1 — 50-sample sanity check for demographic classifiers on Flux Krea.

Pipeline:
  1. Generate 768x1024 neutral anchor (once, cached).
  2. img2img that anchor through Flux Krea for each of the 50 sanity_grid rows
     at denoise=0.9 (aggressive enough to let ethnicity/age/gender come through).
  3. Classify each output with all three classifiers (MiVOLO, FairFace, InsightFace).
  4. Report: prompt-attribute vs classifier agreement; inter-classifier agreement.

Gate: if any classifier disagrees with prompt-attribute on >30% of samples AND
also disagrees with the other two, halt before Stage 2.

Usage:
    uv run python -m src.demographic_pc.stage1_sanity
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import Counter
from pathlib import Path

import cv2

from src.demographic_pc.classifiers import (
    FairFaceClassifier,
    InsightFaceClassifier,
    MiVOLOClassifier,
    predict_all,
)
from src.demographic_pc.comfy_flux import (
    ComfyClient,
    flux_img2img_workflow,
    flux_txt2img_workflow,
)
from src.demographic_pc.prompts import sanity_grid

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "demographic_pc" / "stage1"
ANCHOR_PATH = OUT_DIR / "anchor_768x1024.png"
SAMPLES_DIR = OUT_DIR / "samples"

ANCHOR_PROMPT = (
    "A photorealistic portrait photograph of a person, neutral expression, "
    "plain grey background, studio lighting, sharp focus."
)
ANCHOR_SEED = 42
DENOISE = 0.90  # aggressive — we want prompt to drive demographics


# ── Prompt→prediction mapping ────────────────────────────────────────────────

PROMPT_ETH_TO_FAIRFACE = {
    "East Asian": "East Asian",
    "Southeast Asian": "Southeast Asian",
    "South Asian": "Indian",
    "Black": "Black",
    "White": "White",
    "Hispanic or Latino": "Latino_Hispanic",
    "Middle Eastern": "Middle Eastern",
}

PROMPT_AGE_MIDPOINT = {  # rough center of prompt age term
    "child": 9, "young adult": 24, "adult": 35,
    "middle-aged": 50, "elderly": 70,
}


def prompt_gender_to_MF(g: str) -> str | None:
    if g == "man":
        return "M"
    if g == "woman":
        return "F"
    return None  # non-binary has no binary ground truth


# ── Generation ────────────────────────────────────────────────────────────────

async def ensure_anchor(client: ComfyClient) -> str:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ANCHOR_PATH.exists():
        print(f"[anchor] generating {ANCHOR_PATH.name} ...")
        wf = flux_txt2img_workflow(
            positive=ANCHOR_PROMPT, seed=ANCHOR_SEED,
            width=768, height=1024, prefix="demo_pc_anchor",
        )
        await client.generate(wf, ANCHOR_PATH)
    else:
        print(f"[anchor] reuse {ANCHOR_PATH.name}")
    uploaded = await client.upload_image(ANCHOR_PATH)
    print(f"[anchor] uploaded to ComfyUI as {uploaded}")
    return uploaded


async def generate_samples(client: ComfyClient, anchor_name: str) -> list[Path]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = sanity_grid()
    paths: list[Path] = []
    t0 = time.time()
    for i, row in enumerate(rows, 1):
        dest = SAMPLES_DIR / f"{row.sample_id}.png"
        if dest.exists():
            paths.append(dest)
            continue
        wf = flux_img2img_workflow(
            image_name=anchor_name, positive=row.prompt, seed=row.seed,
            denoise=DENOISE, prefix=f"demo_pc_s1_{row.sample_id}",
        )
        await client.generate(wf, dest)
        paths.append(dest)
        dt = time.time() - t0
        eta = dt / i * (len(rows) - i)
        print(f"  [{i:2d}/{len(rows)}] {row.sample_id:8s} {row.age}/{row.gender}/{row.ethnicity}  dt={dt:.0f}s eta={eta:.0f}s")
    return paths


# ── Classification ────────────────────────────────────────────────────────────

def classify_all(paths: list[Path]) -> list[dict]:
    print("\n[classify] loading 3 classifiers...")
    mv = MiVOLOClassifier()
    ff = FairFaceClassifier()
    ins = InsightFaceClassifier()

    rows = sanity_grid()
    records = []
    for row, p in zip(rows, paths):
        bgr = cv2.imread(str(p))
        r = predict_all(bgr, mv, ff, ins)
        records.append({
            "sample_id": row.sample_id,
            "prompt": row.prompt,
            "prompt_age": row.age, "prompt_gender": row.gender, "prompt_ethnicity": row.ethnicity,
            "mivolo_age": r.mivolo_age, "mivolo_gender": r.mivolo_gender,
            "mivolo_gender_conf": r.mivolo_gender_conf,
            "fairface_age_bin": r.fairface_age_bin, "fairface_gender": r.fairface_gender,
            "fairface_race": r.fairface_race, "fairface_detected": r.fairface_detected,
            "insightface_age": r.insightface_age, "insightface_gender": r.insightface_gender,
            "insightface_detected": r.insightface_detected,
        })
        print(
            f"  {row.sample_id:8s}  "
            f"prompt={row.age[:4]:4s}/{row.gender[:1]}/{row.ethnicity[:4]:4s}  "
            f"M={r.mivolo_age:5.1f}{r.mivolo_gender}  "
            f"F={str(r.fairface_age_bin or 'NA'):>6s}{r.fairface_gender or '?'} {(r.fairface_race or 'NA')[:8]:8s}  "
            f"I={str(round(r.insightface_age) if r.insightface_age else 'NA'):>4s}{r.insightface_gender or '?'}"
        )
    return records


# ── Diagnostics ──────────────────────────────────────────────────────────────

def age_bin_to_mid(b: str | None) -> float | None:
    if not b:
        return None
    m = {
        "0-2": 1, "3-9": 6, "10-19": 15, "20-29": 25,
        "30-39": 35, "40-49": 45, "50-59": 55, "60-69": 65, "70+": 75,
    }
    return m.get(b)


def age_within(pred: float | None, prompt_mid: float, tol: float = 12.0) -> bool:
    return pred is not None and abs(pred - prompt_mid) <= tol


def report(records: list[dict]) -> dict:
    print("\n" + "=" * 78)
    print("STAGE 1 SANITY REPORT")
    print("=" * 78)

    n = len(records)
    # Detection coverage
    ff_det = sum(1 for r in records if r["fairface_detected"])
    ins_det = sum(1 for r in records if r["insightface_detected"])
    print(f"\n[coverage]  N={n}  FairFace={ff_det}/{n}  InsightFace={ins_det}/{n}")

    # Gender agreement with prompt (skip non-binary)
    g_results = {"mivolo": [0, 0], "fairface": [0, 0], "insightface": [0, 0]}
    for r in records:
        gt = prompt_gender_to_MF(r["prompt_gender"])
        if gt is None:
            continue
        for k, pred in [("mivolo", r["mivolo_gender"]),
                        ("fairface", r["fairface_gender"]),
                        ("insightface", r["insightface_gender"])]:
            if pred is None:
                continue
            g_results[k][1] += 1
            if pred == gt:
                g_results[k][0] += 1
    print("\n[gender vs prompt]  (non-binary excluded)")
    for k, (ok, tot) in g_results.items():
        pct = 100 * ok / tot if tot else 0
        print(f"  {k:12s}  {ok}/{tot}  ({pct:5.1f}%)")

    # Age agreement (±12 years of prompt midpoint)
    a_results = {"mivolo": [0, 0], "fairface": [0, 0], "insightface": [0, 0]}
    for r in records:
        mid = PROMPT_AGE_MIDPOINT[r["prompt_age"]]
        for k, pred in [("mivolo", r["mivolo_age"]),
                        ("fairface", age_bin_to_mid(r["fairface_age_bin"])),
                        ("insightface", r["insightface_age"])]:
            if pred is None:
                continue
            a_results[k][1] += 1
            if age_within(pred, mid):
                a_results[k][0] += 1
    print("\n[age vs prompt]  (within ±12 years of prompt midpoint)")
    for k, (ok, tot) in a_results.items():
        pct = 100 * ok / tot if tot else 0
        print(f"  {k:12s}  {ok}/{tot}  ({pct:5.1f}%)")

    # Ethnicity (FairFace only)
    e_ok = e_tot = 0
    per_eth: dict[str, list[int]] = {}
    for r in records:
        if not r["fairface_detected"]:
            continue
        e_tot += 1
        want = PROMPT_ETH_TO_FAIRFACE[r["prompt_ethnicity"]]
        d = per_eth.setdefault(r["prompt_ethnicity"], [0, 0])
        d[1] += 1
        if r["fairface_race"] == want:
            e_ok += 1
            d[0] += 1
    pct = 100 * e_ok / e_tot if e_tot else 0
    print(f"\n[ethnicity vs prompt]  (FairFace only)  {e_ok}/{e_tot} ({pct:.1f}%)")
    for eth, (ok, tot) in sorted(per_eth.items()):
        print(f"  {eth:22s}  {ok}/{tot}")

    # Inter-classifier gender agreement
    inter = Counter()
    for r in records:
        mv, ff, ins = r["mivolo_gender"], r["fairface_gender"], r["insightface_gender"]
        if mv and ff and ins and mv == ff == ins:
            inter["all3"] += 1
        elif mv and ff and mv == ff:
            inter["mv=ff"] += 1
        elif mv and ins and mv == ins:
            inter["mv=ins"] += 1
        elif ff and ins and ff == ins:
            inter["ff=ins"] += 1
        else:
            inter["none"] += 1
    print(f"\n[inter-classifier gender agreement]  {dict(inter)}")

    # Gate
    print("\n" + "-" * 78)
    gate_ok = True
    for k in ("mivolo", "fairface", "insightface"):
        ok, tot = g_results[k]
        if tot and ok / tot < 0.70:
            print(f"  FLAG gender: {k} {100*ok/tot:.1f}% < 70%")
            gate_ok = False
    for k in ("mivolo", "fairface", "insightface"):
        ok, tot = a_results[k]
        if tot and ok / tot < 0.50:
            print(f"  FLAG age: {k} {100*ok/tot:.1f}% < 50%")
            gate_ok = False
    if pct < 50 and e_tot:
        print(f"  FLAG ethnicity: FairFace {pct:.1f}% < 50%")
        gate_ok = False
    print(f"\nGATE: {'✅ GO — proceed to Stage 2' if gate_ok else '❌ HALT — investigate'}")

    return {"n": n, "coverage": {"fairface": ff_det, "insightface": ins_det},
            "gender": g_results, "age": a_results, "ethnicity": {"overall": (e_ok, e_tot), "per": per_eth},
            "inter_gender": dict(inter), "gate_ok": gate_ok}


# ── Main ──────────────────────────────────────────────────────────────────────

async def run_generation() -> list[Path]:
    async with ComfyClient() as client:
        anchor_name = await ensure_anchor(client)
        paths = await generate_samples(client, anchor_name)
    return paths


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = asyncio.run(run_generation())
    records = classify_all(paths)
    summary = report(records)

    out_json = OUT_DIR / "sanity_check_50.json"
    with open(out_json, "w") as f:
        json.dump({"records": records, "summary": summary}, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
