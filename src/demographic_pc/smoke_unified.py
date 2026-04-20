"""Smoke test for the unified classifiers module. Runs all 3 on the same
Flux portraits and prints a comparison row per image."""

from __future__ import annotations

from pathlib import Path

import cv2

from src.demographic_pc.classifiers import (
    FairFaceClassifier,
    InsightFaceClassifier,
    MiVOLOClassifier,
    predict_all,
)

ROOT = Path(__file__).resolve().parents[2]


def list_samples() -> list[Path]:
    out = [ROOT / "output" / "phase1" / "phase1_anchor.png"]
    for d in ["courier_legit", "office_legit", "scam_critical"]:
        p = ROOT / "output" / "phase1" / d
        if p.is_dir():
            imgs = sorted(p.glob("*.png"))
            if imgs:
                out.append(imgs[0])
    return out


def main() -> None:
    mv = MiVOLOClassifier()
    ff = FairFaceClassifier()
    ins = InsightFaceClassifier()
    for p in list_samples():
        if not p.exists():
            continue
        bgr = cv2.imread(str(p))
        r = predict_all(bgr, mv, ff, ins)
        print(
            f"{p.name:42s}  "
            f"M.age={r.mivolo_age:5.1f} {r.mivolo_gender}  "
            f"F.age={r.fairface_age_bin:>6s} {r.fairface_gender} race={r.fairface_race:<17s}  "
            f"I.age={r.insightface_age:5.1f} {r.insightface_gender}  "
            f"FF_det={r.fairface_detected}"
        )


if __name__ == "__main__":
    main()
