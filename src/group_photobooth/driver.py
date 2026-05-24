"""Group photobooth orchestrator.

CLI:
    uv run --no-project python -m group_photobooth.driver \\
        --photo path/to/group.jpg --background blank_studio \\
        --out exp_output/group_photobooth/run01/

Pipeline (Approach A):
    1. detect.detect_people(photo) → list[Person]
    2. for each person: face_renderer.render_doll(face_crop) → portrait
    3. for each portrait: silhouette.cutout(portrait) → RGBA
    4. layout.solve_placements(persons, photo_hw, bg_hw) → list[Placement]
    5. composite.composite(bg, dolls_rgba, placements)
    6. optional polish: lab_match per doll, drop_shadow per placement

All intermediates cache to <out_dir>/people/<i>/ so reruns skip work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from group_photobooth import Person
from group_photobooth.backgrounds import load_background
from group_photobooth.composite import composite, drop_shadow, lab_match
from group_photobooth.detect import detect_people
from group_photobooth.face_renderer import render_doll
from group_photobooth.layout import solve_placements
from group_photobooth.silhouette import cutout

_DEFAULT_DEMO = {"gender": "person", "age_bin": "30s", "race": ""}


def _seed_for_face(photo_id: str, face_index: int) -> int:
    h = hashlib.md5(f"{photo_id}__{face_index}".encode()).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _crop_face(photo_bgr: np.ndarray, person: Person,
               margin_frac: float = 0.35) -> np.ndarray:
    assert person.face_bbox is not None
    x1, y1, x2, y2 = person.face_bbox
    fw, fh = x2 - x1, y2 - y1
    m = int(round(max(fw, fh) * margin_frac))
    H, W = photo_bgr.shape[:2]
    cx1 = max(0, x1 - m); cy1 = max(0, y1 - m)
    cx2 = min(W, x2 + m); cy2 = min(H, y2 + m)
    return photo_bgr[cy1:cy2, cx1:cx2].copy()


def run(photo_path: Path, background_id: str, out_dir: Path,
        comfy_url: str, demo: dict[str, str] | None = None,
        polish: bool = False) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    people_dir = out_dir / "people"
    people_dir.mkdir(exist_ok=True)

    photo = cv2.imread(str(photo_path))
    if photo is None:
        raise FileNotFoundError(f"cannot read photo: {photo_path}")
    persons = detect_people(photo, comfy_url=comfy_url)
    print(f"[group_photobooth] detected {len(persons)} person(s)")
    if not persons:
        bg, _meta = load_background(background_id)
        out_path = out_dir / "result.png"
        cv2.imwrite(str(out_path), bg)
        return out_path

    photo_id = photo_path.stem
    demo = demo or _DEFAULT_DEMO

    dolls_rgba: list[np.ndarray] = []
    for i, person in enumerate(persons):
        pdir = people_dir / f"{i:02d}"
        pdir.mkdir(exist_ok=True)
        face_p = pdir / "face.png"
        doll_p = pdir / "doll.png"
        rgba_p = pdir / "doll_rgba.png"

        if not face_p.exists():
            cv2.imwrite(str(face_p), _crop_face(photo, person))
        if not doll_p.exists():
            face_crop = cv2.imread(str(face_p))
            seed = _seed_for_face(photo_id, i)
            doll = render_doll(face_crop, comfy_url, seed=seed, demo=demo)
            cv2.imwrite(str(doll_p), doll)
        if not rgba_p.exists():
            doll = cv2.imread(str(doll_p))
            rgba = cutout(doll, comfy_url=comfy_url)
            cv2.imwrite(str(rgba_p), rgba)
        dolls_rgba.append(cv2.imread(str(rgba_p), cv2.IMREAD_UNCHANGED))

    bg, meta = load_background(background_id)
    placements = solve_placements(persons, photo.shape[:2], bg.shape[:2])

    if polish:
        target_lab = tuple(meta["palette_lab_mean"])
        for k, doll in enumerate(dolls_rgba):
            alpha = doll[..., 3]
            doll[..., :3] = lab_match(doll[..., :3], alpha, target_lab)
        canvas = bg.copy()
        for k, pl in enumerate(placements):
            doll = dolls_rgba[k]
            doll_w = max(1, int(round(doll.shape[1] * pl.height / doll.shape[0])))
            canvas = drop_shadow(canvas, pl, doll_w=doll_w)
        result = composite(canvas, dolls_rgba, placements)
    else:
        result = composite(bg, dolls_rgba, placements)

    out_path = out_dir / "result.png"
    cv2.imwrite(str(out_path), result)
    print(f"[group_photobooth] wrote {out_path}")
    return out_path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--photo", type=Path, required=True)
    p.add_argument("--background", type=str, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="output directory (intermediates + result.png)")
    p.add_argument("--comfy-url", type=str, default="http://127.0.0.1:8188")
    p.add_argument("--demo-json", type=str, default=None,
                   help="JSON with gender/age_bin/race for prompt injection")
    p.add_argument("--polish", action="store_true",
                   help="enable Lab match + drop shadow polish")
    args = p.parse_args()
    demo = json.loads(args.demo_json) if args.demo_json else None
    run(args.photo, args.background, args.out, args.comfy_url,
        demo=demo, polish=args.polish)


if __name__ == "__main__":
    main()
