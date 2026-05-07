"""Build render_metrics.parquet from a registry of mp4s.

Each registry entry describes one mp4 (or one pane of a side-by-side mp4).
We extract, per frame:
  - mediapipe blendshapes (51-d) and ypr (3-d) on the chosen pane
  - region pixel diffs against a matching teacher_full mp4 if available
    (computed only for `mode=bridge` rows; teacher is the reference)

Output columns:
  run_tag, take, take_name, source_frame_idx, mp4_frame_idx,
  mode, preprocessing, pane, anchor_stem, mp4_path,
  mp_blendshapes (list<f32>[51]), ypr (list<f32>[3]),
  abs_diff_global, region_brow, region_eye_l, region_eye_r, region_mouth

Usage:
  python scripts/build_render_metrics_parquet.py --registry default
  python scripts/build_render_metrics_parquet.py --only-rgb     # just personalive_rgb rows
  python scripts/build_render_metrics_parquet.py --only-bridge  # just bridge rows

Registry covers (when entries exist on disk):
  - data/llf-take-renders/take{1..8}__asian_m.mp4         (personalive_rgb, perframe, 2 panes)
  - data/llf-take-renders/abc/take{2,7}_{A,B,C}_*.mp4     (personalive_rgb, {perframe,nocrop,ema}, 2 panes)
  - exp_output/arkit_bridge/render/teacher_full_cache/*.mp4  (teacher_full, n/a, full)
  - exp_output/arkit_bridge/render/v3_full/*_bridge.mp4   (bridge v3_lam10, n/a, full)
  - exp_output/arkit_bridge/render/v3_compare/*.mp4       (bridge/teacher 60-frame v3_lam10_60f)
"""
from __future__ import annotations
import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _mp_blendshape import make_landmarker, extract_one  # noqa: E402

ANCHOR_STEM = "asian_m__06_neutral.midframe"

REGIONS = {
    "brow":   list(range(46, 56)) + list(range(70, 80)) + [55, 65, 285, 295,
                                                            336, 296, 334, 293,
                                                            300, 276, 283, 282],
    "eye_l":  [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
               160, 161, 246],
    "eye_r":  [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386,
               385, 384, 398],
    "mouth":  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324,
               318, 402, 317, 14, 87, 178, 88, 95, 78, 191, 80, 81, 82, 13,
               312, 311, 310, 415, 308],
}


@dataclass
class Entry:
    run_tag: str
    take: int
    mp4_path: Path
    mode: str            # teacher_full | bridge | personalive_rgb
    preprocessing: str   # perframe | nocrop | ema | n/a
    pane: str            # left | right | full
    stride: int = 1      # mp4 frame i -> source frame i*stride + start_frame
    start_frame: int = 0
    anchor_stem: str = ANCHOR_STEM
    teacher_ref_mp4: Optional[Path] = None  # for region diffs
    clip_id: Optional[str] = None  # e.g. "20260505_MySlate_5_yaw" for auto-cut clips


def take_name(take: int) -> str: return f"20260505_MySlate_{take}"


def default_registry(root: Path) -> list[Entry]:
    """Enumerate everything currently on disk."""
    out: list[Entry] = []

    # personalive_rgb full takes (side-by-side 1024×512 → 2 panes)
    for t in range(1, 9):
        p = root / f"data/llf-take-renders/take{t}__asian_m.mp4"
        if not p.exists(): continue
        for pane in ("left", "right"):
            out.append(Entry("personalive_rgb_full", t, p, "personalive_rgb",
                             "perframe", pane))

    # personalive_rgb abc 60s slices
    abc = {"A": "perframe", "B": "nocrop", "C": "ema"}
    for t in (2, 7):
        for letter, prep in abc.items():
            for fname in (root / "data/llf-take-renders/abc").glob(
                f"take{t}_{letter}_*.mp4"
            ):
                for pane in ("left", "right"):
                    out.append(Entry("personalive_rgb_abc", t, fname,
                                     "personalive_rgb", prep, pane))

    # teacher_full cache (single-pane 512×512, ARKit-driven n=1200 stride=2)
    cache = root / "exp_output/arkit_bridge/render/teacher_full_cache"
    if cache.exists():
        for p in sorted(cache.glob(f"{take_name(1)[:-1]}*__{ANCHOR_STEM}__n*.mp4")):
            m = re.search(r"_(\d+)__", p.name)
            if not m: continue
            t = int(m.group(1))
            sm = re.search(r"n(\d+)_s(\d+)_o(\d+)\.mp4$", p.name)
            stride = int(sm.group(2)) if sm else 1
            start  = int(sm.group(3)) if sm else 0
            out.append(Entry("teacher_full_cache", t, p, "teacher_full",
                             "n/a", "full", stride=stride, start_frame=start))

    # bridge v3_full (1200-frame student, stride=2)
    v3 = root / "exp_output/arkit_bridge/render/v3_full"
    if v3.exists():
        for p in sorted(v3.glob(f"{take_name(1)[:-1]}*_bridge.mp4")):
            m = re.search(r"_(\d+)_bridge\.mp4$", p.name)
            if not m: continue
            t = int(m.group(1))
            ref = (cache / f"{take_name(t)}__{ANCHOR_STEM}__n1200_s2_o0.mp4")
            out.append(Entry("v3_lam10", t, p, "bridge", "n/a", "full",
                             stride=2, start_frame=0,
                             teacher_ref_mp4=ref if ref.exists() else None))

    # auto-discover all `<tag>_full/<take>_bridge.mp4` directories under
    # exp_output/arkit_bridge/render/. Catches v4a_full, v4b_full, v2_120k_full,
    # any future bake-off tag using the same convention. The corresponding
    # teacher_full_cache mp4 is reused as the reference.
    render_root = root / "exp_output/arkit_bridge/render"
    if render_root.exists():
        for tag_dir in sorted(render_root.iterdir()):
            if not tag_dir.is_dir(): continue
            tag = tag_dir.name
            # skip ones we already enumerated above
            if tag in {"v3_full", "v3_compare", "teacher_full_cache",
                       "anchor_pool_teacher_full", "stylized_pool_teacher_full",
                       "teacher_full", "teacher_full_cache",
                       "teacher_rot", "smoke", "smoke_v2", "full",
                       "bridge_v2_fixed", "v3_full_yawfix", "v3_smoke"}:
                continue
            if not tag.endswith("_full"):
                continue
            for p in sorted(tag_dir.glob(f"{take_name(1)[:-1]}*_bridge.mp4")):
                m = re.search(r"_(\d+)_bridge\.mp4$", p.name)
                if not m: continue
                t = int(m.group(1))
                ref = (cache /
                       f"{take_name(t)}__{ANCHOR_STEM}__n1200_s2_o0.mp4")
                run_tag = tag[:-5] if tag.endswith("_full") else tag
                out.append(Entry(run_tag, t, p, "bridge", "n/a", "full",
                                 stride=2, start_frame=0,
                                 teacher_ref_mp4=ref if ref.exists() else None))

    # 60-frame compare set (v3_compare/) — superseded by v3_full but worth indexing
    v3c = root / "exp_output/arkit_bridge/render/v3_compare"
    if v3c.exists():
        for p in sorted(v3c.glob(f"{take_name(1)[:-1]}*_bridge.mp4")):
            m = re.search(r"_(\d+)_bridge\.mp4$", p.name)
            if not m: continue
            t = int(m.group(1))
            tref = v3c / f"{take_name(t)}_teacher_full.mp4"
            out.append(Entry("v3_lam10_60f", t, p, "bridge", "n/a", "full",
                             stride=2, start_frame=0,
                             teacher_ref_mp4=tref if tref.exists() else None))
        for p in sorted(v3c.glob(f"{take_name(1)[:-1]}*_teacher_full.mp4")):
            m = re.search(r"_(\d+)_teacher_full\.mp4$", p.name)
            if not m: continue
            t = int(m.group(1))
            out.append(Entry("v3_lam10_60f", t, p, "teacher_full", "n/a", "full",
                             stride=2, start_frame=0))
    return out


def slice_pane(frame: np.ndarray, pane: str) -> np.ndarray:
    if pane == "full": return frame
    h, w = frame.shape[:2]
    half = w // 2
    return frame[:, :half] if pane == "left" else frame[:, half:]


def landmark_xy(face_mesh, rgb):
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks: return None
    lms = res.multi_face_landmarks[0].landmark
    h, w = rgb.shape[:2]
    return np.array([[lm.x * w, lm.y * h] for lm in lms], dtype=np.float32)


def region_box(landmarks, idxs, pad=8):
    pts = landmarks[idxs]
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    return int(max(0, x0 - pad)), int(max(0, y0 - pad)), int(x1 + pad), int(y1 + pad)


def process_entry(e: Entry, lm, fm, max_frames: int = 0) -> list[dict]:
    cap = cv2.VideoCapture(str(e.mp4_path))
    if not cap.isOpened():
        print(f"  ! cannot open {e.mp4_path}")
        return []

    teacher_cap = None
    if e.teacher_ref_mp4 is not None and e.mode == "bridge":
        teacher_cap = cv2.VideoCapture(str(e.teacher_ref_mp4))

    rows = []
    i = 0
    while True:
        ok, bgr = cap.read()
        if not ok: break
        if max_frames and i >= max_frames: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pane = slice_pane(rgb, e.pane)

        bs, ypr = extract_one(lm, pane)

        # Region diffs vs teacher (bridge rows only)
        abs_diff = np.nan
        regions = {r: np.nan for r in REGIONS}
        if teacher_cap is not None:
            ok_t, bgr_t = teacher_cap.read()
            if ok_t:
                rgb_t = cv2.cvtColor(bgr_t, cv2.COLOR_BGR2RGB)
                pane_t = slice_pane(rgb_t, "full")
                # match pane sizes (both full here)
                if pane.shape == pane_t.shape:
                    diff = np.abs(pane.astype(np.float32) - pane_t.astype(np.float32))
                    abs_diff = float(diff.mean())
                    lms = landmark_xy(fm, pane_t)
                    if lms is not None:
                        for r, idxs in REGIONS.items():
                            x0, y0, x1, y1 = region_box(lms, idxs)
                            if x1 > x0 and y1 > y0:
                                regions[r] = float(diff[y0:y1, x0:x1].mean())

        rows.append({
            "run_tag": e.run_tag,
            "take": e.take,
            "take_name": take_name(e.take),
            "source_frame_idx": i * e.stride + e.start_frame,
            "mp4_frame_idx": i,
            "mode": e.mode,
            "preprocessing": e.preprocessing,
            "pane": e.pane,
            "anchor_stem": e.anchor_stem,
            "clip_id": e.clip_id,
            "mp4_path": str(e.mp4_path.resolve()),
            "mp_blendshapes": bs.tolist(),
            "ypr": ypr.tolist(),
            "abs_diff_global": float(abs_diff) if not np.isnan(abs_diff) else None,
            "region_brow":  float(regions["brow"])  if not np.isnan(regions["brow"])  else None,
            "region_eye_l": float(regions["eye_l"]) if not np.isnan(regions["eye_l"]) else None,
            "region_eye_r": float(regions["eye_r"]) if not np.isnan(regions["eye_r"]) else None,
            "region_mouth": float(regions["mouth"]) if not np.isnan(regions["mouth"]) else None,
        })
        i += 1
    cap.release()
    if teacher_cap is not None: teacher_cap.release()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="exp_output/arkit_bridge/parquet/render_metrics.parquet")
    ap.add_argument("--max_frames_per_entry", type=int, default=0,
                    help="0 = all frames")
    ap.add_argument("--only_rgb", action="store_true")
    ap.add_argument("--only_bridge", action="store_true")
    ap.add_argument("--only_teacher", action="store_true")
    ap.add_argument("--takes", default=None,
                    help="comma list, e.g. 2,3,8 (default: all in registry)")
    ap.add_argument("--limit_entries", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    reg = default_registry(root)
    if args.only_rgb:     reg = [e for e in reg if e.mode == "personalive_rgb"]
    if args.only_bridge:  reg = [e for e in reg if e.mode == "bridge"]
    if args.only_teacher: reg = [e for e in reg if e.mode == "teacher_full"]
    if args.takes:
        keep = {int(t) for t in args.takes.split(",")}
        reg = [e for e in reg if e.take in keep]
    if args.limit_entries:
        reg = reg[: args.limit_entries]

    print(f"registry: {len(reg)} entries")
    for e in reg:
        print(f"  [{e.run_tag}] take={e.take} {e.mode}/{e.preprocessing}/{e.pane}  "
              f"stride={e.stride} start={e.start_frame}  -> {e.mp4_path.name}")

    lm = make_landmarker()
    fm = None
    if any(x.mode == "bridge" for x in reg):
        import mediapipe as mp_mod
        fm = mp_mod.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

    all_rows: list[dict] = []
    for k, e in enumerate(reg, 1):
        print(f"  [{k}/{len(reg)}] {e.mp4_path.name}  pane={e.pane}", flush=True)
        rows = process_entry(e, lm, fm, max_frames=args.max_frames_per_entry)
        print(f"    rows={len(rows)}  bs_nan={sum(1 for r in rows if any(x!=x for x in r['mp_blendshapes']))}")
        all_rows.extend(rows)

    schema = {
        "run_tag": pl.String,
        "take": pl.UInt8,
        "take_name": pl.String,
        "source_frame_idx": pl.UInt32,
        "mp4_frame_idx": pl.UInt32,
        "mode": pl.String,
        "preprocessing": pl.String,
        "pane": pl.String,
        "anchor_stem": pl.String,
        "clip_id": pl.String,
        "mp4_path": pl.String,
        "mp_blendshapes": pl.List(pl.Float32),
        "ypr": pl.List(pl.Float32),
        "abs_diff_global": pl.Float32,
        "region_brow": pl.Float32,
        "region_eye_l": pl.Float32,
        "region_eye_r": pl.Float32,
        "region_mouth": pl.Float32,
    }
    df = pl.DataFrame(all_rows, schema=schema).sort(
        ["run_tag", "take", "mode", "preprocessing", "pane", "mp4_frame_idx"]
    )
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out, compression="zstd")
    print(f"wrote {out}  rows={len(df)}  size={out.stat().st_size/1e6:.1f} MB")
    print(df.group_by(["run_tag", "mode", "preprocessing", "pane"]).len()
            .sort(["run_tag", "mode", "preprocessing", "pane"]))


if __name__ == "__main__":
    main()
