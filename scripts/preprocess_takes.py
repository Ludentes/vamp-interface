"""Preprocess Live Link Face takes: smaller full-take reencodes + auto-cut clips.

The raw `data/llf-takes/20260505_MySlate_*/MySlate_*_iPhone.mov` files are
1.4–3.2 GB each (1440x1080 @ 60 fps prores-y). We rarely need full resolution.
This script produces two derivative trees that drop into anything taking a
`--take_dir` (the apply_bridge_to_personalive contract: a directory containing
`*_iPhone.csv` + `*_iPhone.{mov,mp4}`):

  data/llf-takes-small/<take>/
      MySlate_<N>_iPhone.mp4   h264 720p crf 23 (~50 MB)
      MySlate_<N>_iPhone.csv   verbatim copy
      take.json                verbatim copy
      preprocess.json          provenance

  data/llf-clips-auto/<take>_<axis>/   axis ∈ {yaw, pitch, expr}
      MySlate_<N>_iPhone.mp4   600-frame (10 s @ 60 fps) slice
      MySlate_<N>_iPhone.csv   matching CSV row slice
      clip.json                src_take, src_start_frame, src_end_frame, criterion, peak_value

Clip selection: load the (N, 61) ARKit array via `load_llf_b61`, run a sliding
1-frame-stride window of length `--window_frames` across each criterion, pick
the window with the highest score. Criteria:

  yaw   — mean |b[:, 52]|              (head yaw radians)
  pitch — mean |b[:, 53]|              (head pitch radians)
  expr  — mean Σ_i |b[:, i]|, i in 0..52  (total blendshape activity)

The clip CSV preserves the header row but slices data rows [start:end]; the
`Timecode` column becomes out of sync with absolute time, which is fine since
`load_llf_b61` only uses row index.

Idempotent: skip-if-exists per output, override with `--force`.

Usage:
  # everything for all 8 takes
  python scripts/preprocess_takes.py
  # just the reencodes
  python scripts/preprocess_takes.py --no_clips
  # just the clips (uses already-built small reencodes if present, else mov)
  python scripts/preprocess_takes.py --no_reencode
  # one take only
  python scripts/preprocess_takes.py --take 20260505_MySlate_5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from arkit_bridge.llf_csv import load_llf_b61  # noqa: E402


# Stage 1 — reencode -----------------------------------------------------------

def reencode_take(src_dir: Path, dst_dir: Path, *, height: int, crf: int, force: bool) -> None:
    """Build a downscaled mp4 + csv copy mirror under dst_dir."""
    mov = next(src_dir.glob("*_iPhone.mov"), None)
    csv = next(src_dir.glob("*_iPhone.csv"), None)
    if mov is None or csv is None:
        print(f"  skip {src_dir.name}: missing mov or csv"); return

    dst_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = dst_dir / f"{mov.stem}.mp4"
    out_csv = dst_dir / csv.name

    if out_mp4.exists() and not force:
        print(f"  [reencode] {src_dir.name}: mp4 exists, skip ({out_mp4.stat().st_size/1e6:.1f} MB)")
    else:
        # -vf scale=-2:H keeps the source aspect (1440x1080 → 960x720).
        # -fps_mode passthrough preserves the source's variable / capture cadence
        # so the row ↔ frame mapping in the CSV stays valid.
        # Write to .tmp then atomic replace so a killed run doesn't leave a
        # corrupt mp4 that the next idempotent run silently skips.
        tmp_mp4 = out_mp4.with_suffix(out_mp4.suffix + ".tmp")
        if tmp_mp4.exists():
            tmp_mp4.unlink()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", str(mov),
            "-vf", f"scale=-2:{height}",
            "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough",
            "-an",
            "-f", "mp4",  # explicit since .tmp suffix hides the extension  # audio not needed for our pipeline
            str(tmp_mp4),
        ]
        print(f"  [reencode] {src_dir.name} -> {out_mp4.name} ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            tmp_mp4.unlink(missing_ok=True)
            print(r.stderr[-1500:])
            raise RuntimeError(f"ffmpeg failed for {mov}")
        os.replace(tmp_mp4, out_mp4)
        print(f"            done ({out_mp4.stat().st_size/1e6:.1f} MB)")

    if not out_csv.exists() or force:
        # shutil.copy2 preserves bytes (incl. \r\n line endings) and mtime —
        # safer than text-mode round-trip for round-trippable provenance.
        shutil.copy2(csv, out_csv)
    take_json = src_dir / "take.json"
    if take_json.exists() and (not (dst_dir / "take.json").exists() or force):
        shutil.copy2(take_json, dst_dir / "take.json")

    (dst_dir / "preprocess.json").write_text(json.dumps({
        "src_dir": str(src_dir),
        "src_mov": str(mov),
        "height": height, "crf": crf,
    }, indent=2))


# Stage 2 — clip selection -----------------------------------------------------

def sliding_window_argmax(scores_per_frame: np.ndarray, window: int) -> tuple[int, float]:
    """Return (start_idx, peak_mean_score) for the highest-mean window."""
    if len(scores_per_frame) < window:
        raise ValueError(f"only {len(scores_per_frame)} frames for window {window}")
    cs = np.cumsum(np.concatenate([[0.0], scores_per_frame.astype(np.float64)]))
    sums = cs[window:] - cs[:-window]      # (N - window + 1,)
    means = sums / window
    i = int(np.argmax(means))
    return i, float(means[i])


def cut_clip(*, src_mov: Path, src_csv: Path, dst_dir: Path, start: int, count: int,
             height: int, crf: int, force: bool, sidecar: dict) -> None:
    """Slice [start, start+count) frames from src_mov + matching CSV rows."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = dst_dir / f"{src_mov.stem}.mp4"
    out_csv = dst_dir / src_csv.name

    do_mp4 = force or not out_mp4.exists()
    if not do_mp4:
        print(f"  [clip] {dst_dir.name}: mp4 exists, skip")
    else:
        # `trim=start_frame:end_frame` indexes by *capture-order* input frame
        # number after decoding, which is robust against B-frame reorder
        # vs `select=between(n,...)` which counts post-filter output frames.
        # Stage-1 reencodes are CBR keyframe-friendly so either works on smalls;
        # we use `trim` for safety so this also works on raw mov fallback.
        # `setpts=PTS-STARTPTS` rebases timestamps to 0 for the output.
        tmp_mp4 = out_mp4.with_suffix(out_mp4.suffix + ".tmp")
        if tmp_mp4.exists():
            tmp_mp4.unlink()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
            "-i", str(src_mov),
            "-vf", (f"trim=start_frame={start}:end_frame={start + count},"
                    f"setpts=PTS-STARTPTS,scale=-2:{height}"),
            "-frames:v", str(count),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            "-fps_mode", "passthrough",
            "-an",
            "-f", "mp4",  # explicit since .tmp suffix hides the extension
            str(tmp_mp4),
        ]
        print(f"  [clip] {dst_dir.name} <- frames [{start}:{start+count}] ...", flush=True)
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            tmp_mp4.unlink(missing_ok=True)
            print(r.stderr[-1500:])
            raise RuntimeError(f"ffmpeg failed for clip {dst_dir}")
        os.replace(tmp_mp4, out_mp4)

    # CSV row slice: header + rows[start:start+count]. Rebuild whenever we
    # rebuild the mp4 so the two stay in sync; preserve LLF's \r\n line
    # endings via binary I/O. Atomic replace.
    if do_mp4 or not out_csv.exists():
        with open(src_csv, "rb") as f:
            data = f.read()
        # Robust to \r\n / \n; keepends preserves whatever the source used.
        lines = data.splitlines(keepends=True)
        header, rows = lines[0], lines[1:]
        sl = rows[start:start + count]
        tmp_csv = out_csv.with_suffix(out_csv.suffix + ".tmp")
        with open(tmp_csv, "wb") as f:
            f.write(header)
            f.writelines(sl)
        os.replace(tmp_csv, out_csv)

    (dst_dir / "clip.json").write_text(json.dumps(sidecar, indent=2))


def select_and_cut_clips(src_dir: Path, dst_root: Path, *, window_frames: int,
                         use_small: Path | None, height: int, crf: int, force: bool) -> None:
    csv = next(src_dir.glob("*_iPhone.csv"), None)
    if csv is None:
        print(f"  skip {src_dir.name}: missing csv"); return

    # Source video for clipping: prefer the already-reencoded small mp4 — it
    # is CBR-keyframe-friendly so trim=start_frame:end_frame is well-behaved.
    # Falling back to the raw mov also uses trim (safer than select=between)
    # but is slower and exposes us to source B-frame reorder; emit a warning.
    small_mp4 = (use_small / f"{csv.stem}.mp4") if use_small else None
    if small_mp4 and small_mp4.exists():
        src_mov = small_mp4
    else:
        src_mov = next(src_dir.glob("*_iPhone.mov"), None)
        if src_mov is None:
            print(f"  skip {src_dir.name}: no mov and no small mp4"); return
        print(f"  [clip] {src_dir.name}: WARNING falling back to raw mov "
              f"(no small at {small_mp4}); frame indexing unverified.")

    b_all = load_llf_b61(csv)  # (N, 61)
    n = len(b_all)
    try:
        src_disp = src_mov.relative_to(REPO_ROOT)
    except ValueError:
        src_disp = src_mov.name
    print(f"  [clip] {src_dir.name}: {n} frames, source={src_disp}")

    if n < window_frames:
        print(f"  [clip] {src_dir.name}: only {n} frames < window {window_frames}, skip"); return

    yaw   = np.abs(b_all[:, 52])
    pitch = np.abs(b_all[:, 53])
    expr  = b_all[:, :52].sum(axis=1)  # ARKit blendshapes are already nonneg
    criteria = {"yaw": yaw, "pitch": pitch, "expr": expr}

    for axis, scores in criteria.items():
        i, peak = sliding_window_argmax(scores, window_frames)
        clip_dir = dst_root / f"{src_dir.name}_{axis}"
        sidecar = {
            "src_take": src_dir.name,
            "src_mov":  str(src_mov),
            "src_csv":  str(csv),
            "src_start_frame": i,
            "src_end_frame":   i + window_frames,
            "criterion": axis,
            "peak_value": peak,
            "window_frames": window_frames,
        }
        cut_clip(src_mov=src_mov, src_csv=csv, dst_dir=clip_dir,
                 start=i, count=window_frames, height=height, crf=crf,
                 force=force, sidecar=sidecar)


# Driver -----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--takes_root", default="data/llf-takes",
                    help="parent dir holding 20260505_MySlate_<N> subdirs")
    ap.add_argument("--small_root", default="data/llf-takes-small")
    ap.add_argument("--clips_root", default="data/llf-clips-auto")
    ap.add_argument("--take", default=None,
                    help="restrict to a single take name (e.g. 20260505_MySlate_5)")
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--window_frames", type=int, default=600,
                    help="clip length in frames (10 s @ 60 fps)")
    ap.add_argument("--no_reencode", action="store_true")
    ap.add_argument("--no_clips", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        sys.exit("error: ffmpeg not found in PATH")

    takes_root = Path(args.takes_root)
    small_root = Path(args.small_root)
    clips_root = Path(args.clips_root)

    if args.take:
        take_dirs = [takes_root / args.take]
    else:
        take_dirs = sorted(p for p in takes_root.iterdir() if p.is_dir())

    print(f"preprocess: {len(take_dirs)} take(s) "
          f"reencode={not args.no_reencode} clips={not args.no_clips}")
    for td in take_dirs:
        print(f"\n# {td.name}")
        if not args.no_reencode:
            reencode_take(td, small_root / td.name, height=args.height,
                          crf=args.crf, force=args.force)
        if not args.no_clips:
            select_and_cut_clips(td, clips_root,
                                 window_frames=args.window_frames,
                                 use_small=small_root / td.name,
                                 height=args.height, crf=args.crf, force=args.force)


if __name__ == "__main__":
    main()
