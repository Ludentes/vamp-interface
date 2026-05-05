"""Cut 6 driver clips from the ARKit take.

Reads data/llf-takes/20260505_MySlate_2/ (MOV + CSV @ 60 fps).
Writes data/llf-clips/<label>/clip.mp4 + clip.csv per slot.

CSV slice keeps the original column set so timecode + 52 blendshapes +
head/eye rotation stay aligned 1:1 with the MOV frames.
"""
import csv
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC_DIR = REPO / "data/llf-takes/20260505_MySlate_2"
SRC_MOV = SRC_DIR / "MySlate_2_iPhone.mov"
SRC_CSV = SRC_DIR / "MySlate_2_iPhone.csv"
OUT_DIR = REPO / "data/llf-clips"
FPS = 60
LEN = 5  # seconds

CLIPS = [
    ("01_smile",       3376),
    ("02_jaw_open",    5011),
    ("03_blink",       1453),
    ("04_head_turn",   2207),
    ("05_brow_frown",  4851),
    ("06_neutral",     3866),
]
W = LEN * FPS  # 300 frames

def cut_mov(start_frame: int, dst: Path) -> None:
    start_s = start_frame / FPS
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{start_s:.4f}", "-i", str(SRC_MOV),
        "-t", str(LEN),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-movflags", "+faststart",
        str(dst),
    ]
    subprocess.run(cmd, check=True)

def slice_csv(start_frame: int, dst: Path) -> int:
    with SRC_CSV.open() as f, dst.open("w", newline="") as g:
        reader = csv.reader(f)
        writer = csv.writer(g)
        header = next(reader)
        writer.writerow(header)
        rows = list(reader)
    sl = rows[start_frame:start_frame + W]
    with dst.open("w", newline="") as g:
        writer = csv.writer(g)
        writer.writerow(header)
        writer.writerows(sl)
    return len(sl)

def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    for label, start in CLIPS:
        d = OUT_DIR / label
        d.mkdir()
        mov = d / "clip.mp4"
        csv_path = d / "clip.csv"
        cut_mov(start, mov)
        n = slice_csv(start, csv_path)
        size = mov.stat().st_size / 1e6
        print(f"{label:<16s}  start_frame={start:5d}  rows={n:3d}  mp4={size:5.1f} MB  -> {d.relative_to(REPO)}")

if __name__ == "__main__":
    main()
