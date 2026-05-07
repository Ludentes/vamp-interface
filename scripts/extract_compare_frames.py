"""Pull worst frames from a compare mp4 and dump teacher | bridge | diff stills.

Why: scrubbing the side-by-side mp4 is awkward. This dumps a few PNG triplets
focused on top-K worst-frame indices so the artefact can be inspected statically.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary_json", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    s = json.loads(Path(args.summary_json).read_text())
    base = Path(args.summary_json).parent
    take = s["take"]
    teacher = base / f"{take}_teacher_full.mp4"
    bridge = base / f"{take}_bridge.mp4"
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    def read_all(p):
        cap = cv2.VideoCapture(str(p))
        frs = []
        while True:
            ok, f = cap.read()
            if not ok: break
            frs.append(f)
        cap.release()
        return frs

    t = read_all(teacher); b = read_all(bridge)
    n = min(len(t), len(b))
    worst = s["top10_worst_frames"][:args.n]
    for i in worst:
        if i >= n: continue
        ti, bi = t[i], b[i]
        d = np.clip(np.abs(ti.astype(np.int32) - bi.astype(np.int32)) * 8, 0, 255).astype(np.uint8)
        row = np.concatenate([ti, bi, d], axis=1)
        h = ti.shape[0]
        cv2.putText(row, "TEACHER", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(row, "BRIDGE", (ti.shape[1]+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(row, "DIFFx8", (2*ti.shape[1]+10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(row, f"f{i:03d}", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imwrite(str(out / f"{take}_f{i:03d}.png"), row)
        print(f"wrote {take}_f{i:03d}.png")


if __name__ == "__main__":
    main()
