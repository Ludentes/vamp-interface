#!/usr/bin/env bash
# Render full takes in 600-frame segments under a 45 GB cgroup cap.
# Per saved feedback (feedback_eval_memory_cap.md), wrap the python that
# can OOM in a systemd scope so oomd kills the offender instead of the
# largest unrelated cgroup (Chromium last time, took down the desktop).
#
# Output: exp_output/arkit_bridge/render/full/take<N>_bridge_v1_full.mp4
# Intermediates land in /tmp/arkit_bridge_segs_<N>/seg_<i>.mp4

set -euo pipefail

cd "$(dirname "$0")/.."
PROJ="$PWD"
SEG=600                          # ARKit frames per segment (stride=2 -> 10 s @ 30 fps)
STRIDE=2
ANCHOR="data/llf-phase2/asian_m__06_neutral.midframe.png"
CKPT="runs/student_v1/student_best.pt"
OUTDIR="exp_output/arkit_bridge/render/full"
mkdir -p "$OUTDIR"

# (take_id, total_arkit_frames) — totals from take.json (catalog).
declare -A TOTAL=([2]=7440 [3]=3248 [4]=2063 [5]=7337 [6]=5208 [7]=6434 [8]=2785)

for n in 2 3 4 5 6 7 8; do
  total=${TOTAL[$n]}
  segs_dir="/tmp/arkit_bridge_segs_${n}"
  rm -rf "$segs_dir"; mkdir -p "$segs_dir"
  echo "=== take $n  total=$total  segment=$SEG  stride=$STRIDE ===" >&2

  i=0
  start=0
  while [ $start -lt $total ]; do
    seg_out="$segs_dir/seg_$(printf '%03d' $i).mp4"
    echo "  -> seg $i  start=$start  n=$SEG" >&2
    systemd-run --user --scope -q -p MemoryMax=45G \
      env PYTHONPATH=src \
      "$HOME/w/PersonaLive/.venv/bin/python" scripts/apply_bridge_to_personalive.py \
        --reference "$ANCHOR" \
        --take_dir "data/llf-takes/20260505_MySlate_$n" \
        --ckpt "$CKPT" \
        --out_path "$seg_out" \
        --start_frame "$start" \
        --n_frames "$SEG" \
        --stride "$STRIDE" 2>&1 | grep -E "wrote |Error|Traceback|Killed" >&2 || {
          echo "  seg $i FAILED — aborting take $n" >&2
          break
        }
    if [ ! -f "$seg_out" ]; then
      echo "  seg $i did not produce $seg_out — aborting take $n" >&2
      break
    fi
    i=$((i + 1))
    start=$((start + SEG * STRIDE))
  done

  # Concat segments with ffmpeg.
  list="$segs_dir/list.txt"
  : > "$list"
  for f in "$segs_dir"/seg_*.mp4; do
    [ -f "$f" ] && echo "file '$f'" >> "$list"
  done
  if [ -s "$list" ]; then
    out="$OUTDIR/take${n}_bridge_v1_full.mp4"
    ffmpeg -y -f concat -safe 0 -i "$list" -c copy "$out" 2>&1 | tail -1 >&2
    echo "  WROTE $out (segments: $(wc -l <"$list"))" >&2
    rm -rf "$segs_dir"
  else
    echo "  no segments rendered for take $n" >&2
  fi
done

echo "ALL DONE" >&2
ls -la "$OUTDIR"
