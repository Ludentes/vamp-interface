#!/usr/bin/env bash
# Resumable batch driver: render teacher_full + bridge for a list of takes.
# Skips a take if BOTH mp4s already exist for its <out_dir>/<take>_{teacher_full,bridge}.mp4.
#
# Usage:
#   bash scripts/render_takes_full.sh "1 2 3 6 8" runs/student_v3_lam10/student_best.pt v3_full
set -euo pipefail

TAKES="${1:?takes list e.g. \"1 2 3 6 8\"}"
CKPT="${2:?ckpt path}"
TAG="${3:?out tag e.g. v3_full}"
OUT_DIR="exp_output/arkit_bridge/render/${TAG}"
LOG="runs/${TAG}_render.log"
PY=/home/newub/w/PersonaLive/.venv/bin/python
ANCHOR=data/llf-phase2/asian_m__06_neutral.midframe.png
N_FRAMES=1200
STRIDE=2

mkdir -p "$OUT_DIR"
cd /home/newub/w/vamp-interface

for t in $TAKES; do
  TAKE_DIR="data/llf-takes/20260505_MySlate_${t}"
  TAKE_NAME="20260505_MySlate_${t}"
  TFULL="${OUT_DIR}/${TAKE_NAME}_teacher_full.mp4"
  TBR="${OUT_DIR}/${TAKE_NAME}_bridge.mp4"
  TJSON="${OUT_DIR}/${TAKE_NAME}_compare.json"

  echo "=== take $t ===" | tee -a "$LOG"
  if [[ -s "$TFULL" && -s "$TBR" && -s "$TJSON" ]]; then
    echo "  skip (all artefacts exist)" | tee -a "$LOG"
    continue
  fi

  if [[ -s "$TFULL" && -s "$TBR" ]]; then
    echo "  re-running compare-only (mp4s present, json missing)" | tee -a "$LOG"
    PYTHONPATH=src "$PY" scripts/compare_teacher_vs_bridge.py \
      --reference "$ANCHOR" --take_dir "$TAKE_DIR" --ckpt "$CKPT" \
      --out_dir "$OUT_DIR" --n_frames $N_FRAMES --stride $STRIDE --start_frame 0 \
      --py "$PY" --skip_render >> "$LOG" 2>&1
    continue
  fi

  set +e
  PYTHONPATH=src "$PY" scripts/compare_teacher_vs_bridge.py \
    --reference "$ANCHOR" --take_dir "$TAKE_DIR" --ckpt "$CKPT" \
    --out_dir "$OUT_DIR" --n_frames $N_FRAMES --stride $STRIDE --start_frame 0 \
    --py "$PY" >> "$LOG" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "  FAILED rc=$rc — continuing" | tee -a "$LOG"
  fi
done
echo "all done" | tee -a "$LOG"
