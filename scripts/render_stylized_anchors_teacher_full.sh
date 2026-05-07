#!/usr/bin/env bash
# Run vanilla PersonaLive (--mode teacher_full) on the 5 stylized non-human
# anchors against the same yaw stress driver clip used for the photoreal
# scorecard. Skip-if-exists for resumability. Single GPU, sequential.
#
# Anchors (512² head-aligned crops, manually curated 2026-05-06):
#   data/anchors/other/cropped/{zombie, orc, duck_head, duck_full, demon}.png
#
# Driver: data/llf-clips-auto/20260505_MySlate_5_yaw/ (yaw stress)
# Output: exp_output/arkit_bridge/render/stylized_pool_teacher_full/<tag>.mp4
#
# This batch is the "vanilla baseline" referenced in
# docs/research/2026-05-05-personalive-architecture-notes.md →
# Inference-time control surface → Suggested probe order, step (1).
# Any γ/α/β sweep happens in a follow-up runner.
#
# Usage:
#   bash scripts/render_stylized_anchors_teacher_full.sh
set -uo pipefail

cd /home/newub/w/vamp-interface

PY=/home/newub/w/PersonaLive/.venv/bin/python
CKPT=runs/student_v3_lam10/student_best.pt   # required by CLI even in teacher_full
TAKE_DIR=data/llf-clips-auto/20260505_MySlate_5_yaw
OUT_DIR=exp_output/arkit_bridge/render/stylized_pool_teacher_full
LOG=runs/stylized_pool_teacher_full.log
N_FRAMES=300
STRIDE=2

mkdir -p "$OUT_DIR"

ANCHORS=(
  data/anchors/other/cropped/zombie.png
  data/anchors/other/cropped/orc.png
  data/anchors/other/cropped/duck_head.png
  data/anchors/other/cropped/duck_full.png
  data/anchors/other/cropped/demon.png
)

echo "[pool] ${#ANCHORS[@]} stylized anchors queued" | tee -a "$LOG"
date -Is | tee -a "$LOG"

for ANCHOR in "${ANCHORS[@]}"; do
  TAG=$(basename "$ANCHOR")
  TAG=${TAG%.*}
  OUT="$OUT_DIR/$TAG.mp4"

  if [[ -s "$OUT" ]]; then
    echo "[skip] $TAG (exists)" | tee -a "$LOG"
    continue
  fi

  echo "=== $TAG ===" | tee -a "$LOG"
  # --no_rotate_iphone: cv2 on this box already honors iPhone rotation metadata.
  # --reference_precropped: skip MediaPipe FaceMesh on stylized refs.
  PYTHONPATH=src "$PY" scripts/apply_bridge_to_personalive.py \
    --reference "$ANCHOR" \
    --take_dir "$TAKE_DIR" \
    --ckpt "$CKPT" \
    --out_path "$OUT" \
    --n_frames "$N_FRAMES" \
    --stride "$STRIDE" \
    --mode teacher_full \
    --no_rotate_iphone \
    --reference_precropped \
    >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  FAILED $TAG (exit $rc)" | tee -a "$LOG"
  elif [[ -s "$OUT" ]]; then
    echo "  wrote $OUT" | tee -a "$LOG"
  else
    echo "  FAILED $TAG (no output)" | tee -a "$LOG"
  fi
done

echo "[pool] done $(date -Is)" | tee -a "$LOG"
