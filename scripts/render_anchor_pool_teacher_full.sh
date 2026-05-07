#!/usr/bin/env bash
# Run vanilla PersonaLive (--mode teacher_full) on each anchor in the
# 9-anchor scorecard pool against a fixed driver clip. Skip-if-exists
# for resumability. Single GPU, sequential.
#
# Driver: data/llf-clips-auto/20260505_MySlate_5_yaw/ (600 frames, yaw stress)
# Output: exp_output/arkit_bridge/render/anchor_pool_teacher_full/<anchor>.mp4
#
# Usage:
#   bash scripts/render_anchor_pool_teacher_full.sh
# Continue past per-anchor failures — stylized references (anime / paintings)
# may fail face detection in PersonaLive's reference-side mediapipe crop.
# We want the rest of the batch to still complete and surface the failure
# to the operator at the end.
set -uo pipefail

cd /home/newub/w/vamp-interface

PY=/home/newub/w/PersonaLive/.venv/bin/python
CKPT=runs/student_v3_lam10/student_best.pt   # required by CLI even in teacher_full
TAKE_DIR=data/llf-clips-auto/20260505_MySlate_5_yaw
OUT_DIR=exp_output/arkit_bridge/render/anchor_pool_teacher_full
LOG=runs/anchor_pool_teacher_full.log
N_FRAMES=300
STRIDE=2

mkdir -p "$OUT_DIR"

# Anchors: glob each pool dir for staged PNGs/JPGs.
ANCHORS=()
while IFS= read -r p; do ANCHORS+=("$p"); done < <(
  find data/anchors/photoreal_grid    -maxdepth 1 -name "*.png" 2>/dev/null
  find data/anchors/photoreal_ffhq    -maxdepth 1 -name "*.png" 2>/dev/null
  find data/anchors/anime             -maxdepth 1 -name "*.png" 2>/dev/null
  find data/anchors/oldphoto_tikhonov -maxdepth 1 -name "*.jpg" 2>/dev/null
  find data/anchors/painting_pushkin  -maxdepth 1 -name "*.jpg" 2>/dev/null
)

echo "[pool] ${#ANCHORS[@]} anchors queued" | tee -a "$LOG"
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
  # --no_rotate_iphone: cv2 on this box already honors the iPhone
  # rotation=-90 metadata, so frames arrive portrait. The script's
  # default rotate_iphone=True double-rotates into sideways. (Verified
  # 2026-05-06 by reading the first frame and visualizing.)
  PYTHONPATH=src "$PY" scripts/apply_bridge_to_personalive.py \
    --reference "$ANCHOR" \
    --take_dir "$TAKE_DIR" \
    --ckpt "$CKPT" \
    --out_path "$OUT" \
    --n_frames "$N_FRAMES" \
    --stride "$STRIDE" \
    --mode teacher_full \
    --no_rotate_iphone \
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
