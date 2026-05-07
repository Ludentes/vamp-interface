#!/usr/bin/env bash
# Run vanilla LivePortrait against the same 8 anchors and yaw-stress driver
# clip used for the PersonaLive teacher_full baseline. Skip-if-exists.
#
# Anchors:
#   data/anchors/other/cropped/{zombie,orc,duck_head,duck_full,demon}.png  (5 stylized)
#   data/anchors/photoreal_ffhq/*.png                                       (3 photoreal)
#
# Driver: data/llf-clips-auto/20260505_MySlate_5_yaw/MySlate_5_iPhone.mp4
# Output: exp_output/liveportrait/yaw_stress/<tag>.mp4
#
# Usage: bash scripts/run_liveportrait_anchors.sh
set -uo pipefail

VAMP_ROOT=/home/newub/w/vamp-interface
LP_ROOT=/home/newub/w/LivePortrait
PY="$LP_ROOT/.venv/bin/python"
DRIVER="$VAMP_ROOT/data/llf-clips-auto/20260505_MySlate_5_yaw/MySlate_5_iPhone.mp4"
OUT_DIR="$VAMP_ROOT/exp_output/liveportrait/yaw_stress"
LOG="$VAMP_ROOT/runs/liveportrait_yaw_stress.log"

mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

ANCHORS=(
  "$VAMP_ROOT/data/anchors/other/cropped/zombie.png"
  "$VAMP_ROOT/data/anchors/other/cropped/orc.png"
  "$VAMP_ROOT/data/anchors/other/cropped/duck_head.png"
  "$VAMP_ROOT/data/anchors/other/cropped/duck_full.png"
  "$VAMP_ROOT/data/anchors/other/cropped/demon.png"
  "$VAMP_ROOT/data/anchors/photoreal_ffhq/east_asian__adult__m__41526046.png"
  "$VAMP_ROOT/data/anchors/photoreal_ffhq/south_asian__elderly__f__aeaaffdd.png"
  "$VAMP_ROOT/data/anchors/photoreal_ffhq/white__young__f__c2f7119b.png"
)

echo "[lp] ${#ANCHORS[@]} anchors queued $(date -Is)" | tee -a "$LOG"

cd "$LP_ROOT"
for ANCHOR in "${ANCHORS[@]}"; do
  TAG=$(basename "$ANCHOR" .png)
  OUT="$OUT_DIR/$TAG.mp4"

  if [[ -s "$OUT" ]]; then
    echo "[skip] $TAG (exists)" | tee -a "$LOG"
    continue
  fi

  echo "=== $TAG ===" | tee -a "$LOG"
  # --flag_crop_driving_video: driver is 540x720 portrait, needs face crop.
  # --no-flag_pasteback: keep 512² animated crop (matches PersonaLive output framing).
  "$PY" inference.py \
      --source "$ANCHOR" \
      --driving "$DRIVER" \
      --output_dir "$OUT_DIR/_tmp_$TAG" \
      --flag_crop_driving_video \
      --no-flag_pasteback \
      >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  FAILED $TAG (exit $rc)" | tee -a "$LOG"
    continue
  fi
  # LivePortrait writes <source>--<driver>.mp4 (and a _concat variant). Move
  # the main animation to our naming scheme.
  PROD=$(ls "$OUT_DIR/_tmp_$TAG"/*.mp4 2>/dev/null | grep -v _concat | head -1)
  if [[ -n "$PROD" ]]; then
    mv "$PROD" "$OUT"
    rm -rf "$OUT_DIR/_tmp_$TAG"
    echo "  wrote $OUT" | tee -a "$LOG"
  else
    echo "  FAILED $TAG (no output mp4 found in _tmp dir)" | tee -a "$LOG"
  fi
done

echo "[lp] done $(date -Is)" | tee -a "$LOG"
