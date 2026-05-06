#!/usr/bin/env bash
# LoRA hypothesis test — does PersonaLive's denoising_unet have a LoRA
# surface? Sweep alpha ∈ {0.0, 0.5, 0.8, 1.0} for a Kohya SD1.5 LoRA on
# one Flux-generated photoreal anchor against the yaw-stress driver.
#
# Pre-condition: download the LoRA manually to data/loras/<name>.safetensors.
# Civitai download often requires an account / API token; not auto-fetched.
#
#   mkdir -p data/loras
#   # Studio Ghibli Style LoRA (offset), SD1.5:
#   #   https://civitai.com/models/6526/studio-ghibli-style-lora
#   # Save as data/loras/ghibli_style_offset.safetensors
#
# Usage:
#   bash scripts/render_lora_alpha_sweep.sh [LORA_PATH] [ANCHOR_PATH]
set -uo pipefail
cd /home/newub/w/vamp-interface

LORA="${1:-data/loras/ghibli_style_offset.safetensors}"
ANCHOR="${2:-data/anchors/photoreal_grid/white__young__f__seed8675.png}"

if [[ ! -s "$LORA" ]]; then
  echo "FATAL: LoRA not found at $LORA"
  echo "Download from civitai (model 6526) and save as $LORA"
  exit 2
fi
if [[ ! -s "$ANCHOR" ]]; then
  echo "FATAL: anchor not found at $ANCHOR"
  exit 2
fi

PY=/home/newub/w/PersonaLive/.venv/bin/python
CKPT=runs/student_v3_lam10/student_best.pt
TAKE_DIR=data/llf-clips-auto/20260505_MySlate_5_yaw
LORA_TAG=$(basename "$LORA" .safetensors)
ANCHOR_TAG=$(basename "$ANCHOR" .png)
OUT_DIR=exp_output/personalive/lora_test/${LORA_TAG}_${ANCHOR_TAG}
LOG=runs/lora_alpha_sweep_${LORA_TAG}_${ANCHOR_TAG}.log
N_FRAMES=300
STRIDE=2

mkdir -p "$OUT_DIR" runs
echo "[lora-sweep] LoRA=$LORA anchor=$ANCHOR_TAG" | tee -a "$LOG"
date -Is | tee -a "$LOG"

# Quick key-match probe before burning GPU time on 4 renders.
echo "=== key-match probe ===" | tee -a "$LOG"
PYTHONPATH=src "$PY" -m arkit_bridge.lora_inject diff "$LORA" 2>&1 | tee -a "$LOG"

for ALPHA in 0.0 0.5 0.8 1.0; do
  TAG="alpha${ALPHA//./_}"
  OUT="$OUT_DIR/$TAG.mp4"
  if [[ -s "$OUT" ]]; then
    echo "[skip] $TAG (exists)" | tee -a "$LOG"
    continue
  fi

  echo "=== $TAG ===" | tee -a "$LOG"

  # alpha=0.0 is the control — skip LoRA entirely so we render against the
  # vanilla weights, not a noop-merged copy. The two should be identical
  # but skipping rules out any precision drift in the merge math.
  if [[ "$ALPHA" == "0.0" ]]; then
    LORA_ARGS=""
  else
    LORA_ARGS="--lora_path $LORA --lora_alpha $ALPHA --lora_targets den"
  fi

  PYTHONPATH=src "$PY" scripts/apply_bridge_to_personalive.py \
    --reference "$ANCHOR" \
    --take_dir "$TAKE_DIR" \
    --ckpt "$CKPT" \
    --out_path "$OUT" \
    --n_frames "$N_FRAMES" \
    --stride "$STRIDE" \
    --mode teacher_full \
    --no_rotate_iphone \
    $LORA_ARGS \
    >> "$LOG" 2>&1
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "  FAILED $TAG (exit $rc)" | tee -a "$LOG"
    continue
  fi
  if [[ -s "$OUT" ]]; then
    echo "  wrote $OUT" | tee -a "$LOG"
  else
    echo "  FAILED $TAG (no output mp4)" | tee -a "$LOG"
  fi
done

echo "[lora-sweep] done $(date -Is)" | tee -a "$LOG"
echo "Outputs: $OUT_DIR/"
ls -la "$OUT_DIR"/
