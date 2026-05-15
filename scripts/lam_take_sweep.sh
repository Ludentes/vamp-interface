#!/usr/bin/env bash
# LAM bake-off take sweep: 2 anchors × 7 takes (MySlate_2..8) at full duration.
#
# Stage 1: VHAP video preproc + optimize + export → motion dir per take
# Stage 2: LAM inference for each (anchor, motion_dir) pair
# Outputs: exp_output/lam_bakeoff/take_renders/take{N}__{anchor}.mp4
#
# Designed to run unattended for a few hours. Resumable: each take's motion dir
# and each anchor×take render are checked for existence before re-running.

set -u

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
SWEEP_ROOT=${VAMP}/exp_output/lam_bakeoff/take_runs
RENDER_DIR=${VAMP}/exp_output/lam_bakeoff/take_renders
LOG_DIR=${SWEEP_ROOT}/_logs

mkdir -p "${SWEEP_ROOT}" "${RENDER_DIR}" "${LOG_DIR}"

ANCHORS=(
  "asian_m:${SWEEP_ROOT}/anchors/asian_m.png"
  "young_european_f:${SWEEP_ROOT}/anchors/young_european_f.png"
)
TAKES=(2 3 4 5 6 7 8)

# --- env -------------------------------------------------------------
source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1

cd "${LAM}"

# --- Stage 1: VHAP per take ----------------------------------------
for T in "${TAKES[@]}"; do
  MOV="${VAMP}/data/llf-takes/20260505_MySlate_${T}/MySlate_${T}_iPhone.mov"
  RUN_DIR="${SWEEP_ROOT}/MySlate_${T}_full"
  MOTION_DIR="${RUN_DIR}/export/MySlate_${T}"

  if [[ -f "${MOTION_DIR}/canonical_flame_param.npz" ]]; then
    echo "[skip] motion dir already exists: ${MOTION_DIR}"
    continue
  fi

  # Get full duration from ffprobe (rounded down to int s for slack)
  DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${MOV}")
  DUR_INT=$(printf "%.0f" "${DUR}")
  echo "[take ${T}] duration ${DUR}s, decoding @30fps → ~$((DUR_INT * 30)) frames"

  mkdir -p "${RUN_DIR}"
  LOG="${LOG_DIR}/take${T}_vhap.log"
  t0=$(date +%s)
  if python tools/flame_tracking_video.py \
      --mov "${MOV}" \
      --output_dir "${RUN_DIR}" \
      --seq_name "MySlate_${T}" \
      --fps 30 --duration "${DUR}" \
      > "${LOG}" 2>&1; then
    t1=$(date +%s)
    echo "[take ${T}] VHAP done in $((t1 - t0))s — ${MOTION_DIR}"
  else
    echo "[take ${T}] VHAP FAILED (log: ${LOG}); continuing"
  fi
done

# --- Stage 2: LAM render per (anchor, take) ------------------------
for T in "${TAKES[@]}"; do
  MOTION_DIR="${SWEEP_ROOT}/MySlate_${T}_full/export/MySlate_${T}"
  if [[ ! -f "${MOTION_DIR}/canonical_flame_param.npz" ]]; then
    echo "[render skip] no motion dir for take ${T}"
    continue
  fi

  for ENTRY in "${ANCHORS[@]}"; do
    A_NAME="${ENTRY%%:*}"
    A_PATH="${ENTRY##*:}"
    OUT="${RENDER_DIR}/take${T}__${A_NAME}.mp4"

    if [[ -f "${OUT}" ]]; then
      echo "[skip] render exists: ${OUT}"
      continue
    fi

    LOG="${LOG_DIR}/take${T}__${A_NAME}_render.log"
    t0=$(date +%s)
    if bash scripts/inference.sh \
        configs/inference/lam-20k-8gpu.yaml \
        model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
        "${A_PATH}" \
        "${MOTION_DIR}/" \
        > "${LOG}" 2>&1; then
      # LAM writes to exps/videos/lam/lam_20k/<anchor_stem>.mp4
      STEM=$(basename "${A_PATH}" .png)
      SRC="${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4"
      if [[ -f "${SRC}" ]]; then
        cp "${SRC}" "${OUT}"
        t1=$(date +%s)
        echo "[render] take${T}/${A_NAME} done in $((t1 - t0))s → ${OUT}"
      else
        echo "[render] take${T}/${A_NAME} no output mp4 (log: ${LOG})"
      fi
    else
      echo "[render] take${T}/${A_NAME} FAILED (log: ${LOG})"
    fi
  done
done

echo "=== SWEEP COMPLETE ==="
ls -la "${RENDER_DIR}"/
