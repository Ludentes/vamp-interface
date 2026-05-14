#!/usr/bin/env bash
# Render LAM with pre-generated chibi assets (faster than chibi_anchor_render.sh
# when you're sweeping over chibi_strength on a single anchor).
#
# Usage:
#   bash scripts/chibi_render_with_assets.sh ANCHOR_PNG CHIBI_DIR TAG
set -u
ANCHOR_PNG=$(readlink -f "${1:?usage: $0 ANCHOR_PNG CHIBI_DIR TAG}")
CHIBI_DIR=$(readlink -f "${2}")
TAG="${3}"

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
STEM=$(basename "${ANCHOR_PNG}" .png)
MOTION_DIR=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2
OUT_DIR=${VAMP}/exp_output/lam_chibi/renders
LOG_DIR=${OUT_DIR}/_logs
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1
export LAM_CHIBI_ARKIT_BS=${CHIBI_DIR}/chibi_arkit_bs.npy
export LAM_EDIT_XYZ_OBJ=${CHIBI_DIR}/chibi_textured_mesh.obj
if [[ -f "${CHIBI_DIR}/chibi_scale_ratio.npy" ]]; then
  export LAM_CHIBI_SCALE_RATIO=${CHIBI_DIR}/chibi_scale_ratio.npy
else
  unset LAM_CHIBI_SCALE_RATIO
fi
unset LAM_EDIT_VERTEX_COLORS_OBJ
unset LAM_AUX_SPLATS_PLY

cd "${LAM}"
LOG=${LOG_DIR}/${STEM}_chibi_${TAG}.log
echo "[chibi:${TAG}] render starting"
t0=$(date +%s)
if bash scripts/inference.sh \
    configs/inference/lam-20k-8gpu.yaml \
    model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    "${ANCHOR_PNG}" \
    "${MOTION_DIR}/" \
    > "${LOG}" 2>&1; then
  SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${OUT_DIR}/chibi_${STEM}_${TAG}.mp4"
    t1=$(date +%s)
    echo "[chibi:${TAG}] done in $((t1 - t0))s → ${OUT_DIR}/chibi_${STEM}_${TAG}.mp4"
  else
    echo "[chibi:${TAG}] FAIL: no output (log: ${LOG})"; tail -20 "${LOG}"; exit 2
  fi
else
  echo "[chibi:${TAG}] FAIL (log: ${LOG})"; tail -30 "${LOG}"; exit 2
fi
