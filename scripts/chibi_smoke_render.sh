#!/usr/bin/env bash
# Chibi smoke render: drive a chibi-deformed asian_m through 3s of take-2 ARKit.
#
# Injection points:
#   LAM_CHIBI_ARKIT_BS  → swap flame_arkit_bs.npy at FLAME init
#   LAM_EDIT_XYZ_OBJ    → swap gs_attr.xyz with the chibi-deformed 20018 mesh
#
# After render, builds a side-by-side vs the existing baseline 600-frame render
# (cropped to first 3s) so we can read whether the recipe holds together.

set -u

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
ANCHOR_NAME=asian_m
ANCHOR_PNG=${VAMP}/exp_output/lam_bakeoff/take_runs/anchors/${ANCHOR_NAME}.png
MOTION_DIR=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2
CHIBI_DIR=${VAMP}/exp_output/lam_chibi/${ANCHOR_NAME}
OUT_DIR=${VAMP}/exp_output/lam_chibi/renders
LOG_DIR=${OUT_DIR}/_logs

mkdir -p "${OUT_DIR}" "${LOG_DIR}"

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

# --- the two chibi injections -----------------------------------------------
export LAM_CHIBI_ARKIT_BS=${CHIBI_DIR}/chibi_arkit_bs.npy
export LAM_EDIT_XYZ_OBJ=${CHIBI_DIR}/chibi_textured_mesh.obj

[[ -f "${LAM_CHIBI_ARKIT_BS}" ]] || { echo "missing ${LAM_CHIBI_ARKIT_BS}"; exit 1; }
[[ -f "${LAM_EDIT_XYZ_OBJ}"   ]] || { echo "missing ${LAM_EDIT_XYZ_OBJ}";   exit 1; }
[[ -f "${ANCHOR_PNG}"         ]] || { echo "missing ${ANCHOR_PNG}";         exit 1; }
[[ -f "${MOTION_DIR}/canonical_flame_param.npz" ]] || { echo "missing motion dir"; exit 1; }

cd "${LAM}"

LOG=${LOG_DIR}/chibi_smoke.log
echo "[chibi] starting render"
echo "  anchor : ${ANCHOR_PNG}"
echo "  motion : ${MOTION_DIR}  ($(ls ${MOTION_DIR}/flame_param | wc -l) frames)"
echo "  basis  : ${LAM_CHIBI_ARKIT_BS}"
echo "  xyz    : ${LAM_EDIT_XYZ_OBJ}"
echo "  log    : ${LOG}"

t0=$(date +%s)
if bash scripts/inference.sh \
    configs/inference/lam-20k-8gpu.yaml \
    model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    "${ANCHOR_PNG}" \
    "${MOTION_DIR}/" \
    > "${LOG}" 2>&1; then
  STEM=$(basename "${ANCHOR_PNG}" .png)
  SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${OUT_DIR}/chibi_${ANCHOR_NAME}.mp4"
    t1=$(date +%s)
    echo "[chibi] done in $((t1 - t0))s → ${OUT_DIR}/chibi_${ANCHOR_NAME}.mp4"
  else
    echo "[chibi] FAIL: no output mp4 (log: ${LOG})"
    tail -20 "${LOG}"
    exit 2
  fi
else
  echo "[chibi] FAIL: inference returned non-zero (log: ${LOG})"
  tail -30 "${LOG}"
  exit 2
fi

# --- baseline for side-by-side ----------------------------------------------
BASELINE_FULL=${VAMP}/exp_output/lam_bakeoff/take_renders/take2__${ANCHOR_NAME}__arkit600.mp4
BASELINE_3S=${OUT_DIR}/baseline_${ANCHOR_NAME}_3s.mp4
if [[ -f "${BASELINE_FULL}" && ! -f "${BASELINE_3S}" ]]; then
  ffmpeg -y -i "${BASELINE_FULL}" -t 3 -c:v libx264 -preset ultrafast -crf 18 \
    "${BASELINE_3S}" 2>/dev/null
fi

# --- side-by-side -----------------------------------------------------------
if [[ -f "${BASELINE_3S}" ]]; then
  ffmpeg -y -i "${BASELINE_3S}" -i "${OUT_DIR}/chibi_${ANCHOR_NAME}.mp4" \
    -filter_complex "[0:v]scale=512:-2,drawtext=text='baseline':x=10:y=10:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5[a];[1:v]scale=512:-2,drawtext=text='chibi':x=10:y=10:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5[b];[a][b]hstack" \
    -c:v libx264 -preset ultrafast -crf 18 \
    "${OUT_DIR}/sidebyside_chibi_${ANCHOR_NAME}.mp4" 2>/dev/null
  echo "[sxs] ${OUT_DIR}/sidebyside_chibi_${ANCHOR_NAME}.mp4"
fi

echo "=== CHIBI SMOKE COMPLETE ==="
ls -la "${OUT_DIR}/"
