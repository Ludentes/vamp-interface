#!/usr/bin/env bash
# Lash-doubling gate harness.
#
# Renders frame 328 (full-blink) and frame 599 (half-blink) under three
# conditions and produces a 2x3 collage cropped to the eye region:
#   pre-fix      = chibi xyz override only, no LAM_CHIBI_SCALE_RATIO
#   v2 baseline  = chibi xyz + LAM_CHIBI_SCALE_RATIO from the standard s=2.0 asset
#   candidate    = chibi xyz + LAM_CHIBI_SCALE_RATIO from a custom asset dir
#                  (e.g. one built with --eye_region_scale_cap)
#
# Usage:
#   bash scripts/chibi_lash_gate.sh CAND_ASSET_DIR CAND_TAG
#
# CAND_ASSET_DIR  Asset dir produced by chibi_make_assets.py (must contain
#                 chibi_arkit_bs.npy, chibi_textured_mesh.obj, chibi_scale_ratio.npy)
# CAND_TAG        Short label for the candidate variant (used in collage caption
#                 and output filename), e.g. "cap_1.15"
set -u
CAND_DIR=$(readlink -f "${1:?usage: $0 CAND_ASSET_DIR CAND_TAG}")
CAND_TAG="${2}"

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
ANCHOR=${VAMP}/exp_output/lam_chibi/user_anchor/me.png
STEM=$(basename "${ANCHOR}" .png)
BASELINE_DIR=${VAMP}/exp_output/lam_chibi/${STEM}_s2.0
OUT=${VAMP}/exp_output/lam_chibi/renders/frames_me/lash_gate
LOGS=${OUT}/_logs
mkdir -p "${OUT}" "${LOGS}"

[[ -f "${BASELINE_DIR}/chibi_arkit_bs.npy"     ]] || { echo "missing baseline arkit_bs"; exit 1; }
[[ -f "${BASELINE_DIR}/chibi_textured_mesh.obj" ]] || { echo "missing baseline xyz obj"; exit 1; }
[[ -f "${BASELINE_DIR}/chibi_scale_ratio.npy"  ]] || { echo "missing baseline scale_ratio"; exit 1; }
[[ -f "${CAND_DIR}/chibi_arkit_bs.npy"         ]] || { echo "missing cand arkit_bs"; exit 1; }
[[ -f "${CAND_DIR}/chibi_textured_mesh.obj"     ]] || { echo "missing cand xyz obj"; exit 1; }
[[ -f "${CAND_DIR}/chibi_scale_ratio.npy"      ]] || { echo "missing cand scale_ratio"; exit 1; }

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

cd "${LAM}"

render_one () {
  local TAG="$1"        # e.g. pre_f328
  local MOTION="$2"     # e.g. motion_f328
  local ASSET_DIR="$3"  # baseline or candidate
  local USE_RATIO="$4"  # "1" to export RATIO, "0" to unset

  export LAM_CHIBI_ARKIT_BS=${ASSET_DIR}/chibi_arkit_bs.npy
  export LAM_EDIT_XYZ_OBJ=${ASSET_DIR}/chibi_textured_mesh.obj
  if [[ "${USE_RATIO}" == "1" ]]; then
    export LAM_CHIBI_SCALE_RATIO=${ASSET_DIR}/chibi_scale_ratio.npy
  else
    unset LAM_CHIBI_SCALE_RATIO
  fi
  unset LAM_EDIT_VERTEX_COLORS_OBJ
  unset LAM_AUX_SPLATS_PLY

  local LOG=${LOGS}/${TAG}.log
  local OUT_PNG=${OUT}/${TAG}.png
  echo "[render] ${TAG}  ratio=${USE_RATIO}  log=${LOG}"
  local t0=$(date +%s)
  if bash scripts/inference.sh \
      configs/inference/lam-20k-8gpu.yaml \
      model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
      "${ANCHOR}" \
      "${VAMP}/exp_output/lam_chibi/${MOTION}/" \
      > "${LOG}" 2>&1; then
    local SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
    if [[ -f "${SRC}" ]]; then
      ffmpeg -loglevel error -y -i "${SRC}" -vframes 1 "${OUT_PNG}"
      local t1=$(date +%s)
      echo "  → ${OUT_PNG} ($((t1-t0))s)"
    else
      echo "  FAIL: no output mp4 (log: ${LOG})"; tail -20 "${LOG}"; exit 2
    fi
  else
    echo "  FAIL (log: ${LOG})"; tail -30 "${LOG}"; exit 2
  fi
}

# Render six panels.
render_one pre_f328  motion_f328 "${BASELINE_DIR}" 0
render_one pre_f599  motion_f599 "${BASELINE_DIR}" 0
render_one v2_f328   motion_f328 "${BASELINE_DIR}" 1
render_one v2_f599   motion_f599 "${BASELINE_DIR}" 1
render_one cand_f328 motion_f328 "${CAND_DIR}"     1
render_one cand_f599 motion_f599 "${CAND_DIR}"     1

# Build the collage + run intensity probe.
COLLAGE=${VAMP}/exp_output/lam_chibi/renders/lash_gate_${CAND_TAG}.png
python "${VAMP}/scripts/chibi_lash_gate.py" \
    --pre_f328  "${OUT}/pre_f328.png" \
    --pre_f599  "${OUT}/pre_f599.png" \
    --v2_f328   "${OUT}/v2_f328.png" \
    --v2_f599   "${OUT}/v2_f599.png" \
    --cand_f328 "${OUT}/cand_f328.png" \
    --cand_f599 "${OUT}/cand_f599.png" \
    --cand_tag  "${CAND_TAG}" \
    --out       "${COLLAGE}"

echo ""
echo "=== lash gate done: ${COLLAGE} ==="
