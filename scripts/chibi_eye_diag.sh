#!/usr/bin/env bash
# Single-frame chibi render harness for eye-blink artifact iteration.
#
# Renders frame 599 (= last frame of MySlate_2 take 2, the half-blink + down-gaze
# pose where the iris pokes through the lid on chibi @ s=2.0) under a sweep of
# blink_boost values, and saves one PNG per variant for side-by-side comparison.
#
# Usage:
#   bash scripts/chibi_eye_diag.sh                # default boost sweep
set -u

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
ANCHOR=${VAMP}/exp_output/lam_chibi/user_anchor/me.png
STEM=$(basename "${ANCHOR}" .png)
MOTION=${VAMP}/exp_output/lam_chibi/motion_f599
OUT=${VAMP}/exp_output/lam_chibi/renders/frames_me/eye_diag
BAKED=${VAMP}/exp_output/lam_chibi/${STEM}/${STEM}_textured_mesh_display.obj
LOGS=${OUT}/_logs
mkdir -p "${OUT}" "${LOGS}"

[[ -f "${BAKED}" ]] || { echo "missing display-RGB baked mesh: ${BAKED}"; exit 1; }

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

cd "${LAM}"

STRENGTH=2.0
for BOOST in 1.0 1.3 1.6 2.0; do
  ASSET_DIR=${VAMP}/exp_output/lam_chibi/${STEM}_s${STRENGTH}_b${BOOST}
  mkdir -p "${ASSET_DIR}"

  if [[ ! -f "${ASSET_DIR}/chibi_arkit_bs.npy" ]]; then
    echo "[asset] building s=${STRENGTH} blink_boost=${BOOST}"
    python "${VAMP}/scripts/chibi_make_assets.py" \
        --template "${LAM}/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj" \
        --arkit_bs "${LAM}/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy" \
        --baked    "${BAKED}" \
        --outdir   "${ASSET_DIR}" \
        --y_anchor_frac_baked 0.20 \
        --chibi_strength "${STRENGTH}" \
        --blink_boost "${BOOST}" \
        > "${LOGS}/asset_b${BOOST}.log" 2>&1
  fi

  export LAM_CHIBI_ARKIT_BS=${ASSET_DIR}/chibi_arkit_bs.npy
  export LAM_EDIT_XYZ_OBJ=${ASSET_DIR}/chibi_textured_mesh.obj
  if [[ -f "${ASSET_DIR}/chibi_scale_ratio.npy" ]]; then
    export LAM_CHIBI_SCALE_RATIO=${ASSET_DIR}/chibi_scale_ratio.npy
  else
    unset LAM_CHIBI_SCALE_RATIO
  fi
  unset LAM_EDIT_VERTEX_COLORS_OBJ
  unset LAM_AUX_SPLATS_PLY

  TAG="s${STRENGTH}_b${BOOST}"
  LOG=${LOGS}/render_${TAG}.log
  echo "[render] ${TAG}  log=${LOG}"
  t0=$(date +%s)
  if bash scripts/inference.sh \
      configs/inference/lam-20k-8gpu.yaml \
      model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
      "${ANCHOR}" \
      "${MOTION}/" \
      > "${LOG}" 2>&1; then
    SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
    if [[ -f "${SRC}" ]]; then
      # extract single frame as PNG (the motion has only 1 frame, but ffmpeg helps standardize)
      ffmpeg -loglevel error -y -i "${SRC}" -vframes 1 "${OUT}/eye_${TAG}.png"
      t1=$(date +%s)
      echo "  → ${OUT}/eye_${TAG}.png  (${SRC##*/}, $((t1-t0))s)"
    else
      echo "  FAIL: no output mp4 (log: ${LOG})"; tail -20 "${LOG}"; exit 2
    fi
  else
    echo "  FAIL (log: ${LOG})"; tail -30 "${LOG}"; exit 2
  fi
done

echo "=== eye-diag sweep done ==="
ls -la "${OUT}"/eye_s${STRENGTH}_b*.png
