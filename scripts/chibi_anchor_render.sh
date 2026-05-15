#!/usr/bin/env bash
# End-to-end chibi pipeline for an arbitrary anchor PNG.
#
# 1. Run baseline LAM inference on $ANCHOR_PNG to bake $STEM_textured_mesh.obj
# 2. Run scripts/chibi_make_assets.py to generate chibi_arkit_bs.npy + chibi_textured_mesh.obj
# 3. Run LAM inference again with both chibi hooks set
# 4. Build side-by-side
#
# Usage:
#   bash scripts/chibi_anchor_render.sh ANCHOR_PNG [CHIBI_STRENGTH]
# e.g.:
#   bash scripts/chibi_anchor_render.sh exp_output/lam_chibi/user_anchor/me.png 1.0

set -u
ANCHOR_PNG=$(readlink -f "${1:?usage: $0 ANCHOR_PNG [STRENGTH]}")
STRENGTH=${2:-1.0}

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
STEM=$(basename "${ANCHOR_PNG}" .png)
MOTION_DIR=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2
CHIBI_DIR=${VAMP}/exp_output/lam_chibi/${STEM}
OUT_DIR=${VAMP}/exp_output/lam_chibi/renders
LOG_DIR=${OUT_DIR}/_logs

mkdir -p "${CHIBI_DIR}" "${OUT_DIR}" "${LOG_DIR}"

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

cd "${LAM}"

# --- Stage 1: baseline LAM inference to bake textured_mesh.obj --------------
BAKED_OBJ=${LAM}/exps/cano_gs/${STEM}_textured_mesh.obj
BAKED_DISPLAY=${CHIBI_DIR}/${STEM}_textured_mesh_display.obj
if [[ ! -f "${BAKED_OBJ}" ]]; then
  echo "[stage1] running baseline LAM inference to bake mesh for ${STEM}"
  LOG=${LOG_DIR}/${STEM}_baseline.log
  # explicit: NO chibi hooks set
  unset LAM_CHIBI_ARKIT_BS
  unset LAM_EDIT_XYZ_OBJ
  unset LAM_EDIT_VERTEX_COLORS_OBJ
  unset LAM_AUX_SPLATS_PLY
  if bash scripts/inference.sh \
      configs/inference/lam-20k-8gpu.yaml \
      model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
      "${ANCHOR_PNG}" \
      "${MOTION_DIR}/" \
      > "${LOG}" 2>&1; then
    [[ -f "${BAKED_OBJ}" ]] || { echo "no baked OBJ at ${BAKED_OBJ} after baseline (log: ${LOG})"; tail -20 "${LOG}"; exit 2; }
    cp "${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4" "${OUT_DIR}/baseline_${STEM}.mp4"
    echo "[stage1] baseline mesh: ${BAKED_OBJ}"
  else
    echo "[stage1] baseline FAILED (log: ${LOG})"; tail -30 "${LOG}"; exit 2
  fi
else
  echo "[stage1] baseline mesh already exists: ${BAKED_OBJ}"
fi

# --- Stage 2: SH→display-RGB rebake + chibi asset generation ----------------
if [[ ! -f "${BAKED_DISPLAY}" ]]; then
  echo "[stage2a] baking SH→display-RGB"
  python "${VAMP}/scripts/lam_bake_display_rgb_obj.py" \
      --input "${BAKED_OBJ}" \
      --output "${BAKED_DISPLAY}"
fi

if [[ ! -f "${CHIBI_DIR}/chibi_arkit_bs.npy" ]]; then
  echo "[stage2b] generating chibi assets (strength=${STRENGTH})"
  python "${VAMP}/scripts/chibi_make_assets.py" \
      --template "${LAM}/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj" \
      --arkit_bs "${LAM}/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy" \
      --baked    "${BAKED_DISPLAY}" \
      --outdir   "${CHIBI_DIR}" \
      --chibi_strength "${STRENGTH}"
fi

# --- Stage 3: chibi LAM render ---------------------------------------------
export LAM_CHIBI_ARKIT_BS=${CHIBI_DIR}/chibi_arkit_bs.npy
export LAM_EDIT_XYZ_OBJ=${CHIBI_DIR}/chibi_textured_mesh.obj
unset LAM_EDIT_VERTEX_COLORS_OBJ
unset LAM_AUX_SPLATS_PLY

LOG=${LOG_DIR}/${STEM}_chibi.log
echo "[stage3] chibi render (strength=${STRENGTH})  log=${LOG}"
t0=$(date +%s)
if bash scripts/inference.sh \
    configs/inference/lam-20k-8gpu.yaml \
    model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    "${ANCHOR_PNG}" \
    "${MOTION_DIR}/" \
    > "${LOG}" 2>&1; then
  SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${OUT_DIR}/chibi_${STEM}_s${STRENGTH}.mp4"
    t1=$(date +%s)
    echo "[stage3] done in $((t1 - t0))s → ${OUT_DIR}/chibi_${STEM}_s${STRENGTH}.mp4"
  else
    echo "[stage3] FAIL: no output (log: ${LOG})"
    tail -20 "${LOG}"
    exit 2
  fi
else
  echo "[stage3] FAIL: inference returned non-zero (log: ${LOG})"
  tail -30 "${LOG}"
  exit 2
fi

# --- Stage 4: trim baseline to 3s + side-by-side ----------------------------
B3=${OUT_DIR}/baseline_${STEM}_3s.mp4
ffmpeg -y -i "${OUT_DIR}/baseline_${STEM}.mp4" -t 3 -c:v libx264 -preset ultrafast -crf 18 "${B3}" 2>/dev/null
C3=${OUT_DIR}/chibi_${STEM}_s${STRENGTH}_3s.mp4
ffmpeg -y -i "${OUT_DIR}/chibi_${STEM}_s${STRENGTH}.mp4" -t 3 -c:v libx264 -preset ultrafast -crf 18 "${C3}" 2>/dev/null
SXS=${OUT_DIR}/sidebyside_chibi_${STEM}_s${STRENGTH}.mp4
ffmpeg -y -i "${B3}" -i "${C3}" \
    -filter_complex "[0:v]scale=512:-2,drawtext=text='baseline':x=10:y=10:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5[a];[1:v]scale=512:-2,drawtext=text='chibi s=${STRENGTH}':x=10:y=10:fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5[b];[a][b]hstack" \
    -c:v libx264 -preset ultrafast -crf 18 \
    "${SXS}" 2>/dev/null
echo "[sxs] ${SXS}"

echo "=== ${STEM}@${STRENGTH} DONE ==="
