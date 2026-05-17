#!/usr/bin/env bash
# Render LAM at a chosen flame_subdivide_num, optionally with chibi asset hooks.
# Route-1 spike: bumping subdivide_num raises splat density ~4x per round, the
# uncapped lever for the enlarged-eye blur + iris-through-lid leak.
#
# Usage:
#   chibi_subdiv_render.sh ANCHOR_PNG SUBDIV TAG [CHIBI_DIR]
#     ANCHOR_PNG  anchor image
#     SUBDIV      flame_subdivide_num (1 = stock 20018, 2 = ~80k)
#     TAG         output tag
#     CHIBI_DIR   optional: dir with chibi_{arkit_bs.npy,textured_mesh.obj,
#                 scale_ratio.npy}. Omit for a plain (un-chibi) bake/render.
set -u
ANCHOR_PNG=$(readlink -f "${1:?usage: $0 ANCHOR_PNG SUBDIV TAG [CHIBI_DIR]}")
SUBDIV="${2:?need subdivide_num}"
TAG="${3:?need tag}"
CHIBI_DIR="${4:-}"

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

if [[ -n "${CHIBI_DIR}" ]]; then
  CHIBI_DIR=$(readlink -f "${CHIBI_DIR}")
  export LAM_CHIBI_ARKIT_BS=${CHIBI_DIR}/chibi_arkit_bs.npy
  export LAM_EDIT_XYZ_OBJ=${CHIBI_DIR}/chibi_textured_mesh.obj
  if [[ -f "${CHIBI_DIR}/chibi_scale_ratio.npy" ]]; then
    export LAM_CHIBI_SCALE_RATIO=${CHIBI_DIR}/chibi_scale_ratio.npy
  else
    unset LAM_CHIBI_SCALE_RATIO
  fi
else
  unset LAM_CHIBI_ARKIT_BS LAM_EDIT_XYZ_OBJ LAM_CHIBI_SCALE_RATIO
fi
unset LAM_EDIT_VERTEX_COLORS_OBJ LAM_AUX_SPLATS_PLY

cd "${LAM}"
LOG=${LOG_DIR}/${STEM}_subdiv${SUBDIV}_${TAG}.log
echo "[subdiv${SUBDIV}:${TAG}] render starting (chibi=${CHIBI_DIR:-none})"
t0=$(date +%s)
if CUDA_VISIBLE_DEVICES=0 python -m lam.launch infer.lam \
    --config configs/inference/lam-20k-8gpu.yaml \
    model_name=model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    image_input="${ANCHOR_PNG}" \
    motion_seqs_dir="${MOTION_DIR}/" motion_img_dir=null \
    export_video=true export_mesh=false \
    vis_motion=false motion_img_need_mask=true \
    render_fps=30 motion_video_read_fps=30 \
    save_ply=false save_img=true \
    gaga_track_type="" cross_id=false test_sample=false \
    rank=0 nodes=0 \
    model.flame_subdivide_num="${SUBDIV}" \
    > "${LOG}" 2>&1; then
  SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
  CANO=${LAM}/exps/cano_gs/${STEM}_textured_mesh.obj
  if [[ -f "${SRC}" ]]; then
    cp "${SRC}" "${OUT_DIR}/chibi_${STEM}_subdiv${SUBDIV}_${TAG}.mp4"
    t1=$(date +%s)
    echo "[subdiv${SUBDIV}:${TAG}] done in $((t1 - t0))s → ${OUT_DIR}/chibi_${STEM}_subdiv${SUBDIV}_${TAG}.mp4"
    [[ -f "${CANO}" ]] && echo "[subdiv${SUBDIV}:${TAG}] cano mesh: ${CANO} ($(grep -c '^v ' "${CANO}") verts)"
  else
    echo "[subdiv${SUBDIV}:${TAG}] FAIL: no video (log: ${LOG})"; tail -25 "${LOG}"; exit 2
  fi
else
  echo "[subdiv${SUBDIV}:${TAG}] FAIL (log: ${LOG})"; tail -30 "${LOG}"; exit 2
fi
