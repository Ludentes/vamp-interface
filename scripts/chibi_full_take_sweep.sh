#!/usr/bin/env bash
# Full-take render sweep for chibi iris/lash diagnosis. Renders the same 600-frame
# take under three conditions and assembles a side-by-side mp4:
#
#   prefix    = chibi xyz override only (no LAM_CHIBI_SCALE_RATIO)
#   v2        = chibi xyz + per-vertex ratio
#   magenta   = chibi xyz + per-vertex ratio + magenta-painted eyeball verts
#
# Run from /home/newub/w/vamp-interface.  Output:
#   exp_output/lam_chibi/renders/sweep_<TAG>/{prefix,v2,magenta}.mp4
#   exp_output/lam_chibi/renders/sweep_<TAG>/sxs.mp4
set -u
TAG="${1:-full_sweep}"

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
ANCHOR=${VAMP}/exp_output/lam_chibi/user_anchor/me.png
STEM=$(basename "${ANCHOR}" .png)
ASSET_DIR=${VAMP}/exp_output/lam_chibi/me_s2.0
MAGENTA_OBJ=${VAMP}/exp_output/lam_chibi/me/me_textured_mesh_display_eyemagenta.obj
MOTION_DIR=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2
OUT=${VAMP}/exp_output/lam_chibi/renders/sweep_${TAG}
LOGS=${OUT}/_logs
mkdir -p "${OUT}" "${LOGS}"

for f in "${ASSET_DIR}/chibi_arkit_bs.npy" "${ASSET_DIR}/chibi_textured_mesh.obj" \
         "${ASSET_DIR}/chibi_scale_ratio.npy" "${MAGENTA_OBJ}" "${MOTION_DIR}/transforms.json"; do
  [[ -f "${f}" ]] || { echo "missing: ${f}"; exit 1; }
done

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

cd "${LAM}"

render_take () {
  local VARIANT="$1"  # prefix | v2 | magenta

  export LAM_CHIBI_ARKIT_BS=${ASSET_DIR}/chibi_arkit_bs.npy
  export LAM_EDIT_XYZ_OBJ=${ASSET_DIR}/chibi_textured_mesh.obj
  case "${VARIANT}" in
    prefix)  unset LAM_CHIBI_SCALE_RATIO; unset LAM_EDIT_VERTEX_COLORS_OBJ ;;
    v2)      export LAM_CHIBI_SCALE_RATIO=${ASSET_DIR}/chibi_scale_ratio.npy
             unset LAM_EDIT_VERTEX_COLORS_OBJ ;;
    magenta) export LAM_CHIBI_SCALE_RATIO=${ASSET_DIR}/chibi_scale_ratio.npy
             export LAM_EDIT_VERTEX_COLORS_OBJ=${MAGENTA_OBJ} ;;
    *) echo "bad variant: ${VARIANT}"; return 2 ;;
  esac
  unset LAM_AUX_SPLATS_PLY LAM_CHIBI_SCALE_BOOST

  local LOG=${LOGS}/${VARIANT}.log
  local OUT_MP4=${OUT}/${VARIANT}.mp4
  echo "[render] ${VARIANT}  log=${LOG}"
  local t0=$(date +%s)
  if bash scripts/inference.sh \
      configs/inference/lam-20k-8gpu.yaml \
      model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
      "${ANCHOR}" \
      "${MOTION_DIR}/" \
      > "${LOG}" 2>&1; then
    local SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
    if [[ -f "${SRC}" ]]; then
      cp "${SRC}" "${OUT_MP4}"
      local t1=$(date +%s)
      local frames=$(ffprobe -v error -count_packets -select_streams v:0 -show_entries stream=nb_read_packets -of csv=p=0 "${OUT_MP4}")
      echo "  → ${OUT_MP4}  frames=${frames}  $((t1-t0))s"
    else
      echo "  FAIL: no output mp4 (log: ${LOG})"; tail -20 "${LOG}"; exit 2
    fi
  else
    echo "  FAIL (log: ${LOG})"; tail -30 "${LOG}"; exit 2
  fi
}

render_take prefix
render_take v2
render_take magenta

# Build 3-column side-by-side with text labels.
SXS=${OUT}/sxs.mp4
echo "[sxs] composing ${SXS}"
ffmpeg -loglevel error -y \
    -i "${OUT}/prefix.mp4"  \
    -i "${OUT}/v2.mp4"      \
    -i "${OUT}/magenta.mp4" \
    -filter_complex "
      [0:v]drawtext=text='prefix (no ratio)':fontcolor=white:fontsize=18:x=10:y=10:box=1:boxcolor=black@0.5[a];
      [1:v]drawtext=text='v2 (ratio)':fontcolor=white:fontsize=18:x=10:y=10:box=1:boxcolor=black@0.5[b];
      [2:v]drawtext=text='v2 + magenta eyeballs':fontcolor=white:fontsize=18:x=10:y=10:box=1:boxcolor=black@0.5[c];
      [a][b][c]hstack=inputs=3" \
    "${SXS}"
echo "[sxs] done: ${SXS}"
ls -la "${OUT}"/*.mp4
