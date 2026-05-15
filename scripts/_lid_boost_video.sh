#!/usr/bin/env bash
set -eu
BOOST=$1   # e.g. 1.75 or 2.0
COLOR_MODE=${2:-magenta}   # magenta | normal
VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
ANCHOR=${VAMP}/exp_output/lam_chibi/user_anchor/me.png
ASSET=${VAMP}/exp_output/lam_chibi/me_s2.0
MOTION=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2
MAGENTA=${VAMP}/exp_output/lam_chibi/me/me_textured_mesh_display_eyemagenta.obj
OUT=${VAMP}/exp_output/lam_chibi/renders/lid_video
LOGS=${OUT}/_logs
mkdir -p $OUT $LOGS

export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=$LAM XFORMERS_DISABLED=1 LAM_USE_ARKIT=1
export LAM_CHIBI_ARKIT_BS=${ASSET}/chibi_arkit_bs.npy
export LAM_EDIT_XYZ_OBJ=${ASSET}/chibi_textured_mesh.obj
export LAM_CHIBI_SCALE_RATIO=${ASSET}/chibi_scale_ratio_lidbst${BOOST}.npy
if [[ "${COLOR_MODE}" == "magenta" ]]; then
  export LAM_EDIT_VERTEX_COLORS_OBJ=${MAGENTA}
else
  unset LAM_EDIT_VERTEX_COLORS_OBJ
fi
unset LAM_J_PER_VERT_NPY LAM_AXIS_BOOST_NPY LAM_OPACITY_MUL_NPY LAM_CHIBI_SCALE_BOOST LAM_AUX_SPLATS_PLY

cd $LAM
rm -f exps/videos/lam/lam_20k/me.mp4
echo "[start] lid_boost=$BOOST  at $(date +%H:%M:%S)"
bash scripts/inference.sh \
    configs/inference/lam-20k-8gpu.yaml \
    model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
    "${ANCHOR}" "${MOTION}/" > $LOGS/lid_${BOOST}_${COLOR_MODE}.log 2>&1
cp exps/videos/lam/lam_20k/me.mp4 $OUT/lid_${BOOST}_${COLOR_MODE}.mp4
echo "[done]  lid_boost=$BOOST  at $(date +%H:%M:%S)  → $OUT/lid_${BOOST}_${COLOR_MODE}.mp4"
