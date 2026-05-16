#!/usr/bin/env bash
# nvdiffrast UV-texture bake driver. Activates the lam conda env.
#   bash scripts/bake_uv_texture.sh STEM
set -eu

STEM=${1:?usage: $0 STEM   (e.g. me_512, asian_m or status)}
VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1

python "${VAMP}/scripts/bake_uv_texture.py" \
    --ply  "${LAM}/exps/cano_gs/${STEM}_cano.ply" \
    --mesh "${LAM}/exps/cano_gs/${STEM}_shaped_mesh.obj" \
    --out  "${VAMP}/exp_output/lam_chibi/renders/bake_v3/${STEM}" \
    --stem "${STEM}"
