#!/usr/bin/env bash
# Chibi mesh-pivot driver. Activates the lam conda env (pytorch3d lives there)
# and runs scripts/chibi_mesh_render.py on a v3 textured-mesh anchor.
#   bash scripts/chibi_mesh_render.sh STEM [FIELD_PARAMS_JSON]
set -eu

STEM=${1:?usage: $0 STEM [FIELD_PARAMS_JSON]}
VAMP=/home/newub/w/vamp-interface
FIELD=${2:-${VAMP}/exp_output/lam_chibi/diff_geometry/chibi_field_params.json}
OBJ=${VAMP}/exp_output/lam_chibi/renders/bake_v3/${STEM}/${STEM}_textured.obj
OUT=${VAMP}/exp_output/lam_chibi/renders/mesh_v1

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=/home/newub/w/LAM XFORMERS_DISABLED=1

python "${VAMP}/scripts/chibi_mesh_render.py" \
    --obj "${OBJ}" --field "${FIELD}" --out "${OUT}"
