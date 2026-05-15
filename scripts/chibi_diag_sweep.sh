#!/usr/bin/env bash
# Diagnostic sweep for iris-through-lid mechanism.
# Renders motion_f328 (full blink) under 7 conditions:
#   v2                  baseline (chibi xyz + scale_ratio)
#   opa_eye_0.1         + opacity × 0.1 on eyeball verts  (Exp C1)
#   opa_lid_0.3         + opacity × 0.3 on eye_region     (Exp C2)
#   axis_ax0_x3         + sigma_axis0 × 3 on eye_region   (Exp B-ax0)
#   axis_ax1_x3         + sigma_axis1 × 3 on eye_region   (Exp B-ax1)
#   axis_ax2_x3         + sigma_axis2 × 3 on eye_region   (Exp B-ax2)
#   eyereg_iso_x3       + sigma_all × 3 on eye_region     (Exp A — wild iso)
#
# Output: exp_output/lam_chibi/renders/diag_sweep/<TAG>.png + montage.png
set -u
VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
ANCHOR=${VAMP}/exp_output/lam_chibi/user_anchor/me.png
STEM=$(basename "${ANCHOR}" .png)
ASSET_DIR=${VAMP}/exp_output/lam_chibi/me_s2.0
DIAG=${ASSET_DIR}/diag
MOTION_DIR=${VAMP}/exp_output/lam_chibi/motion_f328
SUFFIX="${SWEEP_SUFFIX:-}"
OUT=${VAMP}/exp_output/lam_chibi/renders/diag_sweep${SUFFIX}
MAGENTA_OBJ=${VAMP}/exp_output/lam_chibi/me/me_textured_mesh_display_eyemagenta.obj
LOGS=${OUT}/_logs
mkdir -p "${OUT}" "${LOGS}"

source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

cd "${LAM}"

run () {
  local TAG="$1"; shift
  # Reset env state.
  export LAM_CHIBI_ARKIT_BS=${ASSET_DIR}/chibi_arkit_bs.npy
  export LAM_EDIT_XYZ_OBJ=${ASSET_DIR}/chibi_textured_mesh.obj
  export LAM_CHIBI_SCALE_RATIO=${ASSET_DIR}/chibi_scale_ratio.npy
  unset LAM_AXIS_BOOST_NPY
  unset LAM_OPACITY_MUL_NPY
  if [[ -n "${USE_MAGENTA:-}" ]]; then
    export LAM_EDIT_VERTEX_COLORS_OBJ=${MAGENTA_OBJ}
  else
    unset LAM_EDIT_VERTEX_COLORS_OBJ
  fi
  unset LAM_AUX_SPLATS_PLY
  unset LAM_CHIBI_SCALE_BOOST
  # Apply variant overrides.
  while [[ $# -gt 0 ]]; do
    eval "export $1"
    shift
  done

  local LOG=${LOGS}/${TAG}.log
  local OUT_PNG=${OUT}/${TAG}.png
  echo "[render] ${TAG}  log=${LOG}"
  local t0=$(date +%s)
  if bash scripts/inference.sh \
      configs/inference/lam-20k-8gpu.yaml \
      model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
      "${ANCHOR}" \
      "${MOTION_DIR}/" \
      > "${LOG}" 2>&1; then
    local SRC=${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
    if [[ -f "${SRC}" ]]; then
      ffmpeg -loglevel error -y -i "${SRC}" -vframes 1 "${OUT_PNG}"
      echo "  → ${OUT_PNG} ($(($(date +%s)-t0))s)"
    else
      echo "  FAIL: no mp4 (log: ${LOG})"; tail -25 "${LOG}"; exit 2
    fi
  else
    echo "  FAIL (log: ${LOG})"; tail -30 "${LOG}"; exit 2
  fi
}

run v2
run opa_eye_0.1     "LAM_OPACITY_MUL_NPY=${DIAG}/opacity_eyeball_0.1.npy"
run opa_lid_0.3     "LAM_OPACITY_MUL_NPY=${DIAG}/opacity_eye_region_0.3.npy"
run axis_ax0_x3     "LAM_AXIS_BOOST_NPY=${DIAG}/axis_boost_ax0_x3.0.npy"
run axis_ax1_x3     "LAM_AXIS_BOOST_NPY=${DIAG}/axis_boost_ax1_x3.0.npy"
run axis_ax2_x3     "LAM_AXIS_BOOST_NPY=${DIAG}/axis_boost_ax2_x3.0.npy"
run eyereg_iso_x3   "LAM_AXIS_BOOST_NPY=${DIAG}/axis_boost_all_x3.0.npy"

# Crop each render to eye strip + build montage labeled.
python - <<'PY'
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import os
OUT = Path(f"/home/newub/w/vamp-interface/exp_output/lam_chibi/renders/diag_sweep{os.environ.get('SWEEP_SUFFIX','')}")
CROP = (100, 250, 412, 420)  # generous eye strip
TAGS = ["v2","opa_eye_0.1","opa_lid_0.3","axis_ax0_x3","axis_ax1_x3","axis_ax2_x3","eyereg_iso_x3"]
strips = []
for t in TAGS:
    p = OUT / f"{t}.png"
    if not p.exists():
        print("missing", p); continue
    im = Image.open(p).convert("RGB").crop(CROP)
    pad = 24
    canvas = Image.new("RGB", (im.width, im.height+pad), (255,255,255))
    canvas.paste(im, (0, pad))
    d = ImageDraw.Draw(canvas)
    try:
        f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except OSError:
        f = ImageFont.load_default()
    d.text((6, 4), t, fill=(0,0,0), font=f)
    strips.append(canvas)
w = strips[0].width; h = strips[0].height
gap = 4
mont = Image.new("RGB", (w, h*len(strips)+gap*(len(strips)-1)), (255,255,255))
for i, s in enumerate(strips):
    mont.paste(s, (0, i*(h+gap)))
mont_path = OUT / "montage.png"
mont.save(mont_path)
print("wrote", mont_path)
PY

echo ""
echo "=== diag sweep done: ${OUT}/montage.png ==="
ls -la "${OUT}"/*.png
