#!/usr/bin/env bash
# Wide α × chibi-strength sweep for the analytical lid-boost formula.
#
#   boost_i = mean(|det J| over lid_mask)^α  if i ∈ lid_mask  else 1.0
#   scale_ratio_final = chibi_scale_ratio · boost   (per-vert)
#
# For each chibi strength s ∈ {1.5, 1.75, 2.0, 2.5}:
#   1. Build chibi assets if missing (mesh, arkit_bs, scale_ratio).
#   2. Build J_per_vert.npy if missing.
#   3. Lid mask is animation-derived, deformation-invariant — reuse me_s2.0/lid_mask_20018.npy.
#   4. For each α ∈ {0.40, 0.50, 0.69}: emit scale_ratio variant, render full-take normal video.
#
# Normal-color only (magenta over-reads — confirmed 2026-05-14).
# Output: exp_output/lam_chibi/renders/alpha_sweep/s{S}_a{A}.mp4 + summary collage.
set -u
VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
STEM=me
ANCHOR=${VAMP}/exp_output/lam_chibi/user_anchor/${STEM}.png
BAKED=${VAMP}/exp_output/lam_chibi/${STEM}/${STEM}_textured_mesh_display.obj
TEMPLATE=${LAM}/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj
ARKIT_BS=${LAM}/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy
MOTION=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600/MySlate_2_arkit/export/MySlate_2
LID_MASK=${VAMP}/exp_output/lam_chibi/me_s2.0/lid_mask_20018.npy
OUT=${VAMP}/exp_output/lam_chibi/renders/alpha_sweep
LOGS=${OUT}/_logs
mkdir -p "${OUT}" "${LOGS}"

export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

S_LIST=(${S_LIST:-0.7 1.0 1.25 1.5 1.75 2.0 2.5})
A_LIST=(${A_LIST:-0.50})

cd "${LAM}"

for S in "${S_LIST[@]}"; do
  ASSET_DIR=${VAMP}/exp_output/lam_chibi/me_s${S}
  mkdir -p "${ASSET_DIR}"

  # 1. Build chibi mesh + arkit_bs + scale_ratio if missing.
  if [[ ! -f "${ASSET_DIR}/chibi_scale_ratio.npy" ]]; then
    echo "[build] chibi assets s=${S}"
    python "${VAMP}/scripts/chibi_make_assets.py" \
      --template "${TEMPLATE}" \
      --arkit_bs "${ARKIT_BS}" \
      --baked    "${BAKED}" \
      --outdir   "${ASSET_DIR}" \
      --y_anchor_frac_baked 0.20 \
      --chibi_strength "${S}" \
      > "${LOGS}/build_s${S}.log" 2>&1 \
      || { echo "  FAIL build s=${S}"; tail -20 "${LOGS}/build_s${S}.log"; exit 2; }
  fi

  # 2. Build J if missing.
  if [[ ! -f "${ASSET_DIR}/J_per_vert.npy" ]]; then
    echo "[build] J s=${S}"
    python "${VAMP}/scripts/chibi_make_j_assets.py" \
      --canonical "${BAKED}" \
      --deformed  "${ASSET_DIR}/chibi_textured_mesh.obj" \
      --template  "${TEMPLATE}" \
      --out       "${ASSET_DIR}/J_per_vert.npy" \
      > "${LOGS}/J_s${S}.log" 2>&1 \
      || { echo "  FAIL J s=${S}"; tail -20 "${LOGS}/J_s${S}.log"; exit 2; }
  fi

  # 3. For each α, emit formula-driven scale_ratio variant.
  for A in "${A_LIST[@]}"; do
    VARIANT=${ASSET_DIR}/chibi_scale_ratio_a${A}.npy
    if [[ ! -f "${VARIANT}" ]]; then
      python - <<PY
import numpy as np
J = np.load("${ASSET_DIR}/J_per_vert.npy")
r = np.load("${ASSET_DIR}/chibi_scale_ratio.npy").astype(np.float32)
lid = np.load("${LID_MASK}").astype(np.int64)
det = np.abs(np.linalg.det(J.astype(np.float64)))
strength = float(np.mean(det[lid]))
boost = strength ** ${A}
boost = max(1.0, boost)
out = r.copy()
out[lid] = (r[lid] * boost).astype(np.float32)
print(f"s=${S} α=${A}: mean_det_lid={strength:.3f} boost={boost:.3f}  scale_ratio[lid] median={np.median(out[lid]):.3f} (was {np.median(r[lid]):.3f})")
np.save("${VARIANT}", out)
PY
    fi

    TAG=s${S}_a${A}
    OUT_MP4=${OUT}/${TAG}.mp4
    if [[ -f "${OUT_MP4}" ]]; then
      echo "[skip ] ${TAG} (mp4 exists)"
      continue
    fi

    export LAM_CHIBI_ARKIT_BS=${ASSET_DIR}/chibi_arkit_bs.npy
    export LAM_EDIT_XYZ_OBJ=${ASSET_DIR}/chibi_textured_mesh.obj
    export LAM_CHIBI_SCALE_RATIO=${VARIANT}
    unset LAM_EDIT_VERTEX_COLORS_OBJ LAM_AXIS_BOOST_NPY LAM_OPACITY_MUL_NPY \
          LAM_J_PER_VERT_NPY LAM_AUX_SPLATS_PLY LAM_CHIBI_SCALE_BOOST

    rm -f ${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4
    echo "[render] ${TAG}  $(date +%H:%M:%S)"
    t0=$(date +%s)
    if bash scripts/inference.sh \
        configs/inference/lam-20k-8gpu.yaml \
        model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
        "${ANCHOR}" \
        "${MOTION}/" \
        > "${LOGS}/render_${TAG}.log" 2>&1; then
      cp ${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4 "${OUT_MP4}"
      echo "  → ${OUT_MP4} ($(($(date +%s)-t0))s)"
    else
      echo "  FAIL ${TAG} (log: ${LOGS}/render_${TAG}.log)"
      tail -25 "${LOGS}/render_${TAG}.log"
    fi
  done
done

# Summary collage: first frame of every render, labeled, grid = strengths × alphas.
python - <<'PY'
import os, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
OUT = Path("/home/newub/w/vamp-interface/exp_output/lam_chibi/renders/alpha_sweep")
S_LIST = ["1.5","1.75","2.0","2.5"]
A_LIST = ["0.40","0.50","0.69"]
TILE_W = 384; TILE_H = 384
LABEL_H = 28
# Sample frame 328 (full blink) from each video.
FRAME_N = 328
def get_frame(p):
    out = p.with_suffix(".png")
    if not out.exists():
        subprocess.run(["ffmpeg","-loglevel","error","-y","-i",str(p),
                        "-vf",f"select=eq(n\\,{FRAME_N})","-frames:v","1",str(out)],
                       check=False)
    if not out.exists():
        return None
    return Image.open(out).convert("RGB").resize((TILE_W, TILE_H))
try:
    F = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
except OSError:
    F = ImageFont.load_default()
canvas = Image.new("RGB",
    (TILE_W*len(A_LIST)+80, (TILE_H+LABEL_H)*len(S_LIST)+40),
    (255,255,255))
d = ImageDraw.Draw(canvas)
for j, A in enumerate(A_LIST):
    d.text((80 + j*TILE_W + TILE_W//2 - 30, 8), f"α={A}", fill=(0,0,0), font=F)
for i, S in enumerate(S_LIST):
    d.text((4, 40 + i*(TILE_H+LABEL_H) + TILE_H//2), f"s={S}", fill=(0,0,0), font=F)
    for j, A in enumerate(A_LIST):
        tag = f"s{S}_a{A}"
        p = OUT / f"{tag}.mp4"
        if not p.exists():
            continue
        im = get_frame(p)
        if im is None: continue
        x = 80 + j*TILE_W
        y = 40 + i*(TILE_H+LABEL_H)
        canvas.paste(im, (x, y))
        d.text((x+4, y+TILE_H+4), tag, fill=(0,0,0), font=F)
collage = OUT / "alpha_sweep_collage.png"
canvas.save(collage)
print("wrote", collage)
PY

echo ""
echo "=== α sweep done: ${OUT}/alpha_sweep_collage.png ==="
ls -la "${OUT}"/*.mp4
