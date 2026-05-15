#!/usr/bin/env bash
# ARKit-LAM spike sweep — paired with lam_take_sweep.sh.
#
# Renders the same (anchor, take) pairs as the PCA sweep but with the LLF-CSV
# ARKit motion folder produced by scripts/llf_csv_to_lam_motion.py, and with
# LAM_USE_ARKIT=1 so gs_renderer.py imports FlameHeadSubdivided from the
# flame_arkit variant.
#
# Designed to run AFTER the PCA sweep completes (so we have apples-to-apples
# baselines in take_renders/take{N}__{anchor}.mp4). Idempotent: re-runs skip
# completed renders.

set -u

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM
SWEEP_ROOT=${VAMP}/exp_output/lam_bakeoff/take_runs
ARKIT_ROOT=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit_600
RENDER_DIR=${VAMP}/exp_output/lam_bakeoff/take_renders
LOG_DIR=${SWEEP_ROOT}/_logs

mkdir -p "${ARKIT_ROOT}" "${RENDER_DIR}" "${LOG_DIR}"

# Minimum spike set: take 2 (long, clean LLF/MOV match), 2 anchors. Add takes
# 3 / 5 / 7 here to expand once the minimum passes.
ANCHORS=(
  "asian_m:${SWEEP_ROOT}/anchors/asian_m.png"
  "young_european_f:${SWEEP_ROOT}/anchors/young_european_f.png"
)
TAKES=(2)

# --- env -------------------------------------------------------------
source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

cd "${LAM}"

# --- Stage 1: build ARKit motion folders from LLF CSVs --------------
for T in "${TAKES[@]}"; do
  CSV="${VAMP}/data/llf-takes/20260505_MySlate_${T}/MySlate_${T}_iPhone.csv"
  VHAP_DIR="${SWEEP_ROOT}/MySlate_${T}_full/export/MySlate_${T}"
  OUT_DIR="${ARKIT_ROOT}/MySlate_${T}_arkit/export/MySlate_${T}"

  if [[ ! -f "${VHAP_DIR}/canonical_flame_param.npz" ]]; then
    echo "[skip] no VHAP motion dir for take ${T} (waiting for PCA sweep)"
    continue
  fi
  if [[ ! -f "${CSV}" ]]; then
    echo "[skip] no LLF CSV for take ${T}"
    continue
  fi

  if [[ -f "${OUT_DIR}/canonical_flame_param.npz" ]]; then
    echo "[skip] ARKit motion dir exists for take ${T}"
    continue
  fi

  echo "[take ${T}] building ARKit motion folder from LLF CSV"
  python "${VAMP}/scripts/llf_csv_to_lam_motion.py" \
    --csv "${CSV}" \
    --vhap_motion_dir "${VHAP_DIR}" \
    --out_motion_dir "${OUT_DIR}" \
    --decimate 2
done

# --- Stage 2: LAM render per (anchor, take) with LAM_USE_ARKIT=1 ----
for T in "${TAKES[@]}"; do
  MOTION_DIR="${ARKIT_ROOT}/MySlate_${T}_arkit/export/MySlate_${T}"
  if [[ ! -f "${MOTION_DIR}/canonical_flame_param.npz" ]]; then
    echo "[render skip] no ARKit motion dir for take ${T}"
    continue
  fi

  for ENTRY in "${ANCHORS[@]}"; do
    A_NAME="${ENTRY%%:*}"
    A_PATH="${ENTRY##*:}"
    OUT="${RENDER_DIR}/take${T}__${A_NAME}__arkit600.mp4"

    if [[ -f "${OUT}" ]]; then
      echo "[skip] render exists: ${OUT}"
      continue
    fi

    LOG="${LOG_DIR}/take${T}__${A_NAME}_arkit600_render.log"
    t0=$(date +%s)
    echo "[render] take${T}/${A_NAME}/arkit600 starting (LAM_USE_ARKIT=1)"
    if bash scripts/inference.sh \
        configs/inference/lam-20k-8gpu.yaml \
        model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
        "${A_PATH}" \
        "${MOTION_DIR}/" \
        > "${LOG}" 2>&1; then
      STEM=$(basename "${A_PATH}" .png)
      SRC="${LAM}/exps/videos/lam/lam_20k/${STEM}.mp4"
      if [[ -f "${SRC}" ]]; then
        cp "${SRC}" "${OUT}"
        # Audio mux variant alongside, mirroring sweep convention
        SRC_AUDIO="${LAM}/exps/videos/lam/lam_20k/${STEM}_audio.mp4"
        if [[ -f "${SRC_AUDIO}" ]]; then
          cp "${SRC_AUDIO}" "${RENDER_DIR}/take${T}__${A_NAME}__arkit600_audio.mp4"
        fi
        t1=$(date +%s)
        echo "[render] take${T}/${A_NAME}/arkit600 done in $((t1 - t0))s → ${OUT}"
      else
        echo "[render] take${T}/${A_NAME}/arkit600 no output mp4 (log: ${LOG})"
      fi
    else
      echo "[render] take${T}/${A_NAME}/arkit600 FAILED (log: ${LOG})"
    fi
  done
done

echo "=== ARKIT SPIKE SWEEP COMPLETE ==="
ls -la "${RENDER_DIR}"/ | grep -E "arkit|take${TAKES[0]}__" || true
