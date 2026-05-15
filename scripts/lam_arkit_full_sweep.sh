#!/usr/bin/env bash
set -u
VAMP=/home/newub/w/vamp-interface
ARKIT_ROOT=${VAMP}/exp_output/lam_bakeoff/take_runs_arkit
RENDER_DIR=${VAMP}/exp_output/lam_bakeoff/take_renders
LOG_DIR=${VAMP}/exp_output/lam_bakeoff/take_runs/_logs
ANCHOR_DIR=${VAMP}/exp_output/lam_bakeoff/take_runs/anchors
mkdir -p "${RENDER_DIR}" "${LOG_DIR}"

ANCHORS=(asian_m young_european_f)
TAKES=(2 3 4)

# Ensure ARKit motion dirs exist for takes 3 and 4 (take 2 is pre-built).
for T in "${TAKES[@]}"; do
  CSV="${VAMP}/data/llf-takes/20260505_MySlate_${T}/MySlate_${T}_iPhone.csv"
  VHAP_DIR="${VAMP}/exp_output/lam_bakeoff/take_runs/MySlate_${T}_full/export/MySlate_${T}"
  OUT_DIR="${ARKIT_ROOT}/MySlate_${T}_arkit/export/MySlate_${T}"
  if [[ ! -f "${OUT_DIR}/canonical_flame_param.npz" ]]; then
    if [[ ! -f "${VHAP_DIR}/canonical_flame_param.npz" ]]; then
      echo "[skip] no VHAP for take ${T}"; continue
    fi
    if [[ ! -f "${CSV}" ]]; then
      echo "[skip] no LLF CSV for take ${T}"; continue
    fi
    echo "[take ${T}] building ARKit motion folder"
    source /home/newub/miniconda3/etc/profile.d/conda.sh
    conda activate lam
    python "${VAMP}/scripts/llf_csv_to_lam_motion.py" \
      --csv "${CSV}" --vhap_motion_dir "${VHAP_DIR}" \
      --out_motion_dir "${OUT_DIR}" --decimate 2
  fi
done

for T in "${TAKES[@]}"; do
  MOTION="${ARKIT_ROOT}/MySlate_${T}_arkit/export/MySlate_${T}"
  if [[ ! -f "${MOTION}/canonical_flame_param.npz" ]]; then
    echo "[render skip] no ARKit motion for take ${T}"; continue
  fi
  for A in "${ANCHORS[@]}"; do
    OUT="${RENDER_DIR}/take${T}__${A}__arkit.mp4"
    LOG="${LOG_DIR}/take${T}__${A}__arkit.log"
    if [[ -f "${OUT}" ]]; then
      echo "[skip] ${OUT}"; continue
    fi
    echo "[full] take${T}/${A} starting"
    bash "${VAMP}/scripts/lam_chunk_render.sh" \
      "${MOTION}" "${ANCHOR_DIR}/${A}.png" "${OUT}" 600 \
      > "${LOG}" 2>&1 \
      && echo "[full] take${T}/${A} → ${OUT}" \
      || echo "[full] take${T}/${A} FAILED (log: ${LOG})"
  done
done

echo "=== FULL SWEEP COMPLETE ==="
ls -la "${RENDER_DIR}"/take{2,3,4}__*__arkit.mp4 2>/dev/null
