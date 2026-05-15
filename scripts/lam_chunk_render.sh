#!/usr/bin/env bash
# Chunked full-take LAM ARKit render.
#
# LAM's infer_single_view materializes every rendered frame in CUDA memory
# before concat (modeling_lam.py:332), so 3720 frames OOMs a 32 GiB card.
# This driver slices a motion folder into N-frame chunks, runs inference per
# chunk, then ffmpeg-concats the per-chunk mp4s into one full-take video.
#
# Resumable: per-chunk mp4s are kept; existing chunks are skipped.
# Silent output: audio mux happens post-hoc from the source .mov if desired.
#
# Usage:
#   lam_chunk_render.sh <full_motion_dir> <anchor_path> <out_mp4> [chunk_size=600]

set -u

FULL_MOTION_DIR=$1   # e.g. exp_output/lam_bakeoff/take_runs_arkit/MySlate_2_arkit/export/MySlate_2
ANCHOR_PATH=$2       # e.g. exp_output/lam_bakeoff/take_runs/anchors/asian_m.png
OUT_MP4=$3           # e.g. exp_output/lam_bakeoff/take_renders/take2__asian_m__arkit.mp4
CHUNK_SIZE=${4:-600}

VAMP=/home/newub/w/vamp-interface
LAM=/home/newub/w/LAM

ANCHOR_STEM=$(basename "${ANCHOR_PATH}" .png)
OUT_STEM=$(basename "${OUT_MP4}" .mp4)
CHUNK_ROOT=${VAMP}/exp_output/lam_bakeoff/_chunk_render/${OUT_STEM}
CHUNK_MOTION_ROOT=${CHUNK_ROOT}/motion
CHUNK_VIDEO_DIR=${CHUNK_ROOT}/videos
mkdir -p "${CHUNK_MOTION_ROOT}" "${CHUNK_VIDEO_DIR}"

# --- env -------------------------------------------------------------
source /home/newub/miniconda3/etc/profile.d/conda.sh
conda activate lam
export PATH=/home/newub/miniconda3/envs/lam/bin:$PATH
export PYTHONPATH=${LAM}
export XFORMERS_DISABLED=1
export LAM_USE_ARKIT=1

# --- Stage 1: build chunked motion folders --------------------------
python - <<PY
import json, shutil
from pathlib import Path

src = Path("${FULL_MOTION_DIR}")
chunk_root = Path("${CHUNK_MOTION_ROOT}")
chunk_size = ${CHUNK_SIZE}

src_canon = src / "canonical_flame_param.npz"
src_tr = src / "transforms.json"
frames = sorted((src / "flame_param").glob("*.npz"))
n = len(frames)
n_chunks = (n + chunk_size - 1) // chunk_size
print(f"src has {n} frames → {n_chunks} chunks of {chunk_size}")

with open(src_tr) as f:
    tr_all = json.load(f)
tr_frames = sorted(tr_all["frames"], key=lambda x: x["flame_param_path"])
assert len(tr_frames) == n, f"transforms.json frames ({len(tr_frames)}) != flame_param count ({n})"

for ci in range(n_chunks):
    a, b = ci * chunk_size, min((ci + 1) * chunk_size, n)
    cdir = chunk_root / f"chunk_{ci:03d}" / "export" / src.name
    fp = cdir / "flame_param"
    if (cdir / "canonical_flame_param.npz").exists() and len(list(fp.glob("*.npz"))) == (b - a):
        continue
    fp.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_canon, cdir / "canonical_flame_param.npz")
    for frame in frames[a:b]:
        dst_path = fp / frame.name
        if not dst_path.exists():
            shutil.copy2(frame, dst_path)
    tr_chunk = dict(tr_all)
    tr_chunk["frames"] = tr_frames[a:b]
    with open(cdir / "transforms.json", "w") as f:
        json.dump(tr_chunk, f, indent=2)
    print(f"  chunk {ci:03d}: frames {a}..{b - 1} ({b - a} frames)")
PY

# --- Stage 2: render per chunk --------------------------------------
cd "${LAM}"
N_CHUNKS=$(ls -d "${CHUNK_MOTION_ROOT}"/chunk_*/ | wc -l)
echo "rendering ${N_CHUNKS} chunks"

for CDIR in "${CHUNK_MOTION_ROOT}"/chunk_*/; do
  CI=$(basename "${CDIR}" | sed 's/chunk_//')
  CHUNK_MOTION="${CDIR}/export/$(basename "${FULL_MOTION_DIR}")/"
  CHUNK_OUT="${CHUNK_VIDEO_DIR}/chunk_${CI}.mp4"
  CHUNK_LOG="${CHUNK_ROOT}/chunk_${CI}.log"

  if [[ -f "${CHUNK_OUT}" ]]; then
    echo "[skip] chunk ${CI} already rendered"
    continue
  fi

  t0=$(date +%s)
  echo "[chunk ${CI}] rendering..."
  if bash scripts/inference.sh \
      configs/inference/lam-20k-8gpu.yaml \
      model_zoo/lam_models/releases/lam/lam-20k/step_045500/ \
      "${ANCHOR_PATH}" \
      "${CHUNK_MOTION}" \
      > "${CHUNK_LOG}" 2>&1; then
    SRC="${LAM}/exps/videos/lam/lam_20k/${ANCHOR_STEM}.mp4"
    if [[ -f "${SRC}" ]]; then
      mv "${SRC}" "${CHUNK_OUT}"
      t1=$(date +%s)
      echo "[chunk ${CI}] done in $((t1 - t0))s → ${CHUNK_OUT}"
    else
      echo "[chunk ${CI}] NO OUTPUT (log: ${CHUNK_LOG})"; exit 2
    fi
  else
    echo "[chunk ${CI}] FAILED (log: ${CHUNK_LOG})"; exit 3
  fi
done

# --- Stage 3: ffmpeg concat -----------------------------------------
CONCAT_LIST="${CHUNK_ROOT}/concat.txt"
: > "${CONCAT_LIST}"
for F in "${CHUNK_VIDEO_DIR}"/chunk_*.mp4; do
  echo "file '${F}'" >> "${CONCAT_LIST}"
done
mkdir -p "$(dirname "${OUT_MP4}")"
ffmpeg -y -f concat -safe 0 -i "${CONCAT_LIST}" -c copy "${OUT_MP4}" < /dev/null > "${CHUNK_ROOT}/concat.log" 2>&1 \
  || { echo "ffmpeg concat failed (log: ${CHUNK_ROOT}/concat.log)"; exit 4; }

echo "=== DONE: ${OUT_MP4} ==="
ls -la "${OUT_MP4}"
