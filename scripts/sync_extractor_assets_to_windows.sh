#!/usr/bin/env bash
# Copy extractor weights + source dependencies to the Windows GPU box.
# Idempotent (scp overwrites unchanged files; total transfer ~500 MB).

set -euo pipefail

REMOTE="videocard@192.168.87.25"
DST_ROOT="C:/arc_distill/repo_assets"

ssh "$REMOTE" "mkdir C:\\arc_distill\\repo_assets 2>nul; mkdir C:\\arc_distill\\repo_assets\\vendor 2>nul; mkdir C:\\arc_distill\\repo_assets\\vendor\\FairFace 2>nul; mkdir C:\\arc_distill\\repo_assets\\models 2>nul; mkdir C:\\arc_distill\\repo_assets\\models\\blendshape_nmf 2>nul; mkdir C:\\arc_distill\\repo_assets\\src 2>nul; exit 0"

echo "[sync] vendor/weights -> $DST_ROOT/vendor/weights"
scp -r vendor/weights                 "$REMOTE:$DST_ROOT/vendor/weights"

echo "[sync] vendor/MiVOLO -> $DST_ROOT/vendor/MiVOLO"
scp -r vendor/MiVOLO                  "$REMOTE:$DST_ROOT/vendor/MiVOLO"

echo "[sync] vendor/FairFace/dlib_models -> $DST_ROOT/vendor/FairFace/dlib_models"
scp -r vendor/FairFace/dlib_models    "$REMOTE:$DST_ROOT/vendor/FairFace/dlib_models"

echo "[sync] models/mediapipe -> $DST_ROOT/models/mediapipe"
scp -r models/mediapipe               "$REMOTE:$DST_ROOT/models/mediapipe"

echo "[sync] models/blendshape_nmf/au_library.npz"
scp models/blendshape_nmf/au_library.npz "$REMOTE:$DST_ROOT/models/blendshape_nmf/au_library.npz"

echo "[sync] src/demographic_pc -> $DST_ROOT/src/demographic_pc"
scp -r src/demographic_pc             "$REMOTE:$DST_ROOT/src/demographic_pc"

echo "[sync] done"
