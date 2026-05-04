@echo off
REM v3_t: noise-conditional MediaPipe-blendshape student. Path 1 (random-t
REM rectified-flow noise during training, no FiLM). Same bs_a architecture
REM and same combined corpus (FFHQ + rendered) as v2c. Warm-start from v2c
REM checkpoint to skip the clean-input learning curve.
REM 40 epochs, ~70-80s/epoch on RTX 4090 (similar to v2c).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v3_t 2>nul
del C:\arc_distill\mediapipe_distill_bs_v3_t.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train_t ^
    --variant bs_a ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_v3_t ^
    --init-from C:\arc_distill\mediapipe_distill\bs_v2c\checkpoint.pt ^
    --epochs 40 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\mediapipe_distill\bs_v3_t\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_v3_t\train.log
    exit /b 1
)

echo mediapipe_distill_bs_v3_t_done > C:\arc_distill\mediapipe_distill_bs_v3_t.done
