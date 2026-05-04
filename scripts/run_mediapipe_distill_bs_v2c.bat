@echo off
REM v2c: train MediaPipe-blendshape student bs_a on FFHQ + rendered combined
REM corpus (~32K rows). Same model architecture as v1; broader expression
REM coverage from the slider-render corpus should help do_not_ship channels.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v2c 2>nul
del C:\arc_distill\mediapipe_distill_bs_v2c.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train ^
    --variant bs_a ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_v2c ^
    --epochs 20 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\mediapipe_distill\bs_v2c\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_v2c\train.log
    exit /b 1
)

echo mediapipe_distill_bs_v2c_done > C:\arc_distill\mediapipe_distill_bs_v2c.done
