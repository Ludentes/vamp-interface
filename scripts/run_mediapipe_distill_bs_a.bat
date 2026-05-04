@echo off
REM Train MediaPipe-blendshape student bs_a on (16, 64, 64) FFHQ latents +
REM 52-d ARKit blendshape labels joined from reverse_index.parquet.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_a 2>nul
del C:\arc_distill\mediapipe_distill_bs_a.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train ^
    --variant bs_a ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_a ^
    --epochs 20 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\mediapipe_distill\bs_a\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_a\train.log
    exit /b 1
)

echo mediapipe_distill_bs_a_done > C:\arc_distill\mediapipe_distill_bs_a.done
