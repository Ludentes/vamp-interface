@echo off
REM v2e: U-Net decoder for landmark head — overnight 80-epoch run.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v2e 2>nul
del C:\arc_distill\mediapipe_distill_bs_v2e.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train_v2e ^
    --variant bs_v2e ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --landmarks C:\arc_distill\arc_full_latent\compact_landmarks.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --rendered-landmarks C:\arc_distill\arc_full_latent\compact_rendered_landmarks.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_v2e ^
    --epochs 120 --batch-size 128 --workers 0 ^
    --lambda-lmk 0.5 --lambda-bs 1.0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\mediapipe_distill\bs_v2e\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_v2e\train.log
    exit /b 1
)

echo mediapipe_distill_bs_v2e_done > C:\arc_distill\mediapipe_distill_bs_v2e.done
