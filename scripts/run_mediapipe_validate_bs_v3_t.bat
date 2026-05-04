@echo off
REM Validate noise-conditional MediaPipe student at t-buckets.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v3_t 2>nul
del C:\arc_distill\mediapipe_validate_bs_v3_t.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_vs_t ^
    --variant bs_a ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_v3_t\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out-json C:\arc_distill\mediapipe_distill\bs_v3_t\validation_report_vs_t.json ^
    --device cuda > C:\arc_distill\mediapipe_distill\bs_v3_t\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\mediapipe_distill\bs_v3_t\validate.log
    exit /b 1
)

echo mediapipe_validate_bs_v3_t_done > C:\arc_distill\mediapipe_validate_bs_v3_t.done
