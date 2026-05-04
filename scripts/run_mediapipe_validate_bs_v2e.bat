@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
del C:\arc_distill\mediapipe_validate_bs_v2e.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_as_loss ^
    --variant bs_v2e ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_v2e\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --out-json C:\arc_distill\mediapipe_distill\bs_v2e\validation_report.json ^
    --atom-library C:\arc_distill\au_library.npz ^
    --device cuda > C:\arc_distill\mediapipe_distill\bs_v2e\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\mediapipe_distill\bs_v2e\validate.log
    exit /b 1
)

echo mediapipe_validate_bs_v2e_done > C:\arc_distill\mediapipe_validate_bs_v2e.done
