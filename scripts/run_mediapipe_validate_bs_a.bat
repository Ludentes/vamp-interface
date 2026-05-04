@echo off
REM Validate mediapipe_distill bs_a checkpoint against the gates.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
del C:\arc_distill\mediapipe_validate_bs_a.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_as_loss ^
    --variant bs_a ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_a\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --out-json C:\arc_distill\mediapipe_distill\bs_a\validation_report.json ^
    --atom-library C:\arc_distill\au_library.npz ^
    --device cuda > C:\arc_distill\mediapipe_distill\bs_a\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\mediapipe_distill\bs_a\validate.log
    exit /b 1
)

echo mediapipe_validate_bs_a_done > C:\arc_distill\mediapipe_validate_bs_a.done
