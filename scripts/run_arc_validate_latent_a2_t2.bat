@echo off
REM Validate Path 2 noise-conditional adapter at t-buckets.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2 2>nul
del C:\arc_distill\arc_validate_latent_a2_t2.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.validate_vs_t2 ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2\validation_report_vs_t.json ^
    --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2\validate.log
    exit /b 1
)

echo arc_validate_latent_a2_t2_done > C:\arc_distill\arc_validate_latent_a2_t2.done
