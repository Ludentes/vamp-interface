@echo off
REM Validate noise-conditional arc adapter at t-buckets {0, 0.1, 0.25, 0.5, 0.75, 1.0}.
REM Gates: T1 — t=0 within 0.01 of 0.881 baseline; T2 — every t within 0.05 of t=0.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t 2>nul
del C:\arc_distill\arc_validate_latent_a2_t.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.validate_vs_t ^
    --variant latent_a2_full_native_shallow ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t\validation_report_vs_t.json ^
    --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t\validate.log
    exit /b 1
)

echo arc_validate_latent_a2_t_done > C:\arc_distill\arc_validate_latent_a2_t.done
