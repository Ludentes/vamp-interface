@echo off
REM Validate latent_a2_full_native_shallow as identity-preservation loss head.
REM Runs Layer 1.1 (teacher cosine), 1.2 (ridge transfer), 1.3 (augmentation
REM TAR@FAR + AUC) and Layer 2 (gradient sanity into latent).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
del C:\arc_distill\validate_full_native_a2.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.validate_as_loss ^
    --variant latent_a2_full_native_shallow ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --face-attrs C:\arc_distill\arc_full_latent\face_attrs.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\validation_report.json ^
    --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\validate.log 2>&1
if errorlevel 1 (
    echo validation failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\validate.log
    exit /b 1
)

echo validate_full_native_a2_done > C:\arc_distill\validate_full_native_a2.done
