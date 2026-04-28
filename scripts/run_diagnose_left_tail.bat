@echo off
REM Diagnose left tail of A2-full-native cos distribution.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\diagnose 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.diagnose_left_tail ^
    --variant latent_a2_full_native_shallow ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --ffhq-parquet-dir C:\arc_distill\ffhq_parquet\data ^
    --out-dir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\diagnose ^
    --k 50 --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\diagnose\diagnose.log 2>&1
