@echo off
REM Latent-A-aligned14: train latent_a_native variant on similarity-aligned
REM (16, 14, 14) latents derived from the (16, 64, 64) corpus + kps_5.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a_aligned14 2>nul
del C:\arc_distill\arc_adapter_latent_a_aligned14.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a_native ^
    --compact C:\arc_distill\arc_full_latent\compact_aligned14.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a_aligned14 ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\latent_a_aligned14\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a_aligned14\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a_native ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a_aligned14\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact_aligned14.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a_aligned14\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a_aligned14\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a_aligned14\train.log
    exit /b 1
)

echo arc_adapter_latent_a_aligned14_done > C:\arc_distill\arc_adapter_latent_a_aligned14.done
