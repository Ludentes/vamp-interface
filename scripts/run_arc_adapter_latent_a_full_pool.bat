@echo off
REM Latent-A-full-pool: adaptive-avg-pool 64->14 + LatentStemNative on the
REM (16, 64, 64) full-image Flux VAE latent corpus. Frozen-backbone baseline.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a_full_pool 2>nul
del C:\arc_distill\arc_adapter_latent_a_full_pool.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a_full_pool ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a_full_pool ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\latent_a_full_pool\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a_full_pool\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a_full_pool ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a_full_pool\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a_full_pool\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a_full_pool\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a_full_pool\train.log
    exit /b 1
)

echo arc_adapter_latent_a_full_pool_done > C:\arc_distill\arc_adapter_latent_a_full_pool.done
