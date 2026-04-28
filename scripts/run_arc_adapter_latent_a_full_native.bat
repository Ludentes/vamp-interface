@echo off
REM Latent-A-full-native: ConvTranspose stride-2 (64->128) + center-crop to 112
REM on the (16, 64, 64) full-image Flux VAE latent corpus. No spatial pooling;
REM ~271K stem params matches LatentStemNative for fair compare with the
REM aligned-crop 14x14 baseline (val cos 0.805).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a_full_native 2>nul
del C:\arc_distill\arc_adapter_latent_a_full_native.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a_full_native ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a_full_native ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\latent_a_full_native\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a_full_native\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a_full_native ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a_full_native\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a_full_native\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a_full_native\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a_full_native\train.log
    exit /b 1
)

echo arc_adapter_latent_a_full_native_done > C:\arc_distill\arc_adapter_latent_a_full_native.done
