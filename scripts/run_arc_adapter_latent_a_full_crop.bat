@echo off
REM Latent-A-full-crop: center-crop (16,64,64)->(16,32,32) + ConvTranspose
REM stride-4 (32->128) + crop to 112. Aligns student input window to teacher's
REM tightly-cropped face view via FFHQ's consistent face centering.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a_full_crop 2>nul
del C:\arc_distill\arc_adapter_latent_a_full_crop.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a_full_crop ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a_full_crop ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\latent_a_full_crop\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a_full_crop\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a_full_crop ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a_full_crop\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a_full_crop\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a_full_crop\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a_full_crop\train.log
    exit /b 1
)

echo arc_adapter_latent_a_full_crop_done > C:\arc_distill\arc_adapter_latent_a_full_crop.done
