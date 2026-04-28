@echo off
REM Latent-A-full-roi: per-row RoIAlign crop using SCRFD-detected face bbox
REM (16,64,64) -> (16,28,28), then ConvTranspose 28->112 stem.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a_full_roi 2>nul
del C:\arc_distill\arc_adapter_latent_a_full_roi.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a_full_roi ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --face-attrs C:\arc_distill\arc_full_latent\face_attrs.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a_full_roi ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\latent_a_full_roi\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a_full_roi\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a_full_roi ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a_full_roi\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --face-attrs C:\arc_distill\arc_full_latent\face_attrs.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a_full_roi\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a_full_roi\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a_full_roi\train.log
    exit /b 1
)

echo arc_adapter_latent_a_full_roi_done > C:\arc_distill\arc_adapter_latent_a_full_roi.done
