@echo off
REM Latent-A2 shallow: latent_a_native stem + IResNet50 layer-1 unfrozen,
REM initialised from latent_a_native's checkpoint. Stem LR 1e-3, backbone LR 1e-4.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_shallow 2>nul
del C:\arc_distill\arc_adapter_latent_a2_shallow.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a2_shallow ^
    --compact C:\arc_distill\arc_pixel_aligned\compact_latent.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a2_shallow ^
    --init-from C:\arc_distill\arc_adapter\latent_a_native\checkpoint.pt ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --backbone-lr 1e-4 --device cuda > C:\arc_distill\arc_adapter\latent_a2_shallow\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a2_shallow\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a2_shallow ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a2_shallow\checkpoint.pt ^
    --compact C:\arc_distill\arc_pixel_aligned\compact_latent.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a2_shallow\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a2_shallow\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a2_shallow\train.log
    exit /b 1
)

echo arc_adapter_latent_a2_shallow_done > C:\arc_distill\arc_adapter_latent_a2_shallow.done
