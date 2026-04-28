@echo off
REM Latent-A2 full-native shallow: latent_a_full_native stem + IResNet50
REM layer-1 unfrozen, init-from latent_a_full_native checkpoint.
REM Stem LR 1e-3, backbone LR 1e-4. (16, 64, 64) full-image input.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow 2>nul
del C:\arc_distill\arc_adapter_latent_a2_full_native_shallow.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant latent_a2_full_native_shallow ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow ^
    --init-from C:\arc_distill\arc_adapter\latent_a_full_native\checkpoint.pt ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --backbone-lr 1e-4 --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant latent_a2_full_native_shallow ^
    --checkpoint C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\train.log
    exit /b 1
)

echo arc_adapter_latent_a2_full_native_shallow_done > C:\arc_distill\arc_adapter_latent_a2_full_native_shallow.done
