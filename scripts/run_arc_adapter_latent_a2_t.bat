@echo off
REM Path 1 noise-conditional retrain of latent_a2_full_native_shallow.
REM Same architecture as the shipped 0.881-cos clean-only adapter, but
REM training inputs are partially noised (z_t = (1-t)*z_0 + t*eps,
REM t ~ Uniform(0,1)). Warm-start from the shipped checkpoint so we don't
REM re-learn the clean Flux latent → ArcFace map from scratch.
REM 40 epochs, ~50-60 min on RTX 4090.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t 2>nul
del C:\arc_distill\arc_adapter_latent_a2_full_native_shallow_t.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter_t ^
    --variant latent_a2_full_native_shallow ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t ^
    --init-from C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\checkpoint.pt ^
    --epochs 40 --batch-size 256 --workers 0 ^
    --lr 1e-3 --backbone-lr 1e-4 --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t\train.log
    exit /b 1
)

echo arc_adapter_latent_a2_full_native_shallow_t_done > C:\arc_distill\arc_adapter_latent_a2_full_native_shallow_t.done
