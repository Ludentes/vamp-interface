@echo off
REM Path 2 noise-conditional retrain: latent_a2_full_native_shallow + 5 FiLM hooks
REM at stage boundaries (Conv_0, Add_17, Add_38, Add_109, Add_125). Frozen
REM IResNet50 backbone weights stay frozen; only stem + layer1 + FiLM/t_embed train.
REM Warm-start from the shipped clean-only checkpoint.
REM 40 epochs, ~50-60 min on RTX 4090.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2 2>nul
del C:\arc_distill\arc_adapter_latent_a2_full_native_shallow_t2.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter_t2 ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2 ^
    --init-from C:\arc_distill\arc_adapter\latent_a2_full_native_shallow\checkpoint.pt ^
    --epochs 40 --batch-size 256 --workers 0 ^
    --lr 1e-3 --backbone-lr 1e-4 --device cuda > C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\latent_a2_full_native_shallow_t2\train.log
    exit /b 1
)

echo arc_adapter_latent_a2_full_native_shallow_t2_done > C:\arc_distill\arc_adapter_latent_a2_full_native_shallow_t2.done
