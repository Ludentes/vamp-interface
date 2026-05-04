@echo off
REM sg_c — noise-conditional SigLIP distill (Path 2 from
REM docs/research/2026-04-30-noise-conditional-distill-design.md).
REM Same v2c trunk + L2-norm head as sg_b, plus per-block FiLM
REM modulation conditioned on t-embedding. Random-t rectified-flow noise
REM (z_t = (1-t)*z_0 + t*eps) sampled per batch.
REM
REM Trains 60 epochs (vs sg_b's 20 — noise-conditional has wider input
REM manifold to fit). ~70-80s/epoch on RTX 4090 → ~75 min.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\siglip_distill\sg_c 2>nul
del C:\arc_distill\siglip_distill_sg_c.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m siglip_distill.train_t ^
    --variant sg_c ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --siglip C:\arc_distill\siglip_distill\compact_siglip.pt ^
    --out-dir C:\arc_distill\siglip_distill\sg_c ^
    --epochs 60 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\siglip_distill\sg_c\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\siglip_distill\sg_c\train.log
    exit /b 1
)

echo siglip_distill_sg_c_done > C:\arc_distill\siglip_distill_sg_c.done
