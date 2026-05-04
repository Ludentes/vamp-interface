@echo off
REM v1 SigLIP-distill: train SigLIPStudent sg_a on (latent, 1152-d siglip emb)
REM pairs over 26,108 FFHQ rows. Loss = 0.5*MSE + 0.5*(1-cos). v2c trunk +
REM Linear(512, 1152) head. ~12 M params, ~1 epoch fits in GPU memory at bs=128.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\siglip_distill\sg_a 2>nul
del C:\arc_distill\siglip_distill_sg_a.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m siglip_distill.train ^
    --variant sg_a ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --siglip C:\arc_distill\siglip_distill\compact_siglip.pt ^
    --out-dir C:\arc_distill\siglip_distill\sg_a ^
    --epochs 20 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\siglip_distill\sg_a\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\siglip_distill\sg_a\train.log
    exit /b 1
)

echo siglip_distill_sg_a_done > C:\arc_distill\siglip_distill_sg_a.done
