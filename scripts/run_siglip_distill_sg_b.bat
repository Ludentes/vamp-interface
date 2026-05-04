@echo off
REM v2 SigLIP-distill (sg_b): same v2c trunk as sg_a, but the head is followed
REM by F.normalize → unit-norm output by construction. Loss simplifies to
REM (1 - cos). Eliminates the pred_norm < 1 magnitude bias that polluted
REM v1's per-probe R² (see 2026-04-30 train-vs-val R² diagnostic).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\siglip_distill\sg_b 2>nul
del C:\arc_distill\siglip_distill_sg_b.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m siglip_distill.train ^
    --variant sg_b ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --siglip C:\arc_distill\siglip_distill\compact_siglip.pt ^
    --out-dir C:\arc_distill\siglip_distill\sg_b ^
    --epochs 20 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\siglip_distill\sg_b\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\siglip_distill\sg_b\train.log
    exit /b 1
)

echo siglip_distill_sg_b_done > C:\arc_distill\siglip_distill_sg_b.done
