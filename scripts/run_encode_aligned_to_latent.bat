@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.encode_aligned_to_latent ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --vae C:\comfy\ComfyUI\models\vae\FLUX1\ae.safetensors ^
    --out C:\arc_distill\arc_pixel_aligned\compact_latent.pt ^
    --batch-size 64 --device cuda > C:\arc_distill\arc_pixel_aligned\encode_latent.log 2>&1
if errorlevel 1 (
    echo encode_latent failed >> C:\arc_distill\arc_pixel_aligned\encode_latent.log
    exit /b 1
)

echo encode_aligned_latent_done > C:\arc_distill\encode_aligned_latent.done
