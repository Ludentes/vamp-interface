@echo off
REM Encode FFHQ-512 full images to (16, 64, 64) Flux VAE latents.
REM Pairs each row in compact.pt (25,696 SCRFD-detected aligned crops + arcface)
REM with the full-image latent of the same SHA from the FFHQ parquet shards.
REM Output:  C:\arc_distill\arc_full_latent\compact.pt  (~3.4 GB).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_full_latent 2>nul
del C:\arc_distill\encode_full_latent.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.encode_full_to_latent ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --ffhq-parquet-dir C:\arc_distill\ffhq_parquet\data ^
    --vae C:\comfy\ComfyUI\models\vae\FLUX1\ae.safetensors ^
    --out C:\arc_distill\arc_full_latent\compact.pt ^
    --resolution 512 --batch-size 32 --device cuda > C:\arc_distill\arc_full_latent\encode.log 2>&1
if errorlevel 1 (
    echo encode_full_latent failed >> C:\arc_distill\arc_full_latent\encode.log
    exit /b 1
)

echo encode_full_latent_done > C:\arc_distill\encode_full_latent.done
