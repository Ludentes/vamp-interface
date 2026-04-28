@echo off
REM Similarity-warp (16, 64, 64) → (16, 14, 14) aligned crops using kps_5
REM from face_attrs.pt (matches insightface face_align.norm_crop in latent space).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
del C:\arc_distill\precompute_aligned_latents.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.precompute_aligned_latents ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --face-attrs C:\arc_distill\arc_full_latent\face_attrs.pt ^
    --out C:\arc_distill\arc_full_latent\compact_aligned14.pt > C:\arc_distill\arc_full_latent\precompute_aligned.log 2>&1
if errorlevel 1 (
    echo precompute failed >> C:\arc_distill\arc_full_latent\precompute_aligned.log
    exit /b 1
)

echo precompute_aligned_latents_done > C:\arc_distill\precompute_aligned_latents.done
