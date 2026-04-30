@echo off
REM Validate bs_v4_pgd checkpoint under PGD attack at eps in
REM {0.0, 0.01, 0.03, 0.05, 0.1}. Also runs the same validator against
REM bs_v3_t for direct comparison (expected to collapse at eps>=0.03).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8

echo === bs_v3_t baseline (expected to collapse at eps^>=0.03) ===
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_pgd ^
    --variant bs_a ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_v3_t\final.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out C:\arc_distill\mediapipe_distill\bs_v3_t\validation_pgd.json ^
    --device cuda

echo.
echo === bs_v4_pgd (expected to be roughly flat across eps) ===
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.validate_pgd ^
    --variant bs_a ^
    --checkpoint C:\arc_distill\mediapipe_distill\bs_v4_pgd\final.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out C:\arc_distill\mediapipe_distill\bs_v4_pgd\validation_pgd.json ^
    --device cuda
