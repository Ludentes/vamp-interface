@echo off
REM v2b: train atom-only student (8-d NMF atoms) on FFHQ + rendered.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v2b 2>nul
del C:\arc_distill\mediapipe_distill_bs_v2b.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train_v2b ^
    --variant bs_v2b ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --atoms C:\arc_distill\arc_full_latent\compact_atoms.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --rendered-atoms C:\arc_distill\arc_full_latent\compact_rendered_atoms.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_v2b ^
    --epochs 20 --batch-size 128 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\mediapipe_distill\bs_v2b\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_v2b\train.log
    exit /b 1
)

echo mediapipe_distill_bs_v2b_done > C:\arc_distill\mediapipe_distill_bs_v2b.done
