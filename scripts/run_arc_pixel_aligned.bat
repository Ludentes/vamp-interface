@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_pixel_aligned 2>nul

REM One-time pack: aligned 112-px crops via InsightFace
if not exist C:\arc_distill\arc_pixel_aligned\compact.pt (
    echo === preparing aligned compact dataset === > C:\arc_distill\arc_pixel_aligned\train.log
    C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.prepare_compact ^
        --shards-dir C:\arc_distill\ffhq_parquet\data ^
        --encoded-dir C:\arc_distill\encoded ^
        --out C:\arc_distill\arc_pixel_aligned\compact.pt ^
        --align >> C:\arc_distill\arc_pixel_aligned\train.log 2>&1
    if errorlevel 1 (
        echo prepare_compact failed >> C:\arc_distill\arc_pixel_aligned\train.log
        exit /b 1
    )
)

echo === training === >> C:\arc_distill\arc_pixel_aligned\train.log
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_pixel ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --out-dir C:\arc_distill\arc_pixel_aligned ^
    --epochs 8 --batch-size 256 --workers 0 ^
    --lr 3e-4 --device cuda >> C:\arc_distill\arc_pixel_aligned\train.log 2>&1
if errorlevel 1 (
    echo train_pixel failed >> C:\arc_distill\arc_pixel_aligned\train.log
    exit /b 1
)

echo === eval === >> C:\arc_distill\arc_pixel_aligned\train.log
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_pixel ^
    --checkpoint C:\arc_distill\arc_pixel_aligned\checkpoint.pt ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --out-json C:\arc_distill\arc_pixel_aligned\eval.json ^
    --device cuda >> C:\arc_distill\arc_pixel_aligned\train.log 2>&1
if errorlevel 1 (
    echo eval_pixel failed >> C:\arc_distill\arc_pixel_aligned\train.log
    exit /b 1
)

echo arc_pixel_aligned_done > C:\arc_distill\arc_pixel_aligned.done
