@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
mkdir C:\arc_distill\arc_pixel 2>nul

REM One-time pack: skip if compact.pt already exists
if not exist C:\arc_distill\arc_pixel\compact.pt (
    echo === preparing compact dataset === > C:\arc_distill\arc_pixel\train.log
    C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.prepare_compact ^
        --shards-dir C:\arc_distill\ffhq_parquet\data ^
        --encoded-dir C:\arc_distill\encoded ^
        --out C:\arc_distill\arc_pixel\compact.pt ^
        --resolution 224 >> C:\arc_distill\arc_pixel\train.log 2>&1
)

echo === training === >> C:\arc_distill\arc_pixel\train.log
C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_pixel ^
    --compact C:\arc_distill\arc_pixel\compact.pt ^
    --out-dir C:\arc_distill\arc_pixel ^
    --epochs 8 --batch-size 128 --workers 0 ^
    --lr 3e-4 --device cuda >> C:\arc_distill\arc_pixel\train.log 2>&1
echo arc_pixel_done > C:\arc_distill\arc_pixel.done
