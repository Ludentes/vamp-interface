@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\pixel_a 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant pixel_a ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --out-dir C:\arc_distill\arc_adapter\pixel_a ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\pixel_a\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\pixel_a\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant pixel_a ^
    --checkpoint C:\arc_distill\arc_adapter\pixel_a\checkpoint.pt ^
    --compact C:\arc_distill\arc_pixel_aligned\compact.pt ^
    --out-json C:\arc_distill\arc_adapter\pixel_a\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\pixel_a\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\pixel_a\train.log
    exit /b 1
)

echo arc_adapter_pixel_a_done > C:\arc_distill\arc_adapter_pixel_a.done
