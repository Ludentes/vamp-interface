@echo off
REM Usage: run_arc_adapter_latent.bat {up|native}
setlocal
set VARIANT=%1
if "%VARIANT%"=="" (
    echo usage: %0 ^{up^|native^}
    exit /b 2
)
if /I "%VARIANT%"=="up" (
    set VARIANT_FULL=latent_a_up
    set OUT_TAG=latent_a_up
) else if /I "%VARIANT%"=="native" (
    set VARIANT_FULL=latent_a_native
    set OUT_TAG=latent_a_native
) else (
    echo unknown variant: %VARIANT%
    exit /b 2
)

cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_adapter\%OUT_TAG% 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.train_adapter ^
    --variant %VARIANT_FULL% ^
    --compact C:\arc_distill\arc_pixel_aligned\compact_latent.pt ^
    --out-dir C:\arc_distill\arc_adapter\%OUT_TAG% ^
    --epochs 20 --batch-size 256 --workers 0 ^
    --lr 1e-3 --device cuda > C:\arc_distill\arc_adapter\%OUT_TAG%\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\arc_adapter\%OUT_TAG%\train.log
    exit /b 1
)

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.eval_adapter ^
    --variant %VARIANT_FULL% ^
    --checkpoint C:\arc_distill\arc_adapter\%OUT_TAG%\checkpoint.pt ^
    --compact C:\arc_distill\arc_pixel_aligned\compact_latent.pt ^
    --out-json C:\arc_distill\arc_adapter\%OUT_TAG%\eval.json ^
    --device cuda >> C:\arc_distill\arc_adapter\%OUT_TAG%\train.log 2>&1
if errorlevel 1 (
    echo eval failed >> C:\arc_distill\arc_adapter\%OUT_TAG%\train.log
    exit /b 1
)

echo arc_adapter_%OUT_TAG%_done > C:\arc_distill\arc_adapter_%OUT_TAG%.done
