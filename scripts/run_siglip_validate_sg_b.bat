@echo off
REM Validate sg_b checkpoint (L2-normed at training time, pure (1-cos) loss).
REM Head-to-head with sg_a's L2-fixed validation (sg_a/validation_report.json).
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\siglip_distill\sg_b 2>nul
del C:\arc_distill\siglip_validate_sg_b.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m siglip_distill.validate_as_loss ^
    --variant sg_b ^
    --checkpoint C:\arc_distill\siglip_distill\sg_b\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --siglip C:\arc_distill\siglip_distill\compact_siglip.pt ^
    --probes C:\arc_distill\siglip_distill\sg_probes.parquet ^
    --out-json C:\arc_distill\siglip_distill\sg_b\validation_report.json ^
    --device cuda > C:\arc_distill\siglip_distill\sg_b\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\siglip_distill\sg_b\validate.log
    exit /b 1
)

echo siglip_validate_sg_b_done > C:\arc_distill\siglip_validate_sg_b.done
