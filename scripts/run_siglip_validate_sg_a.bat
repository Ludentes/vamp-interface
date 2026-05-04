@echo off
REM Validate sg_a checkpoint as a slider/LoRA training loss:
REM   L1 direct cosine, L2 per-probe R² over 12 SigLIP probes, L3a/b invariance,
REM   L4 gradient sanity. Runs against current checkpoint.pt (best val cos so
REM   far). Re-run after training finishes to evaluate the final model.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\siglip_distill\sg_a 2>nul
del C:\arc_distill\siglip_validate_sg_a.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m siglip_distill.validate_as_loss ^
    --variant sg_a ^
    --checkpoint C:\arc_distill\siglip_distill\sg_a\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --siglip C:\arc_distill\siglip_distill\compact_siglip.pt ^
    --probes C:\arc_distill\siglip_distill\sg_probes.parquet ^
    --out-json C:\arc_distill\siglip_distill\sg_a\validation_report.json ^
    --device cuda > C:\arc_distill\siglip_distill\sg_a\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\siglip_distill\sg_a\validate.log
    exit /b 1
)

echo siglip_validate_sg_a_done > C:\arc_distill\siglip_validate_sg_a.done
