@echo off
REM Validate sg_c at t-buckets {0, 0.1, 0.25, 0.5, 0.75, 1.0}.
REM Gates from docs/research/2026-04-30-noise-conditional-distill-design.md:
REM   T1 — clean-input parity (t=0 within 0.01 of sg_b 0.9204)
REM   T2 — schedule coverage (every t bucket within 0.05 of t=0)
REM   T3 — probe R² at t=0.5 within 0.10 of t=0
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\siglip_distill\sg_c 2>nul
del C:\arc_distill\siglip_validate_sg_c.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m siglip_distill.validate_vs_t ^
    --variant sg_c ^
    --checkpoint C:\arc_distill\siglip_distill\sg_c\checkpoint.pt ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --siglip C:\arc_distill\siglip_distill\compact_siglip.pt ^
    --probes C:\arc_distill\siglip_distill\sg_probes.parquet ^
    --out-json C:\arc_distill\siglip_distill\sg_c\validation_report_vs_t.json ^
    --device cuda > C:\arc_distill\siglip_distill\sg_c\validate.log 2>&1
if errorlevel 1 (
    echo validate failed >> C:\arc_distill\siglip_distill\sg_c\validate.log
    exit /b 1
)

echo siglip_validate_sg_c_done > C:\arc_distill\siglip_validate_sg_c.done
