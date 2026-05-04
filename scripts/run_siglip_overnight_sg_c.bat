@echo off
REM Chain training + validation in one launcher so we can fire-and-forget
REM overnight. Validation only runs if training exits 0.
cd /d C:\arc_distill
del C:\arc_distill\siglip_overnight_sg_c.done 2>nul

call C:\arc_distill\run_siglip_distill_sg_c.bat
if errorlevel 1 (
    echo overnight: train failed > C:\arc_distill\siglip_overnight_sg_c.done
    exit /b 1
)

call C:\arc_distill\run_siglip_validate_sg_c.bat
if errorlevel 1 (
    echo overnight: validate failed > C:\arc_distill\siglip_overnight_sg_c.done
    exit /b 1
)

echo overnight_sg_c_done > C:\arc_distill\siglip_overnight_sg_c.done
