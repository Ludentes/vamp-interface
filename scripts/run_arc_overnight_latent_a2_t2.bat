@echo off
REM Chain Path 2 training + t-bucket validation.
cd /d C:\arc_distill
del C:\arc_distill\arc_overnight_latent_a2_t2.done 2>nul

call C:\arc_distill\run_arc_adapter_latent_a2_t2.bat
if errorlevel 1 (
    echo overnight: train failed > C:\arc_distill\arc_overnight_latent_a2_t2.done
    exit /b 1
)

call C:\arc_distill\run_arc_validate_latent_a2_t2.bat
if errorlevel 1 (
    echo overnight: validate failed > C:\arc_distill\arc_overnight_latent_a2_t2.done
    exit /b 1
)

echo overnight_arc_latent_a2_t2_done > C:\arc_distill\arc_overnight_latent_a2_t2.done
