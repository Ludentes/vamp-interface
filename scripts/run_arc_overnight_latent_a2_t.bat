@echo off
REM Chain noise-conditional arc training + t-bucket validation.
cd /d C:\arc_distill
del C:\arc_distill\arc_overnight_latent_a2_t.done 2>nul

call C:\arc_distill\run_arc_adapter_latent_a2_t.bat
if errorlevel 1 (
    echo overnight: train failed > C:\arc_distill\arc_overnight_latent_a2_t.done
    exit /b 1
)

call C:\arc_distill\run_arc_validate_latent_a2_t.bat
if errorlevel 1 (
    echo overnight: validate failed > C:\arc_distill\arc_overnight_latent_a2_t.done
    exit /b 1
)

echo overnight_arc_latent_a2_t_done > C:\arc_distill\arc_overnight_latent_a2_t.done
