@echo off
REM Chain noise-conditional MediaPipe training + t-bucket validation.
cd /d C:\arc_distill
del C:\arc_distill\mediapipe_overnight_bs_v3_t.done 2>nul

call C:\arc_distill\run_mediapipe_distill_bs_v3_t.bat
if errorlevel 1 (
    echo overnight: train failed > C:\arc_distill\mediapipe_overnight_bs_v3_t.done
    exit /b 1
)

call C:\arc_distill\run_mediapipe_validate_bs_v3_t.bat
if errorlevel 1 (
    echo overnight: validate failed > C:\arc_distill\mediapipe_overnight_bs_v3_t.done
    exit /b 1
)

echo overnight_mediapipe_bs_v3_t_done > C:\arc_distill\mediapipe_overnight_bs_v3_t.done
