@echo off
REM Supervisor: wait for v2d training done flag, then launch v2e training.
REM Decouples v2e launch from any SSH session.

set V2D_DONE=C:\arc_distill\mediapipe_distill_bs_v2d.done
set V2E_BAT=C:\arc_distill\repo_assets\scripts\run_mediapipe_distill_bs_v2e.bat

echo waiting for %V2D_DONE% > C:\arc_distill\mediapipe_v2e_supervisor.log

:WAIT
if exist "%V2D_DONE%" goto LAUNCH
timeout /t 30 /nobreak > nul
goto WAIT

:LAUNCH
echo v2d done; launching v2e at %DATE% %TIME% >> C:\arc_distill\mediapipe_v2e_supervisor.log
call "%V2E_BAT%"
echo v2e training finished at %DATE% %TIME% >> C:\arc_distill\mediapipe_v2e_supervisor.log
echo running v2e validation >> C:\arc_distill\mediapipe_v2e_supervisor.log
call "C:\arc_distill\repo_assets\scripts\run_mediapipe_validate_bs_v2e.bat"
echo v2e validation finished at %DATE% %TIME% >> C:\arc_distill\mediapipe_v2e_supervisor.log
