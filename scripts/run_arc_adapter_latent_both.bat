@echo off
cd /d C:\arc_distill
del C:\arc_distill\arc_adapter_latent_both.done 2>nul
del C:\arc_distill\arc_adapter_latent_a_up.done 2>nul
del C:\arc_distill\arc_adapter_latent_a_native.done 2>nul

call C:\arc_distill\run_arc_adapter_latent.bat up
if errorlevel 1 exit /b 1

call C:\arc_distill\run_arc_adapter_latent.bat native
if errorlevel 1 exit /b 1

echo arc_adapter_latent_both_done > C:\arc_distill\arc_adapter_latent_both.done
