@echo off
cd /d C:\arc_distill\repo_assets
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
C:\comfy\ComfyUI\venv\Scripts\python.exe -m siglip_distill.extract_siglip_features ^
    --source flux_solver_a ^
    --scores-parquet %REPO%\output\demographic_pc\solver_a_squint_grid\scores.parquet ^
    --out-parquet C:\arc_distill\siglip_distill\flux_solver_a_siglip_emb.parquet ^
    --batch-size 16 ^
    --flush-every 512 ^
    --log C:\arc_distill\siglip_distill\extract_flux_solver_a.log
echo siglip_solver_a_done > C:\arc_distill\siglip_extract_flux_solver_a.done
