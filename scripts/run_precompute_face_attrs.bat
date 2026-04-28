@echo off
REM Re-run InsightFace buffalo_l on each 512^2 FFHQ image, save bbox + kps +
REM landmark_2d_106 + pose + age + gender + n_faces alongside compact.pt.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\arc_full_latent 2>nul
del C:\arc_distill\precompute_bboxes.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m arc_distill.precompute_bboxes ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --ffhq-parquet-dir C:\arc_distill\ffhq_parquet\data ^
    --out C:\arc_distill\arc_full_latent\face_attrs.pt > C:\arc_distill\arc_full_latent\precompute_bboxes.log 2>&1
if errorlevel 1 (
    echo precompute failed >> C:\arc_distill\arc_full_latent\precompute_bboxes.log
    exit /b 1
)

echo precompute_bboxes_done > C:\arc_distill\precompute_bboxes.done
