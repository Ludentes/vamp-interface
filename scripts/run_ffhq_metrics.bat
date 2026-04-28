@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
C:\comfy\ComfyUI\venv\Scripts\python.exe -m demographic_pc.extract_ffhq_metrics ^
    --shards-dir C:\arc_distill\ffhq_parquet\data ^
    --out-dir C:\arc_distill\metrics ^
    --log C:\arc_distill\metrics.log ^
    --mediapipe-model %REPO%\models\mediapipe\face_landmarker.task ^
    --au-library      %REPO%\models\blendshape_nmf\au_library.npz
echo metrics_done > C:\arc_distill\metrics.done
