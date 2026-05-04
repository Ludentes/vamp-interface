@echo off
cd /d C:\arc_distill\repo_assets
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
C:\comfy\ComfyUI\venv\Scripts\python.exe -m siglip_distill.extract_siglip_features ^
    --source flux_corpus_v3 ^
    --sample-index %REPO%\models\blendshape_nmf\sample_index.parquet ^
    --repo-root %REPO% ^
    --out-parquet C:\arc_distill\siglip_distill\flux_corpus_v3_siglip_emb.parquet ^
    --batch-size 16 ^
    --flush-every 512 ^
    --log C:\arc_distill\siglip_distill\extract_flux_corpus_v3.log
echo siglip_flux_v3_done > C:\arc_distill\siglip_extract_flux_corpus_v3.done
