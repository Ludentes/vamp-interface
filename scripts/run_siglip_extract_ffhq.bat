@echo off
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
C:\comfy\ComfyUI\venv\Scripts\python.exe -m siglip_distill.extract_siglip_features ^
    --source ffhq ^
    --shards-dir C:\arc_distill\ffhq_parquet\data ^
    --out-parquet C:\arc_distill\siglip_distill\ffhq_siglip_emb.parquet ^
    --batch-size 16 ^
    --flush-every 512 ^
    --log C:\arc_distill\siglip_distill\extract_ffhq.log
echo siglip_ffhq_done > C:\arc_distill\siglip_extract_ffhq.done
