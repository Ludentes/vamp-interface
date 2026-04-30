@echo off
REM bs_v4_pgd: PGD-adversarially-robust noise-conditional MediaPipe-blendshape
REM student. Same bs_a architecture as v2c/v3_t, same combined corpus
REM (FFHQ + rendered). Warm-start from v3_t/final.pt (already
REM noise-conditional; this run only specializes to adversarial-robust).
REM
REM Inner loop: 5-step PGD on L-inf-bounded latent delta at eps=0.05.
REM Outer loss: MSE(bs(z_t), y) + 1.0 * MSE(bs(z_t + delta), y).
REM
REM Expected ~5x v3_t per-epoch cost (5 inner forward+backward steps).
REM v3_t was ~80s/epoch -> v4_pgd ~400s/epoch -> ~1.7h for 15 epochs (2h cap).
REM To continue later, relaunch with --init-from bs_v4_pgd/final.pt --epochs N.
cd /d C:\arc_distill
set REPO=C:\arc_distill\repo_assets
set PYTHONPATH=%REPO%\src
set PYTHONIOENCODING=utf-8
mkdir C:\arc_distill\mediapipe_distill\bs_v4_pgd 2>nul
del C:\arc_distill\mediapipe_distill_bs_v4_pgd.done 2>nul

C:\comfy\ComfyUI\venv\Scripts\python.exe -u -m mediapipe_distill.train_t_pgd ^
    --variant bs_a ^
    --compact C:\arc_distill\arc_full_latent\compact.pt ^
    --blendshapes C:\arc_distill\arc_full_latent\compact_blendshapes.pt ^
    --rendered C:\arc_distill\arc_full_latent\compact_rendered.pt ^
    --out-dir C:\arc_distill\mediapipe_distill\bs_v4_pgd ^
    --init-from C:\arc_distill\mediapipe_distill\bs_v3_t\final.pt ^
    --epochs 15 --batch-size 128 --workers 0 ^
    --lr 5e-4 ^
    --pgd-eps 0.05 --pgd-alpha 0.0125 --pgd-k 5 --pgd-lambda 1.0 ^
    --device cuda > C:\arc_distill\mediapipe_distill\bs_v4_pgd\train.log 2>&1
if errorlevel 1 (
    echo train failed >> C:\arc_distill\mediapipe_distill\bs_v4_pgd\train.log
    exit /b 1
)

echo mediapipe_distill_bs_v4_pgd_done > C:\arc_distill\mediapipe_distill_bs_v4_pgd.done
