# Demographic Classifiers — Stage 0 Install Log

**Date:** 2026-04-20
**Related:** `2026-04-20-demographic-classifiers.md` (research), `2026-04-20-demographic-pc-extraction-plan.md` (plan)
**Scope:** Environment setup + smoke test for three classifiers. The third slot pivoted from DEX (blocked) to InsightFace.

## Final Classifier Roster

| Classifier | Install route | Weights | Status |
|---|---|---|---|
| **MiVOLO** volo_d1 face-only | Vendored `WildChlamydia/MiVOLO`, patched for timm 1.0 | `vendor/weights/mivolo_volo_d1_face_age_gender_imdb.pth.tar` via gdown | ✅ Works |
| **FairFace** 7-race res34 | Vendored `dchen236/FairFace`, weights from `yakhyo/fairface-onnx` release mirror | `vendor/weights/fairface/res34_fair_align_multi_7_20190809.pt` via `gh release download` | ✅ Works (without dlib alignment for smoke) |
| **InsightFace** buffalo_l genderage | `uv add insightface onnxruntime-gpu` (PyPI) | auto-downloaded to `~/.insightface/models/buffalo_l/` | ✅ Works |
| ~~**DEX** (original Rothe 2015)~~ | ~~siriusdemon/pytorch-DEX~~ | **blocked** — no prebuilt PyTorch weights; repo ships Caffe→PyTorch convert script only. Swapped for InsightFace. | ❌ Replaced |

Added to `pyproject.toml`: `timm`, `ultralytics`, `gdown`, `dlib`, `insightface`, `onnxruntime-gpu`.

## What Required Patching

### MiVOLO vs timm 1.0 positional-arg shift
timm 1.0 VOLO's `__init__` inserted `pos_drop_rate` at position 15, which shifted every positional argument after it. MiVOLO's `mivolo_model.py` passed args positionally, causing `post_layers=("ca","ca")` to land in the `norm_layer` slot → `TypeError: 'tuple' object is not callable`.

**Fix:** converted `super().__init__(layers, img_size, ...)` to kwargs (`super().__init__(layers=layers, img_size=img_size, ...)`) in `vendor/MiVOLO/mivolo/model/mivolo_model.py`. Also patched `create_timm_model.py` to accept `remap_state_dict` as a fallback for the removed `remap_checkpoint` symbol (though that path isn't exercised by the face-only smoke test, which bypasses `create_timm_model.py` entirely).

With that one edit, `timm.create_model("mivolo_d1_224", num_classes=3, in_chans=3)` + `model.load_state_dict(sd)` loads cleanly (0 missing, 0 unexpected).

### FairFace — no dlib alignment for smoke
FairFace's standard flow is: dlib CNN face-detect → dlib 5-landmark align → 224 crop → res34 predict. Smoke test skipped the dlib stages and fed whole 1024×1024 Flux portraits resized to 224. **Result: low race confidences (0.25–0.30)**, which is expected for misaligned input — the classifier is designed for aligned crops. For Stage 2+ we'll need dlib alignment.

dlib itself installed fine via `uv add dlib` but took 3+ minutes building from source (no wheels for Python 3.12 on PyPI).

### FairFace weights — Google Drive folder blocked, mirror used
`dchen236/FairFace` README points to a Google Drive folder. `gdown --folder ...` returned `status code 404 — may need to change the permission to 'Anyone with the link'`. Workaround: `yakhyo/fairface-onnx` mirrors the res34 weights (`res34_fair_align_multi_7_20190809.pt`) on a GitHub release; downloaded via `gh release download`. This is the 7-race model. (We're not using the 4-race variant.)

### DEX blocked — pivoted to InsightFace
`siriusdemon/pytorch-DEX` does not ship PyTorch weights. The repo's `convert.py` needs `pycaffe` + the original `dex_chalearn_iccv2015.caffemodel` + `age.prototxt`. Modern caffe install on Python 3.12 is an unmaintained nightmare. Couldn't find a pre-converted mirror on HuggingFace (searched: `IMDB-WIKI age`, `fairface_alldata_20191111.pt`, `age_sd.pth pretrained`; only one result — `onnxmodelzoo/vgg_ilsvrc_16_age_imdb_wiki` as ONNX, not ready-made PyTorch).

Considered `deepface` (PyPI) as a DEX substitute but it's VGG-Face transfer-learned, not the original VGG-16 + IMDB-WIKI lineage — so it doesn't preserve the "Rothe 2015 Caffe" pedigree that motivated DEX in the research doc.

**Pivoted to InsightFace** (user suggestion): zero install pain (`uv add insightface onnxruntime-gpu`), bundled SCRFD face detection + alignment, `genderage` ONNX model auto-downloads to `~/.insightface/models/buffalo_l/`. Caveat: no race head — race is now single-sourced from FairFace, which was already flagged as a concentration risk.

## Smoke-Test Outputs

Four Flux-v3 faces from `output/phase1/`: anchor (seed 42 neutral) + one sample each from `courier_legit`, `office_legit`, `scam_critical`.

| Image | MiVOLO (face-only) | FairFace (no align) | InsightFace (buffalo_l) |
|---|---|---|---|
| `phase1_anchor.png` | age **37.6** · male (1.00) | age 20-29 (0.26) · Latino (0.26) · M (1.00) | age **54.0** · male |
| `002ae68a-...` (courier) | age **34.3** · male (1.00) | age 20-29 (0.32) · SE Asian (0.30) · M (1.00) | age **46.0** · male |
| `00a4cf37-...` (office) | age **35.2** · male (1.00) | age 20-29 (0.35) · White (0.25) · M (1.00) | age **48.0** · male |
| `119c7737-...` (scam) | age **33.8** · male (1.00) | age 10-19 (0.45) · SE Asian (0.25) · M (1.00) | age **54.0** · male |

## Load-Bearing Observations

**Age disagreement is enormous.** Same Flux face is read as 20-29 (FairFace), 34-38 (MiVOLO), and 46-54 (InsightFace) — a 20+ year spread on every sample. The research doc flagged "age representation mismatch across the three is a feature, not a bug — disagreement is evidence of surface leakage." The smoke test confirms this qualitatively on N=4. Whether the disagreement is *structured* (e.g. each classifier has a consistent bias direction on Flux output) or *noisy* (random per-sample) is the first thing to measure in Stage 1.

**Gender agreement is unanimous.** All three say male for all four samples at high confidence. The four Flux v3 portraits used for smoke are genuinely male-presenting in the prompt, so this is consistent with prompt intent — not a surprise, but a positive signal.

**FairFace without dlib alignment is degraded.** Race confidences hovering at 0.25–0.30 (near chance for 7-way) suggest the classifier is struggling with un-aligned input. Adding dlib alignment is required before the Stage 2 run — we cannot extract a meaningful race direction from near-uniform softmaxes.

**InsightFace has no race head.** The plan needs one source of race labels → FairFace (with proper dlib alignment). This makes the race subspace single-classifier, which was flagged in the research doc as concentration risk; it is now operational, not hypothetical.

**The Stage 1 sanity check is now load-bearing, not optional.** The age disagreement observed on N=4 could mean:
- (a) each classifier has a consistent bias direction on Flux output — fine, we'll rediscover that per-head in regression and the directions will still be useful
- (b) Flux portraits put classifiers into out-of-distribution noise — the predicted labels are ~random, and any direction we extract is spurious
- Only a 50-sample controlled run with prompt-attribute ground truth can disambiguate. If agreement with the prompt-attribute is weak on all three, we halt before the 1800-sample run.

## What To Do Before Stage 1

1. **Add dlib alignment to the FairFace wrapper.** Current smoke skips it for expediency; production must include it.
2. **Write a unified inference API** in `src/demographic_pc/classifiers.py` that exposes `predict(image_bgr) → {age, gender, race?, conf}` uniformly across all three, so Stage 1 and Stage 2 are a one-liner per sample.
3. **Decide on 50-sample prompt grid** for Stage 1 (subset of the full 105-cell grid). Probably 2 samples per cell × 25 cells = 50.

## Packaging Notes

- **Vendored code:** `vendor/MiVOLO/`, `vendor/FairFace/`, `vendor/pytorch-DEX/` (kept for audit trail, unused)
- **Weights:** `vendor/weights/` (gitignored — total ~190MB); auto-downloaded insightface weights live in `~/.insightface/` (2GB, gitignored by default since outside repo)
- **Smoke scripts:** `src/demographic_pc/smoke_{mivolo,fairface,insightface}.py` — kept as reproducibility artifacts

## Known Risks Carried Forward

- FairFace race is now the *only* race source — direction quality depends entirely on one classifier's bias profile
- InsightFace's genderage ONNX is 96×96 internal — very low-resolution; may be noisy on Flux portraits with fine detail
- MiVOLO volo_d1 weights are from Google Drive only; if the drive link disappears, we have no official mirror
- Smoke test N=4 is not a statistical test; Stage 1 (N=50 with known prompt-attribute ground truth) is the real go/no-go
