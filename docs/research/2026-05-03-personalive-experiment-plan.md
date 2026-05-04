---
status: superseded
topic: neural-deformation-control
superseded_by: 2026-05-03-iphone-pipeline-unified-plan.md
---

> **Superseded 2026-05-03 evening** by `2026-05-03-iphone-pipeline-unified-plan.md`. The unified plan keeps Phases 1–2 verbatim, promotes Phase 4b (the ARKit→latent bridge) to the central milestone, and adds three new phases (pre-flight, stabilizer, iPhone client) that complete the pipeline. The reframing: build the missing plumbing that makes neural deformation the photoreal equivalent of VTube Studio + Live2D — vamp-interface's offline parameter-driven deformation is a strict subset and falls out for free at the bridge milestone. This doc remains the authoritative reference for Phase 1 + 2 step-by-step procedure and the Phase 3/4 option taxonomy.

# PersonaLive / LivePortrait verification plan

**Date:** 2026-05-03
**Goal:** verify the claims in `2026-05-03-liveportrait-x-nemo-analysis.md` and `2026-05-03-neural-deformation-synthesis.md` with the actual installed PersonaLive (15 GB of weights at `/home/newub/w/PersonaLive/pretrained_weights/`) and AdvancedLivePortrait (not yet installed).

Four phases, each gated by a pass criterion. Stop and revise if a phase fails the gate — that's a real signal, not a setup bug.

## Phase 1 — sanity test (1–2 h)

**Claim being tested:** PersonaLive runs end-to-end at the reported speed and quality on our hardware (RTX 4090, no H100), on a Flux-rendered portrait as the source.

### Steps

1. Finish PersonaLive runtime install:
   ```bash
   cd /home/newub/w/PersonaLive
   uv pip install --python .venv -r requirements_base.txt
   ```
2. Pick a single Flux Krea portrait from existing renders — e.g. `output/ai_toolkit_runs/squint_slider_v0/samples/*000000000_*.jpg` for a neutral baseline. Use the european_man seed=1337 row as the canonical sanity source.
3. Pick the demo driver clip shipped under `/home/newub/w/PersonaLive/demo/` (whatever is closest to a single-identity face video, ≤ 5 s).
4. Run offline inference at default config:
   ```bash
   .venv/bin/python inference_offline.py \
     --src_image <flux portrait path> \
     --dri_video <demo driver> \
     --output ./test_output.mp4 \
     --width 512 --height 512
   ```
5. Time it (`time …`). Record: total wall time, FPS, per-frame latency, peak VRAM (`nvidia-smi` polled).

### Pass gate

- Output video plays without crashing.
- Visual: source identity visible throughout; expressions track the driver; no obvious artefacts (occlusion holes, melted features, frame-to-frame jitter beyond what the paper shows).
- Speed within 3× of the paper's H100 number on a 4090 — i.e. **≥ 5 FPS, ≤ 1.0 s latency per frame**. (PersonaLive paper: 15.82 FPS / 0.253 s on H100.)
- VRAM ≤ 16 GB (so we can co-host with Flux later).

### Falsification cases

- **VRAM > 16 GB or OOM.** Try the TinyVAE swap (paper claims 20 FPS with TinyVAE, lower VRAM). If still OOM, this rules out our hardware — escalate to runpod.
- **FPS < 3.** Means either the install fell back to CPU somewhere, or the H100→4090 gap is larger than expected. Check torch.cuda.is_available, check that Sage Attention / xformers are wired.
- **Identity collapse.** If the Flux portrait comes back as a different person, that's the FLUXSynID claim ("LivePortrait closely matches high identity consistency") not generalising to our distribution. **This would be a real falsification of the synthesis doc's recommendation.**

## Phase 2 — recipe-grid sanity (3–4 h)

**Claim being tested:** PersonaLive preserves identity continuously across a curated set of expression edits on Flux portraits — i.e. our use case (offline batch generation of variations from a single anchor) works.

### Steps

1. Pick **3 anchor portraits** spanning the demographic axis we already render: east_asian_man, black_woman, european_man. Use seed=1337 step-0 samples from `squint_slider_v0`.
2. Pick **6 driver clips** spanning the control surface:
   - `neutral → smile`
   - `neutral → jaw-open` (any "ahh" speech segment)
   - `neutral → wink-left, then wink-right`
   - `neutral → brow-up`
   - `head turn left, then right` (~30°)
   - `eyes-closed sustained`
   Source the drivers from the demo set or record 5-second webcam clips with `ffmpeg`.
3. Generate the full 3 × 6 = 18 outputs.
4. Run identity-drift measurement on each output: extract first frame and last frame of each clip, run `insightface buffalo_l`, report ArcFace cosine to the source anchor.
5. Run blendshape readout via MediaPipe Face Landmarker on each output frame (sample every 10th frame). Plot the 52-d trajectory; verify the channel that *should* move (smile → mouthSmile*; jaw-open → jawOpen) actually rises monotonically within each clip.
6. Build a 3-row × 6-col contact sheet (one row per anchor, one col per driver) showing the *peak* expression frame.

### Pass gate

- ArcFace cosine first-frame to last-frame ≥ 0.7 on **all 18 outputs**.
- The targeted channel rises by ≥ 0.4 (on its 0–1 MediaPipe scale) from baseline to peak in **at least 5 of 6 driver categories**, on **at least 2 of 3 anchors**.
- Visual contact sheet shows the right expression on the right person — no swapped identities, no "deformed but wrong person".

### Falsification cases

- **Identity drift > 0.3 cosine.** PersonaLive's training distribution doesn't generalise to Flux outputs as well as we hoped. Check whether it's worse on stylized vs photo-real anchors, then either accept the drift band or pivot to AdvancedLivePortrait alone.
- **Targeted channel doesn't move.** The driver clip wasn't expressive enough, OR PersonaLive's motion extractor doesn't read the channel cleanly. Re-record drivers with exaggerated expressions.

## Phase 3 — slider training-set generation (1 day)

**Claim being tested:** PersonaLive can act as a *data factory* for Concept-Slider training, producing (source, deformed, scalar_strength) triples on the cheap.

This is the high-value experiment. If it works, it replaces the ~1 h-per-axis LoRA training loop with a fully offline data-augmentation step that any axis can plug into.

### The bridge problem

PersonaLive's control surface is **another video**, not a scalar blendshape value. To make it slider-training-friendly we need a way to produce `N` driver clips at controlled blendshape strengths. Three options ranked by effort:

**Option A — slice an existing driver video.** Pick one driver clip that goes "neutral → peak smile" linearly. Generate the full PersonaLive output. Run MediaPipe on every output frame to read out the actual `s_smile` per frame. Now you have (source_anchor, output_frame_t, s_smile_measured) tuples — exactly the (anchor, deformed, strength) triples a slider trainer needs. Fast (1–2 h), but the strength axis is whatever the driver did, not a uniform [0, 1] sweep.

**Option B — interpolate two extreme driver frames.** Find a driver image at "neutral" (b_smile ≈ 0) and one at "peak smile" (b_smile ≈ 1), both of the same identity. Run PersonaLive's motion extractor on both → get the two implicit motion latents `m_neutral`, `m_peak`. Linearly interpolate `m(α) = (1−α) m_neutral + α m_peak` for α ∈ [0, 1] in 9 steps. Inject each `m(α)` into the denoiser, render. Run MediaPipe on each output to confirm `s_smile_measured(α)` rises monotonically. **This requires a small code change to PersonaLive's `inference_offline.py`** to accept a motion vector instead of a driver frame — but the paper's Algorithm 1 already shows the motion vector is a clean injection point.

**Option C — render driver from FLAME.** Render a FLAME mesh at controlled ψ_smile ∈ [0, 1] in 9 steps with neutral identity — flatten to 2D image — feed each as PersonaLive driver. Cleanest because the strength is *prescribed*, not measured. Requires we wire a FLAME renderer (`pyrender` or our existing FLAME stack from `docs/research/2026-04-12-flame-*.md`). Probably 1 day of work.

**Option D — VTuber software as the driver source (discuss before Phase 3 starts).** VTuber pipelines (VSeeFace, VTube Studio, Warudo, Animaze, OBS plugins) already solve the "controlled blendshape value → rendered face video" problem at production quality, with ARKit-52 as the standard control surface (iPhone TrueDepth / MediaPipe ingest). A VRoid / ReadyPlayerMe / MetaHuman avatar driven by a programmable ARKit stream gives us prescribed-strength sweeps for free, headlessly, with the ARKit naming we already use. This collapses Options B+C into off-the-shelf tooling. Trade-off: the driver is a stylised 3D avatar (toon shader or MetaHuman), so PersonaLive sees an OOD source — need to verify motion transfer survives the domain gap. Worth a 30-min spike before committing to A/B/C. Well within project interests since it also gives us a future "VTuber-style live mode" for the vamp interface itself.

### Steps

1. Start with Option A on the smile axis. 1 anchor (european_man), 1 driver clip (recorded webcam, 5 s neutral→smile→neutral, 25 fps = 125 frames).
2. Generate the full PersonaLive output (125 frames).
3. Run MediaPipe Face Landmarker on every frame → 125 × 52 matrix.
4. Plot `mouthSmileLeft + mouthSmileRight` over time. Should be a smooth rise-then-fall.
5. Bin the frames by smile strength into 9 buckets {0.0, 0.125, …, 1.0}. Sample one frame per bucket. You now have (anchor, 9 deformed frames, 9 strengths).
6. Verify visual: contact sheet of 9 frames at increasing smile.
7. (Optional) Repeat for jaw-open with a different driver clip.

### Pass gate

- 9 frames cover the strength range without large gaps (no bucket missing).
- Identity preserved across the 9 frames (ArcFace cosine ≥ 0.7 to anchor).
- The 9 frames look like a continuous smile sweep, not jittery.

### What this enables

If pass: **we have a slider training-set generator that doesn't need 200 steps of LoRA training.** A future Path-B trainer takes (anchor, deformed_image_at_strength_α, α) tuples directly — same shape as our current solver-C-curated FFHQ pair output, but generated offline from PersonaLive in seconds instead of needing a curated FFHQ pair set.

The slider then trained on this corpus could either be:
- The same Concept-Slider LoRA we already have (cheaper than current approach because data is free).
- A learned scalar-to-LoRA-strength mapping that lets us tune slider strength continuously instead of training discrete axes.

### Falsification cases

- **MediaPipe readout doesn't track output smile** (jumps around, doesn't form a clean rise). Means PersonaLive's expression transfer is too noisy frame-to-frame. Try Option B (deterministic motion-vector interp) which should be smoother by construction.
- **9 buckets don't all populate.** Driver clip lacked range. Re-record exaggerated.
- **Identity drift across the 9 frames.** Strong falsification — means PersonaLive shouldn't be used as a data source. Pivot to AdvancedLivePortrait Expression Editor (its sliders are deterministic and identity-stable by construction, just at lower quality).

## Phase 4 — bridge to blendshapes (2 days)

**Claim being tested:** we can drive PersonaLive *directly from an ARKit-52 blendshape vector*, closing the loop with our existing MediaPipe / ARKit infrastructure.

This is two sub-claims; do them in order.

### 4a — read blendshapes OUT of PersonaLive output

The easy direction. Already happens implicitly in Phase 3 step 3 — running MediaPipe on PersonaLive output to measure what blendshape was achieved. Make this a reusable utility:
- `src/blendshape_bridge/measure_alp.py` — given a video file, extract per-frame ARKit-52 vector via MediaPipe Face Landmarker, save as `.parquet`.
- This is the *measurement* side of the bridge. It also doubles as our finegrained eval for any future warp output.

### 4b — drive PersonaLive FROM an ARKit-52 vector

The hard direction. Three viable approaches:

**Approach 1 — render a FLAME driver from b_arkit.** Map ARKit-52 → FLAME ψ via an existing converter (e.g. the SMIRK output mapping from `docs/research/2026-04-12-flame-*`), render FLAME mesh, feed as PersonaLive driver. Highest fidelity but heaviest build.

**Approach 2 — synthesise a driver-image-like placeholder.** Render a generic ARKit-aligned mesh from b_arkit (no FLAME), use that as driver. Lower quality but cheap.

**Approach 3 — fit the 1D motion latent directly.** Train a small MLP `b_arkit → m_personalive` by running PersonaLive's `motion_extractor` on a driver corpus, simultaneously running MediaPipe on the same driver corpus, and regressing `m = f(b)`. ~10K driver-frame pairs should suffice; the MLP has fewer than 10M params. This is the cleanest option because it works with PersonaLive's existing motion-injection point and doesn't require a renderer.

### Steps for Approach 3 (recommended)

1. Pick a driver corpus we already have — VFHQ test split, our own FFHQ-derived video synthesis, or 100 short YouTube clips. Need ~10K frames with both faces and decent expression diversity.
2. For each driver frame:
   - Run PersonaLive's `motion_extractor.pth` → get the 1D motion latent `m_f` (236M-param model, fast inference).
   - Run MediaPipe Face Landmarker → get `b_arkit ∈ R^52`.
3. Save (b, m) pairs.
4. Train a small MLP `g: R^52 → R^d_motion` with MSE loss. ~30 epochs, ~1 h on a 4090.
5. At inference: take a target b, run g(b) → get m_predicted, inject into PersonaLive's denoising UNet via the cross-attention path, render. The 3D-keypoint pose path can use either a fixed neutral pose or be driven separately from b's pose-related channels.

### Pass gate

- The MLP achieves cosine ≥ 0.7 between predicted and ground-truth `m_f` on a held-out split.
- A round-trip experiment (b → m → render → MediaPipe-readout → b') has cosine ≥ 0.7 on the smile/jawOpen/blink/brow channels.
- Visual: rendering with `b_smile = 0.8, others = 0` produces a portrait that obviously smiles.

### Falsification cases

- **MLP can't learn the mapping** (held-out cosine < 0.5). Means the 1D motion latent doesn't have a clean injection from ARKit-52. Falls back to Approach 1 (FLAME-rendered driver).
- **Round-trip cosine < 0.5.** The motion latent isn't sensitive to the channels we care about. Means PersonaLive smooths over fine expression and we should use AdvancedLivePortrait directly (which exposes per-keypoint deltas explicitly).

## Decision tree across phases

```
Phase 1 sanity
  ├─ pass → Phase 2
  └─ fail (perf or identity) → fall back to AdvancedLivePortrait alone (lighter, deterministic)

Phase 2 recipe grid
  ├─ pass → Phase 3 + 4 in parallel
  └─ fail → use PersonaLive only as a re-renderer downstream of AdvancedLivePortrait;
            don't try to use it as the primary slider data source

Phase 3 slider data factory
  ├─ pass → big win, slider corpus is ~free for any axis with a corresponding driver clip
  └─ fail → AdvancedLivePortrait Expression Editor is the deterministic fallback;
            it has 12 sliders, no need for a data factory

Phase 4 blendshape bridge
  ├─ pass → ARKit-52 → PersonaLive direct control loop is real
  └─ fail → bridge through FLAME or AdvancedLivePortrait sliders;
            both work but at the cost of a renderer / partial coverage
```

## What this gets us if everything passes

- A new edit mechanism orthogonal to Concept-Slider LoRAs (Section "Diffusion vs deformation" of the analysis doc).
- A way to generate slider training corpora without per-axis 1 h GPU training.
- A direct ARKit-52 → portrait control surface that bypasses our LoRA path entirely for ~25 of the 52 channels.
- A clear delineation of which channels still need the LoRA path (squint, sneer, dimple, frown, lipRoll, shrug, tongue, fine gaze) and which can be handled by warp.

## What's needed before starting

1. Finish PersonaLive runtime install (`requirements_base.txt`).
2. Pick a Flux portrait as the canonical sanity source. Suggest: `output/ai_toolkit_runs/squint_slider_v0/samples/*0000_european_man*.jpg` step 0 (neutral baseline).
3. Record or download 6 driver clips for Phase 2.
4. Decide whether Phase 4 uses Approach 1 (FLAME renderer) or Approach 3 (learned MLP). Default: Approach 3 for speed.

## Cross-references

- Architecture analysis: `2026-05-03-liveportrait-x-nemo-analysis.md`
- Recipe parking note: `2026-05-03-slider-thread-recipe.md`
- Slider operational handbook: `2026-05-03-slider-operational-handbook.md`
- Practical install: `2026-05-03-neural-deformation-for-blendshape-control-practical.md`
- Topic index: `_topics/neural-deformation-control.md`
