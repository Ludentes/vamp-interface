---
status: live
topic: neural-deformation-control
supersedes: 2026-05-03-personalive-experiment-plan.md
---

# Unified plan: iPhone-driven photoreal portrait pipeline

**Date:** 2026-05-03
**Goal:** build the end-to-end pipeline `iPhone → bridge → PersonaLive → stabilizer → photoreal stream`, accepting that the "parameter-driven offline deformation" capability vamp-interface needs is a strict subset of this pipeline. If we land the full thing, the offline use case falls out for free.

## Reframing

Earlier framing: "verify PersonaLive works, then maybe add an ARKit bridge."
New framing: "build the missing pipeline that turns LivePortrait/PersonaLive into the photoreal equivalent of VTube Studio + Live2D."

The VTuber stack is automated end-to-end (iPhone → ARKit-52 → rigged Live2D character → renderer) but the rig is hand-authored at $200–8000 per character. The neural-deformation stack inverts the cost structure: zero per-face cost, but the *plumbing* is missing. Three pieces, none of which exist off-the-shelf:

| Piece | Purpose | Status |
|---|---|---|
| **Pre-flight** | Profile a source photo's drivability before commit | unbuilt anywhere |
| **Bridge** | ARKit-52 → PersonaLive motion latent | partially explored (FG-Portrait, no code) |
| **Stabilizer** | Identity anchor refresh over long sessions | partial (PersonaLive HKM) |

The **bridge** is load-bearing for both the live pipeline AND the vamp-interface offline use case (parameter-driven deformation = bridge input). Pre-flight and stabilizer are nice-to-haves for live use; vamp-interface offline can ignore them.

## Why this collapses the two goals into one

Vamp-interface needs: "given a Flux portrait + an ARKit-52 vector, render the deformed face." That is exactly `bridge(b_arkit) → m_latent → PersonaLive(source_photo, m_latent)`. No live capture, no streaming, no stabilizer needed for static images — just the bridge running in batch.

Live iPhone use needs: same bridge, plus iPhone-side capture + stream transport + stabilizer for long sessions.

So: **build the bridge first; both use cases unlock immediately, with the live use case needing one more layer of plumbing on top.**

## Revised phase structure

The original four phases (`2026-05-03-personalive-experiment-plan.md`, now superseded) verified PersonaLive in isolation. The unified plan keeps Phases 1–2 as prerequisites, expands Phase 4 (the bridge) into the load-bearing milestone, and adds three new phases for the live-pipeline pieces.

### Phase 1 — PersonaLive sanity (unchanged, prerequisite)

Same as superseded plan §Phase 1. Verifies PersonaLive runs at usable speed on RTX 4090 against a Flux portrait. **Already prepped:** install complete, real CLI documented (`--reference_image`, `--driving_video`, results auto-saved to `results/`).

Pass → Phase 2. Fail → fall back to AdvancedLivePortrait-only path; the unified plan still works but with a smaller motion vocabulary.

### Phase 2 — recipe grid (unchanged, prerequisite)

Same as superseded plan §Phase 2. 3 anchors × 6 driver clips, identity-drift + MediaPipe readout. Confirms PersonaLive generalises to Flux distribution before we invest in the bridge.

### Phase 3 — bridge: `ARKit-52 → PersonaLive motion latent` (was Phase 4b, now central)

**This is the milestone that unlocks both the vamp-interface offline use case AND the live pipeline.**

The bridge is a learned MLP `g: R^52 → R^d_motion` trained on (b_arkit, m_personalive) pairs extracted from a driver corpus. Recipe per superseded plan §Phase 4b Approach 3:

1. Driver corpus: ~10K frames with both faces and decent expression diversity. Sources: VFHQ test split, FFHQ-derived video, or 100 short YouTube clips. Cheap option: record 30 min of webcam from one cooperative actor cycling through ARKit-52 channels deliberately.
2. For each frame: run PersonaLive `motion_extractor` → m_f; run MediaPipe Face Landmarker → b_arkit. Save (b, m) pairs.
3. Train MLP. ~30 epochs, ~1 h on 4090. MSE loss to start; consider per-channel weighting later if some ARKit channels are systematically learned better than others (we already see this on the LoRA side — see `feedback_blendshape_per_channel`).
4. Held-out validation: cosine ≥ 0.7 between predicted and ground-truth m on validation split.
5. Round-trip: `b → g(b) → m → PersonaLive(anchor, m) → MediaPipe → b'`. Cosine(b, b') ≥ 0.7 on smile/jawOpen/blink/brow.

**Vamp-interface gate:** at this point we can render any Flux portrait at any prescribed ARKit-52 setting, fully programmatically. Ship the offline pipeline. Slider data factory becomes trivial — sweep b_smile linearly, render, that's the corpus.

### Phase 4 — pre-flight: portrait profiling (was missing)

**Claim being tested:** different source photos have different drivability profiles, and we can predict this before committing to a render.

This is a small but concrete piece of plumbing that doesn't exist in the public landscape. Output: per-source-photo metadata blob.

#### Steps

1. Take 50 candidate source photos: 20 Flux Krea portraits across demographics, 20 FFHQ real photos, 10 stylised Flux outputs (anime-leaning, painterly).
2. For each, run a fixed *drivability probe*: 6 canonical b_arkit settings (smile peak, jaw-open peak, both eyes closed, brow up, head turn 30°, neutral). Measure:
   - ArcFace cosine of probe-output to source.
   - MediaPipe channel achievement (did the requested channel actually move?).
   - SSIM / LPIPS of background+hair regions (proxy for "warp didn't tear the image apart").
3. Build a per-photo profile: 6 numbers × 3 metrics = 18-d capability fingerprint.
4. Train a small classifier: `is_this_photo_drivable(photo) → confidence ∈ [0,1]`, using the 50-photo corpus as ground truth (label by hand from the contact sheet).
5. At ingest time, run the classifier on any new candidate source photo. Reject low-confidence photos before they enter the pipeline.

#### Pass gate

- Classifier achieves ≥ 0.8 AUC on held-out split (10 of 50 held out).
- The hand-labelled "bad" photos (occlusion, side angle, weird crop) all score < 0.4.

#### Falsification

- Classifier can't distinguish drivable from non-drivable. Means the pipeline is roughly identity-stable across our distribution and pre-flight isn't needed. Skip this layer.

#### What this enables

For vamp-interface: reject Flux portraits that won't deform cleanly *before* generating their full slider corpus. Saves wasted compute. For live use: the iPhone client can warn "your selfie isn't ideal, try better lighting" before starting a session.

### Phase 5 — stabilizer: long-form identity anchor (was missing)

**Claim being tested:** PersonaLive's Historical Keyframe Mechanism handles ~30 s well but drifts on multi-minute sessions; periodic re-anchoring fixes this without hurting motion continuity.

Only relevant for the live use case. Vamp-interface offline can skip.

#### Steps

1. Run a 5-minute PersonaLive session on a single source + continuous driver. Measure ArcFace cosine to source every 5 s. Identify drift onset.
2. Implement re-anchor: every N seconds (start with N=30), re-extract the source's appearance feature `f_app` from the original photo and inject it into the next chunk's denoising context. This is roughly the HKM extended with a hard re-init.
3. Add drift detector: when ArcFace cosine to source drops below 0.7 mid-session, force a re-anchor regardless of N.
4. Repeat the 5-minute test with stabilizer on. Verify cosine stays > 0.7 throughout, no visible discontinuity at re-anchor moments.

#### Pass gate

- 5-minute session never drops below 0.7 cosine to source.
- Visual: no visible jump at re-anchor points (would manifest as a frame where features snap).

#### Falsification

- Re-anchor causes visible flicker. Means we need to blend the anchor injection over a few frames; engineering, not science.
- Drift continues even with re-anchor. Means PersonaLive's identity injection point is wrong — pivot to LivePortrait's pure-deformation path which is GAN-trained with explicit ID loss.

### Phase 6 — iPhone client + transport (was missing)

**Claim being tested:** the iPhone can capture ARKit-52 + RGB at 60 Hz and stream both to a desktop GPU server with end-to-end latency under 200 ms.

Only relevant for the live use case. Vamp-interface offline can skip.

#### Steps

1. Minimal iOS app using `ARFaceTrackingConfiguration`. Captures `blendShapes` dict (52 floats) + RGB buffer (downsample to 256×256) at 30 Hz. Standard Apple sample code is the starting point.
2. Transport: WebRTC for video (low-latency, hardware-accelerated H.264 encoder on iPhone), WebSocket for the 52-d float vector (1 KB/frame, negligible). Both timestamped.
3. Server: small Python service receives both, time-aligns them, hands video frame to PersonaLive, hands b_arkit to the bridge MLP (Phase 3 output) for optional override / blending.
4. Render output → encode H.264 → stream back to iPhone (or to a separate viewer).
5. Measure round-trip: capture timestamp on iPhone → display timestamp on iPhone (via QR code in the rendered frame). Target < 200 ms.

#### Pass gate

- End-to-end latency < 200 ms p50, < 400 ms p99.
- 30 FPS sustained, no frame drops in steady state.
- The override channel works: if the server forces b_smile=1.0 in the bridge, the rendered face smiles even if the iPhone user doesn't.

#### Falsification

- Latency > 500 ms even with optimisation. Means the pipeline isn't viable for live interaction; falls back to "near-real-time" use cases (5 s lag for a "magic mirror" demo) where it's fine.
- Bandwidth too high to be practical (>10 Mbps). Use harder video compression on the iPhone side; ARKit-52 stream is already ~2 KB/s so it's not the issue.

## Decision tree

```
Phase 1 sanity ─pass→ Phase 2 ─pass→ Phase 3 (bridge) ─pass→
                                      ├→ vamp-interface OFFLINE pipeline shipped
                                      ├→ Phase 4 (pre-flight) [optional, unlocks ingest UX]
                                      └→ Phase 5 (stabilizer) ─pass→ Phase 6 (iPhone) ─pass→
                                                                     LIVE pipeline shipped

Phase 1 fail → AdvancedLivePortrait-only path, repeat plan with smaller vocabulary
Phase 3 fail → bridge through FLAME renderer (Phase 4b Approach 1) or accept LoRA-only for vamp
```

## What this gets us at each gate

- **Phase 3 pass:** vamp-interface gets parameter-driven photoreal deformation. The slider thread (LoRA training) becomes complementary, not primary — useful only for the ~25 ARKit channels that don't transfer cleanly through warp (squint, sneer, asymmetric, fine gaze).
- **Phase 4 pass:** vamp-interface gets a "is this Flux portrait deployable?" gate at ingest, saving wasted slider corpus generation.
- **Phase 5 + 6 pass:** vamp-interface gains a "live mode" — analyst or scam-hunter speaks to the camera, the rendered face mirrors them with ARKit overrides applied (e.g., overlay a fraud signal as exaggerated discomfort). This is a future product surface, not a current requirement, but it's the natural endpoint of the same pipeline.

## What's intentionally NOT in scope

- **Audio-driven mode** (Sonic, Hallo, Teller, RAP family). Audio→portrait is a parallel research thread; doesn't share the bridge. Park.
- **3D head reconstruction.** FG-Portrait, MetaHuman, FLAME-direct rendering. Compatible with this plan as a *driver source* (Phase 4b Approach 1) but not the primary path.
- **VTuber-stylised mode.** Live2D / VRoid / VTube Studio is a parallel pipeline for the heavy-stylised regime; we accept that's a different stack and don't try to unify.
- **Replacement of the LoRA slider thread.** LoRAs continue to handle the ~25 channels with no clean ALP/PersonaLive analogue. The two paths coexist per `_topics/neural-deformation-control.md`.

## Cross-references

- Original four-phase plan (now subsumed): `2026-05-03-personalive-experiment-plan.md`
- Architecture analysis: `2026-05-03-liveportrait-x-nemo-analysis.md`
- Synthesis / decision doc: `2026-05-03-neural-deformation-synthesis.md`
- Practical install + slider table: `2026-05-03-neural-deformation-for-blendshape-control-practical.md`
- Slider thread companion: `2026-05-03-slider-thread-recipe.md`
- Open ideas: `2026-05-03-interesting-ideas.md`
- Topic index: `_topics/neural-deformation-control.md`
