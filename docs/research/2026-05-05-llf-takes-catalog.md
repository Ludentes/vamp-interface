---
status: live
topic: neural-deformation-control
---

# LLF takes catalog (2026-05-05 iPhone session)

**Date captured:** 2026-05-05
**Device:** iPhone15,4 (iPhone 15 Pro), Live Link Face v1.6.0 (build 169)
**Capture mode:** ARKit + 1280×720 @ 60 fps, JPEG compression 0.7, "medium" quality
**Identifier prefix:** `20260505_MySlate_<N>`

This is the canonical eight-take batch the PersonaLive long-take work in this session is built on. Everything else (artifact catalog, A/B/C cropper experiment, anchor reinjection design) refers back to these takes by number, so this doc keeps the numbering, paths, and known caveats in one place.

## Where everything lives

- **Source MOVs + ARKit sidecars:** `data/llf-takes/20260505_MySlate_<N>/`
  - `MySlate_<N>_iPhone.mov` — H.264 video, the only file `render_take.py` needs
  - `take.json` — frame count, start/end timecodes, device, app version
  - `video_metadata.json` — 1280×720, 60 fps, Orientation 4 (left-side-up; iPhone honours via `cap.set(CAP_PROP_ORIENTATION_AUTO, 1)`)
  - `frame_log.csv` — per-frame ARKit timestamps (not yet sanity-checked; tracked as task #54)
  - `depth_data.bin` + `depth_metadata.mhaical` — front-camera TrueDepth stream; **unused by current pipeline**, kept for the eventual MetaHuman Animator depth path
  - `audio_metadata.json` — audio side-channel; pipeline drops audio
- **Side-by-side renders against the canonical anchor:** `data/llf-take-renders/take<N>__asian_m.mp4`
  - 1024×512 @ 60 fps, driver left, PersonaLive output right
  - Anchor: `data/llf-phase2/asian_m__06_neutral.midframe.png` (also has matching `.mp4` if needed)
  - Produced by `/home/newub/w/PersonaLive/scripts/render_take.py` against `configs/prompts/personalive_online.yaml`
- **A/B/C crop-strategy renders (60 s of takes 7 + 2):** `data/llf-take-renders/abc/take{2,7}_{A_perframe,B_nocrop,C_ema}.mp4`
- **External archive:** none yet — these MOVs are local-disk only. ~14 GB total across 8 takes; if disk pressure appears, follow `feedback_copy_when_mounted.md` and copy off before any further long jobs.

## The eight takes

| take | dur (s) | frames | mov size | render mp4 | face_mesh fallback (perframe) | notes |
|---|---|---|---|---|---|---|
| 1 | 106 | 6 385 | 613 MB | 52 MB | 10 (0.2 %) | early take, calm |
| 2 | 124 | 7 440 | 3.2 GB | 105 MB | 42 (0.6 %) | **glasses-heavy, longest**; canonical "hard" take for jitter studies |
| 3 |  54 | 3 248 | 1.4 GB | 32 MB | 30 (0.9 %) | short |
| 4 |  34 | 2 063 | 1.1 GB | 21 MB | **856 (41.6 %)** ⚠⚠ | dominant face_mesh failure; cropper is essentially frozen for most of the take |
| 5 | 122 | 7 337 | 2.3 GB | 64 MB | 88 (1.2 %) | long |
| 6 |  87 | 5 208 | 1.7 GB | 51 MB | **0 (0 %)** | clean detection throughout |
| 7 | 107 | 6 434 | 2.2 GB | 62 MB | **0 (0 %)** | **canonical "clean" reference take**; no glasses, no fallback events |
| 8 |  46 | 2 785 | 974 MB | 30 MB | **324 (11.6 %)** ⚠ | secondary high-fallback take |

Total: ~680 s of driver content, ~41 k frames. Ratio of MOV size to duration varies wildly because LLF re-encodes opportunistically based on motion + lighting, not because we changed quality settings — don't read anything into "take 2 is 3.2 GB."

Frame counts above come from `take.json`; `render_take.py` rounds down to the nearest 4-frame chunk before feeding PersonaLive, so the rendered mp4 is up to 3 frames shorter than the source.

## Render pipeline state

Each side-by-side mp4 was produced with:

```
scripts/render_take.py \
  --mov data/llf-takes/20260505_MySlate_<N>/MySlate_<N>_iPhone.mov \
  --anchor data/llf-phase2/asian_m__06_neutral.midframe.png \
  --out data/llf-take-renders/take<N>__asian_m.mp4 \
  --crop-strategy perframe   # legacy default at the time of capture
```

- Sustained ~12 fps render, no model drift across the batch.
- All eight rendered against the **same** anchor (`asian_m__06_neutral.midframe.png`) so outputs are directly comparable.
- `personalive_online.yaml` is the bundled-TRT config (ArchA T4 result); chunks of 4 frames feed `pipeline.process_input`.

The A/B/C subset (`abc/`) reruns takes 7 and 2 for the first 60 s under three crop strategies — `perframe` (legacy `crop_face`), `nocrop` (centre square), `ema` (causal EMA + 10 % forehead bias). Same anchor, same config, same seed-equivalent (the model is deterministic given the same driver tensor sequence).

## Known caveats by take

- **Take 4 (41.6 % face_mesh fallback)** and **take 8 (11.6 %)** — `render_take.py`'s fallback policy is "reuse last good crop". With near-half-frame fallback, take 4's *de facto* crop is a frozen rectangle for long stretches, so any input-jitter metric there is measuring the cropper's failure mode rather than driver instability. Treat these two as adversarial inputs for the failure recovery path, not as fair samples for crop-strategy comparison.
- **Take 2 (glasses-heavy)** — exhibits the "ghost glasses" failure (model renders glasses on the output even though the anchor has none). Per-frame crop becomes especially unstable near the glasses edge, which seeds the hallucination. This is the take to score if a fix targets the glasses-induced confabulation. The 60 s A/B/C subset (`abc/take2_*`) covers the worst stretch.
- **Take 7 (clean reference)** — zero face_mesh fallbacks across 6 432 frames. Whatever you measure on take 7 is a property of PersonaLive itself, not the cut. ArcFace cosine drift on the perframe render is 0.887 → 0.700 over the take (slope ≈ −0.014 per kfr): **the model drifts off-identity on a clean input over ~107 s**, motivating the anchor-reinjection design (see `2026-05-05-personalive-architecture-notes.md`).
- **Takes 1 / 3 / 5 / 6** — uneventful; useful as additional samples but no individual finding rests on them.

## Six failure modes observed across the batch

Numbered as in `2026-05-05-personalive-take-render-observations.md` so cross-references stay stable:

1. **Cut jitter on the driver side.** Per-frame `crop_face` recomputes bbox each frame; the *original* (left pane) wobbles in a way the source MOV does not. See `project_facemesh_crop_wobble.md`.
2. **Head clipping.** Face bbox often crops the top of the head off. Face_mesh landmark 10 is upper-mid forehead/eyebrow region (verify-before-relying on it for a clip metric — open question in the observations doc).
3. **Artifact persistence.** Once an artifact appears it almost never leaves. Autoregressive temporal module locks features in.
4. **Clothing edges seed artifacts.** Garment outlines in the driver appear to nucleate output artifacts.
5. **Glasses → extreme cut jitter.** Cropper instability is amplified when driver wears glasses.
6. **Ghost glasses.** Combined effect of 3 + 5: model paints on glasses even when the anchor has none.

A/B/C result on takes 7 + 2: A (legacy perframe) is dominated by both B (no crop) and C (causal EMA). C wins on identity drift slope and is the realtime default; B is competitive for fixed-framing capture. Detail in the observations doc.

## ARKit blendshape CSV ingestion — verified 2026-05-05

`MySlate_<N>_iPhone.csv` is the load-bearing ARKit sidecar (NOT
`frame_log.csv`, which is per-frame timestamps only). Schema:

- 63 columns: `Timecode`, `BlendshapeCount` (=52), 52 ARKit blendshape
  coefficients in Apple's order (`eyeBlinkLeft` through `tongueOut`),
  `HeadYaw`/`Pitch`/`Roll`, `LeftEye{Yaw,Pitch,Roll}`,
  `RightEye{Yaw,Pitch,Roll}`. All rotations in radians.
- One row per video frame; row `i` aligns with MOV frame `i` modulo
  Live Link's typical 1–2-frame edge slop (negligible for distill
  purposes at stride=2).
- `LeftEyeRoll` and `RightEyeRoll` are always 0.0 (Apple doesn't expose
  eye roll in the wire format), reducing the per-frame independent
  rotation count from 9 to 7.

Loader: `src/arkit_bridge/llf_csv.py:load_llf_b61` returns `(N, 61)`
float32 with the 52 blendshapes re-permuted to canonical Apple order.

**Per-take row counts** (from CSV ingestion during arkit-bridge pair
extraction, 2026-05-05):

| take | CSV rows | stride=2 pairs |
|---|---|---|
| 2 | 7440 | 3720 |
| 3 | 3212 | 1606 |
| 4 |  813 |  407 |
| 5 | 7340 | 3669 |
| 6 | 5208 | 2604 |
| 7 | 6434 | 3217 |
| 8 | 2597 | 1299 |
| **total** | **33044** | **~16522** |

CSV row counts are 1–48 frames *short* of the `take.json` MOV frame
counts (e.g. take 2 MOV=7440 vs CSV=7440 ✓; take 3 MOV=3248 vs
CSV=3212, diff=36). Treat MOV as the authoritative length and trim
trailing MOV frames that exceed CSV range — `extract_arkit_pairs.py`
does this with `if fi >= len(b_all): break`.

## What we don't have yet

- **No depth-stream usage.** TrueDepth `depth_data.bin` is captured for every take but never read. Reserved for the MetaHuman-Animator depth path.
- **No multi-anchor renders.** All 8 takes were rendered against `asian_m__06_neutral.midframe.png` only. Cross-anchor identity drift behaviour is unmeasured.
- **No full-length A/B/C.** The 60 s subset is enough to establish A is dominated; distinguishing C vs B on long-window identity drift would require the full 107 s + 124 s rerun (deferred).
- **No audio alignment.** Pipeline drops audio. If a future demo needs lip-sync grading, audio re-mux from the source MOVs is straightforward but untested.

## Cross-references

- `2026-05-05-personalive-take-render-observations.md` — qualitative observations + A/B/C numbers (this catalog cites it; don't duplicate the tables there)
- `2026-05-05-personalive-architecture-notes.md` — wrapper internals + anchor reinjection design space, motivated by take 7's identity drift
- `_topics/neural-deformation-control.md` — interpretation layer over the dated docs in this thread
- `scripts/render_take.py` (PersonaLive repo) — produced every render listed above
- `vamp-interface/scripts/artifact_score.py` — output-side scorer (ArchA T10)
