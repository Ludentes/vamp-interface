---
status: live
topic: arkit-bridge
---

# LivePortrait streaming: a strong win on latency and stylized-anchor handling

End-to-end webcam → FasterLivePortrait (TRT) → v4l2loopback → OBS, live
on RTX 5090, paired anchor pass on Flux-portrait + Kiprensky's Pushkin
painting. Latency feels near-instant. Anchor handling on a painting
*works* where PersonaLive collapsed.

## Numbers

| | PersonaLive bridge (V2 cohort-stream) | LivePortrait (FasterLivePortrait, TRT) |
|---|---|---|
| per-frame infer | 124 ms (fp16, 4-step DDIM) | **17.8–22.4 ms** (TRT engines) |
| batching | 24-frame cohort, prep 820 ms / batch | none, per-frame |
| structural latency floor | ~3 s (cohort + prep) | ≈80–150 ms (1 cam frame + infer + sink) |
| writer prebuffer needed | `--writer_prebuffer 24` (~3 s) | `--writer_prebuffer 2` (~80 ms) |
| sampler complexity | stateless evenly-spaced packet sampler over LLF ring | none — direct cv2.VideoCapture |
| bridge | ARKit b_61 → student → implicit kp → seams | none — RGB → LP MotionExtractor |

Live-run telemetry (Flux east-asian-adult-male anchor, 12 s):

```
infer median 17.8–22.4 ms, p95 28–36 ms
producer ≈21.5 fps (webcam-bound; infer would allow ~45 fps)
fresh_fps 20.0 / repeat_fps 10 at drain=30 → set drain=20 to zero out
q_mean 0.5–0.7, q_max 1–2, dropped 0
```

## Subjective verdict

User watching OBS with the daemon live:

> "Latency feel is great."

> "Some hair the model fails to keep the part of the head — expected.
> Great. Let's write this down as a strong win."

Tested anchors:

- `data/anchors/photoreal_grid/east_asian__adult__m__seed1212.png` — Flux
  photoreal portrait. Identity preserved, pose tracks, expressions land,
  near-instant feel.
- `data/anchors/painting_pushkin/pushkin__01_kiprensky.jpg` — Kiprensky
  oil painting, c. 1827. Cross-domain warp from photoreal training data
  onto painterly source. Animates. Hair fails to hold the part of the
  head (warp interpolation can't synthesize occluded source texture from
  large rotations) — flagged as expected, not a regression.

This is the same `painting_pushkin` anchor PersonaLive's stylized-anchor
audit on 2026-05-06 hit collapse on (5/5 stylized refs collapsed to
FFHQ-blonde-glasses identity). LivePortrait passes it cleanly.

## Why the latency win is structural, not tunable

PersonaLive's 3 s floor is two things multiplied:

1. **Cohort-batch latency.** `Pose2VideoPipeline_Stream.prepare_v2()` is
   ~820 ms; `step()` runs 6 cohorts of 4 frames each at ~370 ms; that's
   one ~3 s cohort per ~3 s of real time. First cohort can't decode
   until prepare is done. This is intrinsic to RAIN+StreamDiffusion.
2. **Per-frame compute.** 124 ms × 24 frames = 2.97 s/batch on a 5090.

Both are gone in LivePortrait:

- **No batch state.** LP's MotionExtractor is feed-forward on a single
  256² crop. No prepare, no warmup cohorts.
- **No diffusion.** Warping_spade_generator is a single feed-forward
  GAN-class generator; 17 ms of TRT compute, not 124 ms of 4-step
  unrolled UNet.

The per-frame infer gap (~7×) plus the structural-batching gap (~30×) =
the ~30–50× glass-to-OBS gap we measured. Tuning PersonaLive can't
close this without removing diffusion from the path.

## Why the stylized-anchor win matters

The 2026-05-06 pivot memo
[`2026-05-06-pivot-decision-personalive-vs-liveportrait.md`](2026-05-06-pivot-decision-personalive-vs-liveportrait.md)
gated the entire LP-vs-PL decision tree on a 1-hour empirical test that
never ran. Today's Pushkin run is the first leg of that test, in
streaming mode rather than the offline yaw-stress contact sheet
originally specified, but with the same go/no-go question:

> Does LP preserve identity on a non-photoreal anchor where PersonaLive
> collapsed?

Pushkin: yes. We have not yet swept anime/duck/demon/zombie/orc.

Implication for the option space from the pivot memo:

- **Option C (hybrid)** is now the most interesting: PersonaLive for
  photoreal-only paths where its diffusion ceiling matters (mouth
  synthesis, large rotations with occluded inpaint), LivePortrait for
  stylized + the realtime VTuber default. Branch at the API boundary
  by ref-type or by "do you need <100 ms latency."
- **Option B (wholesale pivot)** becomes plausible if the realtime
  default is the only path we ship and the diffusion-only failure
  modes (long-clip drift mitigation via HKM, mouth/teeth synthesis,
  occluded-region inpaint) don't surface in production. We'd lose
  these capabilities but win latency + stylized handling + a much
  simpler stack.

## Code shipped

- `src/arkit_bridge/paced_sink.py` — `PacedSinkWriter` extracted from
  `scripts/streaming_bridge.py` (PersonaLive daemon) so both daemons
  share a single drain implementation. No behavioral change.
- `scripts/streaming_bridge_lp.py` — new daemon. Webcam → FLP TRT pipe
  → `PacedSinkWriter` → `V4L2Sink`. Per-frame, no batching, no bridge
  seams. CLI flags `--cam_index`, `--driver_video` (offline smoke),
  `--mp4_out` (file sink), `--fps`, `--writer_prebuffer`, `--animal`.

Run command:

```bash
PYTHONPATH=/home/newub/w/vamp-interface/src \
/home/newub/w/FasterLivePortrait/.venv/bin/python \
    scripts/streaming_bridge_lp.py \
    --reference data/anchors/painting_pushkin/pushkin__01_kiprensky.jpg \
    --cam_index 0 --device_path /dev/video10 \
    --fps 20 --writer_prebuffer 2
```

Run inside the FasterLivePortrait venv — PersonaLive's venv lacks the
TRT plugin path. Cross-venv import works because `paced_sink.py` and
`v4l2_sink.py` are stdlib + numpy only.

## Crop-variant sweep on Pushkin (live OBS)

Three variants on the Kiprensky anchor with `--crop_driver` ON,
otherwise tweaking `crop_params`:

| variant | src_scale | dri_scale | observed |
|---|---|---|---|
| tight | 1.7 | 2.0 | head-only output, expression visible but framing very close |
| default (yaml) | 2.3 | 2.2 | "samish, not full portrait, just the head" — the framing user saw first |
| **wide** | **3.0** | **2.5** | **"actually better … this result is usable"** |

User feedback at the wide setting:

> "Actually better. But LP crop and rotation is still weird. The
> transform is problematic. But this result is usable."

So:

- Wider source canvas (more of the painting visible around the head)
  helps perceived quality on a non-FFHQ-shaped anchor. Plausibly because
  Kiprensky's painting includes ornate framing / shoulder context, and
  the wider source crop gives LP's WarpingSpadeGenerator more low-
  frequency surround to interpolate against — fewer obvious "flat
  edges" at the crop boundary.
- LP's *transform* (driver-pose → source-keypoint rotation) is still
  visibly off — the user reports rotation feels wrong even at the
  best-found crop. Watch list for follow-up: this could be (a) the
  driver crop's `dri_scale=2.5` over-amplifying head-pose extraction,
  (b) LP's relative-motion mode confusing per-frame rotation against
  the painted source's frontal pose, or (c) the underlying warp's
  failure mode on a high-rotation pose for which the painted source
  has no corresponding profile pixels.
- "Usable" was achieved; "production-ready" was not. Same verdict as
  before.

## "Head pumping" — face scale changes with expression

Surfaced live during the Flux east-asian sweep at `src_scale=1.7
dri_scale=2.0` (tight crop):

> "There is a very interesting effect (now very visible) — the face
> increases based on the expression. I suspect this is again a crop
> issue but somewhere deeper."

This is **LP's known head-pumping quirk**, not a crop bug. The crop is
acting as an amplifier, not the cause:

- LP's MotionExtractor outputs a per-frame scalar `scale` along with
  `R`, `t`, `exp`. Combined keypoints ≈
  `scale_d · (canonical_kp @ R_d + exp_d) + t_d`.
- In relative-motion mode (default), `scale_d` is taken as a ratio
  against driver frame 0. Opening the mouth, raising brows, etc.
  changes the apparent face-bbox vertical extent → MotionExtractor
  emits a different `scale` → the warped face inflates/deflates per
  frame. User during Flux-tight: "When I full jaw drop, the camera
  retreats." Maximum jaw drop = maximum vertical face extent → maximum
  `scale_d` → maximum ratio inflation → composed keypoints push
  outward → warp shrinks the head to fit. Reads to the viewer as the
  camera zooming out.
- Tight source crops fill more of the 512² canvas with face, so the
  same ±5% scale wobble shows as ±25 px head-size oscillation. Wide
  crops *hide* it; they don't fix it.

**Whole-portrait probe falsified, same session:** with
`src_scale=5.0` (face becomes ~1/5 of canvas, full painting in frame),
the warp degraded heavily — confirmed user verdict
"warp [does not work], the mismatch is too big for the model to
handle." LP fundamentally requires face-centered ≈512² source as
trained; "no transform" mode is not viable without retraining the
WarpingSpadeGenerator.

Standard upstream mitigations (none applied this session):

1. Disable relative-motion mode — pin `scale_d` to source's value.
   Kills the pump; loses some head-size dynamics. Search FLP for the
   flag (`flag_relative_motion` or similar in `infer_params`).
2. One-Euro filter on the `scale_d` series, like FLP already does
   for `R_d` (`R_d_smooth = OneEuroFilter(4, 0.3)`).
3. Clamp `scale_d / scale_d_0` to a tight range, e.g. `[0.97, 1.03]`.

**Mitigation 1 tested live on Flux east-asian, wide crop:** kills the
pump. New observations from the user:

- "There is some rigidity." — absolute motion mode over-corrects: face
  no longer breathes/scales naturally with any pose, only with
  expression deltas relative to source. Mitigation 3 (clamp ratio in
  relative-motion mode) becomes the right fix: keep relative motion's
  natural breathing but clamp `x_d_i_info['scale'] / x_d_0_info['scale']`
  to a tight band so the pump can't run away on extreme expressions.
- "While less visible now we still have jitter. Noticeable one. We
  will need to find the source of it." — every-frame MotionExtractor
  output is independent. FLP applies `OneEuroFilter` to `R_d` and
  `exp_d` only when `is_source_video=True`. For image-source (our
  case) there is **zero temporal smoothing** on driver pose/expression.
  Fix: monkey-patch FLP's `pipe.run()` to apply OneEuroFilter to
  `R_d`, `t_d`, `scale_d`, `exp_d` regardless of source type. ~15
  lines.

## Source-crop framing artifacts (Flux + wide src_scale)

Two issues surfaced at `src_scale=3.0` on a Flux portrait:

1. **Black padding visible in output.** Wide src_scale makes the source-
   crop window extend beyond the actual anchor image dimensions. FLP's
   `crop_image()` pads with black, and at scale=3 the padding becomes
   visible at the canvas edges. Three fixes:
   a. Pre-pad anchors with reflection/replication padding before
      passing them in (cheap, content-preserving).
   b. Use a smaller `src_scale` (e.g. 2.5) and accept tighter framing.
   c. Mask the padded regions out of the warp loss (would need patch
      to FLP).
2. **LP-detected source pose isn't axis-aligned.** "LP read the anchor
   as not perfectly staring so the angles on the polygon are no 90,
   meaning not rectangle." FLP's `crop_image_by_bbox` (and its kin)
   compute a *similarity transform* from face landmarks, which
   includes rotation alignment to make the face upright. If the
   detector reads the source as slightly tilted, the crop polygon
   tilts with it — and the 512² crop is no longer an axis-aligned
   sub-rectangle of the source image. The user sees the rectangular
   output, but its mapping back into the source frame is rotated.
   Mitigation: replace the crop call with an axis-aligned bbox crop
   (no rotation alignment), accepting whatever face tilt the source
   has. Trade-off: face is no longer upright in the canvas, which
   could degrade WarpingSpadeGenerator (trained on upright faces).

These are all *implementation* issues with FLP's preprocessing /
post-processing, not fundamental warp limits. All three could be
addressed without retraining the underlying models.

Add to the LP-quality watch list. Doesn't change the production
verdict (LP not production-ready), just clarifies *why* — beyond hair
warp limits and rotation transform feel, there's a structural scale
oscillation at every expression.

## Frame-0 calibration is load-bearing

Surfaced live during the B&W FFHQ anchor test:

> "Also first frames matter a lot. Have to remember we might need a
> calibration."

> "Jitter is noticeable. And pumping is strong." (B&W Flux anchor with
> pasteback, default relative motion, default crop scales — i.e. closer
> to LP-ideal conditions than any other variant tested.)

Mechanism: in relative-motion mode, LP captures `R_d_0`, `x_d_0_info`
(including `scale_d_0`) at `first_frame=True` and treats them as the
neutral anchor. All subsequent driver pose/expression is computed as
a delta against this baseline. If frame 0 is mid-blink, mid-speech,
slightly turned, or off-center, every later frame is implicitly
interpreted as "delta from that already-non-neutral head."

This is why the head pumping, scale wobble, and rotation feel are all
amplified by an unlucky frame 0 — we've been measuring the symptoms
of *first-frame noise* on top of the structural relative-motion
issues. Even on a near-ideal anchor (FFHQ-tight, B&W to strip color
texture noise, pasteback to preserve canvas) the pump and jitter
remain "noticeable / strong." So frame-0 calibration is necessary
but not sufficient — the underlying relative-motion scale ratio and
the per-frame independence of MotionExtractor outputs are the deeper
sources.

Two calibration designs we should ship before next live session:

1. **3-second neutral-hold.** On daemon start, prompt the user
   "look straight, neutral expression, hold 3 s," accumulate
   MotionExtractor outputs over that window, use the median as
   `x_d_0_info` / `R_d_0`. Single user instruction, robust output.
2. **Find-the-neutral-frame.** Collect first ~30 frames silently,
   pick the one whose `R_d` is closest to identity and `exp_d`
   closest to zero, use as anchor. No instruction needed but slower
   start.

Either should monkey-patch `pipe.run()`'s first-frame branch (where
`self.R_d_0` is captured) without touching FLP source.

## Mitigations applied & measured live

Two of the three cheap fixes from the watch list shipped as
`scripts/streaming_bridge_lp.py` flags this session:

- `--smooth_motion` — monkey-patches `MotionExtractor.predict` with
  six `OneEuroFilter`s (mincutoff=4, beta=0.3, matching FLP's existing
  `R_d_smooth` defaults) on driver-side outputs (pitch/yaw/roll/t/exp
  /scale). Patch applied *after* `prepare_source` returns so the
  source pass is unaffected.
- `--scale_clamp BAND` — clamps `scale_d` to first-driver-frame
  `scale_0 × (1 ± BAND)`. Tames pumping while preserving relative-
  motion expression handling. Wired but not yet measured live.

User verdict on B&W Flux anchor + pasteback + relative motion +
`--smooth_motion` (no clamp yet):

> "Jitter is all but gone."

Then layered `--scale_clamp 0.03` on top:

> "Still pumping. Not enough regretably."

Tightened to `--scale_clamp 0.0` (pin scale_d to frame-0 → ratio
always = 1.0 → `scale_new = x_s_info['scale']` flat, equivalent to
absolute mode for *scale only* while relative motion still governs
exp/R/t):

> "Better. Very little left."

So the recipe that lands the pump-and-jitter fix simultaneously is
**`--smooth_motion --scale_clamp 0.0`**. Anything looser than 0.0 lets
the pump back in (3% was already too generous). The architecture
intuition was correct — the right axis to fix the pump on is scale
*alone*, leaving exp/R relative — but the right band is full pin,
not "slightly relaxed."

User verdict on the absolute-motion variant (`--no_relative_motion`),
prior to wiring the smoother:

> "Much better regarding the pump. Jitter is mild. But … the face is
> much more squinted and tight lipped now."

Mechanism: in absolute motion, source `x_s_info['exp']` is *replaced*
by driver `x_d_i_info['exp']` for the indices being animated (rather
than `source_exp + (driver_exp - driver_0_exp)` in relative mode).
The user's natural narrower eyes / tight resting lips overwrite the
source's expression baseline. Confirms: absolute motion is the wrong
axis to fix the pump on; relative-motion + scale clamp (mitigation 3)
is the right combination.

## Recommended LP daemon recipe (end-of-session best)

For shipping LP as the live-default path on photoreal / B&W / FFHQ-
shaped anchors:

```bash
PYTHONPATH=/home/newub/w/vamp-interface/src \
/home/newub/w/FasterLivePortrait/.venv/bin/python \
    scripts/streaming_bridge_lp.py \
    --reference <anchor.png> \
    --cam_index 0 --device_path /dev/video10 \
    --fps 20 --writer_prebuffer 2 \
    --crop_driver --pasteback \
    --smooth_motion --scale_clamp 0.0
```

For non-FFHQ-shaped anchors (paintings, body in frame): add
`--src_scale 3.0 --dri_scale 2.5` (Pushkin "usable" settings).

## Production verdict, end of session

> "Quality not production ready. Latency — amazing." — user, 2026-05-07

Three-way ranking after the live A/B:

1. **LP for live (recommended for shipping latency-bound use cases).**
   ~150 ms glass-to-OBS, no bridge, works on stylized anchors. Quality
   ceiling is acceptable for *some* use cases (low-stakes streaming,
   prototyping, scenarios where motion matters more than texture
   fidelity), not for the VTuber product the user has in mind.
2. **RGB-PL for offline / quality-bound use.** ~1–2 s glass-to-OBS at
   batch=8. Quality is the production bar. Shipped daemon, no further
   work needed.
3. **LLF-PL with the current bridge (v2_120k):** **don't ship.** Visible
   artifacts dominate the ~3 s latency cost; both losses without an
   offsetting win.

Implication for product strategy: the latency-vs-quality Pareto we have
*today* doesn't have a point that satisfies "live + production-quality
VTuber." Everything currently shipped is on one axis or the other.
Closing the gap requires either (a) accelerating PersonaLive (TRT path,
distillation, or the FlashPortrait integration thread), or (b)
end-to-end retraining a bridge directly into a fast pipeline. Bolting
onto v2_120k won't get there.

CLI knobs added late in the session (still in tree, not yet used to
generate evidence):

- `--crop_driver` — enables FLP's internal face-crop on the driver
  stream (retinaface + 106-landmark + `dri_scale=2.2`). Default off
  feeds the whole webcam frame resized to 256² to MotionExtractor.
- `--pasteback` — output = full source frame with animated head
  re-composited (shoulders, background) instead of the 512² head crop
  alone. Useful when the source portrait carries non-face context the
  user wants preserved (hair below the crop, painting frame, etc).

Either flag changes framing/composite, not the underlying quality
ceiling. Above verdict stands either way.

## A/B vs RGB-driven PersonaLive (no bridge)

Same session, same webcam, same anchors (Pushkin, Flux east-asian-adult-male).
PersonaLive driven by raw webcam RGB through its real motion_encoder /
pose_encoder (`streaming_bridge.py --driver rgb`, batch=8, fps=8). No
bridge seams installed; no student involved. This is PersonaLive's full
teacher path — the upper-bound quality this stack can produce in
streaming mode.

User's verdict watching OBS:

> "Latency very noticeable. Quality is better."

Implications, in plain language:

1. **The ARKit→student→implicit-kp bridge is a major source of the
   visible artifacts** — head/face decoupling, expression mismatch,
   identity drift at extremes — that we previously attributed to
   PersonaLive itself. RGB-PL on the same backbone produces visibly
   cleaner output than LLF-PL with the bridge. The 0.0114 ratio_mean
   structural floor on holdout_v3 is real but evidently does not
   capture what the renderer sees at the bridge / pipe seam.
2. **Production-viable LLF-PL needs the bridge baked into a full
   pipeline retrain**, not bolted on at inference. Either co-train the
   motion_encoder on b_61 directly, or distill the full pipeline (not
   just MotEncoder) end-to-end with the bridge in the loss. Today's
   v2_120k is a measurement/diagnosis artifact, not a deploy candidate.
3. **PersonaLive quality > LivePortrait quality, but PL latency kills
   it for live use.** ~1–2 s glass-to-OBS at batch=8 is "very
   noticeable" per the user. ~3 s at batch=24 is unusable. LP at
   ~150 ms is the only currently-running candidate for *live* feel.

The hybrid (Option C from the 2026-05-06 pivot memo) gets sharper:

- **Live VTuber default → LivePortrait.** Sub-150 ms, no bridge,
  acceptable on the anchors we tried (Flux + Pushkin). Hair/teeth/
  occluded-region weaknesses are documented and tolerable.
- **Quality-bound work → PersonaLive RGB-driven (no bridge).** Anything
  not real-time interactive. We already have this daemon shipped.
- **LLF-driven PersonaLive → blocked on retrain.** Don't ship the
  current bridge as the live path. Either invest in end-to-end retrain
  (multi-week, GPU-heavy) or shelve.

## What's deferred to next step
- Stylized gauntlet sweep (anime, duck, demon, zombie, orc) on LP to
  confirm the 5/5 PersonaLive collapse converts into a 5/5 LP pass —
  this completes the gating test from the 2026-05-06 pivot memo.
- Long-clip drift watch — LP's known weakness vs PersonaLive's HKM. Not
  visible in 12 s runs; show up in 5+ minute streams.
- Adding a status overlay (q_mean, infer_med) drawn on the OBS frame
  itself for streamer-side monitoring without the terminal.

## Companion docs

- [`2026-05-06-pivot-decision-personalive-vs-liveportrait.md`](2026-05-06-pivot-decision-personalive-vs-liveportrait.md)
  — the option-space memo this empirical leg resolves part of.
- [`2026-05-07-llf-streaming-pacing-and-sampling.md`](2026-05-07-llf-streaming-pacing-and-sampling.md)
  — the three primitives (paced writer, prebuffer, time-spaced source
  sampling). LP daemon reuses the first two; the third is unneeded
  because per-frame inference produces 1:1 frame-in/frame-out.
- [`2026-05-06-rendering-stack-replacement-options.md`](2026-05-06-rendering-stack-replacement-options.md)
  — broader backbone-swap survey.
