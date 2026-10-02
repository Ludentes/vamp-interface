---
status: live
topic: photobooth-sweep
---

# Qwen-Image-2.1 in ComfyUI — video notes and relevance

**Source:** Nerdy Rodent, "Qwen Image 2.1 - Runs Locally in ComfyUI",
YouTube, 2026-09-25, 16:53 — https://youtu.be/iJoo2RuAfZ0
**Why we watched it:** Qwen-Image-2.1 is the open-weight leader for
multi-reference editing in our landscape refresh
(`2026-09-27-face-identity-landscape-2026h1.md`). This is a hands-on demo,
not a benchmark: every verdict in it is the presenter's eyeball judgement.

## Summary for PMs

The video walks through Alibaba's new Qwen-Image-2.1 image model running
locally in ComfyUI, on the same kind of graphics card we use on the Windows
render box (an RTX 3090).

**Setup facts.**
- The model is a 7.26 GB int8 file and runs on a 3090.
- The license is no longer open source. Showing the images is allowed, but
  commercial use needs a deal with Alibaba.
- There is one new ComfyUI node, *Text Encode Qwen Image 2.1*. It takes a
  prompt plus up to 10 reference images and resizes them automatically.
- Typical settings are 16–30 steps at CFG 1. The video gives no timings in
  seconds.

**What it showed, in order.**

1. **Text to image.** A time machine with seven small text labels, mostly
   pointing at the right parts. A photo of a woman lying on grass came out
   reasonable apart from odd nails. Hands, paws and claws are still the weak
   spot. The presenter's tip is to describe each finger in the prompt.
2. **Mixed styles and extreme sizes.** It followed a prompt mixing a pencil
   sketch with photoreal surroundings, an anime style with thick outlines, and
   a very wide 4096×1024 canvas without breaking.
3. **Upscaling, two ways.** A fast 6-step light refine to 2× stays faithful.
   A prompt-driven "upscale image 1" adds more detail (better hands) but shifts
   the colours. The same trick can do other edits, e.g. "remove the
   background".
4. **Image to image.** Restyling (claymation → anime) only works with denoise
   very high, around 0.85. That is a much narrower useful range than older
   models.
5. **Instruction editing.** "Turn the hummingbird into a beaver" worked, but
   the whole picture shifted slightly. Adding a hand-drawn mask fixed that:
   nothing outside the mask changed.
6. **Style reference.** Given a cubist painting with "use image 1 as a style
   reference", it borrowed the painting's style (smooth curves) without
   copying its layout.
7. **Structure control without a ControlNet.** Feeding a depth map, or an
   OpenPose skeleton, as a reference image was enough. It followed the depth
   map very closely even when the prompt never mentioned it.
8. **Combining references.** Pose from image 1, *the man's face and beard
   from image 2*, clothes from image 3, converted to photorealism. It kept the
   face. A seven-reference scene (two people, a jacket, a crystal ball, a
   painting, a bar, an anime character) also worked, including fitting an
   anime-style person into a photoreal scene.

**Bottom line from the video:** the presenter was impressed by the
prompt-following and multi-reference composition. The known weaknesses are
hands, slight shifting during unmasked edits, and colour drift during
prompt-driven upscales.

## How this relates to our work

### It can do our whole photobooth in one model

Our current photobooth chain is three tools:

1. Z-Image Turbo generates the doll.
2. A ControlNet shapes it.
3. HyperSwap pastes the person's face on afterwards.

Demo 8 is the same job done in a single model: keep a real person's face from
one reference while taking pose and structure from another and clothing or
style from a third. Demos 6 and 7 cover the other two inputs we need. These
are a *style reference* (a matryoshka exemplar) and *structure from a
depth/edge map* (we use a Canny ControlNet for that today). If identity
survives into the doll style, the separate swap stage, along with its
~0.86 identity ceiling and HyperSwap's skin-tone wash, goes away.

The group photobooth is the obvious stretch case. The seven-reference bar
scene is close to "several people → one composed portrait on a chosen
background", which we currently build per-face and composite.

### It overlaps with the inswapper "known style" plan

Two days ago we scoped a zero-training edit to inswapper: per-block identity
strength, plus partly keeping the target's own feature statistics, packaged
as one preset per style. That plan and Qwen-Image-2.1 answer the same
question: given a style we know in advance, put a real identity into it.

- The inswapper route is small, fast (128px, CPU-able), and stays inside the
  current pipeline.
- The Qwen route replaces the pipeline but costs 16–30 steps of a 7B model per
  image, against our 6-step Z-Image.

### What the video does not tell us

- **Identity is never measured.** "It kept the face and beard" is a visual
  impression on a photoreal output. Nobody tests identity *into a stylized
  target*. That is our hard case: PuLID failed it outright, and ArcFace itself
  loses about half its accuracy on stylized faces.
- **Speed.** There are no wall-clock numbers. Our current recipe renders in
  about 7 s.
- **Licence.** It is research/non-commercial. That is fine for personal use,
  and a blocker if the photobooth is ever productised.
- **Pixel stability.** Unmasked edits drift. For the photobooth that argues
  for masked edits around the face, or a composite step.

### Proposed next step

Run a bounded spike on the Windows 3090 using the int8 file. Inputs: one
person photo + one matryoshka style exemplar + optionally the doll's
depth/Canny map. Use the same 20 identities as the photobooth sweeps. Score
them with our existing id_cos scorer and a human judging sheet against the
current production recipe.

The question it answers is whether Qwen-Image-2.1 keeps identity into the
doll style at least as well as generate-then-swap. If yes, the inswapper knob
work becomes a fallback. If no, the knob work goes ahead as planned.
