# Chibi-style head — artist briefing & interview guide

A guide for talking to an artist who understands the chibi aesthetic. The artist does not need to know anything technical. The goal is to come away with answers we can turn into concrete, measurable changes to a 3D head.

## What to give the artist before the conversation

- A current render of the realistic head (our normal LAM output of the anchor face), ideally from the front and a 3/4 angle, and a short clip of it blinking and talking.
- Ask them to bring or pull up **3–5 reference images** of chibi heads — spanning *subtle* (only slightly stylized) to *extreme* (maximum chibi). We need the range, not one example.
- A printout or digital copy of the normal render they can **draw on top of** (overpaint / annotate). A marked-up image is worth more than any verbal description.

## What we can and cannot change

Be explicit with the artist up front, so their advice lands inside what we can build.

**We can change:**
- The **proportions and shape** of every part of the head and face — how big, how round, how long, how far apart, how high or low each feature sits.
- The **surface colour** of the head — skin tone, saturation, flatness vs. shading, blush.

**We cannot change (please don't base the look on these):**
- **Hair** — style, length, volume, colour are locked. The look must work with the existing hair.
- **Accessories** — no glasses, hats, ears-additions, etc.
- **The body** — there is no body. This is a head only. "Big head, tiny body" is not available to us; the chibi-ness must live entirely in the head and face.
- **Adding or removing features** — we reshape what's there; we don't add a feature that isn't.

**It must still move.** The head will blink, open its mouth, and look around (driven by a real performance). Whatever the artist describes has to survive motion — e.g. very large eyes still have to close fully on a blink, a small mouth still has to open to talk. Ask them to keep this in mind.

## The kind of answer we need

For each part of the face, we need a **relative change with a magnitude** — not an absolute description. The frame is always "compared to the realistic head you're looking at."

A usable answer looks like:
> "The eyes get **much bigger** — roughly **double** the height — and move **slightly lower and further apart**. The shape rounds out, less almond, more circular."

A *not* usable answer:
> "Make the eyes cute."

So coach the artist: for every change, give us **direction** (bigger/smaller, higher/lower, rounder/sharper, closer/wider) and **amount** (slightly / noticeably / a lot / halved / doubled / tripled). Reference images and overpaints count as the amount.

## The questions

Go region by region. For each, ask: what changes, which direction, and by how much.

**Overall head shape.** Does the skull/cranium get bigger relative to the face? Does the whole head get rounder? Does the face get shorter top-to-bottom? Where on the head does the "weight" sit?

**Forehead.** Taller or shorter? More or less rounded? Does the hairline effectively sit higher or lower?

**Eyes.** This is usually the biggest single change — spend time here. How much bigger? Does the shape change (almond → round)? Do they move lower, higher, further apart, closer? Do they get *the most* attention, or share it with other features? **And in motion:** when the eyes are this large, what should a blink look like — full closure, or a softer look?

**Eyebrows.** Size, thickness, height above the eye, shape.

**Nose.** Smaller? Much smaller? Does it nearly disappear? Less projecting (flatter to the face)?

**Mouth and lips.** Smaller, wider, narrower? Lips fuller or simplified? Where does it sit vertically? **In motion:** it still has to open to talk — does a small chibi mouth open differently?

**Cheeks.** Fuller and rounder? How much? Do they push outward, giving a soft/puffy look?

**Jaw and chin.** Almost always a major change. How much shorter is the jaw? Does the chin shrink, round off, or nearly vanish? Does the jaw narrow?

**Ears.** Bigger, smaller, repositioned, or left alone?

**Surface and colour.** Should the skin look flatter / more even (less photographic shading)? More saturated? A blush on the cheeks? Should it read as "painted" or stay realistic in colour even with stylized proportions?

**The essence.** If they could only change **one or two things** and had to leave the rest realistic — what would they change to make it most read as chibi? (This tells us what carries the look.)

**The failure modes.** What makes a chibi attempt go wrong — uncanny, off, "trying too hard"? What do amateurs over-do? (This tells us what to *constrain*, not just what to push.)

**The intensity dial.** Imagine a slider from "barely chibi" to "extreme chibi." As you push it up, which features change *first*, which change *most*, and which barely change at all? Does anything change *non-uniformly* — e.g. eyes keep growing but the nose stops shrinking past a point? (This is how we get a principled "more/less chibi" control instead of one fixed look.)

## After the conversation — what to capture

Write up, per facial region: the direction of change, a magnitude in plain words, and any reference image or overpaint. Note separately the artist's answer to *the essence*, *the failure modes*, and *the intensity dial* — those three shape how we build the control, not just the single look.

---

## Appendix — internal: how answers map to our pipeline (not for the artist)

This is the operationalization target. Each artist answer should land on one of these.

- **Per-region proportion change** → a target for the FLAME-mesh warp (`W(mesh; θ)`). Region size/position/roundness answers become landmark-ratio targets: eye-area÷face-area, cranium height, jaw length, cheek fullness, nose projection, inter-ocular distance. These become the explicit "chibi ratio spec" the warp is optimized against.
- **The intensity-dial answer** → defines the chibi-strength axis: which ratios move fast, which saturate, whether the map is non-linear. This replaces the hand-tuned SY×SR spline knots with a justified curve.
- **Motion answers (blink, mouth-open)** → ARKit blendshape-magnitude rescaling, so expression still closes the eye / opens the mouth on the deformed proportions. Directly feeds the iris-leak work.
- **Surface/colour answers** → SH-DC vertex-colour edits (the existing `LAM_EDIT_VERTEX_COLORS_OBJ` path).
- **The essence + failure-mode answers** → loss weighting: what to push hard, what to constrain (regularize) so the optimizer doesn't overshoot into uncanny.

Anything the artist raises that maps to *none* of these (hair, accessories, body, lighting) is out of scope — note it but set it aside.
