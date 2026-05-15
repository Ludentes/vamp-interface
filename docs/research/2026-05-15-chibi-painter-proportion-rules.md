---
status: live
topic: lam-chibi-recipe
---

# Research: Digital-Painter Rules for Drawing Chibi Faces

**Date:** 2026-05-15
**Sources:** 8 sources (Mary Li Art, Clip Studio "Art Rocket", AnimeOutline, MediBang, FIHEROE, Wikipedia, TV Tropes, plus aggregated tutorial search results)

---

## Executive Summary

Chibi face design is a small set of consistent, well-agreed deformation rules — not a precise numeric canon. Across every illustrator tutorial the same moves recur: (1) the head becomes a big rounded *sphere/block*; (2) all facial features are pulled **down and clustered** into a small zone in the lower half of the head; (3) eyes become huge, round, lower-placed, and **spaced ~one eye-width apart**; (4) the nose collapses to a dot or button and the mouth to a small mark; (5) the chin/jaw is rounded off and a generous smooth expanse of "face" sits below the mouth; (6) skin is flat — cheek volume, nasolabial folds, and the nose-bridge ridge are all erased. The recurring failure mode ("creepy chibi") is **stylistic mismatch**: realistic shading or realistic micro-anatomy left on top of chibi proportions. This matters for our pipeline because most of these rules are *geometric vertex displacements* that are naturally differentiable, while the skin-flattening rule is a *shading/appearance* edit that is not geometric.

## Key Findings

### The head is a sphere/block, not a face

Tutorials are unanimous that the chibi head is drawn as a **3D rounded volume** — "the head is a sphere, not a circle" — with sharp jaw edges replaced by smooth curves, and "no chins, at least not sharp ones … usually very rounded for cuteness" [1][2]. The head-to-body ratio collapses to 1.75–3 heads tall (most commonly 2–2.5), making the head roughly one third to one half of total height [2][3][6]. For our purposes the operative claim is the **head shape**: a tall rounded block / sphere, with jaw corners and chin point dissolved into curvature. This is consistent with the user's note that the current pear-shaped `chibi=2` is "somewhat ok" but a more block-shaped head is wanted — the tutorials favor a fuller, rounder cranium than a pear taper.

### Features cluster downward into a small zone

The single most-repeated instruction: "draw everything as close as possible to the middle of the **face** (not the head)" so that eyes, nose, and mouth "can be drawn inside a small circle in the lower part of the head" [1]. The eye horizontal guide line sits **below the head's mid-line** — "a little lower than halfway down the face, or about a third of the way down," "just a smidge below the center horizontal eye-level guide" [4]. The whole eyes-nose-mouth group occupies only about **1/3 to 1/2 of the face** vertically, leaving a large empty rounded forehead above and a large empty rounded chin/jaw below. This directly supports the user's points 2 ("eyes higher" — meaning higher *within the feature cluster*, but the cluster as a whole is low on the head) and 7 (a prominent expanse from mouth to bottom of head).

### Eyes: huge, round, lower, and widely spaced

Eyes "express the most emotion, so draw them very large" — typically spanning **~1/3 of the face width each** [3][4]. Vertical placement is lower than realistic portraiture. Spacing is the load-bearing cuteness lever the user flagged: place eyes **far apart, ~one eye-width between them** [5]. This is wider than the classic realistic portrait rule (where the inter-eye gap also ≈ one eye-width, but eyes are smaller, so proportionally the chibi gap reads as larger). Wide spacing + large size + low placement is what produces "innocent," not "child" — narrow spacing pushes the face toward a literal infant and into the creepy zone (see below).

### Nose collapses to a dot/button; mouth to a small mark

Every source: the nose is "just a tiny dot," or omitted entirely, with all sharp structure removed — "replace sharp parts such as the nose with curves" [1][2][3]. There is **no nose bridge, no visible nostrils, no defined tip-as-separate-form**. The user's brief specifies a "пимпочка" (button): tip narrower than the mouth, no visible nostrils, no bridge ridge. The mouth is small and simple — for an adult-styled chibi a plain **strip/line** rather than a child's "bow" shape, per the artist brief. Both nose and mouth get pulled toward the eye cluster.

### Vertical spacing of the lower face

The artist brief gives a specific ratio our generic tutorials do not state numerically: **nose-tip-to-mouth distance ≈ ½ of mouth-to-chin distance**. Tutorials corroborate the *direction* — "the mouth goes roughly halfway between the nose and the chin" is the realistic rule, and chibi compresses the upper part (nose pulled toward mouth) while *preserving or enlarging* the lower expanse (mouth to chin) [from aggregated proportion search]. The net effect: nose and mouth sit close together, then a long smooth sweep of jaw/chin below. The user's point 7 reframes this as the anti-creepy mechanism: real toddlers have almost no chin, so a chibi that mimics infant proportions exactly reads as uncanny; keeping a generous (rounded, not pointed, not large-jawed) mouth-to-chin span de-infantilizes it.

### Skin is flat — an appearance edit, not geometry

Tutorials describe simplification as "reduce information while leaving key features" and explicitly call out removing anatomical detail [1][3]. The artist brief sharpens this: no visible cheek volume, no nasolabial-fold shadows, no nose-bridge ridge — "perfectly smooth" skin. Critically, the user's own assessment is that flat skin is **mostly post-processing / shading**, not mesh shape. On a Gaussian-splat head this means flattening the SH-DC colour field and suppressing shading cues, plus mildly reducing the actual geometric relief of cheeks and nasolabial region — it is a *mixed* geometry+appearance edit, and the appearance half is not a vertex displacement.

### The "creepy chibi" failure mode

The consistent diagnosis: chibi looks uncanny when it is "too realistic for the exaggerated proportions" — realistic rendering/shading or realistic micro-anatomy mixed with chibi proportions creates discord [6][7]. Two practical implications for us: (a) the skin-flattening step is not optional polish — leaving photoreal skin texture on chibi geometry is *the* documented uncanny trigger; (b) proportions that copy a literal human infant (tiny chin, features crammed very low, eyes close together) also misfire — the user's point 7 is independently corroborated by the "semi-realistic chibi with realistic build feels unsettling" reports.

## Vertical Division Model — thirds → quarters

This is the formulation the user asked for: where the horizontal feature lines sit, expressed as fractions of the **vertical face axis** (crown of the head silhouette `y=0` → bottom of chin `y=1`). It lets us state chibi as a *remapping* of the realistic line positions, which is exactly a per-vertex vertical displacement field. The artist gave the core rule directly: **a standard head is read in thirds, a chibi head in quarters.**

**Realistic adult — thirds.** The classic Loomis rule divides the head crown→chin into **three equal thirds** [9][10], and the artist's brief states the same boundaries:

- First third: **crown → brows** (the whole forehead)
- Second third: **brows → nose tip**
- Third third: **nose tip → chin**

Eyes sit just below the brow line, near the head's vertical **midline** in adults [11].

| Line | Realistic adult `y` |
|---|---|
| Crown | 0.00 |
| Brow | ~0.33 (1st/2nd third boundary) |
| Eye centre | ~0.42 |
| Nose tip | ~0.67 (2nd/3rd third boundary) |
| Mouth (lip line) | ~0.80 |
| Chin | 1.00 |

**Infant.** Diagnostic age marker: brows — not eyes — fall on the midline, so the **eyes sit below the midline**, and the cranium/forehead is enlarged [12]. Copying this literally is the documented creepy trigger (rule 7).

**Chibi — quarters.** The artist's rule: divide the head crown→chin into **four equal quarters** and place features *on the quarter lines*:

- **Eyes on the 2/4 line** (`y = 0.50`) — the head midline
- **Mouth on the 3/4 line** (`y = 0.75`)

Nose sits between eyes and mouth; with the artist-brief ratio (nose→mouth ≈ ½ of mouth→chin) and mouth→chin = `0.25`, the nose lands at `y ≈ 0.625`. Eyes are ~1/4 of the head tall, so the eye span is ≈ `0.375 → 0.625` centred on the 2/4 line.

| Line | Chibi `y` | Move vs realistic |
|---|---|---|
| Crown | 0.00 | — |
| Eye **top** | ~0.375 | — |
| Eye **centre** | **0.50** (2/4 line) | down ~0.08 from brow-adjacent realistic eye |
| Eye **bottom** / nose top | ~0.625 | — |
| Nose (button) | ~0.625 | up + collapsed to dot |
| Mouth (strip) | **0.75** (3/4 line) | up toward nose |
| Chin | 1.00 | — |

The defining chibi move: **thirds become quarters**, and the three feature bands become — counting in quarters — **2 : 1 : 1**.

| Band | Realistic | Chibi | Meaning |
|---|---|---|---|
| Crown → eye-top | ~0.40 (most of 1st third) | **~0.375** (≈ top 1.5 quarters) | big empty rounded forehead |
| Eye-top → mouth | ~0.38 | **~0.375** (eyes quarter + nose quarter) | dense feature cluster — huge eyes + button nose + strip mouth |
| Mouth → chin | ~0.20 | **0.25** (bottom quarter) | smooth empty chin sweep |

The lower-face artist-brief ratio holds: nose `0.625` → mouth `0.75` = `0.125`; mouth `0.75` → chin `1.00` = `0.25` → **exactly 1 : 2** ✓. The eye centre on the 2/4 line is the head midline — so "eyes at half the face" is literal in the quarter grid, no reference-frame juggling needed.

Note this supersedes the earlier synthesis draft (which had eyes at `0.58`, mouth at `0.81`): the artist's quarter grid is a direct instruction and is cleaner — eyes and mouth land exactly on quarter lines, and the 1:2 nose-mouth-chin ratio is exact rather than approximate.

These numbers are a **starting target**, not a canon — the literature gives the *structure* (which lines move which way) but not exact fractions; tune the `y` values against the reference image. The whole model is a 1-D vertical remap `y_chibi = f(y_realistic)` plus the per-feature scale rules below, and `f` is piecewise-linear and differentiable.

## Rule Set → Deformation Recipe Mapping

| # | Painter rule | Mesh operation | Differentiable? |
|---|---|---|---|
| 1 | Head = rounded sphere/block, no sharp jaw/chin | Per-vertex radial inflation of cranium + smooth jaw/chin corners | Yes — vertex displacement field |
| 2 | Features clustered into lower-mid face, 1/3–1/2 of face height | Vertical compression + downward translation of eye/nose/mouth vertex groups | Yes — affine per-region |
| 3 | Eyes huge, round, lower, ~1 eye-width apart | Scale-up eye-region verts about each eye centroid; widen inter-eye gap; lower centroids | Yes — per-region scale + translate |
| 4 | Nose → button: no bridge, no nostrils, tip narrower than mouth | Collapse nose-bridge verts toward face plane; shrink + round tip | Yes — vertex displacement; loss on bridge-ridge height |
| 5 | Mouth → small strip | Shrink mouth-region verts; flatten lip relief | Yes — per-region scale |
| 6 | Nose-to-mouth ≈ ½ of mouth-to-chin | Pull nose+mouth cluster up relative to chin; preserve/extend chin span | Yes — measurable as a vertex-distance ratio loss |
| 7 | Generous rounded mouth-to-chin expanse (anti-creepy) | Keep/extend lower-face verts; round, do not point, the chin | Yes — but needs an explicit "not-infant" guard |
| 8 | Flat skin: no cheek volume, no nasolabial shadow, no bridge ridge | (a) reduce geometric relief of cheek/nasolabial verts; (b) flatten SH-DC colour + kill shading cues | Partly — (a) geometric/differentiable; (b) appearance, NOT a vertex op |
| 9 | More hair volume | Out of scope — hair is not in FLAME topology | No |

## Open Questions

- **Exact numbers are not in the literature.** No tutorial gives precise eye-size-as-%-of-face or feature-cluster height ratios; they teach by proportion sketch. Our recipe will have to pick concrete targets (e.g. eye height = 1/4 face, cluster = 0.45 of face, nose-mouth : mouth-chin = 1:2 from the artist brief) and tune them against the reference image rather than derive them from a published canon.
- **Eye "higher" vs. cluster "lower" tension.** The user wants eyes *larger and higher*; tutorials place the *feature cluster* low on the head. These are compatible — the cluster sits low, but within the cluster the eyes should sit at its top with the long nose-mouth-chin sweep below. Worth confirming against the reference that we are moving eyes up *within the face*, not up the whole head.
- **Skin flatness is appearance, not geometry.** The most documented anti-creepy rule (rule 8b) is precisely the part that cannot be expressed as a differentiable vertex displacement on the LAM splat geometry — it lives in the SH-DC colour field. Any differentiable-recipe plan must treat geometry rules (1–7, 8a) and the appearance rule (8b) as two separate optimization targets.
- **Head "block" vs "sphere".** Sources say sphere; the user wants more block-shaped. These differ mainly in how square the cranium silhouette is. Treat as a tunable — start spherical, square slightly toward the reference.

## Sources

[1] CHYEE / RegCanlas / eonovels et al. "Easy Steps to Creating Chibi Characters" — Clip Studio "Art Rocket". https://www.clipstudio.net/how-to-draw/archives/155423 (Retrieved 2026-05-15)
[2] Mary Li. "How to draw anime chibis – general features" — Mary Li Art. https://maryliart.com/how-to-draw-anime-chibis-general-features (Retrieved 2026-05-15)
[3] CHYEE. "Body Proportion in Chibi Drawing — Art Style Study #2" — Clip Studio Tips. https://tips.clip-studio.com/en-us/articles/4829 (Retrieved 2026-05-15, via search aggregation)
[4] "16 Examples of How to Draw Chibi Anime Facial Expressions" / "How to Draw Chibi Anime Character Step by Step" — AnimeOutline. https://www.animeoutline.com/how-to-draw-chibi-anime-character-step-by-step/ (Retrieved 2026-05-15, via search aggregation)
[5] Aggregated chibi eye-spacing tutorial results (Oreate AI, AnimeOutline, Binge Drawing) — "eyes ~one eye-width apart". https://bingedrawing.com/portrait/place-eyes-on-a-face/ (Retrieved 2026-05-15)
[6] "Chibi (style)" — Wikipedia. https://en.wikipedia.org/wiki/Chibi_(style) (Retrieved 2026-05-15)
[7] "Super-Deformed" — TV Tropes. https://tvtropes.org/pmwiki/pmwiki.php/Main/SuperDeformed (Retrieved 2026-05-15)
[8] "Illustrating Chibi-Character Faces" — MediBang Paint. https://medibangpaint.com/en/use/2021/10/how-to-draw-a-chibi-characters-face/ (Retrieved 2026-05-15, via search snippet — page returned 403 on direct fetch)
[9] "Loomis Method: Draw a Head From Any Angle" — Fine Art Tutorials. https://finearttutorials.com/guide/loomis-method/ (Retrieved 2026-05-15, via search aggregation)
[10] "How to Draw a Face — Facial Proportions" — The Virtual Instructor. https://thevirtualinstructor.com/facialproportions.html (Retrieved 2026-05-15, via search aggregation)
[11] "Drawing Stylized Chibi Characters: Proportions and Expressions" — YouTalent Educational Blog. https://blog.youtalent.com/drawing-stylized-chibi-characters-proportions-expressions/ (Retrieved 2026-05-15)
[12] "Basic Facial Proportions: Infant to Adult" — Drawspace; "Understanding Children's Proportions" — O'Reilly / *Drawing: Faces & Features*. https://lessons.drawspace.com/lessons/1508/basic-facial-proportions-infant-to-adult (Retrieved 2026-05-15, via search aggregation)
