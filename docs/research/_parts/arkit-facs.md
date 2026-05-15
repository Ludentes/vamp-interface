# ARKit's b_61 blendshape set and its relation to FACS

## What ARKit emits

`ARFaceAnchor.blendShapes` exposes a dictionary of 52 named scalar coefficients in `[0.0, 1.0]`, each describing the activation of a specific facial feature relative to a neutral pose [1]. Apple groups them into a small set of "central" channels (`browInnerUp`, `cheekPuff`, `mouthFunnel`, `mouthPucker`, `mouthRollUpper/Lower`, `mouthShrugUpper/Lower`, `mouthClose`, `jawOpen`, `jawForward`, `jawLeft`, `jawRight`, `tongueOut`) and two mirrored sets of 18 left/right channels covering brows, eyelids, gaze, cheeks, and mouth corners [2]. Live Link Face and the b_61 vocabulary used in much downstream avatar work extend this with about a dozen finer tongue shapes (`tongueRoll`, `tongueBendDown`, etc.), but Apple's ARKit tracker itself only solves the 52 — the rest are slots third parties added so head models can carry mocap from elsewhere [2].

## What FACS actually is

FACS, formalised by Ekman and Friesen in 1978 and substantively revised in 2002 with Hager, is *not* a synthesis basis. It is a five-hundred-page manual that trains human coders to decompose an observed face into Action Units (AUs), each corresponding to a discrete, anatomically motivated muscle action, and to score each AU on an A–E intensity ordinal [3,4]. The published inventory contains roughly 28 main AUs for the face proper, plus head-movement codes (51–58), eye-movement codes (61–66), gross behavioural codes (70–98), and "Action Descriptors" (ADs) for movements whose underlying musculature is ambiguous or compound [3]. Inter-coder reliability requires certification; the system was designed to *measure* without committing to emotion labels, which is the opposite of the design pressure on ARKit [3,4]. The total of "44 AUs" sometimes quoted folds in head and eye codes; the muscle-action core is smaller.

## Provenance: faceshift, not Ekman

ARKit's 52 channels are not a clean port of FACS. They descend from faceshift, a Zurich spinout Apple acquired in late 2015; faceshift's runtime mocap pipeline was built around an artist-friendly blendshape rig optimised for real-time solving from RGB+depth, and its shape inventory became the ARKit vocabulary largely unchanged after the TrueDepth camera shipped on iPhone X [5,6]. The lineage matters: the channels were curated for what the faceshift / TrueDepth solver can robustly disambiguate frame-to-frame, not for the discriminations a trained FACS coder cares about. The split into mirrored L/R pairs, the inclusion of gaze (`eyeLookIn/Out/Up/Down`) as if it were a blendshape, and the absence of any forehead-tension or nostril channels are all "engineering" choices in this sense.

## The mapping, and where it breaks

Community mappings — most carefully Ozel's cheat sheet — line ARKit channels up with AUs as follows [7]: `browInnerUp`→AU1, `browOuterUpL/R`→AU2, `browDownL/R`→AU4, `eyeWideL/R`→AU5, `eyeBlinkL/R`→AU45, `cheekSquintL/R`→AU6, `eyeSquintL/R`→AU7, `noseSneerL/R`→AU9, `mouthUpperUpL/R`→AU10, `mouthSmileL/R`→AU12 (zygomaticus major), `mouthDimpleL/R`→AU14, `mouthFrownL/R`→AU15, `mouthLowerDownL/R`→AU16, `mouthShrugUpper/Lower`→AU17, `mouthPucker`→AU18, `mouthStretchL/R`→AU20, `mouthFunnel`→AU22, `mouthPressL/R`→AU24, `mouthRollUpper/Lower`→AU28, `jawOpen`→AU26/AU27, `cheekPuff`→AD34, `jawForward`→AD29, `jawLeft/Right`→AD30 [7]. Gaze channels (`eyeLookIn/Out/Up/Down L/R`) map onto FACS *Movement* codes M61–M64 / AU65–66, which are extra-ocular muscle codes, not facial-muscle AUs [7].

The breaks are systematic. ARKit splits every channel L/R; FACS scores most AUs bilaterally and only adds laterality markers when asymmetry is visible. Conversely, several AUs have no ARKit channel: AU11 (nasolabial deepener), AU13 (sharp lip puller, distinct from the smile-like AU12), AU23 (lip tightener — distinct from `mouthPressL/R`/AU24), AU38, and AU39 (nostril dilator / compressor) [7]. AU7 vs AU6 vs AU44 (squint) collapses into `eyeSquintL/R` + `cheekSquintL/R` without preserving the FACS distinction that AU6 specifically engages the orbital part of orbicularis oculi (Duchenne marker). There is no forehead-tension channel separating AU1+AU2+AU4 co-contraction from a flat forehead; there is no eyebrow-raise-without-frontalis option; there is no asymmetric brow flick beyond the L/R split; pupil dilation and eyelid micro-tremors are absent by construction.

## Discovered vs engineered

The honest answer is *engineered*. The vocabulary is what Apple's TrueDepth-fed solver can reliably output at 60 Hz, exposed as a stable runtime API; the 52 channels are neither a minimal nor a complete basis for human facial motion. They are sufficient for Animoji/Memoji and good enough for VTuber and avatar-puppeting work, which is what the API was built for. The contrast with MPEG-4 Facial Animation Parameters (FAPs, ISO/IEC 14496) is instructive: FAPs define 66 low-level displacements at named feature points on the neutral face plus high-level expression/viseme FAPs, anchored to facial *geometry* rather than muscle action, and were explicitly designed for low-bitrate transmission of animation parameters — yet another engineering vocabulary, with different constraints [8]. JALI takes a third tack: a two-axis (Jaw, Lip) viseme field layered on top of a FACS-style rig, motivated by *psycholinguistic* evidence that visual speech is driven by these two anatomical actions independently [9]. Each of these "vocabularies" optimises a different thing; none of them is the canonical decomposition of facial motion.

## How well can we recover AUs from RGB?

The relevant empirical reference point for treating ARKit-style channels as a stand-in for FACS is OpenFace, which estimates 17 AU intensities (5-point scale) and presences from single RGB frames using static models, and adds person-specific calibration on video via dynamic models [10]. The OpenFace authors are explicit that single-frame static-model accuracy is materially worse than dynamic per-person calibration, and that intensity estimates are noisier than presence detection [10]. OpenFace 3.0 reports improved accuracy and robustness on non-frontal faces but the same model-class limitation [11]. ARKit benefits from depth and a person-specific neutral solve at session start, so its per-frame stability is in practice higher than monocular OpenFace, at the cost of the missing AUs catalogued above.

## Bottom line for our use

ARKit b_61 is a robust runtime mocap signal whose channels approximate, but do not cover, the FACS AU set. Treat the ARKit→AU map as lossy in both directions: many ARKit channels lump >1 AU (squint, press, shrug), and several psychometrically important AUs (11, 13, 23, 38, 39, and the AU6/AU7 separation) have no channel. Anything that requires forehead tension, nostril, asymmetric brow flick, pupil, or fine lip-tightening behaviour is out of vocabulary by construction.

## Sources

1. Apple Developer — ARFaceAnchor blendShapes. https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapes
2. Apple Developer — ARFaceAnchor.BlendShapeLocation. https://developer.apple.com/documentation/arkit/arfaceanchor/blendshapelocation
3. Wikipedia — Facial Action Coding System (history, structure, AU inventory). https://en.wikipedia.org/wiki/Facial_Action_Coding_System
4. Paul Ekman Group — FACS overview. https://www.paulekman.com/facial-action-coding-system/
5. MacRumors (2015-11-24) — Apple confirms faceshift acquisition. https://www.macrumors.com/2015/11/24/apple-faceshift-acquisition-confirmed/
6. Patently Apple (2023) — faceshift patent behind Animoji/Memoji. https://www.patentlyapple.com/2023/02/a-key-patent-that-apple-acquired-from-zurichs-faceshift-to-create-both-animoji-and-memoji-was-published-late-last-month.html
7. Ozel, M. — ARKit to FACS Blendshape Cheat Sheet. https://melindaozel.com/arkit-to-facs-cheat-sheet/
8. Wikipedia / Visage Technologies — MPEG-4 Face Animation Parameter overview. https://en.wikipedia.org/wiki/Face_Animation_Parameter ; https://visagetechnologies.com/uploads/2012/08/MPEG-4FBAOverview.pdf
9. Edwards, Landreth, Fiume, Singh — "JALI: An Animator-Centric Viseme Model for Expressive Lip-Synchronization", SIGGRAPH 2016. https://www.dgp.toronto.edu/~elf/JALISIG16.pdf
10. Baltrušaitis et al. — OpenFace Action Units wiki (static vs dynamic models, intensity scale). https://github.com/TadasBaltrusaitis/OpenFace/wiki/Action-Units
11. OpenFace 3.0 — Lightweight Multitask System for Facial Behavior Analysis. https://arxiv.org/html/2506.02891v1
