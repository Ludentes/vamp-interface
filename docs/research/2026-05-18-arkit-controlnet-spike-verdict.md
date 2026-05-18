---
status: live
topic: arkit-controlnet
---

# ARKit-ControlNet zero-train spike — verdict

Spike of the *zero-train* hypothesis: compose **InfiniteYou** (identity, via the
`ComfyUI_InfiniteYou` InfuseNet node) with **FluxSpace** (expression, via
`FluxSpaceEditPair` attention editing) on FLUX.1-dev in one ComfyUI graph, and
see whether the pair gives a continuous `f(identity, expression) → image`
without training anything.

5 FFHQ identities (one per FairFace race bucket) × the `smile` axis × scale band
`{0.5, 1.0, 1.5, 2.0}`, fixed seed 2026, 25-step euler. RTX 5090, ~60 s/render,
20/20 cases rendered.

- Workflow: `comfyui/workflows/arkit_controlnet_spike.json`
- Runner: `src/arkit_controlnet/run_spike.py`
- Renders + metrics: `exp_output/arkit_controlnet_spike/` (`collage.png`,
  `metrics.parquet`)

## Verdict: composition is mechanically sound, expression channel is not viable

InfuseNet and FluxSpace **co-exist in one graph with no contention** — InfiniteYou
rides the conditioning/ControlNet wire, FluxSpace patches the MODEL wire. That
part of the hypothesis holds. But FluxSpace is too weak and too
identity-dependent to serve as the expression dial, so the zero-train route is
**falsified for the expression channel**.

### Identity injection — works

ArcFace cosine 0.60–0.83 at scale ≤1.0. Source scowls and off-angle crops become
clean front-facing portraits with the identity recognizably preserved. The
identity-only mode (blank `EmptyImage` InfuseNet control) is enough for an
identity anchor.

### Expression edit — weak, narrow, identity-dependent

- **Hard collapse at scale ≥1.5.** Every identity dissolves to pointillist noise;
  ArcFace finds no face (`arcface_cos = -1.0`). Confirms the prior FluxSpace
  finding ([[project_fluxspace_collapse_prediction]] and siblings: "scale>1.5
  breaks down"). Scale cannot be pushed past this ceiling.
- **Within the usable band (0.5–1.0), the smile is inconsistent.** `bs_delta`
  (mean increase in `mouthSmileLeft/Right`): clear on 2/5 (`+0.10` child,
  `+0.21` woman), negligible on 3/5 (`≈0`, `+0.025`, `≈0`).
- **Barely a dial.** `s0.5` and `s1.0` renders are near-identical — FluxSpace
  provides little continuous modulation across the only band where it doesn't
  collapse. A continuous `sus`-style dial needs monotone response; this isn't it.

### Measurement caveat — framing is uncontrolled

The base prompt (`"a portrait photograph of a person, plain background"`) plus a
blank InfuseNet control image does **not lock pose or crop**: one identity
generated hand-over-mouth, so MediaPipe read `bs_delta ≈ 0` from occlusion, not
from a failed edit. Outputs ranged from headshots to full-body shots. Any future
spike here must constrain framing — tighter prompt and/or a real (non-blank)
control image — before `bs_delta` is trustworthy.

## Implication for the thread

This spike tested the *compose-two-pretrained-modules* shortcut, **not** the
topic's Path 1 (a FLAME render into the InfuseNet spatial-control slot). The
shortcut is dead for expression. The result strengthens the case for the
planned route: keep InfuseNet for identity (verified solid here) and **train**
the expression conditioning — CFM against `(photo, FLAME-render)` pairs as in
`2026-05-16-arkit-controlnet-infiniteyou.md`. FluxSpace attention editing is not
a substitute for a trained expression ControlNet.

Path 1 (FLAME render → InfuseNet control slot, zero-train) remains untested and
is the next cheap experiment worth running before committing to a training run.
