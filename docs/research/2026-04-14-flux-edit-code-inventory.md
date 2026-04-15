# Flux-based image editing — code and ComfyUI inventory

**Date:** 2026-04-14
**Purpose:** Concrete inventory of what is installable and runnable *today* for Flux-based image editing. Companion to [2026-04-14-flow-based-foundations.md](2026-04-14-flow-based-foundations.md) and [2026-04-14-rectifid-fluxspace-flowchef-verification.md](2026-04-14-rectifid-fluxspace-flowchef-verification.md). Research-paper survey lives in those docs; this one is "which repo do I `git clone` and which ComfyUI nodes do I install."

**Sources:** tavily pro research (request `e61a113a-65d0-4ff0-b8fd-a48bb7e472e7`) + parallel perplexity research at [runnable-flux-edits.md](runnable-flux-edits.md) + direct GitHub fetches on 2026-04-14 for FluxSpace, Concept Sliders, ComfyUI-Fluxtapoz, and the HF model card for Flux Kontext-dev.

**Key additions from perplexity pass merged below:** Flux Kontext as a first-class entry (a separately-trained 12B rectified-flow transformer for instruction-based editing — this is BFL's answer to Qwen-Edit, not a FluxSpace peer); `sliderspace` as the successor repo to Concept Sliders; tighter license data (BFL code Apache-2.0 but all model weights Flux.1-dev Non-Commercial; FlowEdit MIT; RF-Inversion Apache-2.0; Concept Sliders MIT); VRAM floors (Flux.1-dev runs at ~12 GB fp16, ~8–10 GB with fp8/GGUF, ~16–24 GB comfortable for 1024²).

---

## TL;DR — install today, two paths

**Path A — ready-to-use Flux editing in ComfyUI in under 15 minutes:** install `logtd/ComfyUI-Fluxtapoz` (GPL-3.0, 1.4k stars, 50 commits, actively maintained, available via ComfyUI-Manager). This single custom-node pack gives you RF-Inversion, RF-Edit, FireFlow, FlowEdit, Regional Prompting, PAG, and SEG as ComfyUI nodes on top of our existing Flux.1-dev backbone. All methods are inversion-free or inversion-accelerated, training-free, and run on the same VRAM budget as vanilla Flux (~24 GB for full-precision, ~12–16 GB for fp8-scaled).

**Path B — FluxSpace specifically (the CVPR 2025 method the framework now names as §3.3.1):** install the official `gemlab-vt/FluxSpace` repo (MIT). **There is no ComfyUI node for FluxSpace yet.** It ships as a Python demo notebook + Diffusers pipeline (`flux_semantic_pipeline.py`, `flux_blocks.py`, `utils.py`, `demo.ipynb`) with `torch.bfloat16` on CUDA. We either (a) run it via Diffusers outside ComfyUI, or (b) wrap it in a custom node ourselves. The wrap is bounded-cost — FluxSpace's core operation is an attention-output mod applied via monkey-patched Flux blocks, which is structurally the same shape of integration Fluxtapoz already does for RF-Inversion, so the wrap is feasible. No public wrapper exists as of 2026-04-14.

**These two paths are complementary, not competing.** Fluxtapoz gives us RF-Inversion / FlowEdit / FireFlow editing on Flux in ComfyUI today; FluxSpace gives us disentangled semantic attribute axes (which Fluxtapoz does not) but requires either a Diffusers escape hatch or our own ComfyUI wrapper.

---

## Inventory table

| Project | Repo | License | ComfyUI integration | Base model | Status |
|---|---|---|---|---|---|
| **FluxSpace** | `gemlab-vt/FluxSpace` | MIT | **none** — Diffusers pipeline only | Flux.1-dev | research release (3 commits), notebook + pipeline script |
| **FlowEdit** | `fallenshock/FlowEdit` | MIT | **yes** via `logtd/ComfyUI-Fluxtapoz` (`FlowEditGuider`) and `raindrop313/ComfyUI_SD3_Flowedit` for SD3 | Flux.1-dev, SD3 | runnable, community wrappers mature |
| **RF-Inversion** | `LituRout/RF-Inversion` | not stated | **yes** via Fluxtapoz | Flux.1-dev | runnable example scripts, 28-step inversion |
| **RF-Solver-Edit** (Taming Rectified Flow) | `wangjiangshan0725/RF-Solver-Edit` | not stated | **yes** via Fluxtapoz (`RF-Edit` node) | Flux + HunyuanVideo | ICML 2025, actively maintained, image + video |
| **FireFlow** | `HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing` | not stated | **yes** via Fluxtapoz (`Fireflow` node) | Flux.1-dev | ICML 2025, 8-step inversion, 3× speedup over baselines |
| **Stable Flow: Vital Layers** | `snap-research/stable-flow` | not stated | **no** | Flux (DiT) | CVPR 2025, paper reports experiments on **V100 80 GB** — heavy-memory |
| **FlowChef** | `FlowChef/flowchef` | not stated | **no** | Flux.1-dev, InstaFlow | ICCV 2025 code present; initial configs are 256×256 with annotated masks, not a drop-in Flux editor |
| **BFL Flux.1 Tools** (Fill, Canny, Depth, Redux) | `black-forest-labs/flux` + HF model cards | Flux.1-dev Non-Commercial | **yes** — native ComfyUI support; community nodes (`ComfyUI-FluxSettingsNode`, Redux nodes) | Flux.1-dev (Fill/Canny/Depth/Redux variants) | official, stable, 16 GB VRAM minimum recommended |
| **Concept Sliders — Flux** | `rohitgandikota/sliders` → `flux-sliders/` | MIT (code) | **no** | Flux | **experimental**, README: "doesn't work as good as SDXL," **no pretrained Flux sliders released** |
| **LEDITS++** | `ml-research/ledits_pp` | not stated | **no Flux-specific** | SD1.5, SDXL (evidence); Flux port not found | Diffusers-integrated (`LEditsPPPipeline`) |
| **KV-Edit** | paper arXiv:2502.17363; repo status unclear | — | **no** | Flux.1-dev (28 timesteps) | paper + code references; less mature than FlowEdit |
| **Prompt-to-Prompt / p2p-zero on Flux** | — | — | **no** | — | **no verified Flux port found** |

---

## Per-entry notes

### FluxSpace — `gemlab-vt/FluxSpace`

**Verified 2026-04-14 via direct GitHub fetch.** The official repo exists, is MIT-licensed, and contains:

- `demo.ipynb` — Jupyter notebook demo
- `flux_semantic_pipeline.py` — Diffusers-style pipeline wrapping Flux.1-dev
- `flux_blocks.py` — the monkey-patched attention-block implementations where the orthogonal projection happens
- `utils.py` — helpers
- `requirements.txt` — dependencies (versions not inspected)
- `LICENSE`, `README.md`, `assets/`

**Repo state:** 3 commits total. This is research-code-release velocity, not active library maintenance. That matches what we'd expect for a CVPR 2025 accompanying repo.

**Runtime:** loads Flux.1-dev with `torch.bfloat16` on CUDA. No explicit VRAM declaration in README, but Flux.1-dev bf16 baseline is ~24 GB; with fp8 quantization (which the paper likely does not ship) it would be ~12–16 GB.

**No ComfyUI node exists.** Neither the official repo nor any community repo discovered in the survey wraps FluxSpace as a ComfyUI custom node. This is the key operational gap.

**Path to integration in our stack:**

1. **Minimum effort:** import `flux_semantic_pipeline.py` into a standalone Python script, compute the orthogonal edit direction, render with the pipeline, save the output image, and consume that image from our ComfyUI workflow as a LoadImage. This treats FluxSpace as an external preprocessing step.
2. **Medium effort:** write a thin ComfyUI custom node that wraps `flux_blocks.py`'s monkey-patching mechanism, exposes `λ_fine` and the two prompt strings as node inputs, and hooks into ComfyUI's existing Flux sampler. Structurally similar to how Fluxtapoz wraps RF-Inversion — probably 1–2 days of integration work for someone comfortable with ComfyUI's sampler internals.
3. **Higher effort:** upstream the custom node to Fluxtapoz or submit as a new custom-node pack.

For the framework's §3.3.1 scoring experiment — "score FluxSpace on our corpus under E1–E4 with our own ArcFace measurement" — **option 1 is sufficient**. We don't need ComfyUI integration to run the evaluation; we need a Python script that loops over job embeddings, runs FluxSpace with chosen prompt-pair axes, and dumps ArcFace measurements. Integration into ComfyUI is a deployment concern, not an evaluation concern.

### ComfyUI-Fluxtapoz — `logtd/ComfyUI-Fluxtapoz` — **the main editing node pack**

**Verified 2026-04-14.** GPL-3.0, 1.4k stars, 57 forks, 50 commits, actively maintained. Ships the following nodes, all targeting Flux:

- **RF-Inversion** — inverts a Flux-generated image back to noise so you can re-render with a different prompt
- **RF-Edit** — the RF-Solver-Edit attention-feature-sharing editing
- **FireFlow** — 8-step fast inversion
- **FlowEdit** — inversion-free text-based editing
- **FlowEditGuider** — the guidance variant
- **Regional Prompting** — spatial prompt control
- **PAG, SEG** — guidance methods

The GPL-3.0 is worth noting. For a research-phase experiment it's fine; for a commercial product it means either keeping our usage isolated or choosing a differently-licensed alternative. For vamp-interface V2 in research phase this is not a blocker.

**This is the one package to install first.** It packages the five most relevant papers (RF-Inversion, RF-Solver-Edit, FireFlow, FlowEdit, Flux regional prompting) behind a single ComfyUI-Manager-compatible dependency. Compared to installing each paper's upstream repo and writing custom nodes yourself, Fluxtapoz saves weeks of integration work.

### FlowEdit — `fallenshock/FlowEdit`

MIT license, the most-polished of the community Flux-editing repos. Ships Diffusers scripts for both Flux.1-dev and Stable Diffusion 3 (example `run_script.py`). Requires Diffusers 0.30.1. **Inversion-free** — the key feature vs. RF-Inversion is that you don't need to invert the source image to noise first; the method directly transports the latent along a trajectory that ends at the edited prompt. Faster and more stable than inversion-based editing.

**ComfyUI integration:** two independent wrappers — `raindrop313/ComfyUI_SD3_Flowedit` for SD3 and the `FlowEditGuider` node inside `logtd/ComfyUI-Fluxtapoz` for Flux. Use Fluxtapoz for our Flux-on-Flux use case.

For vamp-interface: FlowEdit is the ready-today alternative to FluxSpace if what we want is "take a Flux-generated face and edit one attribute" rather than "discover a disentangled semantic axis." FlowEdit requires prompt rewriting per edit; FluxSpace gives you a named axis. They are complementary.

### RF-Solver-Edit — `wangjiangshan0725/RF-Solver-Edit`

The "Taming Rectified Flow for Inversion and Editing" paper (OpenReview `uDreZphNky`, ICML 2025). Implements a higher-order Taylor ODE solver to invert rectified flows with lower numerical error, and uses attention-feature sharing to preserve structure during edits. Supports both Flux (images) and HunyuanVideo (video). Runnable upstream; Fluxtapoz wraps the editing module as the `RF-Edit` node.

### FireFlow — `HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing`

ICML 2025. 8-step inversion + 8-step editing on Flux.1-dev, claiming ~3× speedup over baselines with comparable or better results. The "fast inversion" node in Fluxtapoz. If RF-Inversion is too slow in practice, FireFlow is the drop-in replacement on the same backbone.

### Stable Flow: Vital Layers — `snap-research/stable-flow`

CVPR 2025, Snap Research. The paper identifies which DiT layers are "vital" for image editing and injects attention features only in those layers. The paper's reported experiments used a **V100 80 GB** GPU. We do not have that; we have 24 GB / 32 GB at most in realistic setups. The method may still run on smaller cards with reduced batch size or fp8, but the paper's numbers are on heavy-memory hardware. **No ComfyUI wrapper found.** Deprioritize unless someone publishes a memory-optimized port.

### FlowChef — `FlowChef/flowchef`

ICCV 2025. Official implementation. The repo contains Flux.1-dev + InstaFlow support, but the initial dataset-specific configs target AFHQ-Cat and CelebA at **256×256 resolution with annotated masks** — this is the linear-inverse-problems and classifier-guidance regime, not a drop-in Flux editor. Adapting FlowChef to the "give me a linear attribute axis" use case requires either defining a classifier per axis or reframing the edit as an inverse problem. As verified in the verification doc, this matches the conclusion that FlowChef is a **general steering primitive**, not a drop-in editorial mechanism. **No ComfyUI wrapper.** The framework §2.7.5 framing principle #5 formally excludes it from §3.3 admission on this basis.

### Flux Kontext-dev — BFL's instruction-following image editor

**Verified 2026-04-14 via HF model card.** **Kontext is a separately-trained 12-billion-parameter rectified-flow transformer specifically for image editing from text instructions.** Not a LoRA, not a sampler trick, not a fine-tune of Flux.1-dev in the LoRA sense — a full-weight 12B rectified-flow transformer trained with guidance distillation for efficiency.

**Interaction model:** `(source_image, text_instruction) → edited_image`. The HF code example is:

```python
image = pipe(
    image=input_image,
    prompt="Add a hat to the cat",
    guidance_scale=2.5
).images[0]
```

**This is instruction-following editing, structurally equivalent to Qwen-Edit.** You describe the edit in natural language, the model interprets and applies. Multi-turn refinement is supported ("Robust consistency allows users to refine an image through multiple successive edits with minimal visual drift"). Character/style/object reference is supported without fine-tuning.

**Kontext is NOT axis-based or slider-like.** There is no "age += 0.2" operation. The edit is specified as an English sentence. This makes Kontext the *categorical opposite* of FluxSpace in interaction model: Kontext is semantic-parsing-heavy and continuous-control-light; FluxSpace is semantic-parsing-light and continuous-control-heavy.

- License: Flux.1-dev Non-Commercial.
- VRAM: same 12B parameter footprint as Flux.1-dev → ~12 GB fp16, ~8 GB fp8-scaled, ~16–24 GB comfortable for 1024² editing.
- ComfyUI integration: the official ComfyUI examples page has native Kontext workflows — no custom nodes required.

**What Kontext gives you that FluxSpace cannot:** referential editing ("make the second person smile"), compositional edits ("remove the hat, add glasses"), style transfer by description, semantic-scope control, and zero prompt-pair authoring. **What FluxSpace gives you that Kontext cannot:** a continuous scalar slider on a single named axis, precomputed reusable edit directions, and disentangled-by-orthogonal-projection composition of multiple axes at once.

### Black Forest Labs Official Flux.1 Tools — `black-forest-labs/flux`

The official inference repo. Contains demo scripts (`demo_gr.py`, `demo_st.py`, `demo_st_fill.py`) and supports the full Flux.1 family: **dev**, **schnell**, **pro**, **Krea**, **Kontext**, plus the specialized tool variants **Fill** (inpainting), **Canny** (edge-conditioned), **Depth** (depth-conditioned), **Redux** (image variation). ComfyUI has native support for Canny and Depth variants; community nodes extend to Fill and Redux.

**License reality:** Flux.1-dev is **Non-Commercial** (not Apache-2.0). Schnell is Apache-2.0 but is a distilled 1–4-step variant whose edit quality is lower than dev. For research-phase vamp-interface V2 the Non-Commercial dev license is fine; if V2 ever ships to a paying product, we need to either (a) negotiate a commercial license with BFL, (b) run on Schnell, or (c) move to SD3 or another rectified-flow base. This is a downstream concern, not a v0.8 framework concern.

**For our editorial channel work:** the tools aren't what we need. Flux Fill is masked inpainting (not disentangled attribute editing). Flux Canny/Depth are structure conditioning, not semantic edits. Flux Redux is reference-image variation. None of these are the FluxSpace-class "move along a named attribute axis" operation — they are different tools for different jobs.

### Concept Sliders — Flux variant — `rohitgandikota/sliders`

**Verified 2026-04-14.** The main sliders repo (MIT) has a `flux-sliders/` directory that is **experimental**. The README explicitly states: *"You can train sliders for FLUX-1 models. Right now it is experimental!"* and *"it doesn't work as good as SDXL."* **No pretrained Flux sliders are released** — only SDXL sliders have public weights. Training a Flux slider is possible via the notebook in the `flux-sliders/` directory, but the quality is below SDXL-slider quality and the training cost is uncommented in the README.

**Framework implication.** The v0.3 framework claim that "Concept Sliders (ECCV 2024) is the recommended next-step overlay" for the drift channel (§3.2.5) needs a soft correction: **pretrained Flux Concept Sliders do not exist**. If we commit to that path we are either (a) training our own Flux sliders from scratch against the SDXL-slider-quality benchmark handicap, or (b) accepting that the SDXL version is the actual deployment target and using SDXL as a second backbone alongside Flux. Neither matches the "drop-in overlay" framing. This is not a retraction — Concept Sliders on Flux remains a viable candidate — but it is more expensive than §3.2.5 currently implies, and the framework should note that the Flux variant is experimental per the upstream repo.

### LEDITS++ — `ml-research/ledits_pp`

Official Diffusers integration (`LEditsPPPipeline`) — but the pipelines in the repo are `StableDiffusionPipeline_LEDITS` and its XL variant, which are SD1.5 / SDXL. **No verified Flux port.** Not applicable to our use case without a separate porting effort.

### KV-Edit, FluxText, Fluxtools, FireFlow (tooling cluster)

Mature community utilities surveyed but not verified in depth:
- **KV-Edit** (arXiv:2502.17363) — training-free precise background preservation via KV-cache manipulation on Flux
- **FluxText** — scene text editing specific to Flux (not our use case)
- **Fluxtools** (`bethany33/fluxtools`) — general Flux utility suite
- **sayakpaul/flux-image-editing** — image-editing task training for Flux Control

Most relevant of these is KV-Edit if we ever need fine-grained background-preserving local edits. For the editorial-channel work it is not priority-one.

### Prompt-to-Prompt / p2p-zero on Flux — **does not exist in verified form**

The survey found no working port of Prompt-to-Prompt cross-attention editing to Flux or other rectified-flow models. P2P was originally a diffusion cross-attention technique; Flux uses joint attention (MM-DiT), which changes the mechanism fundamentally. **Claimed absence.** If we wanted this, it would be an original research project, not an integration.

---

## Recommendations for vamp-interface

Ordered by immediate leverage:

1. **Install `logtd/ComfyUI-Fluxtapoz` today.** One-step install via ComfyUI-Manager, gives us five editing methods on top of our existing Flux v3 workflow. Start with FlowEdit for inversion-free editing experiments.
2. **Clone `gemlab-vt/FluxSpace` and run its `demo.ipynb` with `torch.bfloat16`.** Verify it loads and runs on our hardware. If it does, write a minimal Python driver that loops over the 543-job corpus, picks 1–2 editorial axes (e.g., "suited professional" vs. "casual" as an age-adjacent axis), generates edited outputs, and dumps ArcFace IR101 cosine drift vs. the unedited baseline. This is the **§3.3.1 E3 measurement experiment** the framework calls for. ~1 day of work.
3. **Defer the ComfyUI wrapping of FluxSpace.** Only wrap it as a custom node after the evaluation in step 2 confirms FluxSpace is worth deploying. If E3 measurements look bad we drop FluxSpace and the wrapping effort is wasted.
4. **Update the framework §3.2.5 Concept Sliders entry** to note that pretrained Flux sliders do not exist and the Flux variant is experimental per the upstream repo. Minor clarification, not a retraction.
5. **Reserve FlowEdit, RF-Solver-Edit, FireFlow as drift-channel candidates for §3.2.** None of them provide a named attribute axis, but they all provide a "trajectory deformation" operator on a Flux generation, which could be reframed as a controllable factor-mismatch injector for D1. Speculative, not a near-term action item.

## Gaps

- **No ComfyUI node for FluxSpace.** This is the single biggest integration-gap the survey found. It is also probably the most valuable thing to publish — the research community would install it.
- **No pretrained Flux Concept Sliders.** The slider infrastructure exists upstream but is experimental and un-released as weights. Concept Sliders on Flux is a training-to-deploy effort, not an install-and-run effort.
- **No verified P2P port to Flux.** Not available; do not plan on it.
- **Stable Flow is heavy.** The paper's 80 GB reference hardware makes it a deprioritize for us unless a memory-optimized port appears.
- **Licenses are inconsistent across community repos.** FluxSpace is MIT (good), Fluxtapoz is GPL-3.0 (watch for commercial), Flux.1-dev itself is Non-Commercial (blocker for paid products but not research). Any V2 deployment plan needs its own license review pass.

## References

- FluxSpace — `gemlab-vt/FluxSpace` · https://github.com/gemlab-vt/FluxSpace (MIT)
- ComfyUI-Fluxtapoz — `logtd/ComfyUI-Fluxtapoz` · https://github.com/logtd/ComfyUI-Fluxtapoz (GPL-3.0)
- FlowEdit — `fallenshock/FlowEdit` · https://github.com/fallenshock/FlowEdit (MIT)
- ComfyUI_SD3_Flowedit — `raindrop313/ComfyUI_SD3_Flowedit` · https://github.com/raindrop313/ComfyUI_SD3_Flowedit
- RF-Inversion — `LituRout/RF-Inversion` · https://github.com/LituRout/RF-Inversion
- RF-Solver-Edit — `wangjiangshan0725/RF-Solver-Edit` · https://github.com/wangjiangshan0725/RF-Solver-Edit
- FireFlow — `HolmesShuan/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing`
- Stable Flow — `snap-research/stable-flow` · https://github.com/snap-research/stable-flow
- FlowChef — `FlowChef/flowchef` · https://github.com/FlowChef/flowchef
- BFL Flux — `black-forest-labs/flux` · https://github.com/black-forest-labs/flux
- Concept Sliders — `rohitgandikota/sliders` (with `flux-sliders/` directory, MIT, experimental) · https://github.com/rohitgandikota/sliders
- LEDITS++ — `ml-research/ledits_pp` · https://github.com/ml-research/ledits_pp
- ComfyUI-FluxSettingsNode — `Light-x02/ComfyUI-FluxSettingsNode` · https://github.com/Light-x02/ComfyUI-FluxSettingsNode
- ComfyUI-Easy-Use — `yolain/ComfyUI-Easy-Use` · https://github.com/yolain/ComfyUI-Easy-Use
- flux-image-editing — `sayakpaul/flux-image-editing` · https://github.com/sayakpaul/flux-image-editing
- KV-Edit paper — arXiv:2502.17363
- FluxTools — `bethany33/fluxtools` · https://github.com/bethany33/fluxtools
- FluxText — `AMAP-ML/FluxText` · https://github.com/AMAP-ML/FluxText
