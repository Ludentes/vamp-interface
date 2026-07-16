# ComfyUI basics — hands-on with music generation

Goal: in under an hour, understand ComfyUI's mental model and generate
a full song locally. We use music (ACE-Step 1.5) instead of images
because it's the fastest feedback loop ComfyUI offers — a complete
4-minute song in seconds — and it exercises exactly the same concepts
the photobooth pipeline is built on.

## What ComfyUI is

ComfyUI is a node-graph frontend + execution engine for diffusion
models. Instead of a monolithic "generate" button, every generation is
an explicit **graph**: nodes (load model, encode text, sample, decode,
save) connected by typed wires. The graph *is* the workflow — it can be
saved as JSON, versioned in git, diffed, and submitted over a REST API
without the GUI at all. That last property is why this project uses it:
every experiment is a JSON file plus a Python driver.

Key vocabulary:

- **Node** — one operation (e.g. `CheckpointLoaderSimple`, `KSampler`,
  `VAEDecode`). Custom nodes are Python classes dropped into
  `custom_nodes/`.
- **Workflow** — the whole graph. Two JSON formats exist, see below.
- **LATENT** — the compressed tensor the diffusion model actually works
  on. Images, video, *and audio* all flow through LATENT; the model and
  VAE decide what it means.
- **Conditioning** — encoded guidance (text prompt, ControlNet hints,
  reference audio). Produced by encoder nodes, consumed by the sampler.
- **KSampler / denoise** — the diffusion loop. `denoise=1.0` means
  generate from scratch; lower values start from an existing latent and
  only partially re-noise it. This one dial gives you img2img,
  audio-to-audio, and (in the photobooth) the refine pass — same
  mechanism everywhere.
- **models directory** — checkpoints go in `models/checkpoints/`, VAEs
  in `models/vae/`, etc. Nodes find models by filename.

## The two workflow JSON formats

This bites everyone once, so learn it now:

- **UI format** (`*.json` saved from the GUI) — includes node positions,
  groups, notes. What the graph editor loads and saves.
- **API format** (`*.api.json`, exported via *Workflow → Export (API)*)
  — just the executable graph, keyed by node id. This is what you POST
  to the server. Everything in `comfyui/workflows/` in this repo is API
  format.

**Presenting an API-format workflow in the GUI:** drag the `.api.json`
file onto the ComfyUI canvas — recent frontends import API JSON and
auto-layout the graph. Good enough for demos and debugging; if you want
a *pretty* presentation layout, arrange it once in the GUI and save a
UI-format copy alongside.

## Setup check

Our local install lives at `/home/newub/w/ComfyUI/` (see project
CLAUDE.md; the production shard is a separate Windows RTX 3090 box —
guide 2 covers it). ACE-Step 1.5 needs ComfyUI ≥ v0.12.0; the local
install is v0.18.1 as of 2026-07-16, so it's fine. To check and start:

```bash
cd /home/newub/w/ComfyUI
grep __version__ comfyui_version.py   # want >= 0.12.0 for ACE-Step 1.5
python main.py --listen 127.0.0.1 --port 8188
```

Open http://127.0.0.1:8188.

If you also want Stable Audio 3 later (needs ≥ v0.22.0), update
ComfyUI first (`git pull` + reinstall requirements).

## Generate your first song (ACE-Step 1.5)

ACE-Step 1.5 (MIT-licensed, ACE Studio + StepFun) is natively supported
in ComfyUI core — no custom nodes. The turbo variant renders a full
4-minute song in **under 10 seconds on an RTX 3090**.

Download the all-in-one checkpoint (~one file, goes in
`models/checkpoints/`):

```bash
cd /home/newub/w/ComfyUI/models/checkpoints
uvx --from huggingface_hub hf download \
  Comfy-Org/ace_step_1.5_ComfyUI_files \
  ace_step_1.5_turbo_aio.safetensors --local-dir .
```

Then in the GUI: **Workflow → Browse Templates → Audio → ACE-Step 1.5
(checkpoint)**. The official template loads a ready graph. This
template browser is also your best tool for *presenting* workflows —
every model ComfyUI supports natively ships a curated example graph.

The graph you'll see, and what each part teaches you:

- `CheckpointLoaderSimple` → loads model + VAE from the AIO file
- `TextEncodeAceStepAudio1.5` → **two text fields**: `tags` (genre,
  mood, instruments — e.g. `synthwave, retro, 120 bpm, female vocal`)
  and `lyrics`, with structure markers `[verse]`, `[chorus]`,
  `[bridge]`. Leave lyrics empty + instrument tags = instrumental.
- `EmptyAceStep1.5LatentAudio` → blank latent, set duration in seconds
- `KSampler` → the diffusion loop (note the `seed` — fixed seed =
  reproducible song, the same convention the photobooth uses per job)
- `VAEDecodeAudio` → latent → waveform
- `SaveAudio` / `PreviewAudio` → FLAC out / play in browser

Queue it (Ctrl+Enter). Iterate on tags a few times to get a feel for
prompt sensitivity.

## The one experiment that teaches the core concept

Do this before moving on — it's the whole photobooth mental model in
miniature. Take your generated song and run **audio-to-audio**:

1. Add `LoadAudio` → `VAEEncodeAudio` (wire the checkpoint's VAE).
2. Feed that latent into the KSampler *instead of* the empty latent.
3. Set `denoise` to 0.3, then 0.5, then 0.8. Change the tags (e.g. from
   `synthwave` to `acoustic folk`).

At low denoise the song keeps its structure and melody but shifts
timbre; at high denoise only a ghost of the original survives. **This
denoise dial is exactly how the photobooth turns a photo into a
matryoshka doll while keeping the person recognizable** — and how the
parent project (vamp-interface) encodes fraud-suspicion as drift from
an anchor face. One knob, three products.

## Where to go deeper

- Official audio tutorials: https://docs.comfy.org/tutorials/audio/ace-step/ace-step-v1-5
- All audio templates: https://comfy.org/templates/tag/audio/
- Field survey with model comparison + licenses:
  `docs/research/2026-07-16-oss-music-generation-comfyui.md`
- Next guide: [Working on ComfyUI with Claude](comfyui-with-claude.md)
  — driving all of this from code instead of the GUI.
