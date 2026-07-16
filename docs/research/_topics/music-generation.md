## Music generation (ComfyUI onboarding playground)

**Status:** surveyed 2026-07-16; no experiments run yet.
**Current belief:** **ACE-Step 1.5 turbo** is the pick — MIT license,
active repo, native in ComfyUI core since v0.12.0 (Feb 2026), and
generates a full 4-minute song in **<10 s on an RTX 3090** (our shard
class). Audio runs through the same LATENT + KSampler machinery as our
image pipelines (denoise dial = audio-to-audio, like img2img), which is
why this thread exists: it's the onboarding exercise for a new
photobooth dev — fast feedback, low stakes, same mental model.

Runner-ups: Tencent LeVo 2 sounds best but is non-commercial-licensed
and its repo/HF access was flaky at survey time; Stable Audio 3 Small
is the zero-GPU instrumental/SFX fallback (native day-0 in core).
Dead/superseded: YuE, DiffRhythm, MusicGen, ACE v1 wrapper packs.

### Read-first

- [`2026-07-16-oss-music-generation-comfyui.md`](../2026-07-16-oss-music-generation-comfyui.md)
  — the full survey: 2026 model landscape (ACE-Step 1.5, LeVo 2,
  HeartMuLa, Khala, Muse, Stable Audio 3), licenses, ComfyUI native
  support matrix, core audio nodes, official templates, custom-node
  pack liveness, 3090 fit. **Start here.**

### Next steps

- Pull `ace_step_1.5_turbo_aio.safetensors` onto the shard, load the
  official `audio_ace_step_1_5_checkpoint` template, generate first
  song via the REST API (same driver pattern as
  `scripts/photobooth_sweep/driver.py`).
- Fold into the new-dev onboarding guide as the hands-on ComfyUI
  exercise — done 2026-07-16: `docs/onboarding/comfyui-basics-music.md`.

### Cross-thread

- [[reference-comfyui-shard-runbook]] — the Windows 3090 host this
  would run on
- `_topics/photobooth-sweep.md` — the pipeline the onboarding leads to
