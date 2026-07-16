---
status: live
topic: music-generation
---

# OSS music generation and ComfyUI support — field survey

Research date: 2026-07-16. Two parallel web sweeps: (a) open-source /
open-weights music-generation models with priority on 2026 releases,
(b) ComfyUI support for audio/music generation. GitHub star counts and
last-push dates verified via GitHub API on the research date. Motivation:
onboarding exercise for a new photobooth dev — music-gen in ComfyUI is a
low-stakes, fast-feedback playground that teaches the same LATENT +
KSampler machinery the photobooth pipeline uses.

## Headline picture

- 2026 is the breakout year for open full-song (vocals + lyrics) models:
  **ACE-Step 1.5** (Jan), **HeartMuLa** (Jan), **Muse** (Jan),
  **SongGeneration 2 / LeVo 2** (Mar), **Khala** (May), plus
  **Stable Audio 3.0** open weights (May) on the instrumental/SFX side.
- Consensus (IT-JIM 2026 hands-on review,
  <https://www.it-jim.com/blog/best-open-source-ai-music-generator/>):
  no open model matches Suno yet, but the gap is "audible, not
  prohibitive". **LeVo 2 sounds best (non-commercial license); ACE-Step
  1.5 is the best permissively-licensed option.**
- The 2025 generation (YuE, DiffRhythm 1, MusicGen) is effectively
  superseded.
- **License is the real decision axis in 2026**: the best-sounding
  models (LeVo 2, Khala) are the non-commercial ones.
- Architecture convergence: the 2026 winners are hybrid
  LM-planner + DiT/diffusion-renderer (ACE-Step 1.5, LeVo 2). Pure-LM
  (HeartMuLa, Khala, Muse, YuE) and pure-diffusion (DiffRhythm) trade
  quality/speed differently.

## Recommendation for our stack

**ACE-Step 1.5 turbo, native in ComfyUI core.** Rationale:

- MIT license, active repo (11.7k stars, pushed 2026-06-26).
- Native ComfyUI support since v0.12.0 (2026-02-03) — no custom nodes.
  Official templates ship with ComfyUI.
- **Full 4-minute song in <10 s on an RTX 3090** (turbo) — exactly our
  shard hardware, and fast enough for an interactive onboarding loop.
- Same LATENT/KSampler/conditioning graph shape as our image workflows;
  the denoise dial gives audio-to-audio exactly like img2img.

Stable Audio 3 Small is the zero-GPU fallback (runs on CPU,
instrumental/SFX only).

## Model survey — 2026 releases

### ACE-Step 1.5 — permissive-license leader

- ACE Studio + StepFun. <https://github.com/ace-step/ACE-Step-1.5> —
  11.7k stars, last push 2026-06-26 (alive, active cadence). Paper:
  <https://arxiv.org/abs/2602.00744>. **MIT** (v1 from 2025 was
  Apache-2.0). XL series (4B DiT) released 2026-04-02; v0.1.8
  (2026-05-18) added Flow-Edit prompt-guided editing + Retake section
  regeneration.
- Full songs with vocals/lyrics, 10 s – 10 min, 50+ languages, covers,
  repainting, track separation, vocal-to-BGM, metadata control (BPM,
  key, time signature), LoRA fine-tuning from a few songs.
- Hybrid: Qwen3-based LM planner (0.6B/1.7B/4B, CoT "song blueprint")
  + DiT renderer (2B base/sft/turbo; 4B XL base/sft/turbo). Intrinsic
  RL, no preference-pair data.
- Hardware: from ≤6 GB VRAM (2B turbo INT8, DiT-only) to ≥24 GB (XL +
  4B LM). <2 s per song on A100, <10 s on RTX 3090 (turbo); ~3–5 min
  per 4-min song full-quality non-turbo on a 4090.
- Quality: paper places it between Suno v4.5 and v5; best-in-open-source
  AudioBox scores (CU 8.09 / PQ 8.35); trails Suno v5 on style/lyric
  alignment. Community flags a "metallic vocal artifact"; v1 had cleaner
  audio but weaker musicality.

### SongGeneration 2 / LeVo 2 (Tencent AI Lab) — best quality, worst license

- <https://github.com/tencent-ailab/SongGeneration> — **repo returned
  404 via GitHub API on 2026-07-16** and the HF card
  (<https://huggingface.co/tencent/SongGeneration>) returned 401
  anonymously; mirrors exist
  (<https://huggingface.co/mlx-community/SongGeneration-v2-large>).
  Verify availability before depending on it.
- v2-large (4B) 2026-03-01; V11.0 update 2026-03-15 (vocal/BGM
  separation, better multilingual).
- **Custom Tencent terms, characterized as non-commercial/research-only**
  (IT-JIM; vLLM-omni issue
  <https://github.com/vllm-project/vllm-omni/issues/3390>).
- Full songs to 4:30; zh/en/es/ja+; structure markers; style tags +
  10-s reference audio; mixed / a-cappella / instrumental / dual-track
  output. PER 8.55% — beats Suno v5 (12.4%) on lyric intelligibility.
- LeLM autoregressive composer over mixed + dual-track token streams +
  diffusion renderer; semi-online DPO with aesthetic scoring.
- 10 GB VRAM (v1, no audio prompt) / 16 GB (with prompt); v2-large
  22–28 GB; RTF 0.82 on H20.
- Verdict: closest open model to Suno, wins on natural sound (IT-JIM;
  SECourses "almost level of Suno").

### HeartMuLa — Apache-2.0 LM-only family

- <https://github.com/HeartMuLa/heartlib> — 3.8k stars, last push
  2026-04-10 (alive but ~3 months quiet). Paper:
  <https://arxiv.org/abs/2601.10547>. **Apache-2.0** (switched
  2026-01-20, code + weights).
- 3B Llama-3.2-based global backbone + 300M local decoder over
  HeartCodec (12.5 Hz codec fusing Whisper+WavLM+MuEncoder). No
  diffusion stage. Full songs, lyrics+tags, "almost all languages",
  default max 240 s. RTF ≈ 1.0; community 8 GB-VRAM fork exists.
  **7B not released** — team claims it's Suno-comparable, unverifiable.
- Verdict mixed: marketing "most powerful open model of 2026" vs
  IT-JIM "generic across styles / underwhelming"; the codec design is
  the novel part.

### Khala — pure-acoustic-token bet (May 2026)

- Central Conservatory of Music + Tsinghua.
  <https://github.com/Khala-Music-AI/Khala>, paper
  <https://arxiv.org/abs/2605.01790>. Weights **CC BY-NC 4.0
  (non-commercial)**. Two months old — longevity unknown.
- Acoustic-token LM only (no semantic tokens, no diffusion): coarse
  tokens over 64-layer RVQ + super-resolution model + decoder.
- ~24 GB VRAM; authors flag inference quality is highly sensitive to
  the GPU/CUDA/Megatron-TransformerEngine environment — use their NGC
  container.
- Hyped as "right behind Suno" in zh/ja coverage; IT-JIM tested "mixed
  results; unstable".

### Muse (Fudan NLP) — reproducibility play, not a quality play

- Paper 2026-01-13: <https://arxiv.org/abs/2601.03973>. MIT code /
  Apache-2.0 weights. **Only project with a fully released training
  corpus** — 7,771 h / 116k songs of fully synthetic Suno-V5 output
  (note the legal wrinkle: it's distilled Suno).
- Deliberately vanilla: 0.6B Qwen3 LM over MuCodec tokens, single-stage
  SFT. Competitive for its size; "generic pop tendency". Value = open
  dataset + reproducible training pipeline.

### Stable Audio 3.0 (Stability AI) — instrumental/SFX lane, licensed data

- Released 2026-05-20.
  <https://stability.ai/news-updates/meet-stable-audio-3-the-model-family-built-for-artistic-experimentation-with-open-weight-models>
- **Stability AI Community License** — weights-available, free
  commercial use under $1M annual revenue; not OSI-open.
- Small SFX 459M (open), Small 459M (open, ≤2 min, runs on-device/CPU),
  Medium 1.4B (open, ≤6:20, needs GPU), Large 2.7B (API-only).
- Music + SFX, 44.1 kHz stereo, inpainting/section editing, causal
  continuation, audio-to-audio, LoRA fine-tuning docs. **Not a Suno
  competitor — vocals/lyrics are not a headline capability.** Trained
  entirely on licensed data (UMG, Warner) — legal cleanliness is its
  differentiator.

## 2025 baseline models — status check

- **YuE** (M-A-P/HKUST) — Apache-2.0, 6.3k stars, last push 2025-06-04:
  **13 months stale, effectively dead** (weights usable; ICLR 2026
  accepted). Was the strongest open lyrics-to-song model in early 2025;
  superseded. ~12× slower than realtime on a 4090.
- **DiffRhythm / DiffRhythm 2** (ASLP-lab) — DiffRhythm 1 Apache-2.0,
  2.3k stars, last push 2025-11-27. DiffRhythm 2 (block flow matching,
  paper rev 2026-02-03) has only 166 stars, last push 2025-11-09 —
  alive academically, minimal traction, not competitive with the 2026
  crop. Fast but sloppy lyric alignment.
- **MusicGen / AudioCraft** (Meta) — repo maintained (push 2026-03-03)
  but no new music model since 2023; CC-BY-NC weights,
  instrumental-only, 30-s chunks. Legacy.
- **Stable Audio Open 1.0 / Small** — superseded by Stable Audio 3
  Small/Medium.
- **SongBloom** — appears in 2026 comparisons only as a baseline; no
  2026 update found.

## ComfyUI support

The ecosystem consolidated hard onto **native core support** through
2025–2026; most wrapper node packs stopped mattering once their model
went native. All version/date claims below from
<https://docs.comfy.org/changelog>.

### Native in ComfyUI core

| Model | Landed | ComfyUI ver | Notes |
|---|---|---|---|
| Stable Audio Open 1.0 | 2024-06 | v0.x | First native audio model |
| ACE-Step v1 (3.5B) | 2025-05-08 | ~v0.3.33 | text2music + music2music (denoise dial); single AIO checkpoint; Apache-2.0 |
| **ACE-Step 1.5** | **2026-02-03** | **v0.12.0** | AIO `ace_step_1.5_turbo_aio.safetensors` or split files from <https://huggingface.co/Comfy-Org/ace_step_1.5_ComfyUI_files>. Fast patch cadence: v0.12.2 4B-LM, v0.12.3 reference-audio + tiled VAE, v0.13.0 no-LLM mode + VRAM fixes, v0.16.0 LoRA/LoKR keys. "Cover"/"Repaint" still "coming soon" in core per docs |
| ACE-Step 1.5 XL | 2026-04-13 | v0.19.0 | Templates `audio_ace_step1_5_xl_{base,sft,turbo}.json` |
| **Stable Audio 3** (Small-SFX/Small/Medium) | **2026-05-20, day-0** | **v0.22.0** | <https://blog.comfy.org/p/stable-audio-3-day-0-support>; Small variants run on CPU; Medium needs GPU. T5Gemma encoder + optional Qwen3.5-2B "category-aware reprompt" subgraph |
| Seed Audio 1.0 (ByteDance) | 2026-07-08 | v0.27.1 | **API/partner node**, not local weights |
| Stable Audio 2.5 | 2025-09-06 | v0.3.58 | API/partner nodes only |
| Sonilo | 2026-04-16 | v0.19.1 | Music-for-video API partner nodes |
| LTX-2 / LTXV audio | 2026 Jan–Jun | v0.16–0.26 | Multimodal video+audio plumbing (audio VAE, reference-audio ID-LoRA) |

### Core audio plumbing

Audio flows through the standard LATENT + KSampler machinery — the
denoise dial gives audio-to-audio exactly like img2img. Built-ins:

- I/O: `LoadAudio`, `SaveAudio{,MP3,Opus,Advanced}`, `PreviewAudio`,
  `RecordAudio` (native mic, v0.3.51), `EmptyAudio`
- Latents: `EmptyLatentAudio`, `EmptyAceStepLatentAudio`,
  `EmptyAceStep1.5LatentAudio`, `VAEEncodeAudio` / `VAEDecodeAudio` /
  `VAEDecodeAudioTiled`
- Conditioning: `TextEncodeAceStepAudio` (separate `tags` + `lyrics`
  fields; `[verse]/[chorus]/[bridge]` structure tags),
  `TextEncodeAceStepAudio1.5`, `ConditioningStableAudio`,
  `ReferenceTimbreAudio` (timbre cloning, v0.12.3)
- DSP: `AudioConcat`, `AudioMerge`, `AudioAdjustVolume`,
  `Split/JoinAudioChannels`, `AudioEqualizer3Band`
- Audio encoders: native wav2vec2 + `models/audio_encoders/` dir
  (`AudioEncoderLoader`/`AudioEncoderEncode`)

### Official workflow templates

<https://github.com/Comfy-Org/workflow_templates>: `audio_ace_step_1_t2a_song`,
`audio_ace_step_1_t2a_instrumentals`, `audio_ace_step_1_m2m_editing`,
`audio_ace_step_1_5_{checkpoint,split,split_4b,split_llm}`,
`audio_ace_step1_5_xl_{base,sft,turbo}`, `audio_stable_audio_example`,
`audio_stable_audio_3_medium{,_base}`,
`audio_melbandroformer_audio_separation` (stem separation), plus
Chatterbox TTS/VC. Browse: <https://comfy.org/templates/tag/audio/>.
Tutorials: docs.comfy.org → Tutorials → Audio.

### Custom node packs (for models not in core)

| Pack | Stars | Last push | Status | Wraps |
|---|---|---|---|---|
| [ComfyUI_SongGeneration](https://github.com/smthemex/ComfyUI_SongGeneration) | 160 | 2026-03-21 | alive-ish | Tencent LeVo; VRAM >12 GB, manual downloads |
| [ComfyUI_FL-SongGen](https://github.com/filliptm/ComfyUI_FL-SongGen) | 61 | 2026-04-25 | alive | Tencent SongGeneration |
| [ACE-Step-ComfyUI](https://github.com/ace-step/ACE-Step-ComfyUI) (official) | 70 | 2026-03-01 | alive | ACE 1.5 features ahead of core (cover/repaint) |
| [ComfyUI_YuE](https://github.com/smthemex/ComfyUI_YuE) | 188 | 2025-02-24 | **dead** | YuE; heavy quantization needed |
| [ComfyUI_ACE-Step](https://github.com/billwuhao/ComfyUI_ACE-Step) | 247 | 2025-05-28 | dead — use native | ACE v1 |
| [ComfyUI_DiffRhythm](https://github.com/billwuhao/ComfyUI_DiffRhythm) | 153 | 2025-05-30 | stale | DiffRhythm |
| [ComfyUI-MMAudio](https://github.com/kijai/ComfyUI-MMAudio) | 574 | 2026-02-01 | alive | video-to-audio foley (adjacent) |
| [audio-separation-nodes-comfyui](https://github.com/christian-byrne/audio-separation-nodes-comfyui) | 587 | 2026-04-14 | alive | Demucs stems, tempo match |

### RTX 3090 (24 GB) fit — our shard

- **ACE-Step 1.5 turbo: <10 s for a full 4-min song** (official claim,
  blog + docs). All LM planner sizes (0.6B/1.7B/4B) fit in 24 GB;
  v0.13.0 added a no-LLM mode.
- ACE-Step v1 3.5B AIO: comfortable.
- Stable Audio 3 Small: CPU-only OK; Medium: well within 24 GB.
- SongGeneration: >12 GB stated — fits.
- YuE: the pain case — 7B two-stage LM, needs quantization, slow.

## Source caveats

- Several "review" sites in this space (heart-mula.com, gaga.art,
  insmelo.com) are SEO/affiliate content; load-bearing claims above are
  anchored to GitHub metadata (API, 2026-07-16), arXiv papers, official
  Comfy-Org blog/docs/changelog, and the IT-JIM hands-on review.
- Star counts are point-in-time.
- comfyui.org is an unofficial mirror (returned 403); official sources
  are comfy.org / docs.comfy.org / blog.comfy.org.
