# New-dev onboarding

Read in order. The first guide gets you generating music in ComfyUI in
under an hour — it exists because music generation uses the exact same
machinery as the photobooth pipeline (node graphs, LATENT flow,
denoise-as-a-dial, REST API), with a much faster feedback loop and zero
setup baggage. Once the mental model clicks there, the photobooth
pipeline reads as "the same thing, with faces".

1. [ComfyUI basics — hands-on with music generation](comfyui-basics-music.md)
   — what ComfyUI is, the graph model, generate your first song with
   ACE-Step 1.5, present workflows in the GUI.
2. [Working on ComfyUI with Claude](comfyui-with-claude.md) — how this
   project drives ComfyUI from code and from Claude Code: API-format
   workflows, driver scripts, custom nodes, the remote 3090 shard, and
   the research-doc conventions.
3. [Photobooth — current state](photobooth-project-state.md) — what the
   product is, what's been swept and judged, the production recipe, and
   the open items you'll likely pick up first.
