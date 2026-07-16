# Working on ComfyUI with Claude

How this project actually uses ComfyUI: not by clicking in the GUI, but
by having Claude Code author API-format workflow JSONs and Python
drivers that submit them over REST. The GUI is for debugging and
presenting; git and the API are for everything else. This guide covers
the loop, the repo conventions, and the remote GPU shard.

## The core loop

Every experiment follows the same shape:

1. **Author the graph once** — either build it in the GUI and *Export
   (API)*, or (more commonly here) ask Claude to write/modify the
   `.api.json` directly. They live in `comfyui/workflows/`.
2. **Drive it from Python** — a script loads the JSON, patches the
   fields that vary per run (prompt, seed, input image, knob values),
   POSTs it, and collects outputs.
3. **Sweep** — wrap the driver in a grid over the axes you care about,
   score results, build contact sheets, judge by eye.

The reference implementation is `scripts/photobooth_sweep/driver.py`.
The REST contract it uses (works against any ComfyUI ≥ 2024):

```python
import requests, json

comfy_url = "http://127.0.0.1:8188"
wf = json.load(open("comfyui/workflows/photobooth_zimage_cn.api.json"))

# patch inputs by node id — the API format is {node_id: {"class_type", "inputs"}}
wf["6"]["inputs"]["text"] = "your prompt"
wf["3"]["inputs"]["seed"] = 12345

pid = requests.post(f"{comfy_url}/prompt", json={"prompt": wf}).json()["prompt_id"]

# poll history until done, then download outputs
h = requests.get(f"{comfy_url}/history/{pid}").json()
# h[pid]["outputs"][node_id]["images"] → fetch each via GET /view?filename=...
```

Upload inputs with `POST /upload/image` (multipart) or write them into
ComfyUI's `input/` directory. Audio works identically — the music
workflow from guide 1 is drivable the same way.

## Working with Claude on this

What works well (patterns this repo has converged on):

- **Claude edits `.api.json` directly.** The API format is plain JSON
  keyed by node id; Claude can add nodes, rewire inputs, and patch
  values reliably. Tell it *what* to change; it knows the format.
- **One workflow file per pipeline stage**, driver patches only
  scalars. Don't generate whole graphs per run — diffable JSON in git
  is a feature.
- **Presenting a workflow**: drag the `.api.json` onto the ComfyUI
  canvas — the frontend imports API JSON and auto-layouts it. For a
  demo-quality layout, arrange once in the GUI and save a UI-format
  copy next to the `.api.json`.
- **Custom nodes**: when a graph needs an op ComfyUI lacks, Claude
  writes a node class into
  `/home/newub/w/ComfyUI/custom_nodes/<name>/__init__.py` (outside the
  repo — see "External paths" in the project CLAUDE.md), then restart
  ComfyUI. Existing examples: `demographic_pc_fluxspace`,
  `demographic_pc_edit`. A stub copy under `comfyui/custom_nodes/` in
  the repo is *not* loaded — don't edit it expecting effects.
- **Sweeps must be resumable** (skip-if-exists per cell) and must do a
  disk-space preflight before long runs. Don't kill a running sweep to
  amend the plan — append and relaunch.
- **Non-trivial code gets a code review** (`superpowers:code-reviewer`
  agent) before the task is declared done. Standing rule.

Gotchas that have burned real days (all in memory/research docs):

- **Remote-shard sweeps: keep local helper models (CLIP scoring etc.)
  on CPU.** Sharing the local GPU with another job got the sweep
  silently OOM-killed mid-run. See `scripts/photobooth_sweep/scorer.py`.
- **Fixed seed per job/identity** — reproducibility is a project-wide
  convention, not an option.
- Model files, per-cell PNGs, and weights **never go into git** — check
  `.gitignore` before committing experiment output; only summaries
  (`scores.parquet`, `manifest.json`) are tracked. GitHub hard-rejects
  files >100 MB and scrubbing history afterwards is painful (we've done
  it twice).

## The remote GPU shard

Production renders run on a Windows RTX 3090 box, not the local
machine. Canonical runbook: `docs/runbooks/comfyui-windows-shard.md`
(`ssh shard`). Read it before touching the box; headline traps:

- Russian-locale `cmd` output is mojibake that *looks* plausible — use
  PowerShell (`(Get-Item ...).GetFiles()`), never trust `dir /b`.
- `scp` paths are `shard:C:/path/...` (native Windows OpenSSH), not
  cygwin-style.
- Files >7 GB stall over scp — run `hf download` on the box instead.

Drivers point at the shard by URL: `--comfy-url http://<shard>:8188`.

## Research + documentation conventions

Every experiment feeds the doc system (this is load-bearing, not
bureaucracy — sessions are stateless and docs are how the next session
resumes):

- **Dated evidence docs** — `docs/research/YYYY-MM-DD-<topic>.md`,
  append-only, with frontmatter (`status` / `topic`).
- **Topic indexes** — `docs/research/_topics/<thread>.md`, the mutable
  "current belief" layer. **Read the topic index first** on any thread;
  update it in the same commit as any new dated doc.
- Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`); commit and
  push after each logical unit.

Suggested first exercise: take the music workflow from guide 1, export
it as API JSON into `comfyui/workflows/`, and write a 20-line driver
that renders the same lyrics across 5 genre tags with a fixed seed.
That's the whole photobooth sweep pattern in miniature.
