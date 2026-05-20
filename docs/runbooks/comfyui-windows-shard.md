---
topic: comfyui-shard
status: live
supersedes: docs/design/comfyui-shard-windows.md
---

# ComfyUI Windows shard — `shard` (192.168.87.25)

Secondary GPU box used as a second ComfyUI endpoint and a place to run generation
sweeps in parallel with the Linux 5090. Standing user authorization for SSH and
ComfyUI/model management on this host.

## Box facts

- **Hostname:** `DESKTOP-BMEMFG1`
- **OS:** Windows 10/11, **Russian locale** (codepage 866 by default — see *gotchas*)
- **GPU:** NVIDIA GeForce RTX 3090 — 24 GB VRAM
- **CUDA / torch:** torch 2.6.0 + cu124 (last verified 2026-05-18); driver supports newer CUDA but stack is pinned
- **Python:** 3.10.3 in the venv at `C:\comfy\ComfyUI\venv`
- **Disk (C:):** ~170 GB free of 1 TB (verified 2026-05-19)
- **ComfyUI:** `C:\comfy\ComfyUI` — full source tree (`main.py`, `models/`, `custom_nodes/`, `venv/`, `output/`, …)

## SSH

Alias `shard` is configured in `~/.ssh/config` (user `videocard`, key `id_ed25519`,
IdentitiesOnly). Always use the alias — don't hard-code the IP:

```bash
ssh shard 'hostname'
```

ComfyUI listens on **0.0.0.0:8188** (no tunnel needed). Firewall rule "ComfyUI 8188"
allows inbound from all profiles. Web UI: `http://192.168.87.25:8188`.

## Service management

ComfyUI runs as a **manual-start Windows service** via WinSW (`winsw.exe` +
`winsw.xml`):

```bash
ssh shard 'C:\comfy\winsw.exe status'   # → Started | Stopped
ssh shard 'C:\comfy\winsw.exe start'
ssh shard 'C:\comfy\winsw.exe stop'
ssh shard 'powershell -Command "Get-Content C:\\comfy\\logs\\winsw.out.log -Tail 50"'
```

Service does not start on boot (manual). On crash, auto-restarts after 10 s.
Logs rotate at 10 MB, keeps 3 files: `C:\comfy\logs\winsw.{out,err}.log`.

## Gotchas (learned the hard way)

- **Russian-locale cmd lies via mojibake.** `dir /b` on an *empty* dir under
  `chcp 866` prints `Не удалось найти файл` which arrives over SSH as
  `���⥬� �� 㤠���� ���� 㪠����� ����`. This looks like "directory missing"
  but usually means "directory exists but empty" or "wildcard found no match".
  **Always inventory via PowerShell, not cmd:**

  ```bash
  ssh shard 'powershell -NoProfile -Command "(Get-Item C:\comfy\ComfyUI\models\diffusion_models).GetFiles() | Select-Object Name,Length"'
  ```

- **`%USERPROFILE%` doesn't always expand** in the SSH default shell. Use the
  full path `C:\Users\videocard\...` instead of relying on `%USERPROFILE%\...`,
  or invoke PowerShell explicitly with `$env:USERPROFILE`.

- **scp uses native Windows paths**, not Cygwin. The shard runs native Windows
  OpenSSH (not Cygwin sshd), so `/cygdrive/c/...` paths silently fail with
  `"No such file or directory"`. Use forward-slash Windows paths and include
  the *destination filename*, not just a dir:

  ```bash
  scp -C local_file 'shard:C:/comfy/ComfyUI/models/insightface/hyperswap_1c_256.onnx'
  ```

  Quoting the remote path matters — backslashes need quoting or doubling. Test
  with a tiny file before pushing GB-sized transfers; the error is silent (exit
  0 in some scp builds) but the file ends up zero-bytes.

- **Large transfers (>~7 GB) stall over scp/curl.** Z-Image bake-off (2026-05-18)
  burned hours on this. Workaround: pull from HuggingFace on the box itself
  using `hf` (HuggingFace CLI is installed; uses xet, resumable):

  ```bash
  ssh shard 'hf download <repo_id> <file> --local-dir C:\comfy\ComfyUI\models\<subdir>'
  ```

  3 GB files do scp fine; only the 7 GB+ checkpoints need `hf`.

- **PyTorch cu130 warning at startup is harmless.** Generation works via legacy
  VRAM path. Don't try to "fix" it.

## Inventory probe (run this at session start)

```bash
ssh shard 'powershell -NoProfile -Command "
  foreach (\$d in @(\"diffusion_models\",\"checkpoints\",\"text_encoders\",\"vae\",\"controlnet\",\"loras\",\"insightface\",\"unet\")) {
    Write-Host \"=== \$d ===\"
    Get-ChildItem \"C:\comfy\ComfyUI\models\\\$d\" -File -ErrorAction SilentlyContinue |
      Select-Object Name, @{n=\"GB\";e={[math]::Round(\$_.Length/1GB,2)}} |
      Format-Table -AutoSize | Out-String
  }
  Write-Host \"=== custom_nodes ===\"
  Get-ChildItem C:\comfy\ComfyUI\custom_nodes -Directory | Select-Object Name | Format-Table -AutoSize | Out-String
"'
```

This is the *only* reliable way to know what's actually there. The
"models installed" lists in older docs go stale fast.

## Models installed (verified 2026-05-19)

Update this table after each material transfer. Use the inventory probe above to
verify before relying on it.

| Path | File | Size | Use |
|------|------|------|-----|
| `diffusion_models/` | `z_image_turbo_bf16.safetensors` | 12.3 GB | Z-Image Turbo base |
| `diffusion_models/` | `flux-2-klein-4b.safetensors` | 7.75 GB | Flux 2 experimental |
| `text_encoders/` | `clip_l.safetensors` | 246 MB | Shared CLIP-L |
| `text_encoders/` | `qwen_3_4b.safetensors` | 8.0 GB | Z-Image text encoder |
| `vae/` | `z_image_ae.safetensors` | (small) | Z-Image VAE |
| `vae/` | `flux2-vae.safetensors` | (small) | Flux 2 VAE |
| `vae/` | `sdxl_vae_fp16_fix.safetensors` | (small) | SDXL VAE |
| `vae/FLUX1/` | `ae.safetensors` | (small) | Flux 1 VAE |
| `controlnet/` | `controlnet-canny-sdxl-1.0.safetensors` | 2.5 GB | SDXL Canny CN |
| `controlnet/` | `Z-Image-Turbo-Fun-Controlnet-Union.safetensors` | 3.1 GB | **transferred 2026-05-19** |
| `insightface/` | `inswapper_128.onnx` | 530 MB | Swap baseline |
| `insightface/` | `hyperswap_1c_256.onnx` | 400 MB | **transferred 2026-05-19** |
| `~/.insightface/models/buffalo_l/` | (bundle) | – | ArcFace+SCRFD for `make_face_app` |

Inherited from earlier work (verify presence before reusing):

- `models/diffusion_models/FLUX1/flux1-krea-dev_fp8_scaled.safetensors`
- `models/vae/FLUX1/ae.safetensors`
- `models/text_encoders/t5/t5xxl_fp8_e4m3fn.safetensors`
- `models/loras/Cursed_LoRA_Flux.safetensors`, `Eerie_horror_portraits.safetensors`, `Strange_and_unsettling.safetensors`, `horror_nova.safetensors`

## Custom nodes installed (verified 2026-05-19)

- `ComfyUI-PuLID-Flux`
- `ComfyUI_IPAdapter_plus`
- `demographic_pc_fluxspace` (one of ours — FluxSpace edit primitives)

Known missing vs the Linux box at `/home/newub/w/ComfyUI/custom_nodes/`, in
roughly descending order of *required-ness* for current workflows:

- `comfyui_controlnet_aux` — Canny + other CN preprocessors. **Required** for
  any Z-Image Turbo + Fun-CN-Union workflow that wants Canny from input photo.
- `ComfyUI-Manager` — UI-driven install of other nodes; install once and the rest
  becomes easy.
- `rgthree-comfy` — workflow QoL.
- `comfyui-kjnodes`, `comfyui_essentials`, `comfyui-easy-use`, `comfyui-impact-pack` —
  general utility, install on demand.
- `demographic_pc_edit`, `vamp_conditioning` — ours; install only if running
  conditioning-edit workflows on the shard.

Install pattern (from the box):

```bash
ssh shard 'powershell -NoProfile -Command "
  cd C:\comfy\ComfyUI\custom_nodes;
  git clone https://github.com/Fannovel16/comfyui_controlnet_aux;
  C:\comfy\ComfyUI\venv\Scripts\python.exe -m pip install -r comfyui_controlnet_aux\requirements.txt
"'
ssh shard 'C:\comfy\winsw.exe stop'
ssh shard 'C:\comfy\winsw.exe start'   # restart picks up new nodes
```

## Pipeline endpoint usage

`generate_v8d.py` and the matryoshka generators accept `--comfy-url`. The
generator auto-detects model paths from the target server, so Windows backslash
paths are transparent.

```bash
# Shard-only run
uv run scripts/<generator>.py --comfy-url http://192.168.87.25:8188 ...

# Parallel both nodes — use different --sample-seed so jobs don't overlap
uv run scripts/<generator>.py --sample-seed 3 &
uv run scripts/<generator>.py --comfy-url http://192.168.87.25:8188 --sample-seed 4 &
```

## Sweep / batch-job pattern

The matryoshka bake-off (2026-05-18) is the working precedent. Pattern:

1. Stage all weights + custom nodes on the box.
2. Restart ComfyUI service so the venv reloads.
3. Push the sweep script + workflow JSON via scp into `C:\comfy\sweeps\<name>\`.
4. Launch via a **SYSTEM scheduled task** (`schtasks /create /sc once /st <time>`)
   so it survives SSH disconnect. Older "nohup-equivalent" tricks via
   `Start-Process -WindowStyle Hidden` work but aren't logged the same way.
5. Tail progress via `Get-Content -Tail 50 -Wait` over SSH.
6. `rsync` (or `scp -r`) the output dir back to `exp_output/<sweep>/`.

The matryoshka bake-off's task was named `bakeoff_sweep`; keep one task per sweep.

## Synced project assets

`scripts/sync_extractor_assets_to_windows.sh` is the canonical sync for
extractor/distill assets to `C:\arc_distill\repo_assets\`. ~500 MB, idempotent
via scp. Edit this script when adding new project assets the shard needs.

## What this runbook supersedes

- `docs/design/comfyui-shard-windows.md` — original 77-line install record.
  Kept for audit trail but the table there is stale; trust this doc instead.

## Open follow-ups

- Add `comfyui_controlnet_aux` (planned 2026-05-19 for the photobooth sweep)
- Consider scripting a `scripts/shard_inventory.sh` wrapper around the
  PowerShell probe so every session starts with one command instead of typing
  it manually.
