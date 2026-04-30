"""Sanity test for blendshape critics (v3_t and v4_pgd).

Encode rendered JPGs at 512² with the Flux VAE, run BlendshapeStudent for
each available checkpoint, print eyeBlink/eyeSquint/jawOpen readings
side-by-side. The G4 gate (see plan): on v1h_bs_only fooling renders the
robust v4_pgd critic should NOT report eyeSquint > 0.30; on v1j_jaw_sanity
real-signal renders it SHOULD still read jawOpen ≥ 0.85.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, "/home/newub/w/vamp-interface/src")

from arc_distill.encode_aligned_to_latent import build_flux_vae, FLUX_VAE_CONFIG
from mediapipe_distill.student import BlendshapeStudent
from mediapipe_distill.build_compact_blendshapes import BLENDSHAPE_CHANNELS

VAE_PATH = Path("/home/newub/w/ComfyUI/models/vae/FLUX1/ae.safetensors")
BS_CKPTS = {
    "v3_t": Path("/home/newub/w/vamp-interface/models/mediapipe_distill/bs_v3_t/final.pt"),
    "v4_pgd": Path("/home/newub/w/vamp-interface/models/mediapipe_distill/bs_v4_pgd/final.pt"),
}
DEVICE = "cuda"
DTYPE = torch.bfloat16

CHAN = {n: i for i, n in enumerate(BLENDSHAPE_CHANNELS)}


def encode(vae, paths):
    shift = float(FLUX_VAE_CONFIG["shift_factor"])
    scale = float(FLUX_VAE_CONFIG["scaling_factor"])
    xs = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((512, 512), Image.LANCZOS)
        t = torch.from_numpy(__import__("numpy").array(im)).permute(2, 0, 1).float() / 255.0
        t = (t - 0.5) * 2.0
        xs.append(t)
    x = torch.stack(xs).to(DEVICE, DTYPE)
    with torch.no_grad():
        z = vae.encode(x).latent_dist.sample()
        z = (z - shift) * scale
    return z  # (B,16,64,64) bf16


def read_bs(student, latents):
    with torch.no_grad():
        out = student(latents.float())  # (B,52)
    return out.float().cpu()


def load_student(ckpt_path: Path) -> BlendshapeStudent:
    student = BlendshapeStudent(variant="bs_a").to(DEVICE).eval()
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    student.load_state_dict(state)
    return student


def main():
    print("loading VAE...")
    vae = build_flux_vae(VAE_PATH, DTYPE, DEVICE)

    students = {}
    for tag, ckpt in BS_CKPTS.items():
        if not ckpt.exists():
            print(f"skip {tag}: {ckpt} not found")
            continue
        print(f"loading {tag} from {ckpt}...")
        students[tag] = load_student(ckpt)
    if not students:
        raise SystemExit("no checkpoints found")

    v0_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_slider_v0/samples")
    v0_paths = sorted(v0_root.glob("*000001800_*.jpg"))
    assert len(v0_paths) == 9, v0_paths
    v1h_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_lora_v1h_bs_only/samples")
    v1h_paths = sorted(v1h_root.glob("*000000200_*.jpg"))
    v1i_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_lora_v1i_sganchor/samples")
    v1i_paths = sorted(v1i_root.glob("*000000200_*.jpg"))
    v1j_root = Path("/home/newub/w/vamp-interface/output/ai_toolkit_runs/squint_lora_v1j_jaw_sanity/samples")
    v1j_paths = sorted(v1j_root.glob("*000000200_*.jpg"))
    v0_step0 = sorted(v0_root.glob("*000000000_*.jpg"))

    all_paths = v0_paths + v1h_paths + v1i_paths + v1j_paths + v0_step0
    z = encode(vae, all_paths)

    sections = [
        ("v0 squint_slider step 1800 (visibly squinting at m=+1.5)", v0_paths),
        ("v1h_bs_only step 200 (m=1.0, NO visible squint — fooling case)", v1h_paths),
        ("v1i_sganchor step 200 (m=1.0, no visible squint)", v1i_paths),
        ("v1j_jaw_sanity step 200 (mouths visibly open)", v1j_paths),
        ("v0 step 0 baseline", v0_step0),
    ]

    eL, eR = CHAN['eyeSquintLeft'], CHAN['eyeSquintRight']
    bL, bR = CHAN['eyeBlinkLeft'], CHAN['eyeBlinkRight']
    jO = CHAN['jawOpen']

    for tag, student in students.items():
        bs = read_bs(student, z)
        print(f"\n\n#### CRITIC: {tag} #################################################")
        offset = 0
        for label, paths in sections:
            n = len(paths)
            print(f"\n=== [{tag}] {label} ===")
            for i in range(n):
                b = bs[offset + i]
                sq = (b[eL].item() + b[eR].item()) / 2
                bk = (b[bL].item() + b[bR].item()) / 2
                jo = b[jO].item()
                print(f"  {Path(paths[i]).name:<55}  "
                      f"squint={sq:.3f}  blink={bk:.3f}  jaw={jo:.3f}")
            offset += n


if __name__ == "__main__":
    main()
