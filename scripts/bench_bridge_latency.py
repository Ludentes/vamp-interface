"""Bridge inference latency micro-bench: MotEncoderStudent + closed_form_pose.

Times one frame's worth of bridge work (student forward + euler_to_rotmat
+ compose_kd) on the GPU. This is everything between an incoming b_61 LLF
packet and the seam-installed PersonaLive pipeline; measures the
ARKit-bridge overhead, not PersonaLive itself.

Gates (per `docs/research/2026-05-06-vtuber-pipeline-priorities.md`,
rescaled from 4080 to 5090):
  - HARD FAIL: > 5 ms/frame (bridge becomes the bottleneck)
  - SOFT GATE: > 2 ms/frame (rescaled 5090 target)
  - PASS: <= 2 ms/frame
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.student import MotEncoderStudent  # noqa: E402
from arkit_bridge.closed_form_pose import (  # noqa: E402
    compose_kd,
    euler_to_rotmat,
)

OUT = ROOT / "exp_output/arkit_bridge/diagnostics/bridge_latency_5090.json"
CKPT = ROOT / "runs/student_v2_120k/student_best.pt"


def main() -> None:
    device = "cuda"
    student = MotEncoderStudent().to(device).eval()
    state = torch.load(CKPT, map_location=device, weights_only=True)
    student.load_state_dict(state)

    n_iters = 1000
    b58 = torch.randn(1, 58, device=device)
    yaw = torch.tensor(0.1, device=device)
    pitch = torch.tensor(0.0, device=device)
    roll = torch.tensor(0.0, device=device)
    kp_ref = torch.randn(1, 21, 3, device=device)
    s = torch.tensor([[1.0]], device=device)
    t = torch.tensor([[0.0, 0.0, 0.0]], device=device)

    # warm
    for _ in range(20):
        with torch.no_grad():
            _ = student(b58)
            R = euler_to_rotmat(yaw, pitch, roll).unsqueeze(0)
            _ = compose_kd(kp_ref, R, s, t)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            _ = student(b58)
            R = euler_to_rotmat(yaw, pitch, roll).unsqueeze(0)
            _ = compose_kd(kp_ref, R, s, t)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms_per_frame = 1000.0 * dt / n_iters

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "device": torch.cuda.get_device_name(0),
        "ckpt": str(CKPT.relative_to(ROOT)),
        "n_iters": n_iters,
        "ms_per_frame_student_plus_pose": ms_per_frame,
        "fps_equivalent": 1000.0 / ms_per_frame,
        "gate_2ms": ms_per_frame <= 2.0,
        "gate_5ms": ms_per_frame <= 5.0,
    }, indent=2))

    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  iters:  {n_iters}")
    print(f"  bridge: {ms_per_frame:.3f} ms/frame  ({1000.0/ms_per_frame:.1f} FPS-equivalent)")
    if ms_per_frame > 5.0:
        print("  HARD FAIL: > 5 ms/frame, bridge is the bottleneck")
        sys.exit(1)
    if ms_per_frame > 2.0:
        print("  SOFT WARN: > 2 ms/frame (5090 target); not blocking")
    else:
        print("  PASS")


if __name__ == "__main__":
    main()
