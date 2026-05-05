"""Per-component latency on RTX 5090 fp16 T=4.

Focused on the question driving the ARKit bridge: is replacing
motion_encoder + motion_extractor with closed-form pose + a small MLP
student materially faster, or is the saving lost in the diffusion UNet?
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PL = Path(os.path.expanduser("~/w/PersonaLive"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(PL))

from arkit_bridge.closed_form_pose import compose_kd, euler_to_rotmat  # noqa: E402
from arkit_bridge.student import MotEncoderStudent                     # noqa: E402
from arkit_bridge.teacher_personalive import (                         # noqa: E402
    load_motion_encoder, load_motion_extractor, load_pose_guider,
)


DEVICE = "cuda"
DTYPE = torch.float16
T = 4
REPS = 20
WARMUP = 5


def timed(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[REPS // 2]


def main():
    os.chdir(PL)
    me = load_motion_encoder(device=DEVICE).to(dtype=DTYPE).eval()
    mx = load_motion_extractor(device=DEVICE).to(dtype=DTYPE).eval()
    pg = load_pose_guider(device=DEVICE).to(dtype=DTYPE).eval()
    student = MotEncoderStudent().to(DEVICE).eval()
    student.load_state_dict(torch.load(
        ROOT / "runs/student_v1/student_best.pt", map_location=DEVICE
    ))

    out = {}

    # 1. Real motion_extractor on T=4 driving frames at 256x256.
    drv = torch.randn(T, 3, 256, 256, device=DEVICE, dtype=DTYPE)
    out["motion_extractor_T4_ms"] = timed(lambda: mx(drv))

    # 2. Closed-form compose_kd for T=4 (CPU euler -> GPU compose).
    kp_ref = torch.randn(T, 21, 3, device=DEVICE, dtype=DTYPE)
    s_ref = torch.ones(T, 1, device=DEVICE, dtype=DTYPE) * 1.2
    t_ref = torch.zeros(T, 3, device=DEVICE, dtype=DTYPE)
    def cf_pose():
        Rs = []
        for k in range(T):
            R = euler_to_rotmat(
                torch.tensor(0.1), torch.tensor(-0.05), torch.tensor(0.02),
            )
            Rs.append(R)
        R = torch.stack(Rs).to(device=DEVICE, dtype=DTYPE)
        compose_kd(kp_ref, R, s_ref, t_ref)
    out["closed_form_kd_T4_ms"] = timed(cf_pose)

    # 3. Real motion_encoder on T=4 chunk (1, 3, T, 224, 224).
    me_in = torch.randn(1, 3, T, 224, 224, device=DEVICE, dtype=DTYPE)
    out["motion_encoder_T4_ms"] = timed(lambda: me(me_in))

    # 4. MotEncoderStudent on T=4 b_expr.
    b = torch.randn(T, 58, device=DEVICE, dtype=torch.float32)
    out["student_T4_ms"] = timed(lambda: student(b))

    # 5. pose_guider on (1, 3, T, 512, 512).
    pg_in = torch.randn(1, 3, T, 512, 512, device=DEVICE, dtype=DTYPE)
    out["pose_guider_T4_ms"] = timed(lambda: pg(pg_in))

    # Comparative summary.
    out["replaced_real_T4_ms"] = (
        out["motion_extractor_T4_ms"] + out["motion_encoder_T4_ms"]
    )
    out["replaced_bridge_T4_ms"] = (
        out["closed_form_kd_T4_ms"] + out["student_T4_ms"]
    )
    out["bridge_speedup_x"] = (
        out["replaced_real_T4_ms"] / out["replaced_bridge_T4_ms"]
        if out["replaced_bridge_T4_ms"] > 0 else None
    )

    out["env"] = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(DTYPE), "T": T, "reps": REPS, "warmup": WARMUP,
    }

    out_path = ROOT / "exp_output/perf/personalive-component-budget.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
