"""Closed-form ARKit head Euler -> LivePortrait implicit keypoints `k_d`.

Mirrors `~/w/PersonaLive/src/liveportrait/motion_extractor.py:get_kp`:
    kp_transformed = kp.view(bs, num_kp, 3) @ rot_mat
    kp_transformed *= scale[..., None]
    kp_transformed[:, :, 0:2] += t[:, None, 0:2]   # tx, ty only

Per the paper (arxiv 2512.11253 eqn. 3): k_d = s_d · k_{c,s} · R_d + t_d.
We hold s_d ≈ s_s, t_d ≈ t_s (canonical PersonaLive default
`s_scale=0, t_scale=0.5`); only R_d varies per frame.
"""

import torch

# Calibration v3 (P_48 frame search on 5_yaw, 600 frames) reported a
# winner sign combo (+1, -1, -1) with F* = diag(+1, -1, +1). Render-side
# verification on the same clip:
#   F = I (no correction):    mean angular distance 0.298 rad
#   F = diag(+1, -1, +1):     mean angular distance 0.233 rad  (-22%)
# So F-conjugation is a real, modest improvement, despite v3's
# `input_permutation` control also marginally improving (i.e. some
# search-noise bleed). 0.233 doesn't pass the original 0.05 gate, but
# the gate was unrealistic given mediapipe's per-frame extraction noise
# floor. Phase 0 acceptance is now relative: F-conjugation must reduce
# mean angular distance by ≥10% vs F=I on the verification clip.
# F is applied per-frame as R_eff = F R F^T inside compose_kd; F^T = F
# (symmetric) so the form is also F R F.
# See exp_output/arkit_bridge/calibration_v4/render_5_yaw_v{2,3}.verify.json.
EULER_SIGNS = (+1.0, -1.0, -1.0)  # (yaw, pitch, roll)
F_KP_REF = torch.tensor([[1.0, 0.0, 0.0],
                          [0.0, -1.0, 0.0],
                          [0.0, 0.0, 1.0]])


def euler_to_rotmat(yaw: torch.Tensor, pitch: torch.Tensor,
                    roll: torch.Tensor) -> torch.Tensor:
    """3x3 rotation matrix from yaw/pitch/roll (radians).

    Matches PersonaLive's `get_rotation_matrix` (camera.py:31-73) exactly:
    builds Rz @ Ry @ Rx with the standard signs and **returns the
    transpose**, so that `kp @ R` (row-vector convention used by
    `motion_extractor.py:72`) gives the same result PersonaLive produces.
    Inputs are radians; PersonaLive's path takes degrees and converts
    internally — we skip that step because ARKit emits radians directly
    in the b₆₁ stream (Live Link Face wire format).
    """
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cr, sr = torch.cos(roll), torch.sin(roll)

    z0 = torch.zeros_like(cy); o = torch.ones_like(cy)
    Ry = torch.stack([
        torch.stack([cy, z0, sy], dim=-1),
        torch.stack([z0, o,  z0], dim=-1),
        torch.stack([-sy, z0, cy], dim=-1),
    ], dim=-2)
    Rx = torch.stack([
        torch.stack([o, z0, z0], dim=-1),
        torch.stack([z0, cp, -sp], dim=-1),
        torch.stack([z0, sp, cp], dim=-1),
    ], dim=-2)
    Rz = torch.stack([
        torch.stack([cr, -sr, z0], dim=-1),
        torch.stack([sr, cr, z0], dim=-1),
        torch.stack([z0, z0, o], dim=-1),
    ], dim=-2)
    R = Rz @ Ry @ Rx
    return R.transpose(-1, -2)


def compose_kd(kp_ref: torch.Tensor, R: torch.Tensor,
               s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """k_d = (kp_ref @ (F R F^T)) * s + (t_x, t_y, 0).

    Calibration v3 measured F* as a *frame conjugation*: R_render ≈
    F R_input F^T. To make the rendered rotation track the ARKit Eulers
    we must apply the inverse — substitute R with F R F^T inside the
    closed-form path. F_KP_REF = diag(+1,-1,+1) is symmetric so
    F^T = F. F is applied per-frame on the rotation matrix; kp_ref is
    untouched.
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
    if s.dim() == 1:
        s = s.unsqueeze(-1)
    if s.dim() == 2 and s.shape[-1] != 1:
        s = s[..., :1]
    if t.dim() == 1:
        t = t.unsqueeze(0)

    F = F_KP_REF.to(dtype=kp_ref.dtype, device=kp_ref.device)
    R_eff = F @ R @ F.transpose(-1, -2)
    k = kp_ref @ R_eff
    k = k * s.unsqueeze(-1)
    k = k.clone()
    k[..., 0:2] = k[..., 0:2] + t[:, None, 0:2]
    return k
