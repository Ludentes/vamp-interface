"""MLP student: ARKit expression channels (58) -> motion_encoder feature.

Input layout (58):
    [0:52]   Apple ARKit blendshape coefficients in [0..1] (canonical order
             from `arkit_bridge.llf_csv.ARKIT_BLENDSHAPE_NAMES`).
    [52:55]  LeftEye yaw/pitch/roll (radians, Live Link Face wire format).
    [55:58]  RightEye yaw/pitch/roll (radians).

Output: (B, 1, 32, 16) — matches PersonaLive `MotEncoder`'s per-frame slice
(PE-included; trained against the teacher's PE-included output).

Zero-initialised head: at step 0, the student emits all-zero `m_f`, which
is a safe identity-like prior under cross-attention.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MotEncoderStudent(nn.Module):
    def __init__(self, in_dim: int = 58, out_l: int = 32, out_c: int = 16,
                 hidden: tuple = (256, 256, 256)):
        super().__init__()
        self.out_l = out_l
        self.out_c = out_c
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.SiLU(inplace=True)]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_l * out_c)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, b_expr: torch.Tensor) -> torch.Tensor:
        h = self.trunk(b_expr)
        return self.head(h).view(-1, 1, self.out_l, self.out_c)
