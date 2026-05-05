"""Parametric PoseGuider student: 61 floats -> (B, 320, 1, 64, 64)."""

import torch
import torch.nn as nn


class ARKitParametricPoseGuider(nn.Module):
    """Drop-in replacement for PersonaLive's PoseGuider at inference.

    Input: (B, 61) — 52 ARKit blendshapes + 9 rotation floats
        b[:, 0:52]   ARKit blendshape coefficients (clamped 0..1 by ARKit)
        b[:, 52:55]  head yaw/pitch/roll (radians)
        b[:, 55:58]  leftEye yaw/pitch/roll (radians)
        b[:, 58:61]  rightEye yaw/pitch/roll (radians)
    Output: (B, 320, 1, 64, 64) — matches PoseGuider on (B, 3, 1, 512, 512) input.
    """

    def __init__(
        self,
        in_dim: int = 61,
        out_channels: int = 320,
        seed_size: int = 8,
        hidden_channels: tuple = (128, 128, 96, 64),
    ):
        super().__init__()
        self.seed_size = seed_size
        self.hidden = hidden_channels
        c0 = hidden_channels[0]
        self.proj = nn.Linear(in_dim, c0 * seed_size * seed_size)
        layers = []
        cin = c0
        for cout in hidden_channels[1:]:
            layers += [
                nn.ConvTranspose2d(cin, cout, kernel_size=4, stride=2, padding=1),
                nn.SiLU(inplace=True),
            ]
            cin = cout
        layers.append(nn.Conv2d(cin, out_channels, kernel_size=3, padding=1))
        self.decoder = nn.Sequential(*layers)
        # PoseGuider's conv_out is zero-init; mimic that so untrained student
        # contributes zero bias (safe drop-in even before training).
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        h = self.proj(b).view(-1, self.hidden[0], self.seed_size, self.seed_size)
        h = self.decoder(h)
        return h.unsqueeze(2)
