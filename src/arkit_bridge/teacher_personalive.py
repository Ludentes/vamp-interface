"""Frozen PersonaLive teacher modules: MotEncoder, MotionExtractor, PoseGuider.

These wrap PersonaLive's own classes with weights loaded and parameters
frozen. PersonaLive's source uses absolute imports of the form
`from src.models...` and `from src.liveportrait...`, so we add the repo
root to sys.path; importing in this order keeps Python's `src.` namespace
pointing at PersonaLive while these helpers are used.
"""

from __future__ import annotations

import os
import sys

import torch

_PERSONA = os.path.expanduser("~/w/PersonaLive")
if _PERSONA not in sys.path:
    sys.path.insert(0, _PERSONA)

PERSONA_ME = os.path.join(_PERSONA, "pretrained_weights/personalive/motion_encoder.pth")
PERSONA_MX = os.path.join(_PERSONA, "pretrained_weights/personalive/motion_extractor.pth")
PERSONA_PG = os.path.join(_PERSONA, "pretrained_weights/personalive/pose_guider.pth")


def _freeze(m: torch.nn.Module) -> torch.nn.Module:
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_motion_encoder(device: str = "cuda") -> torch.nn.Module:
    from src.models.motion_encoder.encoder import MotEncoder
    me = MotEncoder()
    me.load_state_dict(torch.load(PERSONA_ME, map_location="cpu"))
    return _freeze(me).to(device)


def load_motion_extractor(device: str = "cuda") -> torch.nn.Module:
    from src.liveportrait.motion_extractor import MotionExtractor

    class _Wrapped(torch.nn.Module):
        def __init__(self, inner: MotionExtractor):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.inner(x)

        def detect_raw(self, x: torch.Tensor) -> dict:
            return self.inner.detector(x)

    mx = MotionExtractor(num_kp=21)
    mx.load_state_dict(
        torch.load(PERSONA_MX, map_location="cpu"),
        strict=False,
    )
    return _freeze(_Wrapped(mx)).to(device)


def load_pose_guider(device: str = "cuda") -> torch.nn.Module:
    from src.models.pose_guider import PoseGuider
    pg = PoseGuider(
        conditioning_embedding_channels=320,
        block_out_channels=(16, 32, 96, 256),
    )
    pg.load_state_dict(torch.load(PERSONA_PG, map_location="cpu"))
    return _freeze(pg).to(device)
