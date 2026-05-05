import os
import sys

import pytest
import torch

MOORE = os.path.expanduser("~/w/Moore-AnimateAnyone")
PERSONA_PG = os.path.expanduser(
    "~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth"
)

if MOORE not in sys.path:
    sys.path.insert(0, MOORE)


@pytest.mark.skipif(
    not os.path.exists(PERSONA_PG), reason="PersonaLive weights not present"
)
def test_teacher_loads_and_runs():
    from arkit_bridge.teacher import load_frozen_teacher

    teacher = load_frozen_teacher(weights_path=PERSONA_PG, device="cpu")
    assert all(not p.requires_grad for p in teacher.parameters())
    x = torch.randn(1, 3, 1, 512, 512)
    with torch.no_grad():
        y = teacher(x)
    assert y.shape == (1, 320, 1, 64, 64), y.shape
