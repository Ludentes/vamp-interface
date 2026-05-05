import os
import pytest
import torch

PERSONA_ME = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/motion_encoder.pth")
PERSONA_MX = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/motion_extractor.pth")
PERSONA_PG = os.path.expanduser("~/w/PersonaLive/pretrained_weights/personalive/pose_guider.pth")


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in [PERSONA_ME, PERSONA_MX, PERSONA_PG]),
    reason="PersonaLive weights missing",
)
def test_motion_encoder_loads_and_runs():
    from arkit_bridge.teacher_personalive import load_motion_encoder
    me = load_motion_encoder(device="cpu")
    assert all(not p.requires_grad for p in me.parameters())
    with torch.no_grad():
        y = me(torch.zeros(1, 3, 1, 224, 224))
    assert y.shape == (1, 1, 32, 16)


@pytest.mark.skipif(not os.path.exists(PERSONA_MX), reason="weights missing")
def test_motion_extractor_loads_and_runs():
    from arkit_bridge.teacher_personalive import load_motion_extractor
    mx = load_motion_extractor(device="cpu")
    with torch.no_grad():
        kp = mx(torch.zeros(1, 3, 256, 256))
    assert kp.shape == (1, 21, 3)


@pytest.mark.skipif(not os.path.exists(PERSONA_PG), reason="weights missing")
def test_pose_guider_loads_and_runs():
    from arkit_bridge.teacher_personalive import load_pose_guider
    pg = load_pose_guider(device="cpu")
    with torch.no_grad():
        y = pg(torch.zeros(1, 3, 1, 512, 512))
    assert y.shape == (1, 320, 1, 64, 64)
