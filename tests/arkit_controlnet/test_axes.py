from arkit_controlnet.axes import AXES, Axis
from arkit_controlnet.eval_spike import ARKIT_BLENDSHAPE_NAMES


def test_smile_axis_is_defined_and_well_formed():
    smile = AXES["smile"]
    assert isinstance(smile, Axis)
    assert smile.edit_prompt_a and smile.edit_prompt_b
    assert smile.mix_b == 0.5
    # scale band is ascending and inside the verified smile window
    assert smile.scale_band == sorted(smile.scale_band)
    assert min(smile.scale_band) >= 0.0 and max(smile.scale_band) <= 2.3
    # the ARKit channels this axis is expected to move
    assert "mouthSmileLeft" in smile.target_channels
    assert "mouthSmileRight" in smile.target_channels


def test_three_axes_present():
    assert set(AXES) == {"smile", "pucker", "surprise"}


def test_every_target_channel_is_a_real_arkit_name():
    # bs_delta indexes target_channels directly; a typo would KeyError mid-sweep.
    valid = set(ARKIT_BLENDSHAPE_NAMES)
    for axis in AXES.values():
        unknown = [c for c in axis.target_channels if c not in valid]
        assert not unknown, f"{axis.name}: unknown ARKit channels {unknown}"
