from group_photobooth.face_renderer import HEAVY_MIX_CFG


def test_heavy_mix_cfg_matches_phase3_cfg004():
    assert HEAVY_MIX_CFG["swap_weight"] == 0.10
    assert HEAVY_MIX_CFG["face_pixel_budget"] == "natural_1024"
    assert HEAVY_MIX_CFG["cn_condition"] == "canny"
    assert HEAVY_MIX_CFG["canny_preset"] == "soft"
    assert HEAVY_MIX_CFG["cn_strength"] == 0.85
    assert HEAVY_MIX_CFG["refine_denoise"] == 0.00
    assert HEAVY_MIX_CFG["demo_inject"] == "on"
