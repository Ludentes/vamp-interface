"""matryoshka_refine -- model-free driver logic."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from matryoshka_refine import arm_of, denoise_subdir


def test_arm_of_classifies_by_filename_prefix():
    assert arm_of("zimage_turbo_st06_euler_simple_seed74029470.png") == "zimage_turbo"
    assert arm_of("sdxl_lightning_st04_dpmpp_sde_seed73823576.png") == "sdxl_lightning"


def test_arm_of_returns_none_for_unknown_arm():
    assert arm_of("flux_krea_seed73000000.png") is None


def test_denoise_subdir_formats_three_digits():
    assert denoise_subdir(0.55) == "d055"
    assert denoise_subdir(0.4) == "d040"
