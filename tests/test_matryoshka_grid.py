"""Tests for the matryoshka sweep's deterministic grid + workflow substitution.

Only the ComfyUI-free logic is unit-tested here -- build_grid and
build_workflow. The HTTP plumbing is covered by the Phase 0 run gate.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import matryoshka_sweep as ms  # noqa: E402


def test_grid_size():
    # 4 id x 2 cn x 3 pw x 2 ps x 2 seeds
    assert len(ms.build_grid()) == 96


def test_seeds_and_stems_unique():
    grid = ms.build_grid()
    assert len({c["seed"] for c in grid}) == 96
    assert len({c["stem"] for c in grid}) == 96


def test_seed_is_deterministic():
    grid = ms.build_grid()
    assert grid[0]["seed"] == 70_000_000
    assert grid[1]["seed"] == 70_000_000 + 7919


def test_cn0_cell_is_prompt_only():
    grid = ms.build_grid()
    cn0 = [c for c in grid if c["cn_strength"] == 0.0]
    assert len(cn0) == 48
    assert all(c["prompt"] == ms.PROMPT for c in cn0)


def test_workflow_substitution():
    grid = ms.build_grid()
    template = {
        "8": {"class_type": "ApplyPulidFlux", "inputs": {}},
        "13": {"class_type": "ControlNetApplyAdvanced", "inputs": {}},
        "p": {"inputs": {"x": "$$POSITIVE_PROMPT", "s": "$$SEED",
                         "cn": "$$CN_STRENGTH", "pw": "$$PULID_WEIGHT",
                         "canny": "$$CANNY_FILENAME", "la": "$$LORA_A_STRENGTH"}},
    }
    cell = grid[0]
    wf = ms.build_workflow(template, cell, "matryoshka/chibi/test")
    ins = wf["p"]["inputs"]
    assert ins["x"] == cell["prompt"]
    assert ins["s"] == cell["seed"]
    assert ins["cn"] == cell["cn_strength"]
    assert ins["canny"] == ms.CANNY_TEMPLATE
    assert ins["la"] == 0.0  # no style LoRA in v1
    assert wf["13"]["inputs"]["start_percent"] == cell["cn_start"]
    assert wf["8"]["inputs"]["end_at"] == cell["pulid_end"]
