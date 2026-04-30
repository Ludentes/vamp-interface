"""Smoke test: train_t_pgd CLI parses args."""
from __future__ import annotations

import os
import subprocess
import sys


def test_train_t_pgd_help():
    """The script exposes its CLI with PGD-specific flags."""
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/newub/w/vamp-interface/src"
    out = subprocess.check_output(
        [sys.executable, "-m", "mediapipe_distill.train_t_pgd", "--help"],
        text=True,
        cwd="/home/newub/w/vamp-interface",
        env=env,
    )
    assert "--pgd-eps" in out
    assert "--pgd-alpha" in out
    assert "--pgd-k" in out
    assert "--pgd-lambda" in out
