"""ChibiPipeline — compose the chibi re-priming stages.

Stages run global-structure-first: head_block, proportion_remap, then (added
in later tasks) relief_flatten and feature_primitives. `run(verts, faces,
through=)` applies them in order, optionally stopping after a named stage for
inspection. Parameters load from one JSON; absent keys use the dataclass
defaults.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from chibi.stages.head_block import head_block, HeadBlockParams
from chibi.stages.proportion_remap import proportion_remap, RemapParams
from chibi.stages.relief_flatten import relief_flatten, ReliefParams
from chibi.stages.feature_primitives import feature_primitives, FeatureParams

# stage name -> callable. All four re-priming stages.
STAGE_ORDER = ["head_block", "proportion_remap", "relief_flatten",
               "feature_primitives"]


class ChibiPipeline:
    def __init__(self, masks_path: str, params_path: str | None = None):
        self.masks_path = masks_path
        cfg = {}
        if params_path and Path(params_path).exists():
            cfg = json.loads(Path(params_path).read_text())
        self.head_block = HeadBlockParams(**cfg.get("head_block", {}))
        self.remap = RemapParams(**cfg.get("proportion_remap", {}))
        self.relief = ReliefParams(**cfg.get("relief_flatten", {}))
        self.features = FeatureParams(**cfg.get("feature_primitives", {}))

    def run(self, verts: torch.Tensor, faces: torch.Tensor,
            through: str | None = None) -> torch.Tensor:
        """Apply stages in order. `through` stops after that stage (inclusive).
        Returns deformed verts (V,3); faces are never modified."""
        if through is not None and through not in STAGE_ORDER:
            raise ValueError(f"unknown stage {through!r}; "
                             f"known: {STAGE_ORDER}")
        v = verts.to(torch.float64)
        v = head_block(v, faces, self.masks_path, self.head_block)
        if through == "head_block":
            return v
        v = proportion_remap(v, self.remap)
        if through == "proportion_remap":
            return v
        v = relief_flatten(v, faces, self.masks_path, self.relief)
        if through == "relief_flatten":
            return v
        v = feature_primitives(v, faces, self.masks_path, self.features)
        return v
