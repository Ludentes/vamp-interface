"""Axis catalog + LHS sampler for the photobooth Phase-1 sweep.

Six axes post-spike (see spec addendum 2026-05-20):
five categorical + one continuous.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.stats import qmc


@dataclass(frozen=True)
class Axis:
    name: str
    values: tuple[Any, ...] | tuple[float, float]
    kind: str  # "cat" | "cont"


AXES: tuple[Axis, ...] = (
    Axis("face_pixel_budget", ("natural_1024", "tight_1024", "natural_768"), "cat"),
    Axis("cn_condition", ("canny", "depth"), "cat"),
    Axis("cn_strength", (0.80, 1.00), "cont"),
    Axis("canny_preset", ("soft", "default", "aggressive"), "cat"),
    Axis("refine_denoise", (0.00, 0.10, 0.15), "cat"),
    Axis("demo_inject", ("off", "on"), "cat"),
)


def lhs_sample(n_cells: int, seed: int = 0) -> list[dict[str, Any]]:
    """Latin Hypercube over the 6 axes. Returns one dict per cell with the
    string/float value (not the unit-cube coordinate).

    `canny_preset` is set to None whenever `cn_condition == "depth"` (the axis
    is irrelevant for depth)."""
    rng = qmc.LatinHypercube(d=len(AXES), seed=seed,
                             optimization="random-cd")
    u = rng.random(n_cells)  # shape (n_cells, d), all in [0, 1)
    rows: list[dict[str, Any]] = []
    for i in range(n_cells):
        row: dict[str, Any] = {}
        for j, ax in enumerate(AXES):
            uij = float(u[i, j])
            if ax.kind == "cat":
                idx = min(int(uij * len(ax.values)), len(ax.values) - 1)
                row[ax.name] = ax.values[idx]
            else:
                lo, hi = ax.values
                row[ax.name] = round(lo + uij * (hi - lo), 4)
        if row["cn_condition"] == "depth":
            row["canny_preset"] = None
        rows.append(row)
    return rows


def cell_id(photo_id: str, idx: int, seed_iter: int = 0) -> str:
    if seed_iter == 0:
        return f"{photo_id}__cfg{idx:03d}"
    return f"{photo_id}__cfg{idx:03d}_s{seed_iter}"


# Phase 4 — mask-shape × swap_weight sweep (see docs/research/2026-05-20-photobooth-architecture.md).
# Hypothesis (post-Phase-3): swap_weight alone doesn't preserve style because
# the doll's ArcFace target is OOD. Instead control the swap *surface*: erode
# or feather the HyperSwap paste mask so less of the rendered doll is
# overwritten. mask_erode_px / mask_feather_px are in 256-crop pixels.
# Grid: {erode, feather} × {16, 32, 64} × {0.30, 0.10} = 12 cfgs. Reuses
# Phase 3 renders via driver's --reuse-render-from flag.
def phase4_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mode in ("erode", "feather"):
        for radius in (16, 32, 64):
            for w in (0.30, 0.10):
                rows.append({
                    "face_pixel_budget": "natural_1024",
                    "cn_condition": "canny",
                    "cn_strength": 0.85,
                    "canny_preset": "soft",
                    "refine_denoise": 0.00,
                    "demo_inject": "on",
                    "swap_weight": w,
                    "mask_mode": mode,
                    "mask_radius": radius,
                })
    return rows


# Phase 3 — face_swapper_weight sweep (see docs/research/2026-05-20-hyperswap-parameters.md
# and 2026-05-20-photobooth-phase2-findings.md). Locks Phase 2's robust default
# (natural_1024, canny, soft, cn_strength=0.85, refine=0, demo_inject=on) and
# sweeps only the new axis: HyperSwap's embedding-mix weight.
# Lower weight = embedding blends toward the doll's painted face (matryoshka-
# deferred); 0.5 = current pure-source default; >0.5 = amplify identity.
def phase3_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for w in (0.5, 0.4, 0.3, 0.2, 0.1, 0.0):
        rows.append({
            "face_pixel_budget": "natural_1024",
            "cn_condition": "canny",
            "cn_strength": 0.85,
            "canny_preset": "soft",
            "refine_denoise": 0.00,
            "demo_inject": "on",
            "swap_weight": w,
        })
    return rows


# Phase 2 — post-pruning grid (see docs/research/2026-05-20-photobooth-phase1-findings.md).
# Locked: cn_strength=0.85, refine_denoise=0.00, demo_inject="on".
# Open axes: cn_condition × face_pixel_budget × canny_preset (canny only).
def phase2_configs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fpb in ("natural_1024", "tight_1024"):
        for preset in ("soft", "aggressive"):
            rows.append({
                "face_pixel_budget": fpb,
                "cn_condition": "canny",
                "cn_strength": 0.85,
                "canny_preset": preset,
                "refine_denoise": 0.00,
                "demo_inject": "on",
            })
        rows.append({
            "face_pixel_budget": fpb,
            "cn_condition": "depth",
            "cn_strength": 0.85,
            "canny_preset": None,
            "refine_denoise": 0.00,
            "demo_inject": "on",
        })
    return rows
