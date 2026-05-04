"""Validate a MediaPipe-blendshape student against the project's gates.

Mirrors the arc_distill `validate_as_loss.py` shape but for 52-d regression:

  Layer 1.1 — per-channel R² with degenerate-target guard, tiered by val
              variance (high-var top-20 / mid-var next-15 / low-var rest).
  Layer 1.2 — aggregate gates (median R², mean R², fraction-above-0.5,
              fraction-negative, val MSE).
  Layer 1.3a — augmentation invariance: positive-pair L2 vs negative-pair L2
               under {gauss σ=0.02, σ=0.05, shift_h, shift_w}. Excludes hflip
               here — see 1.3b below for the channel-aware version.
  Layer 1.3b — hflip with channel-mirror: hflip the latent, swap left/right
               blendshape channels in the prediction, verify it matches the
               original prediction at L2 < 0.05 mean.
  Layer 2 — gradient sanity: backprop student into a Flux latent toward a
            random other row's blendshape vector, verify finite + descent.

Outputs JSON with per-tier gate results + an overall pass/fail.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import is_held_out
from .student import BlendshapeStudent


# Channels that have a left/right counterpart. For hflip-with-channel-mirror,
# left↔right pairs must be swapped after flipping the latent. Empirically
# verified against MediaPipe via `hflip_diagnostic.py` 2026-04-30: straight
# L↔R swap (NOT eyeLookIn↔Out cross-flip — MediaPipe preserves "looking-in"
# vs "looking-out" semantics relative to the eye, not the image).
LEFT_RIGHT_PAIRS = [
    ("browDownLeft", "browDownRight"),
    ("browOuterUpLeft", "browOuterUpRight"),
    ("cheekSquintLeft", "cheekSquintRight"),
    ("eyeBlinkLeft", "eyeBlinkRight"),
    ("eyeLookDownLeft", "eyeLookDownRight"),
    ("eyeLookInLeft", "eyeLookInRight"),       # straight swap (was cross-flip — wrong)
    ("eyeLookOutLeft", "eyeLookOutRight"),     # straight swap (was cross-flip — wrong)
    ("eyeLookUpLeft", "eyeLookUpRight"),
    ("eyeSquintLeft", "eyeSquintRight"),
    ("eyeWideLeft", "eyeWideRight"),
    ("mouthDimpleLeft", "mouthDimpleRight"),
    ("mouthFrownLeft", "mouthFrownRight"),
    ("mouthLowerDownLeft", "mouthLowerDownRight"),
    ("mouthPressLeft", "mouthPressRight"),
    ("mouthSmileLeft", "mouthSmileRight"),
    ("mouthStretchLeft", "mouthStretchRight"),
    ("mouthUpperUpLeft", "mouthUpperUpRight"),
    ("noseSneerLeft", "noseSneerRight"),
]
# `jawLeft` ↔ `jawRight`, `mouthLeft` ↔ `mouthRight` — direction reverses
# under mirror.
DIRECTIONAL_PAIRS = [
    ("jawLeft", "jawRight"),
    ("mouthLeft", "mouthRight"),
]


def build_mirror_perm(channel_names: list[str]) -> torch.Tensor:
    """Return a (52,) permutation `p` such that p[i] is the index of the
    channel that appears at position i in a mirrored prediction."""
    name_to_idx = {n: i for i, n in enumerate(channel_names)}
    perm = list(range(len(channel_names)))
    for a, b in LEFT_RIGHT_PAIRS + DIRECTIONAL_PAIRS:
        if a in name_to_idx and b in name_to_idx:
            ai, bi = name_to_idx[a], name_to_idx[b]
            perm[ai], perm[bi] = bi, ai
    return torch.tensor(perm, dtype=torch.long)


def load_val(compact_path: Path, blendshapes_path: Path):
    compact = torch.load(compact_path, map_location="cpu", weights_only=False)
    bs_blob = torch.load(blendshapes_path, map_location="cpu", weights_only=False)
    if list(compact["shas"]) != list(bs_blob["shas"]):
        raise ValueError("compact + blendshapes SHA order mismatch")
    detected = bs_blob["detected"]
    val_idx = [i for i, s in enumerate(compact["shas"])
               if is_held_out(s) and bool(detected[i])]
    val_idx_t = torch.tensor(val_idx, dtype=torch.long)
    return {
        "latents": compact["latents"][val_idx_t].to(torch.float32),
        "blendshapes": bs_blob["blendshapes"][val_idx_t].to(torch.float32),
        "channel_names": list(bs_blob["channel_names"]),
    }


def run_inference(model: BlendshapeStudent, latents: torch.Tensor,
                  device: torch.device, batch_size: int = 128) -> torch.Tensor:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, latents.size(0), batch_size):
            x = latents[i:i + batch_size].to(device, non_blocking=True)
            out.append(model(x).cpu())
    return torch.cat(out, dim=0)


def per_channel_r2(pred: torch.Tensor, target: torch.Tensor,
                   var_threshold: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (r2 (C,), valid_mask (C,) bool). Channels with target variance
    below threshold are marked invalid (R² returned as 0 but the mask flags
    them so they don't pollute aggregate stats)."""
    mu = target.mean(dim=0, keepdim=True)
    ss_res = ((target - pred) ** 2).sum(dim=0)
    ss_tot = ((target - mu) ** 2).sum(dim=0)
    valid = ss_tot >= var_threshold
    r2 = torch.where(valid, 1.0 - ss_res / (ss_tot + 1e-12),
                     torch.zeros_like(ss_tot))
    return r2, valid


def layer11_per_channel(pred: torch.Tensor, target: torch.Tensor,
                        channel_names: list[str]) -> dict:
    r2, valid = per_channel_r2(pred, target)
    stds = target.std(dim=0)
    # Sort valid channels by std; top-20 = high-var, next-15 = mid-var.
    valid_idx = torch.where(valid)[0]
    if len(valid_idx) > 0:
        order = torch.argsort(stds[valid_idx], descending=True)
        sorted_idx = valid_idx[order]
        n_valid = len(sorted_idx)
        high = sorted_idx[: min(20, n_valid)].tolist()
        mid = sorted_idx[20: min(35, n_valid)].tolist() if n_valid > 20 else []
        low = sorted_idx[35:].tolist() if n_valid > 35 else []
    else:
        high, mid, low = [], [], []

    def tier(idx_list: list[int]) -> dict:
        if not idx_list:
            return {"n": 0}
        rs = r2[idx_list]
        return {
            "n": len(idx_list),
            "median": float(rs.median()),
            "mean": float(rs.mean()),
            "min": float(rs.min()),
            "max": float(rs.max()),
            "channels": [channel_names[i] for i in idx_list],
            "r2_per_channel": rs.tolist(),
        }

    return {
        "n_channels_total": int(target.size(1)),
        "n_channels_valid": int(valid.sum()),
        "n_channels_degenerate": int((~valid).sum()),
        "degenerate_channels": [channel_names[i]
                                for i in range(len(channel_names)) if not valid[i]],
        "high_var_tier": tier(high),
        "mid_var_tier": tier(mid),
        "low_var_tier": tier(low),
    }


def layer12_aggregate(pred: torch.Tensor, target: torch.Tensor) -> dict:
    r2, valid = per_channel_r2(pred, target)
    rv = r2[valid]
    return {
        "val_mse": float(F.mse_loss(pred, target).item()),
        "r2_median": float(rv.median()) if len(rv) else 0.0,
        "r2_mean": float(rv.mean()) if len(rv) else 0.0,
        "r2_p05": float(rv.kthvalue(max(1, int(0.05 * len(rv)))).values) if len(rv) else 0.0,
        "r2_min": float(rv.min()) if len(rv) else 0.0,
        "frac_above_0p5": float((rv >= 0.5).float().mean()) if len(rv) else 0.0,
        "frac_above_0p7": float((rv >= 0.7).float().mean()) if len(rv) else 0.0,
        "frac_below_zero": float((rv < 0).float().mean()) if len(rv) else 0.0,
        "n_valid": int(valid.sum()),
    }


def layer13a_invariance(model: BlendshapeStudent, latents: torch.Tensor,
                        pred_anchor: torch.Tensor, device: torch.device,
                        seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(latents.size(0), generator=g)
    out = {}
    for kind in ["gauss_002", "gauss_005", "shift_h", "shift_w"]:
        torch.manual_seed(seed)
        if kind == "gauss_002":
            perturbed = latents + 0.02 * torch.randn_like(latents)
        elif kind == "gauss_005":
            perturbed = latents + 0.05 * torch.randn_like(latents)
        elif kind == "shift_h":
            perturbed = torch.roll(latents, shifts=1, dims=-2)
        else:
            perturbed = torch.roll(latents, shifts=1, dims=-1)
        pred_p = run_inference(model, perturbed, device)
        pos_l2 = (pred_anchor - pred_p).pow(2).sum(dim=-1).sqrt()
        neg_l2 = (pred_anchor - pred_p[perm]).pow(2).sum(dim=-1).sqrt()
        out[kind] = {
            "pos_l2_mean": float(pos_l2.mean()),
            "neg_l2_mean": float(neg_l2.mean()),
            "ratio_pos_over_neg": float(pos_l2.mean() / (neg_l2.mean() + 1e-12)),
        }
    return out


def layer13b_hflip(model: BlendshapeStudent, latents: torch.Tensor,
                   pred_anchor: torch.Tensor, channel_names: list[str],
                   device: torch.device) -> dict:
    perm = build_mirror_perm(channel_names)
    flipped = latents.flip(-1)
    pred_flip_raw = run_inference(model, flipped, device)
    # Apply mirror permutation: for each row, channel-mirror the prediction.
    pred_flip_mirrored = pred_flip_raw[:, perm]
    delta = (pred_anchor - pred_flip_mirrored).pow(2).sum(dim=-1).sqrt()
    return {
        "n_pairs_swapped": int((perm != torch.arange(perm.size(0))).sum()) // 2,
        "raw_l2_mean (no mirror — should be larger)":
            float((pred_anchor - pred_flip_raw).pow(2).sum(dim=-1).sqrt().mean()),
        "mirrored_l2_mean (the gate — should be small)": float(delta.mean()),
        "mirrored_l2_p95": float(delta.kthvalue(max(1, int(0.95 * delta.numel()))).values),
    }


def layer2_gradient(model: BlendshapeStudent, latents: torch.Tensor,
                    targets: torch.Tensor, device: torch.device,
                    n_steps: int = 100, lr: float = 0.05, seed: int = 0) -> dict:
    torch.manual_seed(seed)
    n = latents.size(0)
    src = int(torch.randint(0, n, (1,)).item())
    tgt = int(torch.randint(0, n, (1,)).item())
    while tgt == src:
        tgt = int(torch.randint(0, n, (1,)).item())
    x = latents[src:src + 1].clone().to(device).requires_grad_(True)
    target = targets[tgt:tgt + 1].to(device)
    model.eval()
    opt = torch.optim.SGD([x], lr=lr)
    losses, grad_norms = [], []
    grad_finite = True
    for step in range(n_steps + 1):
        opt.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, target)
        if step < n_steps:
            loss.backward()
            grad = x.grad
            assert grad is not None
            if not torch.isfinite(grad).all():
                grad_finite = False
                break
            grad_norms.append(float(grad.norm()))
            opt.step()
        losses.append(float(loss.item()))
    return {
        "src_idx": src,
        "tgt_idx": tgt,
        "grad_finite": grad_finite,
        "grad_norm_mean": (sum(grad_norms) / len(grad_norms)) if grad_norms else None,
        "loss_step0": losses[0],
        "loss_final": losses[-1],
        "loss_reduction_pct": (1.0 - losses[-1] / losses[0]) * 100 if losses[0] > 0 else 0.0,
        "loss_descended_5pct": losses[-1] < 0.95 * losses[0],
    }


def atom_shippability(pred: torch.Tensor, target: torch.Tensor,
                      atom_library_path: Path,
                      channel_names: list[str],
                      ship_threshold: float = 0.5,
                      confident_threshold: float = 0.7) -> dict:
    """Project (pred, target) from 52-d blendshape space into NMF atom
    coordinates and report per-atom shippability.

    Why this matters: blendshapes are correlated; the *intrinsic* dimensionality
    of expression on FFHQ is ≪52. The NMF basis at `models/blendshape_nmf/au_library.npz`
    has 8 atoms with semantic tags (smile_inphase, jaw_inphase, anger, surprise,
    disgust, pucker, lip_press, alpha_interp_attn). Reporting per-atom R² is
    the more interpretable deliverable: 'this model captures atoms {X, Y, Z}
    cleanly'.

    Projection: linear least-squares pseudo-inverse of H (8 × 52); atoms are
    not orthogonal so we don't just dot-product. Reconstructed atom coords
    can go slightly negative — that's a feature of using lstsq instead of NNLS,
    and OK for a regression analysis.
    """
    import numpy as np
    if not atom_library_path.exists():
        return {"available": False, "reason": f"{atom_library_path} not found"}
    blob = np.load(atom_library_path)
    H = blob["H"]              # (8, 52) atoms × channels
    tags = list(blob["tags"])
    lib_names = list(blob["names"])
    if lib_names != channel_names:
        return {"available": False,
                "reason": f"channel order mismatch: lib first 3 = {lib_names[:3]}, "
                          f"ours first 3 = {channel_names[:3]}"}

    H_t = torch.from_numpy(H).to(torch.float32)         # (8, 52)
    H_pinv = torch.linalg.pinv(H_t)                      # (52, 8)
    pred_atoms = pred @ H_pinv                           # (N, 8)
    target_atoms = target @ H_pinv                       # (N, 8)

    r2_atoms, valid_atoms = per_channel_r2(pred_atoms, target_atoms)
    confident, ship, no_ship, degenerate = [], [], [], []
    for i, tag in enumerate(tags):
        info = {"atom": int(i), "tag": tag, "r2": float(r2_atoms[i]),
                "val_std": float(target_atoms[:, i].std())}
        if not valid_atoms[i]:
            degenerate.append(info)
        elif r2_atoms[i] >= confident_threshold:
            confident.append(info)
        elif r2_atoms[i] >= ship_threshold:
            ship.append(info)
        else:
            no_ship.append(info)
    return {
        "available": True,
        "n_atoms": int(H.shape[0]),
        "atom_tags": tags,
        "n_confident_ship": len(confident),
        "n_ship": len(ship),
        "n_do_not_ship": len(no_ship),
        "n_degenerate": len(degenerate),
        "shippable_atoms": [c["tag"] for c in confident + ship],
        "confident_ship": sorted(confident, key=lambda c: -c["r2"]),
        "ship": sorted(ship, key=lambda c: -c["r2"]),
        "do_not_ship": sorted(no_ship, key=lambda c: -c["r2"]),
        "degenerate": degenerate,
    }


def channel_shippability(pred: torch.Tensor, target: torch.Tensor,
                         channel_names: list[str],
                         ship_threshold: float = 0.5,
                         confident_threshold: float = 0.7) -> dict:
    """Bucket each of the 52 channels into one of:

      - confident-ship  (R² ≥ confident_threshold; trust this channel anywhere)
      - ship            (ship_threshold ≤ R² < confident_threshold; usable as
                         a loss term, treat with documented caveat)
      - do-not-ship     (R² < ship_threshold AND target has nonzero variance;
                         the student doesn't learn this channel from this corpus)
      - degenerate      (target variance ≈ 0; channel is essentially constant on
                         FFHQ — don't ship, but it's a corpus issue not a model
                         failure)

    The deliverable for use-as-a-loss is the union (confident-ship ∪ ship); the
    user masks the loss to those channels. Channels never combine into a
    single mean number — heterogeneity across the 52-d output is real.
    """
    r2, valid = per_channel_r2(pred, target)
    stds = target.std(dim=0)
    confident, ship, no_ship, degenerate = [], [], [], []
    for i, name in enumerate(channel_names):
        info = {"channel": name, "r2": float(r2[i]), "val_std": float(stds[i])}
        if not valid[i]:
            degenerate.append(info)
        elif r2[i] >= confident_threshold:
            confident.append(info)
        elif r2[i] >= ship_threshold:
            ship.append(info)
        else:
            no_ship.append(info)
    shippable_names = [c["channel"] for c in confident + ship]
    return {
        "ship_threshold_r2": ship_threshold,
        "confident_threshold_r2": confident_threshold,
        "n_confident_ship": len(confident),
        "n_ship": len(ship),
        "n_do_not_ship": len(no_ship),
        "n_degenerate": len(degenerate),
        "shippable_channels": shippable_names,
        "shippable_channel_indices": [channel_names.index(n) for n in shippable_names],
        "confident_ship": sorted(confident, key=lambda c: -c["r2"]),
        "ship": sorted(ship, key=lambda c: -c["r2"]),
        "do_not_ship": sorted(no_ship, key=lambda c: -c["r2"]),
        "degenerate": degenerate,
    }


def evaluate_gates(report: dict) -> dict:
    """Gates are about *how many channels* are shippable, not aggregate means.
    A model with 25 channels at R²=0.9 and 27 channels at R²=0.0 is shippable
    (for those 25 channels) — the mean R² hides this completely."""
    ship = report["layer_1_1_shippability"]
    agg = report["layer_1_2_aggregate"]
    l2 = report["layer_2_gradient"]
    l13b = report["layer_1_3b_hflip"]
    l13a = report["layer_1_3a_invariance"]

    n_shippable = ship["n_confident_ship"] + ship["n_ship"]
    floors = {
        "geq_15_shippable_channels": n_shippable >= 15,
        "geq_5_confident_ship_channels": ship["n_confident_ship"] >= 5,
        "val_mse_leq_0p005": agg["val_mse"] <= 0.005,
        "gradient_finite_and_descends_5pct": l2["grad_finite"] and l2["loss_descended_5pct"],
        "hflip_mirror_l2_leq_0p05": l13b["mirrored_l2_mean (the gate — should be small)"] <= 0.05,
        "invariance_pos_lt_neg_5x": all(v["ratio_pos_over_neg"] <= 0.2 for v in l13a.values()),
    }
    targets = {
        "geq_25_shippable_channels": n_shippable >= 25,
        "geq_15_confident_ship_channels": ship["n_confident_ship"] >= 15,
        "val_mse_leq_0p003": agg["val_mse"] <= 0.003,
    }
    return {
        "all_floors_pass": all(floors.values()),
        "all_targets_pass": all(targets.values()),
        "n_shippable_channels": n_shippable,
        "n_confident_ship_channels": ship["n_confident_ship"],
        "shippable_channels": ship["shippable_channels"],
        "floors": floors,
        "targets": targets,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="bs_a")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--blendshapes", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--atom-library", type=Path,
                   default=Path("models/blendshape_nmf/au_library.npz"),
                   help="NMF basis for atom-space analysis (skipped if missing)")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    print(f"variant={args.variant} device={device}")
    print("loading val …")
    val = load_val(args.compact, args.blendshapes)
    latents, targets, channel_names = val["latents"], val["blendshapes"], val["channel_names"]
    print(f"  val rows: {latents.size(0)}  channels: {len(channel_names)}")

    print(f"loading checkpoint: {args.checkpoint}")
    if args.variant in ("bs_v2d", "bs_v2e"):
        if args.variant == "bs_v2d":
            from .student_v2d import BlendshapeLandmarkStudent
            base = BlendshapeLandmarkStudent(args.variant).to(device)
        else:
            from .student_v2e import BlendshapeLandmarkStudentUNet
            base = BlendshapeLandmarkStudentUNet(args.variant).to(device)
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        base.load_state_dict(ck["model"])

        class _BSOnly(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m
            def forward(self, x):
                bs, _ = self.m(x)
                return bs

        model = _BSOnly(base).to(device)
    else:
        model = BlendshapeStudent(args.variant).to(device)
        ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])

    t0 = time.time()
    print("inference …")
    pred = run_inference(model, latents, device)
    print(f"  inference: {time.time() - t0:.2f}s")

    report = {
        "variant": args.variant,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(ck.get("epoch", -1)),
        "n_val": int(latents.size(0)),
    }
    print("layer 1.1 — per-channel R² (tiered)")
    report["layer_1_1_per_channel"] = layer11_per_channel(pred, targets, channel_names)
    print(json.dumps({k: v for k, v in report["layer_1_1_per_channel"].items()
                      if k != "high_var_tier" and k != "mid_var_tier" and k != "low_var_tier"
                      and k != "degenerate_channels"}, indent=2))
    print(f"  high_var: median={report['layer_1_1_per_channel']['high_var_tier'].get('median', 0):.3f}")
    print(f"  mid_var:  median={report['layer_1_1_per_channel']['mid_var_tier'].get('median', 0):.3f}")
    print(f"  low_var:  median={report['layer_1_1_per_channel']['low_var_tier'].get('median', 0):.3f}")

    print("layer 1.1b — channel shippability (the actual deliverable)")
    report["layer_1_1_shippability"] = channel_shippability(pred, targets, channel_names)
    s = report["layer_1_1_shippability"]
    print(f"  confident-ship (R² ≥ 0.7): {s['n_confident_ship']}")
    print(f"  ship           (R² ≥ 0.5): {s['n_ship']}")
    print(f"  do-not-ship    (R² < 0.5): {s['n_do_not_ship']}")
    print(f"  degenerate     (val std ≈ 0): {s['n_degenerate']}")
    print(f"  → shippable channels: {s['shippable_channels']}")

    print("layer 1.1c — NMF atom shippability (the interpretable deliverable)")
    report["layer_1_1_atom_shippability"] = atom_shippability(
        pred, targets, args.atom_library, channel_names)
    a = report["layer_1_1_atom_shippability"]
    if a.get("available"):
        print(f"  confident-ship atoms: {a['n_confident_ship']}/{a['n_atoms']}")
        print(f"  ship atoms:           {a['n_ship']}/{a['n_atoms']}")
        print(f"  do-not-ship atoms:    {a['n_do_not_ship']}/{a['n_atoms']}")
        print(f"  → shippable atom tags: {a['shippable_atoms']}")
    else:
        print(f"  skipped: {a.get('reason')}")

    print("layer 1.2 — aggregate (note: heterogeneous across channels;"
          " do not combine into a single mean for ship decisions)")
    report["layer_1_2_aggregate"] = layer12_aggregate(pred, targets)
    print(json.dumps(report["layer_1_2_aggregate"], indent=2))

    print("layer 1.3a — augmentation invariance")
    report["layer_1_3a_invariance"] = layer13a_invariance(model, latents, pred, device)
    print(json.dumps(report["layer_1_3a_invariance"], indent=2))

    print("layer 1.3b — channel-aware hflip")
    report["layer_1_3b_hflip"] = layer13b_hflip(model, latents, pred, channel_names, device)
    print(json.dumps(report["layer_1_3b_hflip"], indent=2))

    print("layer 2 — gradient sanity")
    report["layer_2_gradient"] = layer2_gradient(model, latents, targets, device)
    print(json.dumps(report["layer_2_gradient"], indent=2))

    print("evaluating gates …")
    report["gates"] = evaluate_gates(report)
    print(json.dumps(report["gates"], indent=2))

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with args.out_json.open("w") as f:
        json.dump(report, f, indent=2)
    print(f"\nwrote {args.out_json}")
    print(f"FLOORS PASS: {report['gates']['all_floors_pass']}")
    print(f"TARGETS PASS: {report['gates']['all_targets_pass']}")


if __name__ == "__main__":
    main()
