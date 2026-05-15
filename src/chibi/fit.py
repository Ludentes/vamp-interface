"""Adam fit of a ChibiField to the painter quarter-grid.

Loss = landmark proportion error + Laplacian smoothness of the displacement
field + a minimal-deformation regularizer (the identity guard). CPU, no
renderer, no detector — the targets are vertex-space landmark positions.
"""
from __future__ import annotations
import json
from pathlib import Path
import torch

from chibi.field import ChibiField
from chibi.landmarks import (landmark_lines, landmark_positions, feature_extents,
                             region_falloff_weights, _template_faces,
                             QUARTER_GRID_TARGETS)


def _mesh_laplacian_penalty(disp: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Uniform-Laplacian smoothness of a per-vertex displacement field.

    For each vertex, penalize ||disp_i - mean(disp of 1-ring neighbours)||^2.
    """
    n = disp.shape[0]
    e = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], 0)
    e = torch.cat([e, e.flip(1)], 0)
    nbr_sum = torch.zeros_like(disp).index_add_(0, e[:, 0], disp[e[:, 1]])
    deg = torch.zeros(n).index_add_(0, e[:, 0], torch.ones(e.shape[0]))
    nbr_mean = nbr_sum / deg.clamp_min(1.0)[:, None]
    return ((disp - nbr_mean) ** 2).sum(1).mean()


def chibi_loss(field: ChibiField, verts: torch.Tensor,
               region_weights: dict, faces: torch.Tensor | None = None,
               lam_smooth: float = 1.0, lam_reg: float = 0.005) -> dict:
    """Return {'total','landmark','smooth','reg'} loss tensors."""
    if faces is None:
        faces = torch.as_tensor(_template_faces(), dtype=torch.long)
    deformed = field(verts, region_weights=region_weights)
    lines = landmark_lines(deformed)
    tgt = QUARTER_GRID_TARGETS["lines"]
    l_lm = sum((lines[k] - tgt[k]) ** 2 for k in tgt)

    # feature-size targets: all three are multipliers of the original
    # (undeformed) extent, so every term is reachable on the same scale.
    ext0 = feature_extents(verts)
    ext1 = feature_extents(deformed)
    eye_mul = ext1["eye_y"] / ext0["eye_y"].clamp_min(1e-6)
    nose_mul = ext1["nose_z"] / ext0["nose_z"].clamp_min(1e-6)
    mouth_mul = ext1["mouth_y"] / ext0["mouth_y"].clamp_min(1e-6)
    l_lm = l_lm + (eye_mul - QUARTER_GRID_TARGETS["eye_size_mul"]) ** 2
    l_lm = l_lm + (nose_mul - QUARTER_GRID_TARGETS["nose_depth_mul"]) ** 2
    l_lm = l_lm + (mouth_mul - QUARTER_GRID_TARGETS["mouth_height_mul"]) ** 2

    l_smooth = _mesh_laplacian_penalty(deformed - verts, faces)
    l_reg = (field.remap_incr ** 2).mean() + sum(
        (p ** 2).mean() for p in [field.radial_log, field.s_eye_log,
                                  field.s_nose_xy_log, field.s_nose_z_log,
                                  field.s_mouth_y_log])
    total = l_lm + lam_smooth * l_smooth + lam_reg * l_reg
    return {"total": total, "landmark": l_lm, "smooth": l_smooth, "reg": l_reg}


def fit_chibi_field(verts: torch.Tensor, masks_path: str, *,
                    n_steps: int = 300, lr: float = 0.05,
                    lam_smooth: float = 1.0, lam_reg: float = 0.005,
                    verbose: bool = True) -> ChibiField:
    field = ChibiField(y_crown=float(verts[:, 1].max()),
                       y_chin=float(landmark_positions(verts)[8, 1]),
                       z_center=float(verts[:, 2].mean()))
    rw = region_falloff_weights(verts, masks_path)
    faces = torch.as_tensor(_template_faces(), dtype=torch.long)
    opt = torch.optim.Adam(field.parameters(), lr=lr)
    history = []
    for step in range(n_steps):
        opt.zero_grad()
        loss = chibi_loss(field, verts, rw, faces,
                          lam_smooth=lam_smooth, lam_reg=lam_reg)
        loss["total"].backward()
        opt.step()
        history.append(loss["total"].item())
        if verbose and step % 50 == 0:
            print(f"step {step:4d}  total={loss['total'].item():.5f}  "
                  f"lm={loss['landmark'].item():.5f}")
    field._loss_history = history
    return field


def save_field_params(field: ChibiField, path: str) -> None:
    """Write fitted params + frame buffers to JSON."""
    d = {"remap_incr": field.remap_incr.detach().tolist(),
         "radial_log": field.radial_log.detach().tolist(),
         "s_eye_log": field.s_eye_log.detach().tolist(),
         "s_nose_xy_log": field.s_nose_xy_log.detach().tolist(),
         "s_nose_z_log": field.s_nose_z_log.detach().tolist(),
         "s_mouth_y_log": field.s_mouth_y_log.detach().tolist(),
         "y_crown": float(field.y_crown), "y_chin": float(field.y_chin),
         "z_center": float(field.z_center)}
    Path(path).write_text(json.dumps(d, indent=2))


def load_field_params(path: str) -> ChibiField:
    d = json.loads(Path(path).read_text())
    field = ChibiField(d["y_crown"], d["y_chin"], d["z_center"])
    with torch.no_grad():
        field.remap_incr.copy_(torch.tensor(d["remap_incr"]))
        field.radial_log.copy_(torch.tensor(d["radial_log"]))
        field.s_eye_log.copy_(torch.tensor(d["s_eye_log"]))
        field.s_nose_xy_log.copy_(torch.tensor(d["s_nose_xy_log"]))
        field.s_nose_z_log.copy_(torch.tensor(d["s_nose_z_log"]))
        field.s_mouth_y_log.copy_(torch.tensor(d["s_mouth_y_log"]))
    return field


def _load_obj_verts(path: str) -> torch.Tensor:
    verts = []
    for L in Path(path).read_text().splitlines():
        if L.startswith("v "):
            p = L.split()
            verts.append([float(p[1]), float(p[2]), float(p[3])])
    return torch.tensor(verts, dtype=torch.float32)


if __name__ == "__main__":
    import argparse
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description="Fit a ChibiField to the quarter-grid.")
    ap.add_argument("--template", required=True, help="5023-vert FLAME template .obj")
    ap.add_argument("--masks", required=True, help="FLAME_masks.pkl")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--n_steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    verts = _load_obj_verts(args.template)
    field = fit_chibi_field(verts, args.masks, n_steps=args.n_steps, lr=args.lr)
    save_field_params(field, str(out / "chibi_field_params.json"))

    plt.figure()
    plt.plot(field._loss_history)
    plt.xlabel("step"); plt.ylabel("total loss"); plt.yscale("log")
    plt.title("ChibiField fit")
    plt.savefig(out / "loss_curve.png", dpi=110, bbox_inches="tight")

    rw = region_falloff_weights(verts, args.masks)
    lines = landmark_lines(field(verts, region_weights=rw))
    print("fitted landmark lines:",
          {k: round(float(v), 4) for k, v in lines.items()})
    print(f"wrote {out/'chibi_field_params.json'} and {out/'loss_curve.png'}")
