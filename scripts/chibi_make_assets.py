#!/usr/bin/env python3
"""Generate chibi-deformed assets for a LAM head: deformed 5023-vert FLAME
template + 5023-vert rescaled ARKit blendshape basis + deformed 20018-vert
baked textured_mesh.obj.

Chibi deformation is a y-stratified affine field. For each vertex we compute
a normalized head fraction
    t = clamp((y - y_anchor) / (y_top - y_anchor), 0, 1)
and apply piecewise scales:
    s_y(t)  vertical stretch rate at that height
    s_r(t)  radial (xz) scale at that height
The new y is the cumulative integral of s_y; new xz is the original xz times
s_r centered on the local z-axis of the head. Below y_anchor (neck/shoulders)
nothing changes — keeps the rig anchored.

For blendshape deltas we apply the *local* scale matrix
diag(s_r(t_i), s_y(t_i), s_r(t_i)) per source vertex. This is the
linearization of the deformation field at v_i: small motions of the vert
under blink/jawOpen/etc. land on the deformed mesh proportionally.

The same deformation field is applied to both the 5023-vert template (which
LAM upsamples + uses to build its shapedirs) and the 20018-vert baked
textured_mesh.obj (which we inject via LAM_EDIT_XYZ_OBJ at runtime). They
share y-fraction semantics so the field is identical.

Usage:
    uv run scripts/chibi_make_assets.py \
        --template /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/head_template_mesh.obj \
        --arkit_bs /home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy \
        --baked exp_output/lam_blender_handoff/splats/asian_m_textured_mesh.obj \
        --outdir exp_output/lam_chibi/asian_m \
        --y_anchor_frac 0.20 \
        --chibi_strength 1.0
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


# Chibi knots — see module docstring for what these mean.
# t=0 is the y_anchor (neck base, no change); t=1 is y_top (skull top).
T_KNOTS  = np.array([0.00, 0.30, 0.50, 0.75, 1.00], dtype=np.float64)
SY_KNOTS = np.array([1.00, 0.55, 1.30, 1.30, 1.20], dtype=np.float64)  # local y-stretch
SR_KNOTS = np.array([1.00, 0.75, 1.15, 1.30, 1.30], dtype=np.float64)  # radial xz-scale


def blend(strength: float) -> tuple[np.ndarray, np.ndarray]:
    """Lerp knots from identity (strength=0) toward the chibi profile (strength=1)."""
    sy = 1.0 + strength * (SY_KNOTS - 1.0)
    sr = 1.0 + strength * (SR_KNOTS - 1.0)
    return sy, sr


def build_y_remap(sy_knots: np.ndarray, n_samples: int = 1001) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_grid, y_norm_grid) so that y_norm_grid[i] is the new
    normalized y when the original normalized y was t_grid[i].

    Computed as the cumulative integral of s_y(t) over t∈[0,1], renormalized
    so that t=0 → y_norm=0 and t=1 → y_norm=1. This means the *overall* head
    height is preserved; only the *distribution* of vertical density changes
    (chibi packs more verts up top, fewer down low). The chibi_strength
    parameter alone controls how cartoony; height scale is separate.
    """
    t = np.linspace(0.0, 1.0, n_samples)
    sy = np.interp(t, T_KNOTS, sy_knots)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (sy[:-1] + sy[1:])) / (n_samples - 1)])
    cum /= cum[-1]
    return t, cum


def load_obj(path: Path) -> tuple[np.ndarray, list[str], list[int]]:
    """Return (verts[N,3 or 6], all_lines, vert_indices_in_lines).

    If the OBJ stores per-vert colours (7-col v rows), include them in verts
    as columns 3:6 (raw, untouched by deform). Preserves all non-v lines
    verbatim for clean round-trip.
    """
    lines = path.read_text().splitlines()
    verts, vidx = [], []
    width = None
    for i, L in enumerate(lines):
        if not L.startswith("v "):
            continue
        parts = L.split()
        if len(parts) == 4:
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if width is None: width = 3
        elif len(parts) == 7:
            verts.append([float(x) for x in parts[1:]])
            if width is None: width = 6
        else:
            continue
        vidx.append(i)
    if width is None:
        raise ValueError(f"no parseable v rows in {path}")
    return np.asarray(verts, dtype=np.float64), lines, vidx


def parse_obj_faces(lines: list[str]) -> np.ndarray:
    """Extract triangle faces (0-indexed) from OBJ `f` lines. Quads → 2 tris."""
    faces = []
    for L in lines:
        if not L.startswith("f "):
            continue
        parts = L.split()[1:]
        idx = [int(p.split("/")[0]) - 1 for p in parts]
        if len(idx) == 3:
            faces.append(idx)
        elif len(idx) == 4:
            faces.append([idx[0], idx[1], idx[2]])
            faces.append([idx[0], idx[2], idx[3]])
    return np.asarray(faces, dtype=np.int64)


def per_vert_edge_mean(verts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Mean 1-ring edge length per vertex.

    Used to derive a per-vertex Gaussian-splat scale ratio from the chibi
    stretch: ratio_i = mean_edge_chibi_i / mean_edge_original_i. When applied
    multiplicatively to GSLayer-predicted splat scales, this keeps every
    stretched region densely tiled (closes blink, fills enlarged eye socket).

    Args:
        verts: (N, 3) float
        edges: (E, 2) int64 — undirected; we count each end of each edge.

    Returns:
        (N,) float64 — orphans return 0.0; consumer should guard.
    """
    e_len = np.linalg.norm(verts[edges[:, 0]] - verts[edges[:, 1]], axis=1)
    sums = np.zeros(verts.shape[0], dtype=np.float64)
    counts = np.zeros(verts.shape[0], dtype=np.float64)
    np.add.at(sums, edges[:, 0], e_len)
    np.add.at(counts, edges[:, 0], 1)
    np.add.at(sums, edges[:, 1], e_len)
    np.add.at(counts, edges[:, 1], 1)
    return sums / np.maximum(counts, 1.0)


def subdivided_edges(tpl_verts: np.ndarray, tpl_faces: np.ndarray) -> np.ndarray:
    """Return the (E, 2) edge list of the SubdivideMeshes(tpl) output.

    Same op LAM runs in `flame_arkit.upsample_mesh_cpu` — gives us the
    20018-vert mesh's edge connectivity from the 5023-vert template+faces.
    """
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import SubdivideMeshes
    verts_t = torch.as_tensor(tpl_verts, dtype=torch.float32).unsqueeze(0)
    faces_t = torch.as_tensor(tpl_faces, dtype=torch.int64).unsqueeze(0)
    base = Meshes(verts=verts_t, faces=faces_t)
    sub = SubdivideMeshes()(base)
    return sub.edges_packed().cpu().numpy().astype(np.int64)


def lift_mask_to_subdivided(tpl_verts: np.ndarray, tpl_faces: np.ndarray,
                            mask_5023: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lift a 5023-space index set to the 20018-space produced by one round
    of pytorch3d.SubdivideMeshes (the same op LAM uses in
    `flame_arkit.upsample_mesh_cpu`).

    A midpoint vert at index (5023 + k) corresponds to edges_packed()[k]
    = (v_a, v_b). The midpoint is "in mask" iff EITHER endpoint is in
    `mask_5023` (loose / boundary-inclusive — needed so that midpoints
    along the lid↔cheek boundary, which carry interpolated eyelash and
    iris SH colour, also get pinned).

    Returns:
        lifted_idx : (M,) int64 — combined 0..5022 originals + 5023.. midpoints
        edges      : (E, 2) int64 — for sanity printing
    """
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import SubdivideMeshes  # noqa: F401  (import asserts availability)

    verts_t = torch.as_tensor(tpl_verts, dtype=torch.float32).unsqueeze(0)
    faces_t = torch.as_tensor(tpl_faces, dtype=torch.int64).unsqueeze(0)
    mesh = Meshes(verts=verts_t, faces=faces_t)
    edges = mesh.edges_packed().cpu().numpy().astype(np.int64)   # (E, 2)
    n_orig = tpl_verts.shape[0]
    assert n_orig + edges.shape[0] == 20018, (
        f"unexpected subdivided size: {n_orig} + {edges.shape[0]} = "
        f"{n_orig + edges.shape[0]} (expected 20018)"
    )
    in_mask = np.zeros(n_orig, dtype=bool)
    in_mask[mask_5023] = True
    mid_in = in_mask[edges[:, 0]] | in_mask[edges[:, 1]]
    lifted = np.concatenate([
        np.flatnonzero(in_mask),
        n_orig + np.flatnonzero(mid_in),
    ]).astype(np.int64)
    return lifted, edges


def save_obj(path: Path, verts: np.ndarray, lines: list[str], vidx: list[int]) -> None:
    out = list(lines)
    has_rgb = verts.shape[1] == 6
    for k, vi in enumerate(vidx):
        if has_rgb:
            x, y, z, r, g, b = verts[k]
            out[vi] = f"v {x:.8f} {y:.8f} {z:.8f} {r:.6f} {g:.6f} {b:.6f}"
        else:
            x, y, z = verts[k]
            out[vi] = f"v {x:.8f} {y:.8f} {z:.8f}"
    path.write_text("\n".join(out) + "\n")


def deform_verts(verts: np.ndarray, y_anchor: float, y_top: float,
                 sy_knots: np.ndarray, sr_knots: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply chibi field to vertex positions. Returns (new_verts, t_per_vert).

    t_per_vert is later used to compute per-vert scale matrices for blendshape
    deltas. Verts below y_anchor get t=0 (no deformation).
    """
    t_grid, y_norm_grid = build_y_remap(sy_knots)
    out = verts.copy().astype(np.float64)
    head_h = y_top - y_anchor
    if head_h <= 0:
        raise ValueError(f"y_top ({y_top}) must be > y_anchor ({y_anchor})")
    t_orig = np.clip((verts[:, 1] - y_anchor) / head_h, 0.0, 1.0)
    new_t = np.interp(t_orig, t_grid, y_norm_grid)
    out[:, 1] = y_anchor + new_t * head_h
    sr = np.interp(t_orig, T_KNOTS, sr_knots)
    # z is centered around the local z-axis of the head; FLAME's head leans
    # forward so we recenter relative to the per-mesh z mean above y_anchor.
    head_mask = verts[:, 1] >= y_anchor
    z_center = float(verts[head_mask, 2].mean()) if head_mask.any() else 0.0
    out[:, 0] = verts[:, 0] * sr   # x around 0 (symmetric)
    out[:, 2] = z_center + (verts[:, 2] - z_center) * sr
    # Verts below the anchor: no change at all.
    out[~head_mask] = verts[~head_mask]
    return out, t_orig


def rescale_arkit_bs(arkit_bs: np.ndarray, t_per_vert: np.ndarray,
                     sy_knots: np.ndarray, sr_knots: np.ndarray,
                     z_identity_rows: list[int] | None = None) -> np.ndarray:
    """For each vert i with normalized head fraction t_i, scale its blendshape
    delta by diag(s_r(t_i), s_y(t_i), s_r(t_i)).

    arkit_bs shape: (52, 5023, 3). We multiply column-wise (broadcast over
    the 52 blendshapes). Verts below y_anchor (t=0) get identity scale.

    If z_identity_rows is given, those blendshape indices keep z scale = 1.0
    (used to test the hypothesis that s_r amplification of the small forward-
    curl z-component of eyeBlinkL/R is what makes the lid sail past the iris).
    """
    sy = np.interp(t_per_vert, T_KNOTS, sy_knots).astype(np.float64)
    sr = np.interp(t_per_vert, T_KNOTS, sr_knots).astype(np.float64)
    out = arkit_bs.astype(np.float64).copy()
    out[..., 0] *= sr[None, :]
    out[..., 1] *= sy[None, :]
    out[..., 2] *= sr[None, :]
    if z_identity_rows:
        # Recompute z for these rows from the ORIGINAL basis (un-rescaled in z).
        orig = arkit_bs.astype(np.float64)
        rows = np.asarray(z_identity_rows, dtype=np.int64)
        out[rows, :, 2] = orig[rows, :, 2]
    return out


def auto_anchor_top(verts: np.ndarray, y_anchor_frac: float) -> tuple[float, float]:
    """Pick y_anchor and y_top from the mesh's y-extent.

    y_anchor = y_min + y_anchor_frac * (y_max - y_min)   default 0.20 leaves
                                                          shoulders/neck untouched
    y_top    = y_max                                      skull crown
    """
    y_min = float(verts[:, 1].min())
    y_max = float(verts[:, 1].max())
    y_anchor = y_min + y_anchor_frac * (y_max - y_min)
    return y_anchor, y_max


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True, help="FLAME 5023-vert head_template_mesh.obj")
    ap.add_argument("--arkit_bs", required=True, help="flame_arkit_bs.npy (52, 5023, 3)")
    ap.add_argument("--baked",    required=True, help="LAM-baked 20018-vert textured_mesh.obj (display-RGB or raw)")
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--y_anchor_frac_template", type=float, default=0.20,
                    help="below this fraction of template y-extent, no deformation")
    ap.add_argument("--y_anchor_frac_baked", type=float, default=0.50,
                    help="baked mesh includes shoulders; default 0.50 keeps neck+shoulders fixed")
    ap.add_argument("--chibi_strength", type=float, default=1.0,
                    help="0 = identity, 1 = full chibi knots, can overshoot to 1.5 for dramatic")
    ap.add_argument("--blink_boost", type=float, default=1.0,
                    help="Manual multiplier on eyeBlinkL/R (ch8/9) and eyeSquintL/R (ch18/19, "
                         "half-strength) basis rows. Default 1.0 = no boost. "
                         "WARNING: empirically harmful on chibi geometry — the iris-through-lid "
                         "artifact is z-axis (eyeball protrudes anterior to lid surface in chibi "
                         "proportions). Boosting blink only pulls the lid further past the eyeball "
                         "and exposes MORE eye. Use --preserve_eyeballs instead. Kept as a knob in "
                         "case future deformation profiles want it.")
    ap.add_argument("--preserve_eyeballs", type=int, default=0,
                    help="0 (default) = let the chibi field deform the eyeballs along with the "
                         "rest of the head, so they stay where the chibi face geometry expects "
                         "them. 1 = pin FLAME left/right_eyeball mask verts to their original "
                         "(pre-chibi) positions. EMPIRICALLY WRONG on this stack — produces a "
                         "second pair of eyes stuck on the un-chibi face plane while the chibi "
                         "head deforms around them. Kept only as an experimentally accessible "
                         "knob; do not turn on without re-validating against the three-way "
                         "compare under renders/frames_me/eye_diag/eye_three_way_pinning.png.")
    ap.add_argument("--preserve_eye_region", type=int, default=0,
                    help="0 (default). 1 = also pin the lid-skin verts (FLAME 'eye_region' "
                         "mask, 751 verts) to identity. Same wrongness as --preserve_eyeballs.")
    ap.add_argument("--blink_rows_identity", type=int, default=0,
                    help="1 = leave the ENTIRE eyeBlinkL/R (8,9) and eyeSquintL/R (18,19) "
                         "basis rows at their original (un-rescaled) values. Diagnostic "
                         "control: does the chibi rescale itself break closure, or only "
                         "one axis of it? Takes precedence over --blink_z_identity.")
    ap.add_argument("--blink_z_identity", type=int, default=0,
                    help="1 = leave the z-component of eyeBlinkL/R (rows 8,9) and "
                         "eyeSquintL/R (rows 18,19) blink deltas at their original "
                         "(un-radial-scaled) magnitude. Diagnostic for the hypothesis "
                         "that s_r amplification of the lid's forward-curl is what "
                         "makes the lid sail past the chibi-pushed-forward iris.")
    ap.add_argument("--flame_masks", type=str,
                    default="/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl",
                    help="Path to FLAME_masks.pkl. Required when preserve_eyeballs=1.")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sy_knots, sr_knots = blend(args.chibi_strength)
    blink_boost = args.blink_boost
    print(f"chibi strength={args.chibi_strength}: sy={sy_knots.round(3)}  sr={sr_knots.round(3)}")

    # Load FLAME mask indices for any preserved regions.
    eyeball_idx = None
    if args.preserve_eyeballs or args.preserve_eye_region:
        import pickle
        masks = pickle.load(open(args.flame_masks, "rb"), encoding="latin1")
        parts = []
        if args.preserve_eyeballs:
            parts.extend([masks["left_eyeball"], masks["right_eyeball"]])
        if args.preserve_eye_region:
            parts.append(masks["eye_region"])
        eyeball_idx = np.unique(np.concatenate(parts).astype(np.int64))
        print(f"  preserve eye regions: pinning {eyeball_idx.size} template verts to identity "
              f"(eyeballs={int(bool(args.preserve_eyeballs))}, eye_region={int(bool(args.preserve_eye_region))})")

    # --- 1) FLAME 5023 template + blendshape basis ---
    tpl_verts, tpl_lines, tpl_vidx = load_obj(Path(args.template))
    assert tpl_verts.shape[0] == 5023, f"expected 5023 verts in template, got {tpl_verts.shape[0]}"
    y_a, y_t = auto_anchor_top(tpl_verts[:, :3], args.y_anchor_frac_template)
    print(f"template: y in [{tpl_verts[:,1].min():.4f},{tpl_verts[:,1].max():.4f}]  anchor={y_a:.4f}  top={y_t:.4f}")
    tpl_new, t_pv = deform_verts(tpl_verts[:, :3], y_a, y_t, sy_knots, sr_knots)
    # Pin eyeball verts to their original positions BEFORE writing. This stops
    # the chibi radial expansion from projecting the eyeball sphere forward in z.
    if eyeball_idx is not None:
        tpl_new[eyeball_idx] = tpl_verts[eyeball_idx, :3]
    save_obj(outdir / "chibi_template.obj", tpl_new, tpl_lines, tpl_vidx)
    print(f"  wrote {outdir / 'chibi_template.obj'}")

    arkit = np.load(args.arkit_bs)
    assert arkit.shape == (52, 5023, 3), f"expected (52,5023,3), got {arkit.shape}"
    z_id_rows = [8, 9, 18, 19] if args.blink_z_identity else None
    arkit_new = rescale_arkit_bs(arkit, t_pv, sy_knots, sr_knots,
                                 z_identity_rows=z_id_rows)
    if z_id_rows:
        print(f"  blink_z_identity: kept z component of rows {z_id_rows} at original magnitude")
    if args.blink_rows_identity:
        # ARKit-52 alphabetical, all rows whose geometry touches eye_region:
        # 0-4 brows (browDownL/R, browInnerUp, browOuterUpL/R)
        # 8-9 eyeBlink L/R
        # 10-17 eyeLook (Down/In/Out/Up) L/R
        # 18-19 eyeSquint L/R
        # 20-21 eyeWide L/R
        eye_rows = np.arange(0, 22, dtype=np.int64)
        arkit_new[eye_rows] = arkit.astype(np.float64)[eye_rows]
        print(f"  blink_rows_identity: restored full identity on rows {eye_rows.tolist()} "
              f"(brows + eye_look + eye_blink + eye_squint + eye_wide)")
    # Restore unscaled basis for eyeball verts — eye-look channels translate
    # the sphere rigidly, and if we rescale those deltas the sphere ends up at
    # wrong gaze positions on chibi proportions.
    if eyeball_idx is not None:
        arkit_new[:, eyeball_idx, :] = arkit.astype(np.float64)[:, eyeball_idx, :]
        print(f"  preserve_eyeballs: restored unscaled ARKit basis for {eyeball_idx.size} verts")
    # --- blink boost ---------------------------------------------------------
    # ARKit-52 ordering is alphabetical: eyeBlinkLeft=8, eyeBlinkRight=9,
    # eyeSquintLeft=18, eyeSquintRight=19. After chibi rescale the lids still
    # don't reach the (now larger) eyeball at partial blink — iris pokes through.
    # Multiplying the basis row scales the *amplitude* of the per-frame delta,
    # leaving the spatial pattern intact (verts the channel touches are unchanged).
    if blink_boost != 1.0:
        sq_boost = 1.0 + 0.5 * (blink_boost - 1.0)   # half-strength on squint
        arkit_new[8]  *= blink_boost
        arkit_new[9]  *= blink_boost
        arkit_new[18] *= sq_boost
        arkit_new[19] *= sq_boost
        print(f"  blink_boost={blink_boost:.3f}  squint_boost={sq_boost:.3f}")
    arkit_new = arkit_new.astype(arkit.dtype)
    np.save(outdir / "chibi_arkit_bs.npy", arkit_new)
    print(f"  wrote {outdir / 'chibi_arkit_bs.npy'} (max |delta| went "
          f"{float(np.abs(arkit).max()):.4f} → {float(np.abs(arkit_new).max()):.4f})")

    # --- 2) Baked 20018 textured_mesh ---
    baked_verts, baked_lines, baked_vidx = load_obj(Path(args.baked))
    print(f"baked: shape={baked_verts.shape}  y in [{baked_verts[:,1].min():.4f},{baked_verts[:,1].max():.4f}]")
    y_a_b, y_t_b = auto_anchor_top(baked_verts[:, :3], args.y_anchor_frac_baked)
    print(f"  baked anchor={y_a_b:.4f}  top={y_t_b:.4f}")
    baked_xyz_new, _ = deform_verts(baked_verts[:, :3], y_a_b, y_t_b, sy_knots, sr_knots)
    # Pin baked-mesh eye verts. The baked mesh is 20018 = 5023 original FLAME
    # verts + ~15000 edge-midpoint verts appended by pytorch3d.SubdivideMeshes
    # (same op LAM runs in upsample_mesh_cpu). The FLAME eye masks live in
    # 5023-space; if we only pin those, the midpoint verts inside the eye
    # region (indices ≥ 5023) still drift under the chibi affine, dragging
    # their photo-baked SH colour (iris/sclera/lash) up the forehead. Result:
    # ghost second pair of eyes. Lift the mask through one round of
    # SubdivideMeshes so we also pin every midpoint whose parent edge touches
    # an eye vert.
    if eyeball_idx is not None and baked_verts.shape[0] == 20018:
        tpl_faces = parse_obj_faces(tpl_lines)
        assert tpl_faces.size > 0, "no faces parsed from FLAME template OBJ"
        eyeball_idx_20018, edges = lift_mask_to_subdivided(
            tpl_verts[:, :3], tpl_faces, eyeball_idx,
        )
        n_orig = (eyeball_idx_20018 < 5023).sum()
        n_mid = (eyeball_idx_20018 >= 5023).sum()
        baked_xyz_new[eyeball_idx_20018] = baked_verts[eyeball_idx_20018, :3]
        print(f"  preserve_eyeballs: pinned {eyeball_idx_20018.size} baked verts "
              f"({n_orig} originals + {n_mid} midpoints, from {edges.shape[0]} total edges)")
    elif eyeball_idx is not None and baked_verts.shape[0] >= 5023:
        # Fallback: non-20018 baked mesh (e.g. raw template). Pin originals only.
        baked_xyz_new[eyeball_idx] = baked_verts[eyeball_idx, :3]
        print(f"  preserve_eyeballs: pinned {eyeball_idx.size} baked verts "
              f"(non-20018 mesh: skipping midpoint lift)")
    if baked_verts.shape[1] == 6:
        baked_out = np.concatenate([baked_xyz_new, baked_verts[:, 3:]], axis=1)
    else:
        baked_out = baked_xyz_new
    save_obj(outdir / "chibi_textured_mesh.obj", baked_out, baked_lines, baked_vidx)
    print(f"  wrote {outdir / 'chibi_textured_mesh.obj'}")

    # --- 3) Per-vertex splat-scale ratio (chibi / original) -----------------
    # GSLayer in LAM predicts per-splat Gaussian sigmas at original FLAME
    # vertex density. After we override `_gm.xyz` with the chibi-stretched
    # mesh, those sigmas no longer tile the surface — visible as iris leaking
    # through closed lid and small-looking irises in oversized sockets.
    # Emit a (20018,) float32 ratio = mean_edge_chibi / mean_edge_original
    # so the runtime hook (`LAM_CHIBI_SCALE_RATIO`) can multiply the splat
    # scales back into proportion.
    if baked_verts.shape[0] == 20018:
        tpl_faces_for_edges = parse_obj_faces(tpl_lines)
        assert tpl_faces_for_edges.size > 0, "no faces parsed from FLAME template OBJ"
        edges20018 = subdivided_edges(tpl_verts[:, :3], tpl_faces_for_edges)
        assert edges20018.max() < 20018, (
            f"edges reference vert {edges20018.max()} >= 20018 — "
            "subdivision produced unexpected indexing"
        )
        el_orig = per_vert_edge_mean(baked_verts[:, :3], edges20018)
        el_chibi = per_vert_edge_mean(baked_xyz_new, edges20018)
        scale_ratio = (el_chibi / np.maximum(el_orig, 1e-8)).astype(np.float32)
        # Clamp pathological corners (orphan verts, degenerate edges) to 1.0
        # so the multiplicative hook is a no-op there rather than collapsing
        # or exploding splats.
        bad = ~np.isfinite(scale_ratio) | (el_orig <= 1e-6)
        scale_ratio[bad] = 1.0
        np.save(outdir / "chibi_scale_ratio.npy", scale_ratio)
        print(
            f"  wrote {outdir / 'chibi_scale_ratio.npy'} "
            f"(min={scale_ratio.min():.3f} median={np.median(scale_ratio):.3f} "
            f"max={scale_ratio.max():.3f} bad={int(bad.sum())})"
        )
    else:
        print(f"  baked mesh has {baked_verts.shape[0]} verts (not 20018) — "
              "skipping chibi_scale_ratio.npy emit")

    print(f"\nDone. Inject at runtime via:")
    print(f"  LAM_CHIBI_ARKIT_BS={outdir / 'chibi_arkit_bs.npy'}")
    print(f"  LAM_EDIT_XYZ_OBJ={outdir / 'chibi_textured_mesh.obj'}")
    if baked_verts.shape[0] == 20018:
        print(f"  LAM_CHIBI_SCALE_RATIO={outdir / 'chibi_scale_ratio.npy'}")


if __name__ == "__main__":
    main()
