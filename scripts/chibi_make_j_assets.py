"""Build per-vertex Jacobian J (3x3) for anisotropic v3 splat correction.

Given a canonical 20018-vert mesh (LAM bake of the anchor) and a deformed
20018-vert mesh (chibi/fridge/snout/etc. — same topology), compute per-vert
J_i = least-squares fit such that  e_k^1 ≈ J_i · e_k^0  over the one-ring
edges (e_k^0 = canon[nbr_k] - canon[i],  e_k^1 = deformed[nbr_k] - deformed[i]).

This is the linear deformation gradient at vert i. Used by the runtime hook
to push forward each splat's covariance:  Sigma' = J Sigma J^T, then re-SVD
to recover updated (rotation, scaling) for the rasterizer.

Output: J_per_vert.npy of shape (N, 3, 3), float32. ~720 KB for N=20018.

Generic in the deformation: same script works for any mesh-pair with shared
topology. No per-transform knobs.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chibi_make_assets import load_obj, parse_obj_faces  # noqa: E402


def subdivided_edges(tpl_verts: np.ndarray, tpl_faces: np.ndarray) -> np.ndarray:
    """Return undirected edges of the once-subdivided 20018-vert mesh.

    pytorch3d.SubdivideMeshes builds edges from the original mesh once, then
    each face becomes 4 sub-faces. We rebuild the same edge list manually so
    we don't depend on pytorch3d at runtime.
    """
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import SubdivideMeshes

    verts_t = torch.as_tensor(tpl_verts, dtype=torch.float32).unsqueeze(0)
    faces_t = torch.as_tensor(tpl_faces, dtype=torch.int64).unsqueeze(0)
    mesh = Meshes(verts=verts_t, faces=faces_t)
    sub = SubdivideMeshes()(mesh)
    edges = sub.edges_packed().cpu().numpy().astype(np.int64)
    assert edges.max() < 20018, f"unexpected vert count in subdivided edges: {edges.max()+1}"
    return edges


def per_vert_neighbors(edges: np.ndarray, n_verts: int) -> list[list[int]]:
    nbrs: list[list[int]] = [[] for _ in range(n_verts)]
    for a, b in edges:
        nbrs[a].append(b)
        nbrs[b].append(a)
    return nbrs


def per_vert_jacobian(canon: np.ndarray, deformed: np.ndarray,
                      nbrs: list[list[int]]) -> np.ndarray:
    """For each vert i, fit J in R^{3x3} s.t. e1_k = J · e0_k for one-ring edges.
    Closed-form LS:  J = E1 · E0^T · (E0 · E0^T)^{-1}  where columns of E0/E1
    are edge vectors. Falls back to identity when the one-ring is rank-deficient.
    """
    N = canon.shape[0]
    J = np.tile(np.eye(3, dtype=np.float32)[None], (N, 1, 1))
    n_fallback = 0
    for i in range(N):
        idx = nbrs[i]
        if len(idx) < 3:
            n_fallback += 1
            continue
        E0 = (canon[idx] - canon[i]).T.astype(np.float64)       # (3, k)
        E1 = (deformed[idx] - deformed[i]).T.astype(np.float64)  # (3, k)
        # SVD pseudoinverse is well-defined even on rank-deficient one-rings;
        # if the smallest singular value falls below rcond·largest, drop to J=I.
        try:
            E0_pinv = np.linalg.pinv(E0, rcond=1e-8)              # (k, 3)
        except np.linalg.LinAlgError:
            n_fallback += 1
            continue
        sv = np.linalg.svd(E0, compute_uv=False)
        if not np.isfinite(sv).all() or sv[-1] < 1e-8 * sv[0]:
            n_fallback += 1
            continue
        Ji = E1 @ E0_pinv                                         # (3, 3)
        J[i] = Ji.astype(np.float32)
    if n_fallback:
        print(f"  rank-deficient one-ring fallback (J=I) on {n_fallback}/{N} verts")
    return J


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True,
                    help="20018-vert canonical mesh OBJ (LAM bake, pre-deformation)")
    ap.add_argument("--deformed", required=True,
                    help="20018-vert deformed mesh OBJ (chibi/fridge/etc.)")
    ap.add_argument("--template", required=True,
                    help="FLAME 5023 head_template_mesh.obj (for edge topology)")
    ap.add_argument("--out", required=True,
                    help="path to write J_per_vert.npy (N,3,3) float32")
    args = ap.parse_args()

    canon_verts, _, _ = load_obj(Path(args.canonical))
    deformed_verts, _, _ = load_obj(Path(args.deformed))
    tpl_verts, tpl_lines, _ = load_obj(Path(args.template))

    assert canon_verts.shape[0] == 20018, f"canonical has {canon_verts.shape[0]} verts (expected 20018)"
    assert deformed_verts.shape[0] == 20018, f"deformed has {deformed_verts.shape[0]} verts (expected 20018)"
    assert tpl_verts.shape[0] == 5023, f"template has {tpl_verts.shape[0]} verts (expected 5023)"

    canon_xyz = canon_verts[:, :3]
    deformed_xyz = deformed_verts[:, :3]

    tpl_faces = parse_obj_faces(tpl_lines)
    edges = subdivided_edges(tpl_verts[:, :3], tpl_faces)
    print(f"edges in 20018 mesh: {edges.shape[0]}")

    nbrs = per_vert_neighbors(edges, 20018)
    avg_ring = np.mean([len(n) for n in nbrs])
    print(f"avg one-ring size: {avg_ring:.2f}")

    print("computing per-vert J (least squares over one-ring)...")
    J = per_vert_jacobian(canon_xyz, deformed_xyz, nbrs)

    # Stats: det J distribution + isotropy.
    dets = np.linalg.det(J)
    sigmas = np.linalg.svd(J, compute_uv=False)  # (N, 3)
    iso_mean = sigmas.mean(axis=1)               # geometric stretch proxy
    aniso = sigmas[:, 0] / np.maximum(sigmas[:, 2], 1e-8)  # max/min stretch ratio
    print(f"det J: median={np.median(dets):.3f} p5={np.percentile(dets,5):.3f} p95={np.percentile(dets,95):.3f}")
    print(f"iso  : median={np.median(iso_mean):.3f} p5={np.percentile(iso_mean,5):.3f} p95={np.percentile(iso_mean,95):.3f}")
    print(f"aniso ratio (σ_max/σ_min): median={np.median(aniso):.3f} p95={np.percentile(aniso,95):.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, J.astype(np.float32))
    print(f"wrote {out_path}  shape={J.shape}  dtype={J.dtype}  size={J.nbytes/1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
