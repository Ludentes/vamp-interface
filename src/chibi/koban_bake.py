"""Single-view projective bake: a warped chibi portrait -> canonical UV atlas.

rasterize_uv_attrs gives per-texel 3D position + normal; bake_points projects
those through the frontal View and samples the (already-registered) portrait.
mirror_fill + dilate_texture + skin_fallback complete the non-frontal texels.

NOTE ON vt2v: rasterize_uv_attrs interpolates per-uv-vertex attributes
attr_pos = verts[vt2v] / attr_nrm = vn[vt2v], so `vt2v` is load-bearing for
the position/normal rasterisation. KobanMesh does not store vt2v directly, but
the correspondence is recoverable: `faces` and `uv_faces` are paired row-for-row
(same F, same corner order), so faces[f][k] is the mesh vertex of uv-vertex
uv_faces[f][k]. _koban_vt2v rebuilds the map by scattering those pairs. Koban's
mapping is non-injective for body loops (many verts share one body texel), which
is exactly why load_flame_uv's uniqueness assertion can't be reused; for the
face island the mapping is consistent, and last-write-wins is harmless for the
collapsed body texel. Passing torch.arange would be a garbage bake.
"""
from __future__ import annotations
import torch
from chibi.bake import bake_points
from chibi.camera_rig import View
from chibi.texture_bake import rasterize_uv_attrs, dilate_texture
from chibi.uv_template import FlameUV
from chibi.koban_asset import KobanMesh


def _koban_vt2v(faces: torch.Tensor, uv_faces: torch.Tensor,
                n_uv: int) -> torch.Tensor:
    """Build the uv-vertex -> mesh-vertex map from the row-paired face arrays.

    faces[f][k] and uv_faces[f][k] are the mesh-vertex and uv-vertex indices of
    the same face corner. Scatter every corner; last-write-wins. Koban's body
    loops collapse to a single texel (non-injective), so a vt may legitimately
    be written by several different verts — we do not assert uniqueness."""
    vt2v = torch.zeros(n_uv, dtype=torch.int64)
    fv = faces.reshape(-1).to(torch.int64)
    fvt = uv_faces.reshape(-1).to(torch.int64)
    vt2v[fvt] = fv
    return vt2v


def mirror_fill(texture: torch.Tensor, seen: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
    """Fill unseen texels from their left-right mirror partner. The canonical
    UV is laid out left-right-symmetric across the W axis, so a texel (h,w)
    mirrors to (h, W-1-w). Returns (texture, seen) with mirror-filled texels
    now marked seen."""
    out = texture.clone()
    out_seen = seen.clone()
    mir = torch.flip(texture, dims=[1])
    mir_seen = torch.flip(seen, dims=[1])
    fillable = (~seen) & mir_seen
    out[fillable] = mir[fillable]
    out_seen[fillable] = True
    return out, out_seen


def skin_fallback(texture: torch.Tensor, seen: torch.Tensor) -> torch.Tensor:
    """Fill any still-unseen texel with the median colour of seen texels
    (a flat skin tone). Back-of-head texels are covered by hair geometry, so a
    constant is acceptable."""
    out = texture.clone()
    if seen.any():
        med = texture[seen].median(dim=0).values
    else:
        med = torch.tensor([0.5, 0.5, 0.5])
    out[~seen] = med
    return out


def bake_portrait_to_uv(koban: KobanMesh, portrait: torch.Tensor,
                        view: View, tex_size: int = 1024,
                        depth: torch.Tensor | None = None) -> torch.Tensor:
    """portrait: (H,W,3) float[0,1], already TPS-registered to `view`.
    Returns the (tex_size,tex_size,3) float[0,1] UV atlas, top-origin."""
    vt2v = _koban_vt2v(koban.faces, koban.uv_faces, koban.uv.shape[0])
    fuv = FlameUV(uv=koban.uv, uv_faces=koban.uv_faces, vt2v=vt2v)
    pos_map, nrm_map, mask = rasterize_uv_attrs(
        koban.verts, koban.faces, fuv, tex_size)
    pts = pos_map[mask]                                  # (N,3)
    nrm = nrm_map[mask]
    img_u8 = (portrait.clamp(0, 1) * 255).to(torch.uint8)[None]   # (1,H,W,3)
    if depth is None:                                    # no occlusion test
        H, W, _ = portrait.shape
        depth = torch.full((1, H, W), 1e9)
    rgb, seen = bake_points(pts, nrm, img_u8, depth, [view],
                            mode="best", min_facing=0.1)
    tex = torch.full((tex_size, tex_size, 3), 0.5)
    seen_map = torch.zeros(tex_size, tex_size, dtype=torch.bool)
    tex[mask] = rgb
    seen_map[mask] = seen
    tex, seen_map = mirror_fill(tex, seen_map)
    tex = dilate_texture(tex, seen_map)
    tex = skin_fallback(tex, seen_map)
    return tex
