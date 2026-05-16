"""nvdiffrast UV-space bake: splat appearance -> FLAME UV texture atlas.

Rasterise the mesh in UV space once to get a per-texel canonical 3D position +
normal, run the v2 projection bake on those texel points, dilate UV-gutter
holes. nvdiffrast's raster origin is bottom-left; maps are flipped to top-origin
so the atlas, the PNG, and pytorch3d TexturesUV agree.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch

from chibi.bake import bake_points, _vertex_normals
from chibi.camera_rig import View
from chibi.uv_template import FlameUV


def rasterize_uv_attrs(verts: torch.Tensor, faces: torch.Tensor,
                       flame_uv: FlameUV, tex_size: int
                       ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rasterise the mesh in UV space. Returns top-origin maps:
    pos_map (tex,tex,3), nrm_map (tex,tex,3), mask (tex,tex) bool. tex_size
    must be a multiple of 8 (nvdiffrast requirement)."""
    import nvdiffrast.torch as dr
    assert tex_size % 8 == 0, "tex_size must be a multiple of 8"
    dev = torch.device("cuda")
    uv = flame_uv.uv.to(torch.float32).to(dev)                 # (Nvt,2)
    uv_faces = flame_uv.uv_faces.to(torch.int32).contiguous().to(dev)
    vt2v = flame_uv.vt2v
    vn = _vertex_normals(verts, faces)                         # (V,3)
    attr_pos = verts.to(torch.float32)[vt2v].to(dev)           # (Nvt,3)
    attr_nrm = vn[vt2v].to(dev)                                # (Nvt,3)
    nvt = uv.shape[0]
    clip = torch.cat([uv * 2.0 - 1.0,
                      torch.zeros(nvt, 1, device=dev),
                      torch.ones(nvt, 1, device=dev)], dim=1)
    clip = clip[None].contiguous()                             # (1,Nvt,4)

    glctx = dr.RasterizeCudaContext()
    rast, _ = dr.rasterize(glctx, clip, uv_faces,
                           resolution=[tex_size, tex_size])
    pos_map, _ = dr.interpolate(attr_pos[None].contiguous(), rast, uv_faces)
    nrm_map, _ = dr.interpolate(attr_nrm[None].contiguous(), rast, uv_faces)
    mask = rast[..., 3] > 0                                    # (1,H,W)
    # nvdiffrast raster origin is bottom-left -> flip rows to top-origin.
    pos_map = torch.flip(pos_map[0], dims=[0]).cpu()
    nrm_map = torch.flip(nrm_map[0], dims=[0]).cpu()
    mask = torch.flip(mask[0], dims=[0]).cpu()
    return pos_map, nrm_map, mask


def bake_texture(verts: torch.Tensor, faces: torch.Tensor, flame_uv: FlameUV,
                 images: torch.Tensor, depth: torch.Tensor,
                 views: Sequence[View], tex_size: int
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Bake a (tex,tex,3) float32 texture atlas. Returns (texture, filled),
    where `filled` (tex,tex) bool marks texels a view actually saw — the
    complement is what dilate_texture must fill."""
    pos_map, nrm_map, mask = rasterize_uv_attrs(verts, faces, flame_uv, tex_size)
    flat_pos = pos_map.reshape(-1, 3)
    flat_nrm = nrm_map.reshape(-1, 3)
    idx = mask.reshape(-1).nonzero(as_tuple=True)[0]
    if idx.numel() == 0:
        raise ValueError("no texel covered by the UV layout")
    nrm = flat_nrm[idx]
    nrm = nrm / nrm.norm(dim=1, keepdim=True).clamp_min(1e-9)
    rgb, seen = bake_points(flat_pos[idx], nrm, images, depth, views)

    texture = torch.full((tex_size * tex_size, 3), 0.5, dtype=torch.float32)
    texture[idx] = rgb
    filled = torch.zeros(tex_size * tex_size, dtype=torch.bool)
    filled[idx] = seen
    return (texture.reshape(tex_size, tex_size, 3),
            filled.reshape(tex_size, tex_size))


def dilate_texture(texture: torch.Tensor, mask: torch.Tensor,
                   iters: int = 16) -> torch.Tensor:
    """Fill `~mask` texels from filled 4-neighbours, iteratively. Any texel
    still unfilled after `iters` passes takes the mean of all filled texels."""
    tex = texture.clone()
    m = mask.clone()
    for _ in range(iters):
        if bool(m.all()):
            break
        acc = torch.zeros_like(tex)
        cnt = torch.zeros(m.shape, dtype=torch.float32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            sh_t = torch.roll(tex, shifts=(dy, dx), dims=(0, 1))
            sh_m = torch.roll(m, shifts=(dy, dx), dims=(0, 1))
            acc += sh_t * sh_m.unsqueeze(-1).float()
            cnt += sh_m.float()
        newly = (~m) & (cnt > 0)
        tex[newly] = acc[newly] / cnt[newly].unsqueeze(-1)
        m = m | newly
    if not bool(m.all()) and bool(m.any()):
        tex[~m] = tex[m].mean(0)
    return tex
