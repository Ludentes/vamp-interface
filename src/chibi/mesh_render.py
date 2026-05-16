"""Render ChibiMesh frames with pytorch3d.

`render` handles both cases the pivot needs: a mesh *sequence* from one camera
angle (a driven animation — one ChibiMesh per frame), and a single mesh from
*many* angles (a turntable). Whichever list has length 1 is broadcast.

Unlit: AmbientLights with full ambient and no diffuse/specular, so the output
is the per-vertex colour interpolated across triangles — the flat chibi look,
and it isolates geometry from shading for the quality verdict.

Recentre + unit-scale are computed once from the first mesh and reused for the
whole sequence, so a driven animation does not jitter or breathe.
"""
from __future__ import annotations
from collections.abc import Sequence
import torch

from chibi.mesh import ChibiMesh


def render(meshes: Sequence[ChibiMesh], azims: Sequence[float], *,
           image_size: int = 256, dist: float = 2.7, elev: float = 0.0,
           device: str = "cuda") -> torch.Tensor:
    """Render `T = max(len(meshes), len(azims))` frames. The length-1 list is
    broadcast to T. Returns (T, image_size, image_size, 3) uint8 RGB on CPU."""
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        TexturesVertex, FoVPerspectiveCameras, RasterizationSettings,
        MeshRenderer, MeshRasterizer, SoftPhongShader, AmbientLights,
        look_at_view_transform,
    )
    meshes = list(meshes)
    azims = list(azims)
    n = max(len(meshes), len(azims))
    if len(meshes) == 1:
        meshes = meshes * n
    if len(azims) == 1:
        azims = azims * n
    assert len(meshes) == len(azims) == n, \
        "meshes and azims must have equal length or one of them length 1"

    dev = torch.device(device)
    # fixed recentre/scale from frame 0 — keeps a driven animation steady.
    ref = meshes[0].verts.to(torch.float32)
    centre = ref.mean(0, keepdim=True).to(dev)
    scale = (ref.to(dev) - centre).abs().max().clamp_min(1e-6)

    raster = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                   faces_per_pixel=1)
    lights = AmbientLights(device=dev)            # unlit: flat vertex colour

    frames = []
    for mesh, azim in zip(meshes, azims):
        verts = (mesh.verts.to(torch.float32).to(dev) - centre) / scale
        faces = mesh.faces.to(torch.int64).to(dev)
        rgb = mesh.rgb.to(torch.float32).to(dev)
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim,
                                      device=dev)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=dev)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster),
            shader=SoftPhongShader(device=dev, cameras=cameras, lights=lights),
        )
        p3d = Meshes(verts=[verts], faces=[faces],
                     textures=TexturesVertex(verts_features=[rgb]))
        img = renderer(p3d)[0, ..., :3]
        frames.append((img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu())
    return torch.stack(frames)
