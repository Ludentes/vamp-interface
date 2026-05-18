import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.primitives import Box, project_superellipsoid
from chibi.chibi_metrics import box_residual, relief_energy, foldover_count


def _grid_mesh(n=7):
    g = torch.linspace(-1, 1, n)
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    verts = torch.cat([xy, torch.zeros(xy.shape[0], 1)], 1).double()
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = a + n, b + n
            faces += [[a, b, c], [b, d, c]]
    return verts, torch.tensor(faces, dtype=torch.int64)


def test_box_residual_zero_on_surface():
    box = Box(center=torch.zeros(3), half=torch.ones(3))
    pts = torch.randn(50, 3).double()
    on = project_superellipsoid(pts, box, exponent=8.0)
    assert box_residual(on, box, exponent=8.0) < 1e-5
    off = on * 1.5
    assert box_residual(off, box, exponent=8.0) > 0.1


def test_relief_energy_drops_when_flat():
    verts, faces = _grid_mesh()
    w = torch.ones(verts.shape[0], dtype=torch.float64)
    flat = relief_energy(verts, faces, w)
    verts[verts.shape[0] // 2, 2] = 1.0
    bumpy = relief_energy(verts, faces, w)
    assert bumpy > flat


def test_foldover_count_zero_for_identity():
    verts, faces = _grid_mesh()
    assert foldover_count(verts, verts, faces) == 0
    flipped = verts.clone()
    flipped[:, 0] = -flipped[:, 0]   # mirror in-plane -> winding reverses
    assert foldover_count(verts, flipped, faces) > 0
