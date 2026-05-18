import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch
from chibi.primitives import (Box, fit_box, project_superellipsoid,
                              laplacian_smoothed)


def test_fit_box_recovers_known_extent():
    # points filling a [-2,2]x[-1,1]x[-3,3] box
    g = torch.linspace(-1, 1, 8)
    pts = torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    pts = pts * torch.tensor([2.0, 1.0, 3.0])
    box = fit_box(pts)
    assert torch.allclose(box.center, torch.zeros(3, dtype=torch.float64),
                          atol=1e-4)
    assert torch.allclose(box.half, torch.tensor([2.0, 1.0, 3.0],
                          dtype=torch.float64), atol=1e-4)


def test_project_superellipsoid_high_exponent_is_boxy():
    box = Box(center=torch.zeros(3), half=torch.ones(3))
    # a point on the +x face diagonal; n large -> projects near the box face
    p = torch.tensor([[0.5, 0.5, 0.0]])
    out = project_superellipsoid(p, box, exponent=16.0)
    # boxy: the dominant axis lands on the face (coord ~= +-1)
    assert out.abs().max() > 0.95


def test_project_superellipsoid_n2_is_sphere():
    box = Box(center=torch.zeros(3), half=torch.ones(3))
    p = torch.tensor([[3.0, 4.0, 0.0]])          # radius 5
    out = project_superellipsoid(p, box, exponent=2.0)
    assert torch.allclose(out.norm(), torch.tensor(1.0, dtype=torch.float64),
                          atol=1e-4)


def test_laplacian_smoothed_reduces_a_bump():
    # flat grid with one raised vertex -> smoothing lowers the bump
    g = torch.linspace(-1, 1, 7)
    xy = torch.stack(torch.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    verts = torch.cat([xy, torch.zeros(xy.shape[0], 1)], 1).double()
    bump = verts.shape[0] // 2
    verts[bump, 2] = 1.0
    # build faces from the grid
    faces = []
    n = 7
    for i in range(n - 1):
        for j in range(n - 1):
            a, b = i * n + j, i * n + j + 1
            c, d = a + n, b + n
            faces += [[a, b, c], [b, d, c]]
    faces = torch.tensor(faces, dtype=torch.int64)
    sm = laplacian_smoothed(verts, faces, iters=10, lam=0.5)
    assert sm[bump, 2] < 0.5            # bump pulled down
    assert verts[bump, 2] == 1.0        # input not mutated
