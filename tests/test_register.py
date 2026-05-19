import torch
from chibi.register import fit_tps, warp_image

def test_tps_recovers_affine():
    # 12 source points; target = known affine of source.
    torch.manual_seed(0)
    src = torch.rand(12, 2) * 200 + 28
    A = torch.tensor([[1.1, 0.05], [-0.03, 0.95]])
    b = torch.tensor([10.0, -6.0])
    dst = src @ A.T + b
    tps = fit_tps(src, dst)
    pred = tps(src)
    assert torch.allclose(pred, dst, atol=1e-3), (pred - dst).abs().max()

def test_warp_image_shape_preserved():
    img = torch.rand(256, 256, 3)
    src = torch.rand(8, 2) * 200 + 28
    dst = src + torch.randn(8, 2) * 3.0
    tps = fit_tps(src, dst)
    out = warp_image(img, tps)
    assert out.shape == img.shape
