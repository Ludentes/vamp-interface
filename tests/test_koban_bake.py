import torch
from chibi.koban_bake import mirror_fill, skin_fallback


def test_mirror_fill_copies_seen_partner():
    tex = torch.zeros(8, 8, 3)
    seen = torch.zeros(8, 8, dtype=torch.bool)
    # left half seen with value 0.7, right half unseen
    tex[:, :4] = 0.7
    seen[:, :4] = True
    out, out_seen = mirror_fill(tex, seen)   # mirror across the W axis
    assert torch.allclose(out[:, 4:], torch.full((8, 4, 3), 0.7))
    assert out_seen.all()


def test_skin_fallback_fills_unseen_with_constant():
    tex = torch.zeros(8, 8, 3)
    seen = torch.zeros(8, 8, dtype=torch.bool)
    tex[0, 0] = torch.tensor([0.6, 0.4, 0.3])
    seen[0, 0] = True
    out = skin_fallback(tex, seen)
    assert torch.allclose(out[4, 4], torch.tensor([0.6, 0.4, 0.3]))
