import torch
from chibi.koban_bake import mirror_fill, skin_fallback, _koban_vt2v


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


def test_mirror_fill_leaves_doubly_unseen():
    tex = torch.zeros(8, 8, 3)
    seen = torch.zeros(8, 8, dtype=torch.bool)
    # only column 0 seen; its mirror (col 7) gets filled, the rest stay unseen
    tex[:, 0] = 0.7
    seen[:, 0] = True
    out, out_seen = mirror_fill(tex, seen)
    assert out_seen[:, 7].all()           # mirror partner filled
    assert not out_seen[:, 1:7].any()     # interior columns still unseen
    assert torch.allclose(out[:, 7], torch.full((8, 3), 0.7))


def test_skin_fallback_uses_median_of_seen():
    tex = torch.zeros(4, 4, 3)
    seen = torch.zeros(4, 4, dtype=torch.bool)
    vals = torch.tensor([[0.1, 0.1, 0.1], [0.5, 0.5, 0.5], [0.9, 0.9, 0.9]])
    tex[0, 0], tex[0, 1], tex[0, 2] = vals[0], vals[1], vals[2]
    seen[0, 0] = seen[0, 1] = seen[0, 2] = True
    out = skin_fallback(tex, seen)
    assert torch.allclose(out[3, 3], vals[1])   # median of the 3 seen texels


def test_koban_vt2v_reconstructs_correspondence():
    # 2 triangles; uv-vertex i corresponds to mesh-vertex (i + 10)
    faces = torch.tensor([[10, 11, 12], [11, 12, 13]], dtype=torch.int64)
    uv_faces = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.int64)
    n_uv = 4
    vt2v = _koban_vt2v(faces, uv_faces, n_uv)
    assert vt2v.shape == (n_uv,)
    assert torch.equal(vt2v, torch.tensor([10, 11, 12, 13]))
