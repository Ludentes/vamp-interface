import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, pytest
from chibi.fit import _load_obj_verts
from chibi.landmarks import FLAME_TEMPLATE, _template_faces

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
needs_flame = pytest.mark.skipif(not os.path.exists(MASKS),
                                 reason="FLAME assets not present")


def _flame_mesh():
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = torch.as_tensor(_template_faces(), dtype=torch.int64)
    return v, f


@needs_flame
def test_head_block_identity_strength_is_noop():
    from chibi.stages.head_block import head_block, HeadBlockParams
    v, f = _flame_mesh()
    out = head_block(v, f, MASKS,
                     HeadBlockParams(exponent=8.0, strength_cranium=0.0,
                                     strength_face=0.0))
    assert torch.allclose(out, v, atol=1e-9)


@needs_flame
def test_head_block_lowers_box_residual():
    from chibi.stages.head_block import head_block, HeadBlockParams
    from chibi.primitives import fit_box
    from chibi.chibi_metrics import box_residual
    v, f = _flame_mesh()
    p = HeadBlockParams(exponent=8.0, strength_cranium=0.9, strength_face=0.3)
    out = head_block(v, f, MASKS, p)
    box = fit_box(v)
    assert box_residual(out, box, 8.0) < box_residual(v, box, 8.0)
    assert out.shape == v.shape
