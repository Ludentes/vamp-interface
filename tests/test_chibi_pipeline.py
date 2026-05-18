import sys, pathlib, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import torch, pytest
from chibi.fit import _load_obj_verts
from chibi.landmarks import FLAME_TEMPLATE, _template_faces

MASKS = "/home/newub/w/LAM/model_zoo/human_parametric_models/flame_assets/flame/FLAME_masks.pkl"
needs_flame = pytest.mark.skipif(not os.path.exists(MASKS),
                                 reason="FLAME assets not present")


@needs_flame
def test_pipeline_through_stops_after_named_stage():
    from chibi.pipeline import ChibiPipeline
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = torch.as_tensor(_template_faces(), dtype=torch.int64)
    pipe = ChibiPipeline(MASKS)
    only_block = pipe.run(v, f, through="head_block")
    full = pipe.run(v, f)
    assert only_block.shape == v.shape
    # stopping early differs from the full run
    assert not torch.allclose(only_block, full)


@needs_flame
def test_pipeline_keeps_mesh_coherent():
    from chibi.pipeline import ChibiPipeline
    from chibi.chibi_metrics import foldover_count
    v = _load_obj_verts(FLAME_TEMPLATE).double()
    f = torch.as_tensor(_template_faces(), dtype=torch.int64)
    out = ChibiPipeline(MASKS).run(v, f)
    # no catastrophic self-fold (the wasp-waist class of failure)
    assert foldover_count(v, out, f) < 0.02 * f.shape[0]
