from pathlib import Path

import torch
import torch.nn.functional as F

from arc_distill.dataset import CompactFFHQDataset, is_held_out
from arc_distill.model import ArcStudentResNet18, cosine_distance_loss


def test_is_held_out_f_prefix_held():
    assert is_held_out("f1234567" + "0" * 56) is True


def test_is_held_out_other_prefix_train():
    assert is_held_out("a1234567" + "0" * 56) is False
    assert is_held_out("01234567" + "0" * 56) is False


def test_is_held_out_uppercase_ok():
    assert is_held_out("F1234567" + "0" * 56) is True


def _make_compact_blob(tmp_path: Path) -> Path:
    """Tiny compact .pt with mixed SHA prefixes for split testing."""
    shas = [
        "a" + "0" * 63,
        "f" + "0" * 63,
        "1" + "0" * 63,
        "f" + "1" + "0" * 62,
        "b" + "0" * 63,
    ]
    n = len(shas)
    images = torch.randint(0, 256, (n, 3, 8, 8), dtype=torch.uint8)
    arcface = torch.randn(n, 512, dtype=torch.float32)
    p = tmp_path / "compact.pt"
    torch.save({"images_u8": images, "arcface": arcface, "shas": shas,
                "resolution": 8, "format_version": 1}, p)
    return p


def test_compact_dataset_split_and_normalisation(tmp_path):
    p = _make_compact_blob(tmp_path)
    train = CompactFFHQDataset(p, split="train")
    val = CompactFFHQDataset(p, split="val")
    # 5 rows, 2 begin with 'f' → val=2, train=3.
    assert len(val) == 2
    assert len(train) == 3
    assert len(train) + len(val) == 5

    x, y = train[0]
    assert x.shape == (3, 8, 8)
    assert x.dtype == torch.float32
    assert y.shape == (512,)
    assert y.dtype == torch.float32
    # ImageNet-normalised values must include negatives (mean ≈ 0.5)
    assert x.min().item() < 0.0


def test_model_output_shape_and_l2_norm():
    m = ArcStudentResNet18(pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 512)
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_cosine_distance_loss_zero_when_aligned():
    a = F.normalize(torch.randn(4, 512), dim=-1)
    assert cosine_distance_loss(a, a).item() < 1e-6


def test_cosine_distance_loss_two_when_opposite():
    a = F.normalize(torch.randn(4, 512), dim=-1)
    assert abs(cosine_distance_loss(a, -a).item() - 2.0) < 1e-5


def test_adapter_pixel_a_shapes_and_grads():
    from arc_distill.adapter import AdapterStudent
    m = AdapterStudent(variant="pixel_a")
    x = torch.randn(2, 3, 112, 112)
    z = m(x)
    assert z.shape == (2, 512)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)
    z.sum().backward()
    deep = [n for n, p in m.named_parameters()
            if p.grad is not None and ("Conv_3" in n or "BatchNormalization_8" in n or "Conv_124" in n)]
    assert deep == [], f"deep backbone params received gradient: {deep[:3]}"


def test_adapter_latent_a_up_shapes():
    from arc_distill.adapter import AdapterStudent
    m = AdapterStudent(variant="latent_a_up")
    z = m(torch.randn(2, 16, 14, 14))
    assert z.shape == (2, 512)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_adapter_latent_a_native_shapes():
    from arc_distill.adapter import AdapterStudent
    m = AdapterStudent(variant="latent_a_native")
    z = m(torch.randn(2, 16, 14, 14))
    assert z.shape == (2, 512)
    assert torch.allclose(z.norm(dim=-1), torch.ones(2), atol=1e-5)


def test_adapter_train_mode_keeps_backbone_in_eval():
    """When the trainer calls m.train(), only stem submodules switch; deep BN
    must stay in eval so its running stats don't drift."""
    from arc_distill.adapter import AdapterStudent
    m = AdapterStudent(variant="pixel_a")
    m.train()
    # stem is in train
    assert m.backbone.Conv_0.training is True
    assert m.backbone.BatchNormalization_2.training is True
    # a deep BN is in eval
    assert m.backbone.BatchNormalization_8.training is False
    assert m.backbone.BatchNormalization_126.training is False
