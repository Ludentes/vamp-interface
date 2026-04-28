from pathlib import Path

import pyarrow.parquet as pq
import torch

import torch.nn.functional as F

from arc_distill.dataset import FFHQPixelDataset, is_held_out
from arc_distill.model import ArcStudentResNet18, cosine_distance_loss


def test_is_held_out_f_prefix_held():
    assert is_held_out("f1234567" + "0" * 56) is True


def test_is_held_out_other_prefix_train():
    assert is_held_out("a1234567" + "0" * 56) is False
    assert is_held_out("01234567" + "0" * 56) is False


def test_is_held_out_uppercase_ok():
    assert is_held_out("F1234567" + "0" * 56) is True


def _make_smoke_pt(tmp_path: Path, n_rows: int) -> tuple[Path, list[str], list[bool]]:
    """Build a fake encoded .pt aligned with a smoke parquet's rows.

    Mix SHA prefixes so the train/val split has both sides; mark one row
    detected=False so we test the detection filter.
    """
    shas = [
        "a" + "0" * 63,
        "f" + "0" * 63,
        "1" + "0" * 63,
        "f" + "1" + "0" * 62,
        "b" + "0" * 63,
        "c" + "0" * 63,
    ][:n_rows]
    detected = [True, True, True, True, False, True][:n_rows]
    pt_path = tmp_path / "shard-00000.pt"
    torch.save(
        {
            "image_sha256": shas,
            "arcface_fp32": torch.randn(n_rows, 512, dtype=torch.float32),
            "detected": torch.tensor(detected),
            "format_version": 1,
        },
        pt_path,
    )
    return pt_path, shas, detected


def test_dataset_join_filters_detected_and_split(tmp_path):
    smoke = Path("tests/fixtures/ffhq_smoke.parquet")
    n_rows = pq.read_table(smoke, columns=["image"]).num_rows
    pt, shas, detected = _make_smoke_pt(tmp_path, n_rows)

    ds_train = FFHQPixelDataset(
        parquet_path=smoke, encoded_pt_path=pt, split="train", resolution=224,
    )
    ds_val = FFHQPixelDataset(
        parquet_path=smoke, encoded_pt_path=pt, split="val", resolution=224,
    )

    n_detected = sum(detected)
    n_detected_held = sum(
        1 for s, d in zip(shas, detected) if d and s[:1].lower() == "f"
    )
    assert len(ds_val) == n_detected_held
    assert len(ds_train) == n_detected - n_detected_held
    assert len(ds_train) + len(ds_val) == n_detected

    img, tgt = ds_train[0]
    assert img.shape == (3, 224, 224)
    assert img.dtype == torch.float32
    assert tgt.shape == (512,)
    assert tgt.dtype == torch.float32


def test_dataset_row_count_mismatch_raises(tmp_path):
    smoke = Path("tests/fixtures/ffhq_smoke.parquet")
    n_rows = pq.read_table(smoke, columns=["image"]).num_rows
    pt, _, _ = _make_smoke_pt(tmp_path, n_rows - 1)

    import pytest
    with pytest.raises(ValueError, match="row count"):
        FFHQPixelDataset(parquet_path=smoke, encoded_pt_path=pt, split="train")


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
