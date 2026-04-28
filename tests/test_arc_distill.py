from pathlib import Path

import pyarrow.parquet as pq
import torch

from arc_distill.dataset import FFHQPixelDataset, is_held_out


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
