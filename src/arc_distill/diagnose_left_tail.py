"""Diagnose the left tail of A2-full-native cos distribution.

Pulls bottom-K, top-K, and median-K rows by student-teacher cos, looks them up
in the FFHQ parquet shards by SHA, writes PNG grids and stats to identify what
distinguishes failure rows from success rows.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from arc_distill.adapter import AdapterStudent
from arc_distill.dataset import CompactLatentDataset, is_held_out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def grid_image(images: list[Image.Image], cols: int = 10, tile: int = 96) -> Image.Image:
    n = len(images)
    rows = (n + cols - 1) // cols
    out = Image.new("RGB", (cols * tile, rows * tile), (32, 32, 32))
    for i, im in enumerate(images):
        r, c = divmod(i, cols)
        tt = im.resize((tile, tile), Image.LANCZOS).convert("RGB")
        out.paste(tt, (c * tile, r * tile))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", default="latent_a2_full_native_shallow")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--compact", type=Path, required=True)
    p.add_argument("--ffhq-parquet-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--k", type=int, default=50, help="rows per band")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"loading compact: {args.compact}")
    blob = torch.load(args.compact, map_location="cpu", weights_only=False)
    shas = list(blob["shas"])
    arcface = blob["arcface"]
    val_idx = [i for i, s in enumerate(shas) if is_held_out(s)]
    print(f"  val rows: {len(val_idx)}")

    ds = CompactLatentDataset(args.compact, "val")
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)

    print("loading model")
    m = AdapterStudent(args.variant).to(device)
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    m.load_state_dict(ck["model"])
    m.eval()

    cos_all = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            z = m(x)
            y_n = F.normalize(y, dim=-1)
            cos_all.append((z * y_n).sum(dim=-1).cpu())
    cos = torch.cat(cos_all)
    print(f"  cos: mean={cos.mean():.4f} median={cos.median():.4f}")

    arcface_val = arcface[val_idx]
    teacher_n = F.normalize(arcface_val, dim=-1)
    teacher_norm = arcface_val.norm(dim=-1)
    val_shas = [shas[i] for i in val_idx]

    order = torch.argsort(cos)
    bottom = order[: args.k].tolist()
    top = order[-args.k:].tolist()
    mid_lo = (len(order) - args.k) // 2
    median = order[mid_lo: mid_lo + args.k].tolist()

    bands = {"bottom": bottom, "median": median, "top": top}
    band_shas = {name: [val_shas[i] for i in idxs] for name, idxs in bands.items()}
    target_sha_to_band = {}
    for name, ss in band_shas.items():
        for s in ss:
            target_sha_to_band[s] = name

    sha_to_image: dict[str, Image.Image] = {}
    print(f"scanning FFHQ shards for {len(target_sha_to_band)} target shas")
    shards = sorted(args.ffhq_parquet_dir.glob("train-*.parquet"))
    for s_idx, s_path in enumerate(shards):
        if len(sha_to_image) >= len(target_sha_to_band):
            break
        table = pq.read_table(s_path, columns=["image"])
        for row in table.column("image").to_pylist():
            if not row:
                continue
            img_bytes = row.get("bytes") if isinstance(row, dict) else row
            if not img_bytes:
                continue
            sha = sha256_bytes(img_bytes)
            if sha in target_sha_to_band and sha not in sha_to_image:
                try:
                    sha_to_image[sha] = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception:
                    pass
        if (s_idx + 1) % 20 == 0:
            print(f"  shard {s_idx+1}/{len(shards)} found={len(sha_to_image)}/{len(target_sha_to_band)}")

    print(f"matched {len(sha_to_image)} / {len(target_sha_to_band)} images")

    stats = {}
    for name, idxs in bands.items():
        cs = cos[torch.tensor(idxs)]
        t_n = teacher_n[torch.tensor(idxs)]
        intra_t_cos = (t_n @ t_n.T).fill_diagonal_(0).sum() / (len(idxs) * (len(idxs) - 1))
        t_norm = teacher_norm[torch.tensor(idxs)]
        stats[name] = {
            "n": len(idxs),
            "cos_mean": float(cs.mean()),
            "cos_median": float(cs.median()),
            "cos_min": float(cs.min()),
            "cos_max": float(cs.max()),
            "teacher_norm_mean": float(t_norm.mean()),
            "teacher_norm_median": float(t_norm.median()),
            "teacher_norm_min": float(t_norm.min()),
            "teacher_norm_max": float(t_norm.max()),
            "intra_band_teacher_cos_mean": float(intra_t_cos),
            "sha_prefix_dist": {h: sum(1 for s in band_shas[name] if s[1] == h)
                                 for h in "0123456789abcdef"},
        }

    inter_bottom_top = (teacher_n[torch.tensor(bottom)] @ teacher_n[torch.tensor(top)].T).mean()
    stats["inter_bottom_top_teacher_cos_mean"] = float(inter_bottom_top)

    print("writing image grids and stats")
    for name, idxs in bands.items():
        ss = [val_shas[i] for i in idxs]
        imgs = [sha_to_image[s] for s in ss if s in sha_to_image]
        if imgs:
            grid_image(imgs, cols=10, tile=96).save(args.out_dir / f"{name}.png")

    with (args.out_dir / "stats.json").open("w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
