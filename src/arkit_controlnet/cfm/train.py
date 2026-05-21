"""CFM training loop for InfuseNet expression control."""
from __future__ import annotations

import argparse
import csv
import math
import os
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader

from arkit_controlnet.cfm.dataset import CfmPairDataset
from arkit_controlnet.cfm.model import (
    build_model, velocity, pack_latents, prepare_latent_image_ids,
    prepare_text_ids,
)

CTRL_SIZE = 512
LATENT_H = LATENT_W = CTRL_SIZE // 8   # 64
SHIFT = 3.0


def _atomic_save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)


def _load_text_embeds(path: str, device: str, dtype: torch.dtype):
    d = torch.load(path, map_location="cpu")
    return d["t5"].to(device, dtype), d["pooled"].to(device, dtype)


def _flux_sigma(rand_normal: torch.Tensor) -> torch.Tensor:
    t = torch.sigmoid(rand_normal)
    return (SHIFT * t) / (1 + (SHIFT - 1) * t)


def train(
    out_dir: str,
    max_steps: int = 20000,
    save_every: int = 500,
    eval_every: int = 50,
    eval_dense_until: int = 1000,
    eval_sparse_every: int = 200,
    lr: float = 1e-4,
    warmup: int = 200,
    grad_accum: int = 4,
    seed: int = 0,
    dataset_filter_n: Optional[int] = None,
    text_embeds_path: str = "output/cfm_precompute/text_embeds.pt",
):
    torch.manual_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "step_log.csv"
    log_new = not log_path.exists()
    log_f = log_path.open("a", newline="")
    log_w = csv.writer(log_f)
    if log_new:
        log_w.writerow(["step", "loss", "grad_norm", "lr",
                        "sigma_mean", "wall_s"])

    device, dtype = "cuda", torch.bfloat16
    model = build_model()
    t5_seq, pooled_one = _load_text_embeds(text_embeds_path, device, dtype)

    ds = CfmPairDataset(split="train")
    if dataset_filter_n is not None:
        ds.df = ds.df.head(dataset_filter_n).reset_index(drop=True)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2,
                        pin_memory=True, drop_last=True, persistent_workers=True)

    opt = torch.optim.AdamW(model.trainable_params, lr=lr, weight_decay=0.01)

    img_ids = prepare_latent_image_ids(LATENT_H, LATENT_W, device, dtype)
    id_txt_ids = prepare_text_ids(8, device, dtype)
    t5_txt_ids = prepare_text_ids(t5_seq.shape[1], device, dtype)
    guidance = torch.full((1,), 3.5, device=device, dtype=dtype)

    ckpt = out / "latest.pt"
    step = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu")
        missing, unexpected = model.infusenet.load_state_dict(
            state["infusenet"], strict=False)
        if unexpected:
            raise RuntimeError(
                f"resume: unexpected keys in checkpoint — LoRA layout drift? "
                f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
        if missing:
            print(f"[resume] {len(missing)} frozen-base keys not in ckpt "
                  f"(expected for strict=False on partial save)")
        opt.load_state_dict(state["opt"])
        step = state["step"]
        print(f"[resume] step={step}")

    t0 = time.time()
    opt.zero_grad()
    grad_count = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break

            photo_latent = batch["photo_latent"].to(device, dtype)
            id_tokens = batch["id_tokens"].to(device, dtype)
            ctrl_latent = batch["ctrl_latent"].to(device, dtype)

            z0 = photo_latent
            eps = torch.randn_like(z0)
            sigma = _flux_sigma(torch.randn(1, device=device))
            s = sigma.view(-1, 1, 1, 1).to(dtype)
            z_t = (1 - s) * z0 + s * eps
            target_packed = pack_latents(eps - z0)
            z_t_packed = pack_latents(z_t)
            control_packed = pack_latents(ctrl_latent)

            warm_lr = lr * min(1.0, (step + 1) / max(1, warmup))
            for g in opt.param_groups:
                g["lr"] = warm_lr

            v_pred = velocity(model, z_t_packed, sigma, id_tokens,
                              t5_seq.expand(1, -1, -1), pooled_one,
                              id_txt_ids, t5_txt_ids, img_ids,
                              control_packed, guidance)
            loss = torch.nn.functional.mse_loss(
                v_pred.float(), target_packed.float()) / grad_accum
            loss.backward()
            grad_count += 1

            if grad_count >= grad_accum:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.trainable_params, 1.0)
                opt.step()
                opt.zero_grad()
                grad_count = 0
                step += 1
                log_w.writerow([step, f"{loss.item()*grad_accum:.5f}",
                                f"{float(grad_norm):.4e}", f"{warm_lr:.2e}",
                                f"{float(sigma):.3f}", f"{time.time()-t0:.1f}"])
                log_f.flush()
                if step % save_every == 0:
                    _atomic_save({"step": step,
                                  "infusenet": model.infusenet.state_dict(),
                                  "opt": opt.state_dict()}, ckpt)

                eval_cadence = (eval_every if step <= eval_dense_until
                                else eval_sparse_every)
                if step % eval_cadence == 0:
                    from arkit_controlnet.cfm.eval import dump_samples
                    dump_samples(model, step, out_dir=out_dir,
                                 text_embeds=(t5_seq, pooled_one))

    log_f.close()
    _atomic_save({"step": step, "infusenet": model.infusenet.state_dict(),
                  "opt": opt.state_dict()}, ckpt)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="exp_output/cfm_train/run01")
    p.add_argument("--max-steps", type=int, default=20000)
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--dataset-filter-n", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    a = _cli()
    train(out_dir=a.out_dir, max_steps=a.max_steps,
          save_every=a.save_every, eval_every=a.eval_every,
          dataset_filter_n=a.dataset_filter_n)
