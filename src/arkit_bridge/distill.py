"""Distill loop: per-frame MSE on (b_expr, m_f) pairs.

If a holdout dir is provided, runs the Tier-1+Tier-2 viability eval
(arkit_bridge.eval.main) every `ckpt_every` steps, writes one JSON per
checkpoint, and tracks the best held-out ratio_mean for early-stop.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent


def train(
    pairs_dir: str | Path,
    out_dir: str | Path,
    *,
    holdout_dir: str | Path | None = None,
    batch_size: int = 64,
    lr: float = 5e-4,
    steps: int = 20000,
    log_every: int = 100,
    ckpt_every: int = 2000,
    device: str = "cuda",
    early_stop_patience: int = 3,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = PairDataset(pairs_dir)
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        num_workers=2, drop_last=True, persistent_workers=True,
    )
    s = MotEncoderStudent().to(device)
    opt = torch.optim.AdamW(s.parameters(), lr=lr)

    log: list[dict] = []
    eval_log: list[dict] = []
    best_ratio = float("inf")
    best_r2 = -float("inf")
    best_step = 0
    plateau = 0

    step = 0
    t0 = time.time()

    def do_eval(ckpt_path: Path, step: int):
        nonlocal best_ratio, best_step, plateau
        if holdout_dir is None:
            return
        from arkit_bridge.eval import main as eval_main
        eval_path = out_dir / f"eval_step{step:06d}.json"
        eval_main(str(ckpt_path), str(holdout_dir), str(eval_path), device=device)
        with open(eval_path) as f:
            payload = json.load(f)
        ratio = payload["tier1"]["ratio_mean"]
        r2_above = payload["tier1"]["r2_above_0_7_fraction"]
        eval_log.append({
            "step": step, "ratio_mean": ratio,
            "r2_above_0_7_fraction": r2_above,
            "passes_ratio_0_10": payload["tier1"]["passes_ratio_0_10"],
            "passes_r2_mask": payload["tier1"]["passes_r2_mask"],
        })
        with open(out_dir / "eval_log.json", "w") as f:
            json.dump(eval_log, f, indent=2)
        nonlocal best_r2
        improved_ratio = ratio < best_ratio - 1e-5
        improved_r2 = r2_above > best_r2 + 1e-5
        tags = []
        if improved_ratio:
            best_ratio = ratio
            best_step = step
            plateau = 0
            torch.save(s.state_dict(), out_dir / "student_best.pt")
            tags.append("BEST_RATIO")
        if improved_r2:
            best_r2 = r2_above
            torch.save(s.state_dict(), out_dir / "student_best_r2.pt")
            tags.append("BEST_R2")
        if not (improved_ratio or improved_r2):
            plateau += 1
            tail = f"(no improvement, plateau={plateau}/{early_stop_patience})"
        else:
            if not improved_ratio:
                # R²-only win: don't reset plateau (ratio is the primary metric)
                plateau += 1
                tail = f"*{'+'.join(tags)}* (plateau={plateau}/{early_stop_patience})"
            else:
                tail = f"*{'+'.join(tags)}*"
        print(f"  [eval@{step}] ratio={ratio:.5f} R²≥0.7={r2_above:.3f} {tail}",
              flush=True)
        return plateau >= early_stop_patience

    while step < steps:
        for b, m in dl:
            b = b.to(device); m = m.to(device)
            loss = F.mse_loss(s(b), m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            if step % log_every == 0:
                rate = step / max(1e-6, time.time() - t0)
                msg = f"step {step:6d}  loss {loss.item():.5f}  ({rate:.1f}/s)"
                print(msg, flush=True)
                log.append({"step": step, "loss": float(loss.detach())})
            if step % ckpt_every == 0 or step >= steps:
                ckpt = out_dir / f"student_step{step:06d}.pt"
                torch.save(s.state_dict(), ckpt)
                with open(out_dir / "log.json", "w") as f:
                    json.dump(log, f)
                stop = do_eval(ckpt, step)
                if stop:
                    print(f"early stop at step {step} (best={best_ratio:.5f}@{best_step})",
                          flush=True)
                    return s
            if step >= steps:
                break
    print(f"done. best_ratio={best_ratio:.5f} @ step {best_step}", flush=True)
    return s
