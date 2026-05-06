"""Distill loop: per-frame MSE on (b_expr, m_f) pairs.

If a holdout dir is provided, runs the Tier-1+Tier-2 viability eval
(arkit_bridge.eval.main) every `ckpt_every` steps, writes one JSON per
checkpoint, and tracks the best held-out ratio_mean for early-stop.

v2 additions (2026-05-05 evening): loss_mode + sampler arguments.
  loss_mode='varnorm_std_tail' uses
      varnorm-MSE + λ_std·std_match + λ_tail·tail
  with stats loaded from teacher_stats.npz. sampler='active_channel'
  uses WeightedRandomSampler over precomputed sample_weights.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

from arkit_bridge.dataset import PairDataset
from arkit_bridge.student import MotEncoderStudent


def _make_loss(loss_mode, stats, device, lam_std=1.0, lam_tail=0.5,
               tail_z=2.0, lam_jvp=0.1, alpha=0.5, anneal_to_step=0,
               eps=1e-3):
    """Returns a closure (student_pred, teacher, **kwargs) -> (loss, parts).

    Modes that need extras must receive them as kwargs:
      - weighted_mse:               needs b
      - varnorm_jvp:                needs b and model
      - weighted_mse_jvp_anneal:    needs b, model, and step
        Linearly anneals A+B+C augmentation toward pure varnorm_std_tail
        (v2's loss) between step 0 and step `anneal_to_step`. After that,
        loss is exactly varnorm_std_tail. At step 0, full A+B+C augmentation.
    """
    if loss_mode == "plain":
        def fn_plain(s, t, **_):
            l = F.mse_loss(s, t)
            return l, {"mse": float(l.detach())}
        return fn_plain

    sigma = torch.from_numpy(stats["teacher_std"]).to(device).clamp(min=eps)
    mean = torch.from_numpy(stats["teacher_mean"]).to(device)

    if loss_mode == "varnorm":
        def fn_varnorm(s, t, **_):
            l = ((s - t) / sigma).pow(2).mean()
            return l, {"varnorm": float(l.detach())}
        return fn_varnorm

    if loss_mode == "varnorm_std_tail":
        def fn_full(s, t, **_):
            # main term: varnorm pointwise
            l_var = ((s - t) / sigma).pow(2).mean()
            # std match: per-cell batch std of student vs teacher
            #   t.std might collapse if all batch samples are near-identical
            #   (rare with weighted sampler); cap floor with eps
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            # tail mining: |teacher_z| > tail_z
            z = (t - mean) / sigma
            tail_mask = (z.abs() > tail_z).float()
            denom = tail_mask.sum().clamp(min=1.0)
            l_tail = ((s - t).pow(2) * tail_mask).sum() / denom
            loss = l_var + lam_std * l_std + lam_tail * l_tail
            return loss, {
                "varnorm": float(l_var.detach()),
                "std_match": float(l_std.detach()),
                "tail": float(l_tail.detach()),
            }
        return fn_full

    if loss_mode == "weighted_mse":
        if "freq_k" not in stats or "C" not in stats:
            raise ValueError("weighted_mse requires stats with freq_k and C")
        freq_k = torch.from_numpy(stats["freq_k"]).to(device).clamp(min=1e-4)
        w_k = (1.0 / freq_k).pow(alpha)                              # (58,)
        b_p95 = torch.from_numpy(stats["b_p95"]).to(device).clamp(min=eps)
        C = torch.from_numpy(stats["C"]).to(device)                  # (512, 58)
        # cell_weight[j] = sum_k w_k * C[j,k] / sum_k C[j,k]; normalized to mean=1.
        # cell_w >= 0 by construction (w_k > 0 since freq_k clamped, C >= 0).
        num = (C * w_k[None, :]).sum(dim=1)                          # (512,)
        den = C.sum(dim=1).clamp(min=eps)
        cell_w = num / den
        cell_w = (cell_w / cell_w.mean().clamp(min=eps)).reshape(*sigma.shape)

        def fn_weighted(s, t, *, b=None, **_):
            if b is None:
                raise ValueError("weighted_mse needs b kwarg")
            # per-sample weight = max_k (|b_k|/b_p95_k * w_k), clipped + mean-norm
            score = (b.abs() / b_p95[None, :]) * w_k[None, :]        # (B, 58)
            sw = score.max(dim=1).values.clamp(0.1, 10.0)            # (B,)
            sw = sw / sw.mean().clamp(min=eps)
            err2 = ((s - t) / sigma).pow(2)                          # (B, ..., 32, 16)
            err2 = err2 * cell_w                                      # broadcasts on trailing dims
            # reduce all non-batch dims so we can apply per-sample weight
            err2 = err2.mean(dim=tuple(range(1, err2.ndim)))         # (B,)
            l_main = (err2 * sw).mean()
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            loss = l_main + lam_std * l_std
            return loss, {
                "weighted_mse": float(l_main.detach()),
                "std_match": float(l_std.detach()),
                "sample_w_mean": float(sw.mean().detach()),
            }
        return fn_weighted

    if loss_mode == "varnorm_jvp":
        if "freq_k" not in stats or "C" not in stats:
            raise ValueError("varnorm_jvp requires stats with freq_k and C")
        freq_k = torch.from_numpy(stats["freq_k"]).to(device).clamp(min=1e-4)
        w_k = (1.0 / freq_k).pow(alpha)
        probs = w_k / w_k.sum().clamp(min=eps)                        # (58,)
        C = torch.from_numpy(stats["C"]).to(device)
        target_norm = C.norm(dim=0)                                   # (58,)

        def fn_jvp(s, t, *, b=None, model=None, **_):
            if b is None or model is None:
                raise ValueError("varnorm_jvp needs b and model kwargs")
            l_var = ((s - t) / sigma).pow(2).mean()
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            # importance-sample one channel by w_k; JVP of student w.r.t. e_k.
            k = int(torch.multinomial(probs, num_samples=1).item())
            e_k = torch.zeros(b.shape[1], device=device)
            e_k[k] = 1.0
            # constant tangent: same e_k applied to every batch row; we average
            # ‖J_s e_k‖ over batch to get a noise-reduced estimate per step.
            tangent = e_k.unsqueeze(0).expand_as(b).contiguous()
            _, jvp_out = torch.func.jvp(model, (b.detach(),), (tangent,))
            student_norm = jvp_out.flatten(1).norm(dim=1).mean()
            l_jvp = (student_norm - target_norm[k]).pow(2)
            loss = l_var + lam_std * l_std + lam_jvp * l_jvp
            return loss, {
                "varnorm": float(l_var.detach()),
                "std_match": float(l_std.detach()),
                "jvp": float(l_jvp.detach()),
                "k": k,
                "student_norm": float(student_norm.detach()),
                "target_norm": float(target_norm[k].detach()),
            }
        return fn_jvp

    if loss_mode == "weighted_mse_jvp_anneal":
        if "freq_k" not in stats or "C" not in stats:
            raise ValueError("weighted_mse_jvp_anneal requires freq_k and C")
        if anneal_to_step <= 0:
            raise ValueError(
                "weighted_mse_jvp_anneal needs --anneal_to_step > 0; "
                "without it the mode silently degenerates to varnorm_std_tail"
            )
        # weighted_mse precompute
        freq_k = torch.from_numpy(stats["freq_k"]).to(device).clamp(min=1e-4)
        w_k = (1.0 / freq_k).pow(alpha)
        b_p95 = torch.from_numpy(stats["b_p95"]).to(device).clamp(min=eps)
        C = torch.from_numpy(stats["C"]).to(device)
        num = (C * w_k[None, :]).sum(dim=1)
        den = C.sum(dim=1).clamp(min=eps)
        cell_w = num / den
        cell_w = (cell_w / cell_w.mean().clamp(min=eps)).reshape(*sigma.shape)
        # JVP precompute
        probs = w_k / w_k.sum().clamp(min=eps)
        target_norm = C.norm(dim=0)
        anneal_denom = max(anneal_to_step, 1)

        def fn_anneal(s, t, *, b=None, model=None, step=0, **_):
            if b is None or model is None:
                raise ValueError("weighted_mse_jvp_anneal needs b and model")
            # β: 1.0 at step 0 → 0.0 at and after anneal_to_step.
            beta = max(0.0, 1.0 - step / anneal_denom)

            # always-on v2 components (varnorm + std + tail)
            l_var = ((s - t) / sigma).pow(2).mean()
            s_std = s.std(dim=0, unbiased=False)
            t_std = t.std(dim=0, unbiased=False)
            l_std = (s_std - t_std).pow(2).mean()
            z = (t - mean) / sigma
            tail_mask = (z.abs() > tail_z).float()
            denom_tail = tail_mask.sum().clamp(min=1.0)
            l_tail = ((s - t).pow(2) * tail_mask).sum() / denom_tail

            # augmentation: per-cell + per-sample reweighted varnorm
            score = (b.abs() / b_p95[None, :]) * w_k[None, :]
            sw = score.max(dim=1).values.clamp(0.1, 10.0)
            sw = sw / sw.mean().clamp(min=eps)
            err2 = ((s - t) / sigma).pow(2) * cell_w
            err2 = err2.mean(dim=tuple(range(1, err2.ndim)))
            l_weighted = (err2 * sw).mean()

            # JVP regularizer (skip after anneal to save the JVP cost)
            if beta > 0.0:
                k = int(torch.multinomial(probs, num_samples=1).item())
                e_k = torch.zeros(b.shape[1], device=device)
                e_k[k] = 1.0
                tangent = e_k.unsqueeze(0).expand_as(b).contiguous()
                _, jvp_out = torch.func.jvp(
                    model, (b.detach(),), (tangent,),
                )
                student_norm = jvp_out.flatten(1).norm(dim=1).mean()
                l_jvp = (student_norm - target_norm[k]).pow(2)
                k_val = float(k)
                tn = float(target_norm[k].detach())
            else:
                l_jvp = torch.zeros((), device=device)
                student_norm = torch.zeros((), device=device)
                k_val = -1.0
                tn = 0.0

            # main term blends: β=1 → weighted_mse; β=0 → varnorm
            l_main = beta * l_weighted + (1.0 - beta) * l_var
            loss = (l_main + lam_std * l_std + lam_tail * l_tail
                    + beta * lam_jvp * l_jvp)
            return loss, {
                "varnorm": float(l_var.detach()),
                "weighted": float(l_weighted.detach()),
                "std_match": float(l_std.detach()),
                "tail": float(l_tail.detach()),
                "jvp": float(l_jvp.detach()),
                "student_norm": float(student_norm.detach()),
                "target_norm": tn,
                "k": k_val,
                "beta": float(beta),
            }
        return fn_anneal

    raise ValueError(f"unknown loss_mode={loss_mode}")


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
    loss_mode: str = "plain",
    sampler: str = "uniform",
    stats_path: str | Path | None = None,
    lam_std: float = 1.0,
    lam_tail: float = 0.5,
    tail_z: float = 2.0,
    lam_jvp: float = 0.1,
    alpha: float = 0.5,
    anneal_to_step: int = 0,
    seed: int = 0,
    lr_schedule: str = "constant",
    lr_min: float = 5e-5,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if seed:
        import random
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
    ds = PairDataset(pairs_dir)

    stats = None
    if stats_path is not None:
        stats_path = Path(stats_path)
        s_npz = np.load(stats_path, allow_pickle=True)
        stats = {k: s_npz[k] for k in s_npz.files}

    if sampler == "active_channel":
        if stats is None:
            raise ValueError("active_channel sampler requires --stats")
        # align stats['paths'] with ds.paths
        path_to_w = dict(zip(
            [str(x) for x in stats["paths"]],
            stats["sample_weights"].astype(np.float32),
        ))
        weights = np.array(
            [path_to_w[p.name] for p in ds.paths], dtype=np.float32,
        )
        if (weights < 0).any() or not np.isfinite(weights).all():
            raise ValueError("bad sample weights")
        wrs = WeightedRandomSampler(
            weights.astype(np.float64).tolist(),
            num_samples=len(ds), replacement=True,
        )
        dl = DataLoader(
            ds, batch_size=batch_size, sampler=wrs,
            num_workers=2, drop_last=True, persistent_workers=True,
        )
    else:
        dl = DataLoader(
            ds, batch_size=batch_size, shuffle=True,
            num_workers=2, drop_last=True, persistent_workers=True,
        )

    s = MotEncoderStudent().to(device)
    opt = torch.optim.AdamW(s.parameters(), lr=lr)

    sched = None
    if lr_schedule == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=steps, eta_min=lr_min,
        )
    elif lr_schedule != "constant":
        raise ValueError(f"unknown lr_schedule={lr_schedule}")

    loss_fn = _make_loss(loss_mode, stats, device,
                         lam_std=lam_std, lam_tail=lam_tail, tail_z=tail_z,
                         lam_jvp=lam_jvp, alpha=alpha,
                         anneal_to_step=anneal_to_step)

    log: list[dict] = []
    eval_log: list[dict] = []
    best_ratio = float("inf")
    best_r2 = -float("inf")
    best_step = 0
    plateau = 0

    step = 0
    t0 = time.time()

    def do_eval(ckpt_path: Path, step: int):
        nonlocal best_ratio, best_step, plateau, best_r2
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
        improved_ratio = ratio < best_ratio - 1e-5
        improved_r2 = r2_above > best_r2 + 1e-5
        tags = []
        if improved_ratio:
            best_ratio = ratio; best_step = step; plateau = 0
            torch.save(s.state_dict(), out_dir / "student_best.pt")
            tags.append("BEST_RATIO")
        if improved_r2:
            best_r2 = r2_above
            torch.save(s.state_dict(), out_dir / "student_best_r2.pt")
            tags.append("BEST_R2")
        if not improved_ratio:
            plateau += 1
        suffix = ("*" + "+".join(tags) + "*" if tags else
                  f"(plateau={plateau}/{early_stop_patience})")
        print(f"  [eval@{step}] ratio={ratio:.5f} R²≥0.7={r2_above:.3f} {suffix}",
              flush=True)
        return plateau >= early_stop_patience

    while step < steps:
        for b, m in dl:
            b = b.to(device); m = m.to(device)
            pred = s(b)
            loss, parts = loss_fn(pred, m, b=b, model=s, step=step)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if sched is not None:
                sched.step()
            step += 1
            if step % log_every == 0:
                rate = step / max(1e-6, time.time() - t0)
                pstr = " ".join(f"{k}={v:.4f}" for k, v in parts.items())
                print(f"step {step:6d}  loss {loss.item():.5f}  {pstr}  ({rate:.1f}/s)",
                      flush=True)
                log.append({"step": step, "loss": float(loss.detach()), **parts})
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
