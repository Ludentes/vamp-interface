"""Side-by-side bake-off scorecard for v4 student variants.

Reads two checkpoints (v4a weighted_mse, v4b varnorm_jvp), runs offline
diagnostics on a heldout pair set, emits CSV + Markdown summary.

Metrics per arm:
  - tail_recovery_median  (per-cell std ratio on |t_z|>2 frames)
  - n_channels_above_0_7  (per-channel input_sensitivity ratio >= 0.7)
  - jvp_norm_match_median (|‖J_s e_k‖ - ‖C[:,k]‖| / ‖C[:,k]‖, all 58 channels)
  - heldout_r2_median     (per-cell R² on heldout)
  - heldout_std_ratio_median (student/teacher std ratio on heldout)

Decision: arm wins ≥4/5 metrics → ship. Otherwise flag tie for follow-up.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.func

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arkit_bridge.student import MotEncoderStudent  # noqa: E402


def load_pairs(pairs_dir: Path):
    paths = sorted(pairs_dir.glob("*frame_*.pkl"))
    bs, ms = [], []
    for p in paths:
        with open(p, "rb") as f:
            d = pickle.load(f)
        bs.append(np.asarray(d["b_expr"], dtype=np.float32))
        ms.append(np.asarray(d["m_f"], dtype=np.float32).squeeze(0))
    return np.stack(bs), np.stack(ms)


def load_student(ckpt: str, device: str):
    s = MotEncoderStudent().to(device)
    s.load_state_dict(torch.load(ckpt, map_location=device))
    s.eval()
    return s


@torch.no_grad()
def predict(s, b: np.ndarray, device: str):
    out = s(torch.from_numpy(b).to(device)).cpu().numpy()
    # student emits (B, 1, 32, 16); heldout m is (N, 32, 16) after squeeze.
    if out.ndim == 4 and out.shape[1] == 1:
        out = out.squeeze(1)
    return out


def tail_recovery_median(t: np.ndarray, p: np.ndarray, mean: np.ndarray, std: np.ndarray):
    z = (t - mean) / np.maximum(std, 1e-6)
    mask = np.abs(z) > 2.0
    out = []
    for j in range(t.shape[1]):
        for k in range(t.shape[2]):
            m = mask[:, j, k]
            if m.sum() < 8:
                continue
            ts = t[m, j, k].std()
            ps = p[m, j, k].std()
            if ts < 1e-4:
                continue
            out.append(ps / ts)
    return float(np.median(out)) if out else float("nan")


def per_channel_sensitivity(s, b: np.ndarray, m: np.ndarray, device: str, delta=0.05):
    """For each input channel k, perturb b → b + δ·e_k, return per-channel
    student response norm averaged across the heldout batch (rough analog
    of finite-difference Jacobian column norm)."""
    n_b = b.shape[1]
    out = np.zeros(n_b)
    b_t = torch.from_numpy(b).to(device)
    with torch.no_grad():
        m_base = s(b_t).cpu().numpy()
    for k in range(n_b):
        b_pert = b_t.clone()
        b_pert[:, k] += delta
        with torch.no_grad():
            m_pert = s(b_pert).cpu().numpy()
        out[k] = np.linalg.norm(m_pert - m_base) / max(delta, 1e-6) / max(np.sqrt(b.shape[0]), 1)
    return out


def jvp_norm(s, b: torch.Tensor, k: int):
    e_k = torch.zeros(b.shape[1], device=b.device)
    e_k[k] = 1.0
    _, jvp_out = torch.func.jvp(s, (b,), (e_k.unsqueeze(0).expand_as(b),))
    return jvp_out.flatten(1).norm(dim=1).mean().item()


def evaluate_arm(name: str, ckpt: str, b: np.ndarray, m: np.ndarray,
                 stats: dict, device: str):
    s = load_student(ckpt, device)
    pred = predict(s, b, device)

    tr = tail_recovery_median(m, pred, stats["teacher_mean"], stats["teacher_std"])

    sens = per_channel_sensitivity(s, b, m, device)
    target = np.linalg.norm(stats["C"], axis=0)
    ratio = sens / np.maximum(target, 1e-6)
    n_above_07 = int((ratio >= 0.7).sum())

    b_t = torch.from_numpy(b[:64]).to(device)
    sn = np.array([jvp_norm(s, b_t, k) for k in range(b.shape[1])])
    jvp_match = float(np.median(np.abs(sn - target) / np.maximum(target, 1e-6)))

    err = ((m - pred) ** 2).mean(axis=0)
    var = m.var(axis=0)
    r2 = 1.0 - err / np.maximum(var, 1e-6)
    r2_med = float(np.median(r2.flatten()))
    s_std = pred.std(axis=0)
    t_std = m.std(axis=0)
    sr = s_std / np.maximum(t_std, 1e-6)
    sr_med = float(np.median(sr.flatten()))

    return {
        "name": name,
        "ckpt": ckpt,
        "tail_recovery_median": tr,
        "n_channels_above_0_7": n_above_07,
        "jvp_norm_match_median": jvp_match,
        "heldout_r2_median": r2_med,
        "heldout_std_ratio_median": sr_med,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_a", required=True, help="v4a weighted_mse")
    ap.add_argument("--ckpt_b", required=True, help="v4b varnorm_jvp")
    ap.add_argument("--heldout", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s_npz = np.load(args.stats, allow_pickle=True)
    stats = {k: s_npz[k] for k in s_npz.files}

    b, m = load_pairs(Path(args.heldout))
    print(f"heldout: {b.shape[0]} pairs")

    a = evaluate_arm("v4a_weighted_mse", args.ckpt_a, b, m, stats, args.device)
    bb = evaluate_arm("v4b_varnorm_jvp", args.ckpt_b, b, m, stats, args.device)

    metrics_higher_better = {
        "tail_recovery_median": True,
        "n_channels_above_0_7": True,
        "jvp_norm_match_median": False,
        "heldout_r2_median": True,
        "heldout_std_ratio_median": "closer_to_1",
    }
    a_wins = 0; b_wins = 0
    for k, hb in metrics_higher_better.items():
        va = a[k]; vb = bb[k]
        if hb is True:
            (a_wins, b_wins) = (a_wins + 1, b_wins) if va > vb else (a_wins, b_wins + 1)
        elif hb is False:
            (a_wins, b_wins) = (a_wins + 1, b_wins) if va < vb else (a_wins, b_wins + 1)
        else:
            (a_wins, b_wins) = (a_wins + 1, b_wins) if abs(va - 1) < abs(vb - 1) else (a_wins, b_wins + 1)

    decision = ("v4a" if a_wins >= 4 else
                "v4b" if b_wins >= 4 else
                "tie_render_yaw_clip")

    payload = {"a": a, "b": bb, "a_wins": a_wins, "b_wins": b_wins, "decision": decision}
    with open(out_dir / "scorecard.json", "w") as f:
        json.dump(payload, f, indent=2)

    rows = ["metric,v4a_weighted_mse,v4b_varnorm_jvp"]
    for k in metrics_higher_better:
        rows.append(f"{k},{a[k]},{bb[k]}")
    rows.append(f"wins,{a_wins},{b_wins}")
    rows.append(f"decision,{decision},")
    (out_dir / "scorecard.csv").write_text("\n".join(rows) + "\n")

    md = [
        f"# v4 Bake-off Scorecard",
        f"",
        f"Heldout: {args.heldout} ({b.shape[0]} pairs)",
        f"",
        f"| Metric | v4a (A+B weighted_mse) | v4b (C varnorm_jvp) |",
        f"|---|---|---|",
    ]
    for k in metrics_higher_better:
        md.append(f"| `{k}` | {a[k]:.4f} | {bb[k]:.4f} |")
    md += [
        f"",
        f"**Wins:** v4a={a_wins}, v4b={b_wins}",
        f"",
        f"**Decision:** `{decision}`",
    ]
    (out_dir / "scorecard.md").write_text("\n".join(md) + "\n")

    print("\n".join(md))
    print(f"\nwrote {out_dir}/scorecard.{{json,csv,md}}")


if __name__ == "__main__":
    main()
