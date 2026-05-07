"""Two diagnostic queries against the four-table parquet.

Q1 SIGN AGREEMENT (per blendshape channel):
  For each (take, frame) where the channel is "active" enough on both sides,
  does sign(delta_bridge) == sign(delta_teacher)?
  A channel with <50 % agreement on active frames is *wired backward*.

Q2 THREE-AMP DECOMPOSITION (per blendshape channel and per ypr axis):
  amp_personalive = ‖rgb_out_delta‖ / ‖rgb_in_delta‖   (RGB→RGB ceiling, 8-take RGB)
  amp_arkit_path  = ‖teacher_full_delta‖ / ‖input_delta‖ (ARKit b_61 → motion_encoder)
  amp_student     = ‖bridge_delta‖ / ‖teacher_full_delta‖ (the bridge contribution)

Inputs:
  exp_output/arkit_bridge/parquet/{anchors,frames,render_metrics}.parquet

Output:
  - signs_<run_tag>.csv  (51 rows: channel, n_active, agree_frac, status)
  - amps_<run_tag>.csv   (51 + 3 rows: channel, amp_personalive, amp_arkit_path, amp_student)
  - prints both tables sorted by problematic-first
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import polars as pl

from _mp_blendshape import NAMES_51

MP_NAMES_52 = ["_neutral"] + NAMES_51   # mediapipe order; we drop _neutral upstream
ARKIT_B58_NAMES = NAMES_51 + ["tongueOut"]  # ARKit 52; b_expr also has 6 head-rotation channels (52..57)


def _arr(s: pl.Series) -> np.ndarray:
    """Polars list<f32> → numpy (N, K) array."""
    return np.stack([np.asarray(x, dtype=np.float32) for x in s.to_list()])


def load_parquets(root: Path):
    a = pl.read_parquet(root / "anchors.parquet")
    f = pl.read_parquet(root / "frames.parquet")
    r = pl.read_parquet(root / "render_metrics.parquet")
    return a, f, r


def anchor_bs(anchors: pl.DataFrame, stem: str) -> np.ndarray:
    row = anchors.filter(pl.col("anchor_stem") == stem)
    if len(row) == 0:
        raise SystemExit(f"anchor {stem} not in anchors.parquet")
    return np.asarray(row["mp_blendshapes"][0].to_list(), dtype=np.float32)


def collect_deltas(rm: pl.DataFrame, *, run_tag: str | None, mode: str,
                   pane: str = "full", preprocessing: str | None = None,
                   anchor_bs_vec: np.ndarray, with_ypr: bool = True):
    """Return dict[take] -> dict[mp4_frame_idx -> delta_51_or_54].

    delta = mp_blendshapes − anchor_bs. If with_ypr, append (yaw, pitch, roll)
    raw radians at the end (no anchor subtract — head pose 'neutral' is 0 rad).
    Caveat: for `pane='left'` (driver crops) the anchor subtract mixes identity
    into the delta — only `std_vec` (dispersion) is meaningful there, not mean.
    """
    sel = rm.filter((pl.col("mode") == mode) & (pl.col("pane") == pane))
    if run_tag is not None:
        sel = sel.filter(pl.col("run_tag") == run_tag)
    if preprocessing is not None:
        sel = sel.filter(pl.col("preprocessing") == preprocessing)
    out: dict[int, dict[int, np.ndarray]] = {}
    if len(sel) == 0: return out
    bs = _arr(sel["mp_blendshapes"])
    delta = bs - anchor_bs_vec[None, :]
    if with_ypr:
        ypr = _arr(sel["ypr"])
        delta = np.concatenate([delta, ypr], axis=1)  # (N, 54)
    takes = sel["take"].to_list()
    fidx = sel["mp4_frame_idx"].to_list()
    for t, i, d in zip(takes, fidx, delta):
        out.setdefault(int(t), {})[int(i)] = d
    return out


def sign_agreement(bridge: dict[int, dict[int, np.ndarray]],
                   teacher: dict[int, dict[int, np.ndarray]],
                   *, active_thresh: float = 0.05, k: int = 51):
    """Per channel, fraction of frames where sign(bridge) == sign(teacher)
    on frames where |teacher| > active_thresh. Uses first `k` channels only.
    """
    K = k
    agree = np.zeros(K, dtype=np.int64)
    active = np.zeros(K, dtype=np.int64)
    for take, frames_b in bridge.items():
        frames_t = teacher.get(take, {})
        for i, db in frames_b.items():
            dt = frames_t.get(i)
            if dt is None: continue
            mask = np.abs(dt) > active_thresh
            if not mask.any(): continue
            active += mask.astype(np.int64)
            same = (np.sign(db) == np.sign(dt)) & mask
            agree += same.astype(np.int64)
    frac = np.where(active > 0, agree / np.maximum(active, 1), np.nan)
    return active, frac


def std_vec(deltas: dict, *, k: int = 51) -> np.ndarray:
    """std across all frames per channel (anchor-relative or pane-relative — all
    we need is dispersion, not absolute level)."""
    arrs = []
    for take, frames in deltas.items():
        for i, d in frames.items():
            arrs.append(d[:k])
    if not arrs:
        return np.full(k, np.nan, dtype=np.float64)
    M = np.stack(arrs)
    return np.nanstd(M, axis=0).astype(np.float64)


def amps_std(numer_std: np.ndarray, denom_std: np.ndarray, *, eps: float = 1e-3):
    out = np.full_like(numer_std, np.nan)
    mask = denom_std > eps
    out[mask] = numer_std[mask] / denom_std[mask]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet_root", default="exp_output/arkit_bridge/parquet")
    ap.add_argument("--run_tag", default="v3_lam10")
    ap.add_argument("--rgb_run_tag", default="personalive_rgb_full")
    ap.add_argument("--rgb_preprocessing", default="perframe")
    ap.add_argument("--anchor", default="asian_m__06_neutral.midframe")
    ap.add_argument("--out_dir", default="exp_output/arkit_bridge/diagnostics")
    args = ap.parse_args()

    root = Path(args.parquet_root)
    anchors, frames, rm = load_parquets(root)
    a_bs = anchor_bs(anchors, args.anchor)
    print(f"anchor {args.anchor}: bs[browDownLeft]={a_bs[0]:.3f} (sanity)")

    # All deltas computed wrt anchor blendshapes (51-d)
    bridge_dt   = collect_deltas(rm, run_tag=args.run_tag, mode="bridge",
                                  anchor_bs_vec=a_bs)
    teacher_dt  = collect_deltas(rm, run_tag=None, mode="teacher_full",
                                  anchor_bs_vec=a_bs)
    rgb_in_dt   = collect_deltas(rm, run_tag=args.rgb_run_tag, mode="personalive_rgb",
                                  pane="left", preprocessing=args.rgb_preprocessing,
                                  anchor_bs_vec=a_bs)
    rgb_out_dt  = collect_deltas(rm, run_tag=args.rgb_run_tag, mode="personalive_rgb",
                                  pane="right", preprocessing=args.rgb_preprocessing,
                                  anchor_bs_vec=a_bs)

    print(f"bridge   takes={sorted(bridge_dt)}  total frames={sum(len(v) for v in bridge_dt.values())}")
    print(f"teacher  takes={sorted(teacher_dt)}  total frames={sum(len(v) for v in teacher_dt.values())}")
    print(f"rgb_in   takes={sorted(rgb_in_dt)}  total frames={sum(len(v) for v in rgb_in_dt.values())}")
    print(f"rgb_out  takes={sorted(rgb_out_dt)}  total frames={sum(len(v) for v in rgb_out_dt.values())}")

    # Q1 sign agreement: bridge vs teacher (51 blendshapes + 3 ypr)
    NAMES_54 = NAMES_51 + ["yaw", "pitch", "roll"]
    n_active, agree = sign_agreement(bridge_dt, teacher_dt, k=54,
                                     active_thresh=0.05)
    # ypr active threshold should be in radians, not blendshape units. Recompute.
    n_active_y, agree_y = sign_agreement(
        {t: {i: d[51:54] for i, d in fr.items()} for t, fr in bridge_dt.items()},
        {t: {i: d[51:54] for i, d in fr.items()} for t, fr in teacher_dt.items()},
        k=3, active_thresh=0.10)  # ~6° in radians
    n_active[51:54] = n_active_y; agree[51:54] = agree_y

    print("\n# Q1 sign agreement (bridge vs teacher) — sorted worst-first:")
    print(f"  {'ch':<22} {'n_active':>9} {'agree':>7}  status")
    rows_sign = []
    for c, name in enumerate(NAMES_54):
        s = agree[c]
        st = "FLIP" if (n_active[c] >= 200 and s < 0.5) else \
             "SUS"  if (n_active[c] >= 200 and s < 0.6) else \
             "OK"   if (n_active[c] >= 200) else "low_active"
        rows_sign.append({"channel": name, "n_active": int(n_active[c]),
                          "agree_frac": float(s), "status": st})
    rows_sign_sorted = sorted(rows_sign, key=lambda r: (
        {"FLIP":0,"SUS":1,"OK":2,"low_active":3}[r["status"]], r["agree_frac"]))
    for r in rows_sign_sorted[:18]:
        print(f"  {r['channel']:<22} {r['n_active']:>9} {r['agree_frac']:>7.3f}  {r['status']}")

    # Q2 three-amp using std (dispersion) per channel — robust to identity-offset
    print("\n# Q2 three amplitudes — std(numer)/std(denom) per channel")
    s_rgb_in  = std_vec(rgb_in_dt,  k=54)
    s_rgb_out = std_vec(rgb_out_dt, k=54)
    s_teacher = std_vec(teacher_dt, k=54)
    s_bridge  = std_vec(bridge_dt,  k=54)
    amp_pl = amps_std(s_rgb_out, s_rgb_in)            # PersonaLive RGB→RGB attenuation
    amp_ap = amps_std(s_teacher, s_rgb_out)           # ARKit-path vs RGB-path ceiling
    amp_st = amps_std(s_bridge,  s_teacher)           # student attenuation

    print(f"  {'ch':<22} {'std_in':>8} {'amp_PL':>7} {'amp_arkit':>10} {'amp_student':>12}  flag")
    rows_amp = []
    for c, name in enumerate(NAMES_54):
        flag = ""
        if not np.isnan(amp_st[c]) and amp_st[c] < 0.5: flag += " STU<0.5"
        if not np.isnan(amp_st[c]) and amp_st[c] > 1.5: flag += " STU>1.5"
        rows_amp.append({"channel": name,
                         "std_rgb_in": float(s_rgb_in[c]),
                         "amp_personalive": float(amp_pl[c]),
                         "amp_arkit_path": float(amp_ap[c]),
                         "amp_student": float(amp_st[c]),
                         "flag": flag.strip()})
    # Sort by |amp_student − 1| descending (most-broken first), gated on activity
    def _sort_key(r):
        s = r["amp_student"]
        if np.isnan(s) or r["std_rgb_in"] < 0.01:
            return -1.0
        return -abs(s - 1.0)
    for r in sorted(rows_amp, key=_sort_key)[:25]:
        sin = f"{r['std_rgb_in']:>8.3f}"
        s = f"{r['amp_personalive']:>7.2f}" if not np.isnan(r['amp_personalive']) else f"{'-':>7}"
        a = f"{r['amp_arkit_path']:>10.2f}" if not np.isnan(r['amp_arkit_path'])  else f"{'-':>10}"
        b = f"{r['amp_student']:>12.2f}"    if not np.isnan(r['amp_student'])     else f"{'-':>12}"
        print(f"  {r['channel']:<22} {sin} {s} {a} {b}  {r['flag']}")

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows_sign).write_csv(out_dir / f"signs_{args.run_tag}.csv")
    pl.DataFrame(rows_amp ).write_csv(out_dir / f"amps_{args.run_tag}.csv")
    print(f"\nwrote {out_dir}/signs_{args.run_tag}.csv and amps_{args.run_tag}.csv")


if __name__ == "__main__":
    main()
