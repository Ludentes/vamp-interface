"""Frame-aware calibration of ARKit→LivePortrait Euler signs (v3).

Replaces v2 (Pearson-per-axis), which empirically failed: kp_ref's frame is
not promised axis-aligned to the canonical face frame, so any constant
axis-swap or off-diagonal mirror between kp_ref and canonical makes per-axis
Pearson degenerate. See `docs/research/2026-05-06-arkit-vs-mediapipe-pose-conventions.md`.

Metric: rotation-matrix-direct, with a constant frame transform F absorbed
out by exhaustive enumeration over the 48-element signed-permutation group
P_48 ⊂ O(3) (i.e. all matrices that permute and sign-flip the three axes).
This is the natural F search space: kp_ref is a learned implicit-keypoint
basis whose plausible mismatch with the canonical face frame is exactly an
axis-swap + sign-flip constellation. Larger O(3) (general orthogonal) is
unnecessary and the log-map approximation we'd need to make it tractable
breaks for det F = −1 (mirrors), which is exactly the case that motivated
the upgrade.

Procedure:

  For each (sy, sp, sr) ∈ {±1}³ (8 combos):
    1. Build R_input(t) = (Rz(sr·roll) · Ry(sy·yaw) · Rx(sp·pitch))^T from
       input ARKit eulers (LivePortrait closed-form-pose convention).
    2. Render N frames through PersonaLive `bridge` with --euler_signs
       (reuse mp4 if already on disk).
    3. Extract R_render(t) = mediapipe facial_transformation_matrix[:3,:3].
    4. For each F ∈ P_48, compute mean angular distance over valid frames:
            d(F) = mean_t ang(F · R_input(t) · F^T, R_render(t)).
       Take inner_score(combo) = min_F d(F); record arg-min F*.
  Combo-winner: arg-min over the 8 combos.

Negative-control battery (each is a regression test for a different way the
metric could be wrong):

  * synthetic-mirror — generate R_re := F* R_in F*^T for known F* ∈ P_48
    (det F* = ±1) and verify the recovered F equals F* and score ≈ 0.
    This catches B1 from review (log-map approximation issues) directly.
  * sequence-shuffle — score the winner with R_render time-shuffled.
    Should be much worse, otherwise the metric ranks on rotation magnitude
    only and ignores temporal correspondence.
  * input-permutation — rebuild R_input with yaw↔roll swap on winner's
    signs; if F genuinely absorbs frame mismatch, score should be unchanged.

Cost: ~30s startup + ~4s render per combo (skipped if mp4 exists) + 48
angular-distance averages × 8 combos × ~600 frames ≈ subsecond per combo
after extraction.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from _mp_blendshape import make_landmarker  # noqa: E402
from arkit_bridge.llf_csv import load_llf_b61  # noqa: E402

Signs = tuple[float, float, float]


# ---------- Rotation helpers ------------------------------------------------

def rotmat_from_euler_lp(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Mirror src/arkit_bridge/closed_form_pose.py: R = (Rz·Ry·Rx)^T."""
    cy, sy = np.cos(yaw),   np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll),  np.sin(roll)
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]], dtype=np.float64)
    return (Rz @ Ry @ Rx).T


def angular_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic distance on SO(3): arccos((tr(R1ᵀR2)-1)/2)."""
    tr = np.clip((np.trace(R1.T @ R2) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(tr))


def signed_permutation_group() -> list[np.ndarray]:
    """All 48 elements of P_48 ⊂ O(3): 6 permutations × 8 sign patterns."""
    out = []
    for perm in itertools.permutations(range(3)):
        P = np.zeros((3, 3), dtype=np.float64)
        for i, j in enumerate(perm):
            P[i, j] = 1.0
        for s in itertools.product([+1.0, -1.0], repeat=3):
            S = np.diag(s)
            out.append(S @ P)
    assert len(out) == 48
    return out


P_48 = signed_permutation_group()


def fmt_F(F: np.ndarray) -> str:
    """Compact one-line repr of a signed-permutation matrix."""
    rows = []
    axes = ("x", "y", "z")
    for i in range(3):
        j = int(np.argmax(np.abs(F[i])))
        sgn = "+" if F[i, j] > 0 else "-"
        rows.append(f"{sgn}{axes[j]}")
    return f"[{','.join(rows)}]  det={np.linalg.det(F):+.0f}"


# ---------- Pipeline pieces -------------------------------------------------

def load_input_R_seq(take_dir: Path, n_frames: int, stride: int, start: int,
                    signs: Signs) -> np.ndarray:
    csv = next(take_dir.glob("*_iPhone.csv"))
    b_all = load_llf_b61(csv)
    idxs = start + np.arange(n_frames) * stride
    idxs = idxs[idxs < len(b_all)]
    yaw = b_all[idxs, 52].astype(np.float64)
    pit = b_all[idxs, 53].astype(np.float64)
    rol = b_all[idxs, 54].astype(np.float64)
    sy, sp, sr = signs
    return np.stack([rotmat_from_euler_lp(sy * y, sp * p, sr * r)
                     for y, p, r in zip(yaw, pit, rol)])


def extract_R_render_seq(mp4: Path, lm) -> np.ndarray:
    cap = cv2.VideoCapture(str(mp4))
    out = []
    while True:
        ok, bgr = cap.read()
        if not ok: break
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = mp.Image(image_format=mp.ImageFormat.SRGB,
                       data=np.ascontiguousarray(rgb))
        res = lm.detect(img)
        if res.facial_transformation_matrixes:
            M = np.asarray(res.facial_transformation_matrixes[0], dtype=np.float64)
            out.append(M[:3, :3])
        else:
            out.append(np.full((3, 3), np.nan))
    cap.release()
    return np.stack(out) if out else np.zeros((0, 3, 3))


def render_one(*, py: str, take_dir: Path, anchor: Path, ckpt: Path,
               out: Path, n_frames: int, stride: int, signs: Signs):
    cmd = [
        py, "scripts/apply_bridge_to_personalive.py",
        "--reference",   str(anchor),
        "--take_dir",    str(take_dir),
        "--ckpt",        str(ckpt),
        "--out_path",    str(out),
        "--n_frames",    str(n_frames),
        "--stride",      str(stride),
        "--start_frame", "0",
        "--mode",        "bridge",
        # '=' joining keeps argparse from treating '-1,...' as a flag (v2 bug).
        f"--euler_signs={','.join(f'{s:+.0f}' for s in signs)}",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:])
        raise RuntimeError(f"render failed for signs={signs}")


def score_combo(R_input: np.ndarray, R_render: np.ndarray) -> dict:
    """Inner-loop F-search over P_48; return min mean angular distance + best F.

    R_input (T,3,3) drives the head; R_render (T,3,3) is mediapipe's measure.
    We compute, per F ∈ P_48, the mean over valid frames of
    angular_distance(F R_input,t F^T, R_render,t), and take the F that
    minimizes that mean. This is exact (no log-map approximation), fast
    (vectorized over T), and exhaustive within the natural frame-mismatch
    search space."""
    n = min(len(R_input), len(R_render))
    valid = ~np.isnan(R_render[:n]).any(axis=(1, 2))
    if int(valid.sum()) < 20:
        return {"score_rad": float("inf"), "n_valid": int(valid.sum())}
    Ri = R_input[:n][valid]   # (V, 3, 3)
    Rr = R_render[:n][valid]  # (V, 3, 3)

    means: list[float] = []
    medians: list[float] = []
    p90s: list[float] = []
    for F in P_48:
        # F R F^T applied to all frames at once via einsum
        FRF = np.einsum("ij,tjk,lk->til", F, Ri, F)
        # angular distance per frame: arccos((tr(FRF^T · Rr)-1)/2)
        cosang = (np.einsum("tij,tij->t", FRF, Rr) - 1.0) * 0.5
        cosang = np.clip(cosang, -1.0, 1.0)
        d = np.arccos(cosang)
        means.append(float(d.mean()))
        medians.append(float(np.median(d)))
        p90s.append(float(np.percentile(d, 90)))
    means_arr = np.asarray(means)
    k = int(np.argmin(means_arr))
    return {
        "score_rad":  float(means_arr[k]),
        "median_rad": medians[k],
        "p90_rad":    p90s[k],
        "n_valid":    int(valid.sum()),
        "F":          P_48[k].tolist(),
        "F_repr":     fmt_F(P_48[k]),
    }


def shuffle_score(R_input: np.ndarray, R_render: np.ndarray, *, seed: int = 0) -> float:
    """Score with R_render rows time-shuffled — should be >> in-order score
    if the metric truly cares about temporal correspondence."""
    rng = np.random.default_rng(seed)
    n = min(len(R_input), len(R_render))
    perm = rng.permutation(n)
    return score_combo(R_input, R_render[:n][perm])["score_rad"]


# ---------- Driver ----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--take_dir", default="data/llf-clips-auto/20260505_MySlate_5_yaw",
                    help="prefer the auto-cut yaw clip — high-yaw motion makes "
                         "the alignment well-conditioned")
    ap.add_argument("--anchor", default="data/llf-phase2/asian_m__06_neutral.midframe.png")
    ap.add_argument("--ckpt",   default="runs/student_v3_lam10/student_best.pt")
    ap.add_argument("--out_dir", default="exp_output/arkit_bridge/calibration_v3")
    ap.add_argument("--n_frames", type=int, default=600)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--py", default="/home/newub/w/PersonaLive/.venv/bin/python")
    ap.add_argument("--reuse_v2", default="exp_output/arkit_bridge/calibration_v2",
                    help="symlink-reuse v2 mp4s if take/anchor/ckpt match (and "
                         "--reuse_v2_take matches the v2 default)")
    ap.add_argument("--reuse_v2_take", default="data/llf-takes/20260505_MySlate_5",
                    help="the take_dir v2 actually used; reuse only valid if "
                         "--take_dir matches this")
    ap.add_argument("--skip_render", action="store_true")
    ap.add_argument("--negative_control", action="store_true",
                    help="run the synthetic-mirror + shuffle + input-permutation battery")
    ap.add_argument("--force", action="store_true",
                    help="re-render combos even if mp4 already on disk")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    take_dir = Path(args.take_dir).resolve()
    anchor = Path(args.anchor).resolve()
    ckpt = Path(args.ckpt).resolve()

    can_reuse_v2 = (Path(args.reuse_v2).exists()
                    and Path(args.take_dir).resolve() == Path(args.reuse_v2_take).resolve())
    if can_reuse_v2:
        print(f"reuse_v2: take_dir matches; will symlink missing mp4s from {args.reuse_v2}")

    lm = make_landmarker()
    combos: list[Signs] = list(itertools.product([+1.0, -1.0], repeat=3))  # type: ignore[arg-type]
    results: dict[str, dict] = {}

    for k, signs in enumerate(combos, 1):
        tag = f"y{int(signs[0]):+d}_p{int(signs[1]):+d}_r{int(signs[2]):+d}"
        mp4 = out / f"render_{tag}.mp4"
        if (not mp4.exists() or args.force) and can_reuse_v2:
            v2_mp4 = Path(args.reuse_v2) / f"render_{tag}.mp4"
            if v2_mp4.exists():
                if mp4.exists() or mp4.is_symlink(): mp4.unlink()
                mp4.symlink_to(v2_mp4.resolve())
                print(f"[{k}/8] {tag}: symlinked v2 mp4")
        if (not mp4.exists() or args.force) and not args.skip_render:
            print(f"[{k}/8] render signs={signs} -> {mp4.name}", flush=True)
            render_one(py=args.py, take_dir=take_dir, anchor=anchor, ckpt=ckpt,
                       out=mp4, n_frames=args.n_frames, stride=args.stride, signs=signs)
        if not mp4.exists():
            print(f"[{k}/8] {tag}: no mp4, skipping (--skip_render set)")
            continue
        R_input = load_input_R_seq(take_dir, args.n_frames, args.stride,
                                   start=0, signs=signs)
        R_render = extract_R_render_seq(mp4, lm)
        sc = score_combo(R_input, R_render)
        results[tag] = sc
        print(f"   F*={sc.get('F_repr','?')}  score={sc['score_rad']:+.4f} rad  "
              f"median={sc.get('median_rad',float('nan')):+.4f}  "
              f"p90={sc.get('p90_rad',float('nan')):+.4f}  n={sc['n_valid']}")

    if not results:
        sys.exit("no combos scored")
    finite = {k: v for k, v in results.items() if np.isfinite(v["score_rad"])}
    if not finite:
        sys.exit("all combos returned non-finite scores")
    best = min(finite, key=lambda k: finite[k]["score_rad"])
    print("\n# Sign combo (sy,sp,sr) — frame-aware mean angular distance (rad, lower better):")
    for k, v in sorted(results.items(), key=lambda kv: kv[1]["score_rad"]):
        marker = "  <-- BEST" if k == best else ""
        print(f"  {k}  mean={v['score_rad']:+.4f}  median={v.get('median_rad',float('nan')):+.4f}  "
              f"p90={v.get('p90_rad',float('nan')):+.4f}  F*={v.get('F_repr','?')}  n={v['n_valid']}{marker}")

    out_doc = {"results": results, "best": best, "n_frames": args.n_frames,
               "take_dir": str(take_dir)}

    if args.negative_control:
        # Build R_input/R_render once for the winner combo for all controls.
        toks = best.split("_")
        best_signs: Signs = (float(toks[0][1:]), float(toks[1][1:]), float(toks[2][1:]))
        R_in_w = load_input_R_seq(take_dir, args.n_frames, args.stride,
                                  start=0, signs=best_signs)
        mp4 = out / f"render_{best}.mp4"
        R_re_w = extract_R_render_seq(mp4, lm)
        orig = results[best]["score_rad"]

        # (1) synthetic-mirror: build R_re* := F* R_in F*^T for known F* with
        # det = ±1; verify F search recovers F* and score ≈ 0.
        controls: dict[str, dict] = {"synthetic": {}}
        for F_name, F_star in [("identity",   np.eye(3)),
                               ("diag_-+_-",  np.diag([-1.0, +1.0, -1.0])),  # det=+1
                               ("diag_-_++",  np.diag([-1.0, +1.0, +1.0])),  # det=-1 (mirror)
                               ("yz_swap",    np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]],
                                                       dtype=np.float64))]:  # det=-1
            R_re_star = np.einsum("ij,tjk,lk->til", F_star, R_in_w, F_star)
            sc = score_combo(R_in_w, R_re_star)
            controls["synthetic"][F_name] = {
                "F_star":    F_star.tolist(),
                "F_star_det": float(np.linalg.det(F_star)),
                "F_recovered": sc.get("F_repr"),
                "score_rad":  sc["score_rad"],
                "pass":       sc["score_rad"] < 1e-3,
            }

        # (2) sequence-shuffle: temporal correspondence test.
        s_shuffle = shuffle_score(R_in_w, R_re_w, seed=0)
        controls["shuffle"] = {
            "orig_score":     orig,
            "shuffled_score": s_shuffle,
            "ratio":          (s_shuffle / orig) if orig > 1e-9 else float("inf"),
            "pass":           s_shuffle > 2.0 * orig,
        }

        # (3) input-permutation: yaw↔roll swap. F absorbs → score unchanged.
        csv = next(take_dir.glob("*_iPhone.csv"))
        b_all = load_llf_b61(csv)
        idxs = np.arange(args.n_frames) * args.stride
        idxs = idxs[idxs < len(b_all)]
        yaw_in   = b_all[idxs, 52].astype(np.float64)
        pitch_in = b_all[idxs, 53].astype(np.float64)
        roll_in  = b_all[idxs, 54].astype(np.float64)
        sy, sp, sr = best_signs
        # Swap: pass roll into yaw arg, yaw into roll arg.
        R_perm = np.stack([rotmat_from_euler_lp(yaw=sy * r_raw, pitch=sp * p_raw,
                                                roll=sr * y_raw)
                           for y_raw, p_raw, r_raw
                           in zip(yaw_in, pitch_in, roll_in)])
        sc_perm = score_combo(R_perm, R_re_w)
        delta = abs(sc_perm["score_rad"] - orig)
        controls["input_permutation"] = {
            "permutation":   "yaw↔roll",
            "orig_score":    orig,
            "perm_score":    sc_perm["score_rad"],
            "delta":         delta,
            "pass":          delta < 0.05 * orig + 1e-3,
        }

        out_doc["controls"] = controls

        print("\n# Negative controls:")
        for name, results_per in controls["synthetic"].items():
            tag2 = "PASS" if results_per["pass"] else "FAIL"
            print(f"  synthetic[{name:14s}] det={results_per['F_star_det']:+.0f} "
                  f"score={results_per['score_rad']:.4e} rad  "
                  f"recovered={results_per['F_recovered']}  [{tag2}]")
        print(f"  shuffle           orig={orig:+.4f}  shuf={s_shuffle:+.4f}  "
              f"ratio={controls['shuffle']['ratio']:.2f}×  "
              f"[{'PASS' if controls['shuffle']['pass'] else 'FAIL'}]")
        print(f"  input_permutation orig={orig:+.4f}  perm={sc_perm['score_rad']:+.4f}  "
              f"Δ={delta:+.4f}  "
              f"[{'PASS' if controls['input_permutation']['pass'] else 'FAIL'}]")

    (out / "calibration_v3.json").write_text(json.dumps(out_doc, indent=2))
    print(f"\nwrote {out}/calibration_v3.json")
    print(f"\nWINNER: {best}  F*={results[best].get('F_repr','?')}")


if __name__ == "__main__":
    main()
