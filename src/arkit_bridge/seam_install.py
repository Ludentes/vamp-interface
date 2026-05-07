"""Install ARKit-bridge seams onto a Pose2VideoPipeline_Stream.

Replaces pose_encoder + motion_encoder hot paths with closed-form pose +
trained MotEncoderStudent. Two modes:

  array:    b_seq + ypr_seq are precomputed numpy arrays indexed by the
            internal cursor (offline rendering path).
  provider: caller supplies provider(start, n) -> (b_chunk, ypr_chunk) so
            data can be swapped per pipe-call (streaming daemon).

Provider-mode contract: provider(start, n) returns numpy arrays of shape
(n, 58) and (n, 3), where row k corresponds to absolute frame index
(start + k). The caller is responsible for any clipping / boundary policy
the offline path performed via np.clip on b_seq.

Indexing convention: the internal helpers operate on an arbitrary
``indices`` array (consistent with the pre-refactor function); for
provider-mode that array is required to be a contiguous strictly-
increasing range, EXCEPT for the single leading-zero pad pattern emitted
by ``patched_interpolate_kps_online`` (``[0]*(num_interp-1) + range(n)``)
which we detect and serve with two provider calls.
"""
from __future__ import annotations

from types import MethodType

import numpy as np
import torch

from arkit_bridge.closed_form_pose import (
    EULER_SIGNS, euler_to_rotmat, compose_kd,
)


def _provider_fetch(provider, indices):
    """Service ``indices`` from a (start, n)->(b, ypr) callable.

    Fast paths:
    - Empty indices: pass through.
    - Strictly increasing contiguous: one provider call.
    - Leading-zero pad ([0]*k + arange(0, m)): two provider calls
      (provider(0, 1) for the pad, provider(0, m) for the body) then
      concatenate.
    Otherwise: AssertionError. The seam patches don't emit any other
    pattern, so we'd rather fail loudly than silently re-issue calls.
    """
    indices = np.asarray(indices)
    n_total = indices.shape[0]
    if n_total == 0:
        return (np.zeros((0, 58), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32))

    # Contiguous strictly-increasing.
    diffs = np.diff(indices)
    if n_total == 1 or np.all(diffs == 1):
        start = int(indices[0])
        b_chunk, ypr_chunk = provider(start, n_total)
        return np.asarray(b_chunk), np.asarray(ypr_chunk)

    # Leading-zero pad pattern emitted by patched_interpolate_kps_online:
    #   [0]*(num_interp-1) + list(range(0, motion.shape[0]))
    # which in concrete form is `[0]*pad_k + [0, 1, ..., m-1]`. Detect by
    # finding the longest trailing arange-from-0; the remaining prefix
    # must be all zeros.
    # Find the largest m such that indices[-m:] == arange(0, m). The trailing
    # arange has strictly-increasing diffs of 1; its first element is 0.
    # Scan from the end while diff==1, then the run-start position must hold
    # the value 0.
    m = 1
    while m < n_total and indices[n_total - m - 1] + 1 == indices[n_total - m]:
        m += 1
    if indices[n_total - m] != 0:
        m = 0  # trailing run doesn't start at 0; not the pad pattern
    pad_k = n_total - m
    if m > 0 and pad_k >= 0 and np.all(indices[:pad_k] == 0):
        b_body, y_body = provider(0, m)
        b_body = np.asarray(b_body)
        y_body = np.asarray(y_body)
        if pad_k == 0:
            return b_body, y_body
        # Pad rows = body row 0 (frame index 0) repeated pad_k times.
        b_pad = np.repeat(b_body[:1], pad_k, axis=0)
        y_pad = np.repeat(y_body[:1], pad_k, axis=0)
        return (np.concatenate([b_pad, b_body], axis=0),
                np.concatenate([y_pad, y_body], axis=0))

    raise AssertionError(
        f"provider-mode received unsupported index pattern: {indices!r}"
    )


def install_arkit_seams(pipe, b_seq, ypr_seq, student, device, dtype, *,
                        patch_pose=True, patch_motion=True, mf_log=None,
                        euler_signs=None, provider=None):
    """Patch pipe.pose_encoder + pipe.motion_encoder to consume ARKit data.

    Modes (mutually exclusive):
      array:    pass b_seq (T, 58) float32 + ypr_seq (T, 3) float32 radians,
                provider=None.
      provider: pass provider=callable(start, n) -> (b_chunk, ypr_chunk),
                with b_seq=ypr_seq=None. Caller swaps data per pipe-call
                (streaming daemon).

    student: trained MotEncoderStudent on device, dtype=float32.

    patch_pose: replace pose_encoder seams with closed-form ARKit path.
    patch_motion: replace motion_encoder driving-side calls with student.
    mf_log: dict with key 'records' (list) — if provided, every motion_encoder
        forward call appends {'role': 'ref'|'driving', 'mf': ndarray} so we
        can save teacher OR student m_f for analysis.
    """
    if provider is None:
        assert b_seq is not None and ypr_seq is not None, \
            "array-mode requires b_seq and ypr_seq"
    else:
        assert b_seq is None and ypr_seq is None, \
            "provider-mode forbids b_seq/ypr_seq"

    # State threaded through closures.
    ref_kp_canonical = {"kp": None, "t": None, "scale": None}
    chunk_cursor = {"i": 0}

    def _fetch(indices):
        """Return (b_chunk, ypr_chunk) for the given absolute frame indices."""
        if provider is not None:
            return _provider_fetch(provider, indices)
        idxs = np.clip(indices, 0, len(b_seq) - 1)
        return b_seq[idxs], ypr_seq[idxs]

    def cf_kd_for_indices(indices):
        """Run closed-form compose_kd for a list of frame indices."""
        kp_ref = ref_kp_canonical["kp"]   # (1, 21, 3) on device
        t_ref = ref_kp_canonical["t"]      # (1, 3)
        s_ref = ref_kp_canonical["scale"]  # (1, 1)
        sy, sp, sr = euler_signs if euler_signs is not None else EULER_SIGNS
        _, ypr = _fetch(indices)
        T = ypr.shape[0]
        Rs = []
        for k in range(T):
            R = euler_to_rotmat(
                torch.tensor(sy * ypr[k, 0].item()),
                torch.tensor(sp * ypr[k, 1].item()),
                torch.tensor(sr * ypr[k, 2].item()),
            )
            Rs.append(R)
        R = torch.stack(Rs, dim=0).to(device=device, dtype=dtype)  # (T, 3, 3)
        kp_ref_T = kp_ref.expand(T, -1, -1).to(device=device, dtype=dtype)
        s_T = s_ref.expand(T, -1).to(device=device, dtype=dtype)
        t_T = t_ref.expand(T, -1).to(device=device, dtype=dtype)
        return compose_kd(kp_ref_T, R, s_T, t_T)  # (T, 21, 3)

    def patched_interpolate_kps_online(self, ref, motion, num_interp,
                                       t_scale=0.5, s_scale=0):
        # Compute the canonical reference once from the actual ref RGB.
        kp1 = self.detector(ref.to(self.dtype))
        ref_kp_canonical["kp"] = kp1["kp"].reshape(1, -1, 3).detach()
        ref_kp_canonical["t"] = kp1["t"].detach()
        ref_kp_canonical["scale"] = kp1["scale"].detach()

        # Driving: motion has padding_num+1 frames; map to ARKit indices.
        # The first chunk receives padding_num+1 stand-in frames; we want
        # the *last* of those to be "frame 0" of our ARKit sequence and
        # the preceding (num_interp-1) to interpolate from ref pose to it.
        idxs = np.array([0] * (num_interp - 1) + list(range(motion.shape[0])))
        kp_intrep = cf_kd_for_indices(idxs)  # (n, 21, 3)
        kp_frame1 = self.detector(motion[:1].to(self.dtype))
        chunk_cursor["i"] = motion.shape[0]
        return kp_intrep, kp1, kp_frame1, None

    def patched_get_kps(self, kp_ref, kp_frame1, motion, t_scale=0.5, s_scale=0):
        start = chunk_cursor["i"]
        n = motion.shape[0]
        idxs = np.arange(start, start + n)
        kp_d = cf_kd_for_indices(idxs)
        chunk_cursor["i"] += n
        return kp_d, None

    if patch_pose:
        pipe.pose_encoder.interpolate_kps_online = MethodType(
            patched_interpolate_kps_online, pipe.pose_encoder
        )
        pipe.pose_encoder.get_kps = MethodType(
            patched_get_kps, pipe.pose_encoder
        )

    # motion_encoder seam: dispatch ref vs driving by time-dim.
    real_me_forward = pipe.motion_encoder.forward
    me_cursor = {"i": 0}

    def patched_me_forward(self, x):
        # x shape: (B, C, T, H, W). T==1 -> reference path; T>=2 -> driving.
        T = x.shape[2]
        if T == 1:
            mf = real_me_forward(x)
            if mf_log is not None:
                mf_log["records"].append({
                    "role": "ref", "start": -1,
                    "mf": mf.detach().cpu().float().numpy(),
                })
            return mf
        if patch_motion:
            # Driving: pull T b_expr starting at me_cursor.
            start = me_cursor["i"]
            idxs = np.arange(start, start + T)
            b_chunk, _ = _fetch(idxs)
            b = torch.from_numpy(np.asarray(b_chunk).astype(np.float32)).to(device)
            with torch.no_grad():
                mf = student(b)  # (T, 1, 32, 16)
            mf = mf.squeeze(1).unsqueeze(0)  # (1, T, 32, 16)
            me_cursor["i"] += T
            mf_out = mf.to(dtype=self.dtype)
        else:
            mf_out = real_me_forward(x)
            start = me_cursor["i"]
            me_cursor["i"] += T
        if mf_log is not None:
            mf_log["records"].append({
                "role": "driving", "start": int(start),
                "mf": mf_out.detach().cpu().float().numpy(),
            })
        return mf_out

    pipe.motion_encoder.forward = MethodType(patched_me_forward, pipe.motion_encoder)
