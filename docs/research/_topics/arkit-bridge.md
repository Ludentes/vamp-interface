## ARKit → PersonaLive bridge

**Status:** shipped. Closed 2026-05-06. ARKit Live Link Face b_61 +
HeadYaw/Pitch/Roll → PersonaLive's implicit-keypoint head + closed-form
pose, no learned pose path. The student replaces only the motion
encoder; pose is closed-form.

**Final config**:
- Student: `runs/student_v2_120k/student_best.pt`
  (ratio_mean 0.0114 on holdout_v3, structural floor confirmed)
- Pose: `src/arkit_bridge/closed_form_pose.py` —
  `EULER_SIGNS=(+1,-1,+1)`, `F_KP_REF=I`
- Render driver: `scripts/apply_bridge_to_personalive.py`. Raw .mov →
  default `--rotate_iphone=True`; transcoded mp4 → `--no_rotate_iphone`.
  Flag only matters in `teacher_full`/`teacher_motion` modes; bridge
  mode is rotation-invariant.

### Read-first

- [`2026-05-06-arkit-bridge-final-pose-config.md`](../2026-05-06-arkit-bridge-final-pose-config.md)
  — wrap-up, full lessons learned. **Start here on any new session
  touching ARKit-bridge.**
- [`2026-05-06-vtuber-pipeline-priorities.md`](../2026-05-06-vtuber-pipeline-priorities.md)
  — strategic framing: Regime A vs Regime B, four products, acceptance
  gates per product. Read this before scoping any new work that
  consumes the bridge.
- [`2026-05-06-rendering-stack-replacement-options.md`](../2026-05-06-rendering-stack-replacement-options.md)
  — backbone-swap survey (FLUX, Z-Image, FasterLivePortrait, Hyper-SD)
  for when PersonaLive's photoreal/stylized tradeoffs become a blocker.

### Streaming pipeline (LLF → OBS)

- Spec: [`docs/superpowers/specs/2026-05-06-llf-to-obs-pipeline-design.md`](../../superpowers/specs/2026-05-06-llf-to-obs-pipeline-design.md)
- Plan: [`docs/superpowers/plans/2026-05-06-llf-obs-streaming.md`](../../superpowers/plans/2026-05-06-llf-obs-streaming.md)
  (Task 0 acceptance gates + V1 batch + V2 cohort-streaming + next-product
  roadmap)
- Streaming research notes: [`2026-05-06-rain-streaming-research.md`](../2026-05-06-rain-streaming-research.md)
  — PersonaLive *is* a RAIN+StreamDiffusion pipeline; cohort latency is
  intrinsic, V2 is mechanical state-hoist.
- LLF UDP wire-format verification: [`2026-05-07-llf-udp-protocol-verification.md`](../2026-05-07-llf-udp-protocol-verification.md)
  — `llf_udp.py` float decoding is correct (tail-slice 244 bytes); the
  prefix layout doc and `subject` extraction are wrong (real prefix is
  45 bytes: 4-byte LE version + 37-byte UUID, name length at offset 41).
  Functional impact: `subject` is always "?". Float payload, port
  11111, 52+9 ordering all confirmed against PyLiveLinkFace + UE forum.

### Current beliefs

- The 0.0114 ratio_mean is a structural floor for the
  (MotEncoderStudent, b_expr → m_f, varnorm_std_tail loss, this corpus)
  tuple. Loss design (v4 family), function-class kernel methods,
  mirror-flip augmentation (v5), and LR/seed/schedule sweeps all
  landed within seed noise. See
  [`2026-05-06-arkit-bridge-lr-seed-sweep.md`](../2026-05-06-arkit-bridge-lr-seed-sweep.md),
  [`2026-05-06-arkit-bridge-v5-falsified.md`](../2026-05-06-arkit-bridge-v5-falsified.md),
  [`2026-05-06-arkit-bridge-function-class-pareto.md`](../2026-05-06-arkit-bridge-function-class-pareto.md).
- Offline ratio_mean does not capture rendered-quality regressions
  (e.g. the OLD pose config crushed pitch amplitude to 13% of teacher
  with no offline signal). Rendered-side metrics in
  `render_metrics.parquet` are the gate that matters; offline ratio is
  the convergence signal.
- F-conjugation in `compose_kd` is a no-op now (F = I). The earlier
  `F = diag(1,-1,1)` from calibration v3 was confounded by the
  rotation-flag bug and couples pitch/roll signs together — fixed
  by per-axis EULER_SIGNS with F = I.

### Falsified

- Cached-δ replay editing (cross-thread, but same architectural
  conclusion): editing must go through live forward passes
- v4 loss-design tweaks beat v2 baseline (none did, within seed noise)
- v5 mirror-flip + p99 filter as hard targets: PersonaLive MotEncoder
  is not flip-equivariant; flipped pairs as hard targets are
  contaminated. Use soft-consistency loss instead.
- Calibration v3's `F* = diag(1,-1,1)` as the right correction —
  confounded by the rotation-flag bug

### Companion docs

- [`2026-05-05-arkit-bridge-v1-design.md`](../2026-05-05-arkit-bridge-v1-design.md)
  / [`2026-05-05-arkit-bridge-v1-readout.md`](../2026-05-05-arkit-bridge-v1-readout.md)
  / [`2026-05-05-arkit-bridge-v1-viability.md`](../2026-05-05-arkit-bridge-v1-viability.md)
- [`2026-05-05-arkit-bridge-v2-loss-redesign.md`](../2026-05-05-arkit-bridge-v2-loss-redesign.md)
- [`2026-05-05-arkit-bridge-parquet-plan.md`](../2026-05-05-arkit-bridge-parquet-plan.md)
- [`2026-05-06-arkit-bridge-next-experiments.md`](../2026-05-06-arkit-bridge-next-experiments.md)
- [`2026-05-06-arkit-vs-mediapipe-pose-conventions.md`](../2026-05-06-arkit-vs-mediapipe-pose-conventions.md)
- [`2026-05-06-yaw-sign-flip-fix.md`](../2026-05-06-yaw-sign-flip-fix.md)
  (superseded by final-pose-config doc above)
