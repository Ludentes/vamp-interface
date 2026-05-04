# v2c vs v2e

- v2c: `output/mediapipe_distill/validation_report_v2c.json`
- v2e: `output/mediapipe_distill/validation_report_v2e.json`

## Atom shippability (8 NMF atoms)

| atom_idx | tag | v2c R² | tier | v2e R² | tier | Δ |
|---:|---|---:|---|---:|---|---:|
| atom | smile_inphase | 0.779 | confident_ship | 0.740 | confident_ship | -0.039 |
| atom | jaw_inphase | 0.870 | confident_ship | 0.868 | confident_ship | -0.002 |
| atom | alpha_interp_attn | 0.840 | confident_ship | 0.871 | confident_ship | +0.031 |
| atom | anger_rebalance | 0.830 | confident_ship | 0.868 | confident_ship | +0.038 |
| atom | surprise_rebalance | 0.795 | confident_ship | 0.777 | confident_ship | -0.019 |
| atom | disgust_rebalance | 0.946 | confident_ship | 0.942 | confident_ship | -0.004 |
| atom | pucker_rebalance | 0.964 | confident_ship | 0.964 | confident_ship | +0.000 |
| atom | lip_press_rebalance | 0.852 | confident_ship | 0.836 | confident_ship | -0.016 |

## Channel shippability (52 channels)

| channel | v2c R² | tier | v2e R² | tier | Δ | move |
|---|---:|---|---:|---|---:|---|
| _neutral | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| browDownLeft | 0.789 | confident_ship | 0.827 | confident_ship | +0.038 | = stable ship |
| browDownRight | 0.802 | confident_ship | 0.845 | confident_ship | +0.043 | = stable ship |
| browInnerUp | 0.775 | confident_ship | 0.817 | confident_ship | +0.042 | = stable ship |
| browOuterUpLeft | 0.787 | confident_ship | 0.807 | confident_ship | +0.020 | = stable ship |
| browOuterUpRight | 0.778 | confident_ship | 0.788 | confident_ship | +0.010 | = stable ship |
| cheekPuff | -10.525 | do_not_ship | 0.573 | ship | +11.097 | ↑ promoted |
| cheekSquintLeft | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| cheekSquintRight | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| eyeBlinkLeft | 0.839 | confident_ship | 0.817 | confident_ship | -0.023 | = stable ship |
| eyeBlinkRight | 0.829 | confident_ship | 0.796 | confident_ship | -0.033 | = stable ship |
| eyeLookDownLeft | 0.827 | confident_ship | 0.813 | confident_ship | -0.014 | = stable ship |
| eyeLookDownRight | 0.825 | confident_ship | 0.822 | confident_ship | -0.003 | = stable ship |
| eyeLookInLeft | 0.870 | confident_ship | 0.857 | confident_ship | -0.013 | = stable ship |
| eyeLookInRight | 0.873 | confident_ship | 0.870 | confident_ship | -0.003 | = stable ship |
| eyeLookOutLeft | 0.881 | confident_ship | 0.865 | confident_ship | -0.017 | = stable ship |
| eyeLookOutRight | 0.861 | confident_ship | 0.852 | confident_ship | -0.008 | = stable ship |
| eyeLookUpLeft | 0.780 | confident_ship | 0.765 | confident_ship | -0.015 | = stable ship |
| eyeLookUpRight | 0.787 | confident_ship | 0.778 | confident_ship | -0.009 | = stable ship |
| eyeSquintLeft | 0.856 | confident_ship | 0.820 | confident_ship | -0.036 | = stable ship |
| eyeSquintRight | 0.865 | confident_ship | 0.806 | confident_ship | -0.059 | = stable ship |
| eyeWideLeft | 0.780 | confident_ship | 0.782 | confident_ship | +0.002 | = stable ship |
| eyeWideRight | 0.777 | confident_ship | 0.714 | confident_ship | -0.063 | = stable ship |
| jawForward | 0.112 | do_not_ship | 0.698 | ship | +0.586 | ↑ promoted |
| jawLeft | 0.044 | do_not_ship | 0.067 | do_not_ship | +0.023 | = no-ship |
| jawOpen | 0.785 | confident_ship | 0.777 | confident_ship | -0.008 | = stable ship |
| jawRight | -0.066 | do_not_ship | -0.256 | do_not_ship | -0.190 | = no-ship |
| mouthClose | 0.022 | do_not_ship | 0.334 | do_not_ship | +0.312 | = no-ship |
| mouthDimpleLeft | 0.478 | do_not_ship | 0.587 | ship | +0.108 | ↑ promoted |
| mouthDimpleRight | 0.448 | do_not_ship | 0.473 | do_not_ship | +0.025 | = no-ship |
| mouthFrownLeft | 0.061 | do_not_ship | 0.259 | do_not_ship | +0.198 | = no-ship |
| mouthFrownRight | 0.070 | do_not_ship | 0.333 | do_not_ship | +0.263 | = no-ship |
| mouthFunnel | 0.583 | ship | 0.665 | ship | +0.082 | = stable ship |
| mouthLeft | 0.009 | do_not_ship | 0.039 | do_not_ship | +0.030 | = no-ship |
| mouthLowerDownLeft | 0.743 | confident_ship | 0.693 | ship | -0.050 | ↓ demoted |
| mouthLowerDownRight | 0.793 | confident_ship | 0.762 | confident_ship | -0.031 | = stable ship |
| mouthPressLeft | 0.688 | ship | 0.667 | ship | -0.021 | = stable ship |
| mouthPressRight | 0.721 | confident_ship | 0.706 | confident_ship | -0.015 | = stable ship |
| mouthPucker | 0.791 | confident_ship | 0.773 | confident_ship | -0.018 | = stable ship |
| mouthRight | 0.008 | do_not_ship | -0.086 | do_not_ship | -0.094 | = no-ship |
| mouthRollLower | 0.707 | confident_ship | 0.715 | confident_ship | +0.008 | = stable ship |
| mouthRollUpper | 0.605 | ship | 0.614 | ship | +0.009 | = stable ship |
| mouthShrugLower | 0.727 | confident_ship | 0.722 | confident_ship | -0.006 | = stable ship |
| mouthShrugUpper | 0.591 | ship | 0.584 | ship | -0.007 | = stable ship |
| mouthSmileLeft | 0.964 | confident_ship | 0.964 | confident_ship | +0.001 | = stable ship |
| mouthSmileRight | 0.963 | confident_ship | 0.963 | confident_ship | +0.000 | = stable ship |
| mouthStretchLeft | 0.594 | ship | 0.650 | ship | +0.056 | = stable ship |
| mouthStretchRight | 0.642 | ship | 0.673 | ship | +0.032 | = stable ship |
| mouthUpperUpLeft | 0.943 | confident_ship | 0.939 | confident_ship | -0.004 | = stable ship |
| mouthUpperUpRight | 0.945 | confident_ship | 0.936 | confident_ship | -0.008 | = stable ship |
| noseSneerLeft | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| noseSneerRight | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |

## Summary
- channels promoted: **3**
- channels demoted: **1**
- stable ship: 35
- stable no-ship: 8
- stable degenerate: 5

### Aggregate
| metric | v2c | v2e | Δ |
|---|---:|---:|---:|
| val_mse | 0.0030 | 0.0031 | +0.0001 |
| r2_median | 0.7783 | 0.7648 | -0.0135 |
| r2_mean | 0.4048 | 0.6652 | +0.2603 |
| frac_above_0p5 | 0.7660 | 0.8298 | +0.0638 |
| frac_above_0p7 | 0.6383 | 0.6170 | -0.0213 |
