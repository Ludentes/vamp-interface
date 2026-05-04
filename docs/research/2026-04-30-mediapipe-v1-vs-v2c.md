# v1 vs v2c

- v1: `output/mediapipe_distill/validation_report_v1_hflipfix.json`
- v2c: `output/mediapipe_distill/validation_report_v2c.json`

## Atom shippability (8 NMF atoms)

| atom_idx | tag | v1 R² | tier | v2c R² | tier | Δ |
|---:|---|---:|---|---:|---|---:|
| atom | smile_inphase | 0.717 | confident_ship | 0.779 | confident_ship | +0.062 |
| atom | jaw_inphase | 0.873 | confident_ship | 0.870 | confident_ship | -0.002 |
| atom | alpha_interp_attn | 0.810 | confident_ship | 0.840 | confident_ship | +0.030 |
| atom | anger_rebalance | 0.814 | confident_ship | 0.830 | confident_ship | +0.016 |
| atom | surprise_rebalance | 0.792 | confident_ship | 0.795 | confident_ship | +0.003 |
| atom | disgust_rebalance | 0.937 | confident_ship | 0.946 | confident_ship | +0.009 |
| atom | pucker_rebalance | 0.949 | confident_ship | 0.964 | confident_ship | +0.015 |
| atom | lip_press_rebalance | 0.851 | confident_ship | 0.852 | confident_ship | +0.001 |

## Channel shippability (52 channels)

| channel | v1 R² | tier | v2c R² | tier | Δ | move |
|---|---:|---|---:|---|---:|---|
| _neutral | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| browDownLeft | 0.782 | confident_ship | 0.789 | confident_ship | +0.008 | = stable ship |
| browDownRight | 0.763 | confident_ship | 0.802 | confident_ship | +0.039 | = stable ship |
| browInnerUp | 0.827 | confident_ship | 0.775 | confident_ship | -0.052 | = stable ship |
| browOuterUpLeft | 0.712 | confident_ship | 0.787 | confident_ship | +0.075 | = stable ship |
| browOuterUpRight | 0.563 | ship | 0.778 | confident_ship | +0.216 | ↑ promoted |
| cheekPuff | -12.491 | do_not_ship | -10.525 | do_not_ship | +1.966 | = no-ship |
| cheekSquintLeft | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| cheekSquintRight | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| eyeBlinkLeft | 0.801 | confident_ship | 0.839 | confident_ship | +0.038 | = stable ship |
| eyeBlinkRight | 0.823 | confident_ship | 0.829 | confident_ship | +0.005 | = stable ship |
| eyeLookDownLeft | 0.818 | confident_ship | 0.827 | confident_ship | +0.009 | = stable ship |
| eyeLookDownRight | 0.816 | confident_ship | 0.825 | confident_ship | +0.009 | = stable ship |
| eyeLookInLeft | 0.819 | confident_ship | 0.870 | confident_ship | +0.051 | = stable ship |
| eyeLookInRight | 0.847 | confident_ship | 0.873 | confident_ship | +0.025 | = stable ship |
| eyeLookOutLeft | 0.858 | confident_ship | 0.881 | confident_ship | +0.023 | = stable ship |
| eyeLookOutRight | 0.801 | confident_ship | 0.861 | confident_ship | +0.060 | = stable ship |
| eyeLookUpLeft | 0.765 | confident_ship | 0.780 | confident_ship | +0.015 | = stable ship |
| eyeLookUpRight | 0.772 | confident_ship | 0.787 | confident_ship | +0.015 | = stable ship |
| eyeSquintLeft | 0.844 | confident_ship | 0.856 | confident_ship | +0.012 | = stable ship |
| eyeSquintRight | 0.863 | confident_ship | 0.865 | confident_ship | +0.002 | = stable ship |
| eyeWideLeft | 0.602 | ship | 0.780 | confident_ship | +0.178 | ↑ promoted |
| eyeWideRight | 0.666 | ship | 0.777 | confident_ship | +0.110 | ↑ promoted |
| jawForward | -0.102 | do_not_ship | 0.112 | do_not_ship | +0.213 | = no-ship |
| jawLeft | 0.041 | do_not_ship | 0.044 | do_not_ship | +0.003 | = no-ship |
| jawOpen | 0.780 | confident_ship | 0.785 | confident_ship | +0.005 | = stable ship |
| jawRight | -0.049 | do_not_ship | -0.066 | do_not_ship | -0.017 | = no-ship |
| mouthClose | 0.005 | do_not_ship | 0.022 | do_not_ship | +0.017 | = no-ship |
| mouthDimpleLeft | 0.251 | do_not_ship | 0.478 | do_not_ship | +0.227 | = no-ship |
| mouthDimpleRight | 0.269 | do_not_ship | 0.448 | do_not_ship | +0.179 | = no-ship |
| mouthFrownLeft | 0.056 | do_not_ship | 0.061 | do_not_ship | +0.005 | = no-ship |
| mouthFrownRight | 0.044 | do_not_ship | 0.070 | do_not_ship | +0.025 | = no-ship |
| mouthFunnel | 0.479 | do_not_ship | 0.583 | ship | +0.104 | ↑ promoted |
| mouthLeft | 0.003 | do_not_ship | 0.009 | do_not_ship | +0.006 | = no-ship |
| mouthLowerDownLeft | 0.689 | ship | 0.743 | confident_ship | +0.054 | ↑ promoted |
| mouthLowerDownRight | 0.752 | confident_ship | 0.793 | confident_ship | +0.041 | = stable ship |
| mouthPressLeft | 0.590 | ship | 0.688 | ship | +0.098 | = stable ship |
| mouthPressRight | 0.429 | do_not_ship | 0.721 | confident_ship | +0.292 | ↑ promoted |
| mouthPucker | 0.787 | confident_ship | 0.791 | confident_ship | +0.004 | = stable ship |
| mouthRight | 0.021 | do_not_ship | 0.008 | do_not_ship | -0.013 | = no-ship |
| mouthRollLower | 0.665 | ship | 0.707 | confident_ship | +0.042 | ↑ promoted |
| mouthRollUpper | 0.334 | do_not_ship | 0.605 | ship | +0.272 | ↑ promoted |
| mouthShrugLower | 0.694 | ship | 0.727 | confident_ship | +0.033 | ↑ promoted |
| mouthShrugUpper | 0.550 | ship | 0.591 | ship | +0.041 | = stable ship |
| mouthSmileLeft | 0.955 | confident_ship | 0.964 | confident_ship | +0.009 | = stable ship |
| mouthSmileRight | 0.940 | confident_ship | 0.963 | confident_ship | +0.023 | = stable ship |
| mouthStretchLeft | 0.618 | ship | 0.594 | ship | -0.024 | = stable ship |
| mouthStretchRight | 0.615 | ship | 0.642 | ship | +0.027 | = stable ship |
| mouthUpperUpLeft | 0.938 | confident_ship | 0.943 | confident_ship | +0.005 | = stable ship |
| mouthUpperUpRight | 0.930 | confident_ship | 0.945 | confident_ship | +0.015 | = stable ship |
| noseSneerLeft | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |
| noseSneerRight | 0.000 | degenerate | 0.000 | degenerate | +0.000 | = degenerate |

## Summary
- channels promoted: **9**
- channels demoted: **0**
- stable ship: 27
- stable no-ship: 11
- stable degenerate: 5

### Aggregate
| metric | v1 | v2c | Δ |
|---|---:|---:|---:|
| val_mse | 0.0036 | 0.0030 | -0.0006 |
| r2_median | 0.6940 | 0.7783 | +0.0843 |
| r2_mean | 0.3093 | 0.4048 | +0.0956 |
| frac_above_0p5 | 0.7021 | 0.7660 | +0.0638 |
| frac_above_0p7 | 0.4894 | 0.6383 | +0.1489 |
