# axis report: v3_lam10

ckpt: `runs/student_v3_lam10/student_best.pt`

## summary
- **amp_geomean**: 0.4314
- **amp_pass_frac**: 0.5357
- **r_pass_frac**: 0.7500
- **r2_ge_0p7_frac**: 1.0000
- **region_p90_mean_per255**: 49.5010
- **score**: 0.5664

## input channels (sorted by amp_med ascending — worst first)

| status | name | category | amp_med | amp_iqr | abs_r_med | driv_std | rend_std |
|---|---|---|---|---|---|---|---|
| BROKEN | browDownLeft | brow | 0.023 | 0.036 | 0.164 | 0.1537 | 0.0035 |
| BROKEN | mouthLowerDownLeft | mouth | 0.032 | 0.212 | 0.899 | 0.2488 | 0.0047 |
| BROKEN | browDownRight | brow | 0.035 | 0.275 | 0.084 | 0.1297 | 0.0045 |
| BROKEN | mouthLowerDownRight | mouth | 0.044 | 0.242 | 0.912 | 0.2714 | 0.0119 |
| BROKEN | browOuterUpRight | brow | 0.080 | 0.874 | 0.186 | 0.1286 | 0.0155 |
| WEAK | mouthClose | mouth | 0.206 | 0.085 | 0.224 | 0.0158 | 0.0033 |
| WEAK | mouthSmileLeft | mouth | 0.238 | 0.345 | 0.721 | 0.0122 | 0.0005 |
| WEAK | eyeLookDownRight | eye | 0.246 | 0.077 | 0.595 | 0.0815 | 0.0210 |
| WEAK | eyeLookDownLeft | eye | 0.269 | 0.104 | 0.563 | 0.0886 | 0.0238 |
| WEAK | browOuterUpLeft | brow | 0.295 | 1.219 | 0.254 | 0.1120 | 0.0330 |
| WEAK | eyeBlinkLeft | eye | 0.320 | 0.180 | 0.285 | 0.0423 | 0.0134 |
| WEAK | eyeWideRight | eye | 0.391 | 0.412 | 0.433 | 0.0117 | 0.0019 |
| PASS | eyeSquintRight | eye | 0.565 | 0.074 | 0.504 | 0.0846 | 0.0575 |
| PASS | mouthDimpleLeft | mouth | 0.583 | 0.359 | 0.907 | 0.0341 | 0.0212 |
| PASS | eyeLookOutRight | eye | 0.644 | 0.195 | 0.514 | 0.0154 | 0.0072 |
| PASS | mouthDimpleRight | mouth | 0.647 | 2.449 | 0.902 | 0.0318 | 0.0050 |
| PASS | eyeLookInLeft | eye | 0.660 | 0.472 | 0.428 | 0.0190 | 0.0117 |
| PASS | mouthFunnel | mouth | 0.692 | 0.340 | 0.795 | 0.0100 | 0.0075 |
| PASS | eyeBlinkRight | eye | 0.756 | 0.096 | 0.309 | 0.0233 | 0.0192 |
| PASS | jawOpen | jaw | 0.857 | 0.472 | 0.979 | 0.2242 | 0.2679 |
| PASS | mouthSmileRight | mouth | 0.881 | 0.528 | 0.669 | 0.0101 | 0.0007 |
| PASS | eyeSquintLeft | eye | 0.966 | 0.279 | 0.390 | 0.0721 | 0.0533 |
| PASS | eyeLookInRight | eye | 1.599 | 0.386 | 0.469 | 0.0224 | 0.0309 |
| PASS | eyeLookUpRight | eye | 1.762 | 0.756 | 0.384 | 0.0186 | 0.0492 |
| WEAK | eyeLookOutLeft | eye | 2.453 | 0.448 | 0.277 | 0.0165 | 0.0384 |
| PASS | eyeLookUpLeft | eye | 2.524 | 1.270 | 0.541 | 0.0198 | 0.0549 |
| PASS | browInnerUp | brow | 2.923 | 5.637 | 0.326 | 0.0486 | 0.1149 |
| PASS | mouthPucker | mouth | 3.275 | 3.973 | 0.506 | 0.0648 | 0.2123 |
| INACTIVE | mouthFrownLeft | mouth | 0.042 | 0.023 | 0.079 | 0.0043 | 0.0001 |
| INACTIVE | mouthFrownRight | mouth | 0.045 | 0.036 | 0.350 | 0.0060 | 0.0002 |
| INACTIVE | jawLeft | jaw | 0.060 | 0.020 | 0.596 | 0.0054 | 0.0003 |
| INACTIVE | cheekPuff | cheek | 0.065 | 0.110 | 0.977 | 0.0004 | 0.0000 |
| INACTIVE | mouthStretchLeft | mouth | 0.086 | 0.051 | 0.377 | 0.0073 | 0.0002 |
| INACTIVE | mouthLeft | mouth | 0.114 | 0.051 | 0.578 | 0.0056 | 0.0008 |
| INACTIVE | jawForward | jaw | 0.134 | 0.042 | 0.974 | 0.0005 | 0.0001 |
| INACTIVE | cheekSquintRight | cheek | 0.165 | 0.027 | 0.873 | 0.0000 | 0.0000 |
| INACTIVE | cheekSquintLeft | cheek | 0.173 | 0.099 | 0.862 | 0.0000 | 0.0000 |
| INACTIVE | mouthUpperUpLeft | mouth | 0.187 | 0.105 | 0.815 | 0.0054 | 0.0002 |
| INACTIVE | noseSneerRight | nose | 0.207 | 0.116 | 0.925 | 0.0000 | 0.0000 |
| INACTIVE | noseSneerLeft | nose | 0.266 | 0.303 | 0.881 | 0.0000 | 0.0000 |
| INACTIVE | mouthUpperUpRight | mouth | 0.267 | 0.342 | 0.835 | 0.0032 | 0.0001 |
| INACTIVE | mouthStretchRight | mouth | 0.282 | 0.362 | 0.548 | 0.0029 | 0.0015 |
| INACTIVE | mouthPressLeft | mouth | 0.678 | 0.159 | 0.376 | 0.0086 | 0.0037 |
| INACTIVE | eyeWideLeft | eye | 0.717 | 0.304 | 0.371 | 0.0049 | 0.0024 |
| INACTIVE | mouthPressRight | mouth | 0.836 | 1.472 | 0.161 | 0.0042 | 0.0131 |
| INACTIVE | mouthShrugUpper | mouth | 1.012 | 0.397 | 0.428 | 0.0017 | 0.0018 |
| INACTIVE | mouthShrugLower | mouth | 2.155 | 8.612 | 0.430 | 0.0015 | 0.0052 |
| INACTIVE | mouthRollUpper | mouth | 2.368 | 1.041 | 0.326 | 0.0045 | 0.0086 |
| INACTIVE | mouthRollLower | mouth | 2.470 | 1.108 | 0.489 | 0.0043 | 0.0106 |
| INACTIVE | jawRight | jaw | 4.281 | 0.816 | 0.788 | 0.0010 | 0.0043 |
| INACTIVE | mouthRight | mouth | 24.279 | 16.332 | 0.033 | 0.0000 | 0.0006 |

## bottom-15 output cells by R²

| row | col | R² |
|---|---|---|
| 7 | 3 | +0.7235 |
| 29 | 6 | +0.7259 |
| 18 | 12 | +0.7332 |
| 18 | 7 | +0.7560 |
| 15 | 8 | +0.7656 |
| 18 | 1 | +0.7673 |
| 7 | 9 | +0.7723 |
| 7 | 7 | +0.7737 |
| 17 | 4 | +0.7742 |
| 1 | 4 | +0.7762 |
| 30 | 7 | +0.7803 |
| 14 | 1 | +0.7881 |
| 10 | 13 | +0.7889 |
| 21 | 10 | +0.7901 |
| 16 | 6 | +0.7915 |

## face regions (pixel abs diff /255, teacher vs bridge, averaged over takes)

| region | mean_avg | p90_avg | status |
|---|---|---|---|
| brow | 23.91 | 32.45 | OK |
| eye_l | 50.10 | 64.75 | HIGH |
| eye_r | 43.55 | 60.74 | HIGH |
| mouth | 33.80 | 40.06 | HIGH |

## per-take regions

| take | brow | eye_l | eye_r | mouth | global |
|---|---|---|---|---|---|
| 20260505_MySlate_2 | 15.70 | 35.52 | 31.84 | 25.46 | 5.01 |
| 20260505_MySlate_3 | 24.95 | 62.49 | 52.49 | 41.43 | 7.14 |
| 20260505_MySlate_8 | 31.07 | 52.30 | 46.33 | 34.50 | 12.19 |