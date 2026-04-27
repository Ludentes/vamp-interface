---
status: live
topic: corpus-rebalance
summary: First corpus-problem scan over the 4942-sample index (3502 training + 1440 overnight_drift). Surfaces detection failures, coverage gaps, per-seed variance, axis-atom trackers, and cross-base spread.
---

# Corpus problems — 2026-04-23 scan

Total samples: 5,038.

## MediaPipe detection failures

0 / 5038 (0.00%) samples have `_neutral > 0.99` or all-zero blendshapes.

## Coverage

192 overnight cells across (axis × subtag × base × scale). 0 cells have fewer than 5 seeds.

### Beard axis is structurally unbalanced

```
base    asian_m  elderly_latin_m  european_m
subtag                                      
add           4                0           4
remove        0                4           0
```

`beard/add` only covers 2 male bases; `beard/remove` only covers 1. Cross-base generalisation is not testable here; this axis needs at least `remove` on a second bearded base before slope fits mean much.

## Noisiest cells (per-seed std, top 10)

Cells with high per-seed std need more seeds before a mean is trustworthy.

| axis | subtag | base | scale | n | atom | std | mean |
|---|---|---|---|---|---|---|---|
| beard_rebalance | remove | european_m_bearded | +0.40 | 8 | atom_12 | 4.794 | 2.344 |
| beard_rebalance | remove | european_m_bearded | +0.00 | 8 | atom_12 | 3.949 | 2.078 |
| anger | anger | asian_m | +1.20 | 8 | atom_04 | 3.425 | 1.810 |
| beard_rebalance | remove | european_m_bearded | +0.70 | 8 | atom_12 | 3.241 | 1.679 |
| beard_rebalance | remove | european_m_bearded | +1.00 | 8 | atom_12 | 3.228 | 1.639 |
| anger | anger | asian_m | +0.80 | 8 | atom_04 | 2.317 | 1.058 |
| surprise | surprise | black_f | +1.20 | 8 | atom_09 | 2.030 | 1.653 |
| beard | add | european_m | +1.00 | 8 | atom_12 | 1.718 | 0.808 |
| surprise | surprise | european_m | +1.20 | 8 | atom_14 | 1.493 | 0.808 |
| surprise | surprise | southasian_f | +1.20 | 8 | atom_09 | 1.422 | 1.132 |

## Axis → atom tracker

For each (axis, subtag) the 3 atoms with the largest median |slope| vs scale across bases. An axis with all small slopes has no atom that tracks it — raw channels will be needed.

### anger / anger

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_15 | 0.822 | 0.961 | 0.973 |
| atom_18 | 0.281 | 0.778 | 0.516 |
| atom_04 | 0.223 | 0.902 | 0.409 |

### beard / add

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_05 | 0.780 | 0.774 | 0.780 |
| atom_12 | 0.422 | 0.586 | 0.422 |
| atom_09 | 0.419 | 0.559 | 0.414 |

### beard / remove

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_15 | 0.210 | 0.958 | 0.210 |
| atom_04 | 0.145 | 0.744 | 0.151 |
| atom_05 | 0.137 | 0.552 | 0.173 |

### beard_rebalance / remove

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_05 | 0.171 | 0.871 | 0.421 |
| atom_12 | 0.118 | 0.520 | 0.479 |
| atom_09 | 0.100 | 0.824 | 0.123 |

### pucker / pucker

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_17 | 0.275 | 0.911 | 0.343 |
| atom_05 | 0.172 | 0.600 | 0.230 |
| atom_09 | 0.136 | 0.768 | 0.179 |

### smile / broad

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_16 | 1.277 | 0.817 | 1.324 |
| atom_17 | 0.411 | 0.682 | 0.426 |
| atom_09 | 0.337 | 0.588 | 0.380 |

### smile / faint

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_07 | 0.978 | 0.895 | 0.906 |
| atom_17 | 0.226 | 0.868 | 0.235 |
| atom_18 | 0.181 | 0.792 | 0.214 |

### smile / manic

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_19 | 0.813 | 0.950 | 0.780 |
| atom_16 | 0.423 | 0.493 | 0.539 |
| atom_17 | 0.409 | 0.673 | 0.423 |

### smile / warm

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_16 | 0.958 | 0.882 | 0.962 |
| atom_17 | 0.403 | 0.723 | 0.414 |
| atom_07 | 0.223 | 0.751 | 0.242 |

### surprise / surprise

| atom | median \|slope\| | median R² | median range |
|---|---|---|---|
| atom_09 | 0.735 | 0.896 | 0.897 |
| atom_01 | 0.507 | 0.850 | 0.615 |
| atom_18 | 0.261 | 0.783 | 0.308 |

## Cross-base response spread (top 15)

At these cells the response-atom's mean differs most across bases. Large spread → per-base calibration strictly needed, single global scale won't hit target uniformly.

| axis | subtag | scale | n_bases | atom | range | mean |
|---|---|---|---|---|---|---|
| beard_rebalance | remove | +0.40 | 3 | atom_12 | 2.046 | 1.096 |
| beard_rebalance | remove | +0.00 | 3 | atom_12 | 1.871 | 0.838 |
| anger | anger | +1.20 | 6 | atom_04 | 1.638 | 0.583 |
| beard_rebalance | remove | +0.70 | 3 | atom_12 | 1.512 | 0.717 |
| beard_rebalance | remove | +1.00 | 3 | atom_12 | 1.414 | 0.716 |
| surprise | surprise | +1.20 | 6 | atom_09 | 1.387 | 0.927 |
| beard | add | +1.00 | 2 | atom_05 | 1.216 | 1.088 |
| smile | faint | +1.00 | 6 | atom_07 | 1.066 | 1.045 |
| smile | faint | +0.70 | 6 | atom_07 | 0.991 | 0.987 |
| beard | add | +0.70 | 2 | atom_05 | 0.986 | 0.981 |
| anger | anger | +0.80 | 6 | atom_04 | 0.920 | 0.525 |
| anger | anger | +0.40 | 6 | atom_18 | 0.859 | 0.417 |
| beard | add | +0.40 | 2 | atom_05 | 0.853 | 0.546 |
| pucker | pucker | +0.80 | 6 | atom_09 | 0.818 | 0.266 |
| surprise | surprise | +0.80 | 6 | atom_09 | 0.783 | 0.658 |
