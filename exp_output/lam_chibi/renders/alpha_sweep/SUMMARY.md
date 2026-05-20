# α × chibi-strength sweep — analytical lid-boost generalization

Sweep launched 2026-05-14 ~13:24, completed 13:40 (16 min wall; ~70 s/render).

## Formula tested

```
mask     = blink-displacement verts (1662 / 20018, animation-derived, deformation-invariant)
strength = mean(|det J|) over mask     # per-deformation scalar
boost    = max(1, strength^α)          # single constant α across all chibi strengths
scale_ratio_final = chibi_scale_ratio · boost   # multiply on lid mask only
```

Mask reused from `me_s2.0/lid_mask_20018.npy` (vertex indices invariant under chibi).

## Per-cell numbers

| s    | mean&#124;det J&#124; @ lid | α=0.40 boost | α=0.50 boost | α=0.69 boost | r[lid] med (v2 base) | total scale_ratio[lid] @ α=0.50 |
|------|---------------:|-----:|-----:|-----:|-----:|-----:|
| 1.5  | 2.226 | 1.377 | 1.492 | 1.737 | 1.299 | 1.94 |
| 1.75 | 2.488 | 1.440 | 1.577 | 1.876 | 1.348 | 2.13 |
| 2.0  | 2.767 | 1.503 | 1.664 | 2.018 | 1.397 | 2.33 |
| 2.5  | 3.379 | 1.628 | 1.838 | 2.317 | 1.494 | 2.75 |

The formula auto-scales the lid boost with chibi strength as designed: at s=1.5 boost is 1.38–1.74×, at s=2.5 it rises to 1.63–2.32×.

## Calibration

α=0.50 at s=2.0 → boost 1.66, which sits at the upper end of the working window confirmed on the prior dedicated sweep (1.5×–2.0× pass normal-render gate). α=0.40 → 1.50 (floor).

## Renders

12 full-take normal videos: `s{S}_a{A}.mp4` for S ∈ {1.5, 1.75, 2.0, 2.5} and A ∈ {0.40, 0.50, 0.69}.

Frame-328 collage: `alpha_sweep_collage.png` (blink frame, but lower face cropped by the diag-tile sampler — judge from the videos themselves, not the collage).

## Decision template

Pick the α whose s=2.0 video matches the prior lid_1.75_normal.mp4 quality, then sanity-check that the same α renders at s=1.5, 1.75, 2.5 don't regress (smear at large s, leak at small s).

Stale flag remains on `docs/research/2026-05-14-chibi-splat-scale-fix.md` and topic `_topics/lam-chibi-recipe.md` until α decision lands.
