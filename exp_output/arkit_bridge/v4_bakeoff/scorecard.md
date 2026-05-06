# v4 Bake-off Scorecard

Heldout: data/arkit_bridge_pairs/holdout_v3 (822 pairs)

| Metric | v4a (A+B weighted_mse) | v4b (C varnorm_jvp) |
|---|---|---|
| `tail_recovery_median` | 0.9485 | 0.8379 |
| `n_channels_above_0_7` | 47.0000 | 58.0000 |
| `jvp_norm_match_median` | 0.3020 | 0.0504 |
| `heldout_r2_median` | 0.8928 | 0.7665 |
| `heldout_std_ratio_median` | 0.9586 | 0.8780 |

**Wins:** v4a=3, v4b=2

**Decision:** `tie_render_yaw_clip`
