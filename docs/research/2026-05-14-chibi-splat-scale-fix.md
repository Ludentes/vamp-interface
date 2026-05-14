---
status: live
topic: lam-chibi-recipe
---

# Chibi-mesh splat-scale compensation (eye-occlusion fix)

**Date:** 2026-05-14
**Status:** Shipped. Closes the iris-through-lid artifact that has tracked the chibi recipe since the recipe first ran end-to-end.

## TL;DR

LAM's `GSLayer` predicts per-vertex Gaussian splat sigmas at the canonical FLAME vertex density. When the chibi recipe overrides `_gm.xyz` with a 1.4×-radially-stretched 20018-vert mesh, those sigmas no longer tile the surface — gaps between splats open up wherever the local edge length grew. The eye region is the worst-case: lid splats no longer overlap densely enough to occlude the eyeball, so the iris reads through partially-closed lids and the eyeball itself reads as a "tiny dot in an oversized socket."

Fix: emit a per-vertex `chibi_scale_ratio.npy = mean_edge_chibi / mean_edge_original` (over the 20018-vert subdivided template), and multiply LAM's `_gm.scaling` by this ratio at runtime. One env var — `LAM_CHIBI_SCALE_RATIO=<path>` — wired into the same hook point as the existing `LAM_EDIT_XYZ_OBJ` override. Closes blink cleanly, sizes irises proportionally, preserves cheek/forehead detail (which v1 uniform-boost did not — see falsification ladder below).

## Falsification ladder

This was the leading hypothesis after seven other candidates were ruled out at frame 599 of MySlate_2 take 2 (chibi_strength=2.0, eyeBlinkRight=0.86):

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Lid verts pinned to original (preserve_eyeballs=1) | ✗ | "Two pairs of eyes" — original eyes ghosted onto stretched head |
| `s_r` amplifying lid forward-curl on the z axis | ✗ | `--blink_z_identity` render identical to baseline |
| Basis rescale of blink/squint rows (8, 9, 18, 19) | ✗ | `--blink_rows_identity` render identical to baseline |
| Basis rescale of all 22 eye-region rows (0–21) | ✗ | 22-row identity render identical to baseline |
| Any basis rescale | ✗ | Complete original-basis copy + chibi xyz override still leaked |
| `deform_verts` skipping verts below anchor | ✗ | 0/751 eye_region + 0/1092 eyeball verts below anchor |
| LAM not upsampling basis midpoints | ✗ | `flame_arkit.py:725-745` concatenates ARKit basis into shapedirs, upsamples both via pytorch3d |
| **Per-vertex Gaussian scales fixed at original FLAME density** | **✓** | Continuous degradation with chibi strength; `LAM_CHIBI_SCALE_BOOST=1.5` closes blink |
| LBS joint transforms in FLAME-canonical applied to chibi-displaced verts | — | Plausible secondary; v2 closes the visible artifact so not investigated further |

## v1: uniform `LAM_CHIBI_SCALE_BOOST` (sanity)

Multiplies every splat's linear scale by a constant. Implementation: ~5 lines next to the existing OBJ-edit hook in `modeling_lam.py`. Important: `_gm.scaling` is the **linear** sigma (already `trunc_exp`'d at construction time — see `gaussian_model.py:160` and the `np.log()` in the save path at line 76). Initial implementation added `log(r)` to `_gm.scaling`, which blew every splat into screen-spanning ovals. Corrected to direct multiplication.

`LAM_CHIBI_SCALE_BOOST=1.5` at f599 closes the half-blink completely — confirming splat density is the dominant cause. Side effect: the entire face becomes blurrier (uniform boost = uniform blur).

## v2: per-vertex `LAM_CHIBI_SCALE_RATIO` (proper fix)

`scripts/chibi_make_assets.py` emits `chibi_scale_ratio.npy` alongside `chibi_arkit_bs.npy` and `chibi_textured_mesh.obj`:

```python
# pseudocode
edges20018 = subdivided_edges(tpl_5023, tpl_faces)        # pytorch3d.SubdivideMeshes
el_orig    = per_vert_edge_mean(baked_verts,    edges20018)
el_chibi   = per_vert_edge_mean(baked_xyz_new,  edges20018)
ratio      = (el_chibi / el_orig).astype(np.float32)      # (20018,)
np.save(outdir / "chibi_scale_ratio.npy", ratio)
```

For `me` anchor: `s=1.0 → min=0.569 median=1.177 max=1.599`; `s=2.0 → min=0.275 median=1.358 max=2.368`. The 1.358 median matches our rough hand-calculation of the chibi radial stretch in the upper face. Min<1 happens near `y_anchor` where the chibi blend wraps — splats there get slightly smaller, fine.

Runtime hook in `modeling_lam.py` (same conditional block as the boost):

```python
ratio_t = torch.from_numpy(np.load(path)).to(_gm.scaling.device, dtype=_gm.scaling.dtype)
_gm.scaling = _gm.scaling * ratio_t.unsqueeze(-1)
```

## Three-moment visual gate

Verdict at `exp_output/lam_chibi/renders/v2_three_moment_gate.png`:

- **f150 (max eyeWide)**: pre-fix has small irises pinned in oversized sockets; v2 reads proportional.
- **f328 (max full-blink, both eyes closed)**: pre-fix shows iris through closed lid; v2 shows clean dark lids.
- **f599 (half-blink down-gaze, eyeBlinkRight=0.86)**: the canonical chase frame; pre-fix had iris poking out; v2 fully closes.

## Env-var surface

Three env vars now drive the chibi pipeline:

| Variable | Asset | Purpose |
|---|---|---|
| `LAM_CHIBI_ARKIT_BS` | `chibi_arkit_bs.npy` | Swap ARKit basis at FLAME init |
| `LAM_EDIT_XYZ_OBJ` | `chibi_textured_mesh.obj` | Override canonical xyz before LBS |
| `LAM_CHIBI_SCALE_RATIO` | `chibi_scale_ratio.npy` | Per-vertex splat-scale compensation |

`LAM_CHIBI_SCALE_BOOST=<float>` is also available as a sanity/fallback knob (applies uniformly across all splats).

## Open knobs

- **Anime-style oversized eyes.** Chibi recipe currently sizes eyes proportionally to the radial stretch (median 1.36×). For canonical anime "huge eyes" you'd multiply `chibi_scale_ratio` only on eyeball + eye_region verts by an additional `eye_boost`. Not implemented — current proportional sizing reads as "chibi-correct" enough.
- **Anisotropic v3.** If the per-vertex isotropic ratio leaves residual streaks along stretched directions, replace the scalar with a 2×2 tangent-plane covariance ratio applied in the splat's local frame. Not needed at current chibi strengths.

## Falsified earlier-session heuristics now retired

- "Some lid vertices were forgotten during transformation" — architecturally no: 0/751 eye-region verts and 0/1092 eyeball verts are below `y_anchor`, and the basis upsample at `flame_arkit.py:725-745` concatenates shape+expression dirs and upsamples both together. Confirmed by frame-599 probe.
- "Try a smaller chibi_strength as the production fix" — unnecessary now that the splat-scale matches the mesh stretch. s=2.0 is shippable.

## Files changed

- `/home/newub/w/LAM/lam/models/modeling_lam.py` — splat-scale compensation block (boost + ratio)
- `scripts/chibi_make_assets.py` — `per_vert_edge_mean()`, `subdivided_edges()`, ratio emit
- `scripts/chibi_eye_diag.sh` — wire `LAM_CHIBI_SCALE_RATIO`
- `scripts/chibi_render_with_assets.sh` — wire `LAM_CHIBI_SCALE_RATIO`

## Verdict artifact

`/home/newub/w/vamp-interface/exp_output/lam_chibi/renders/v2_three_moment_gate.png` — 3×2 panel (pre / v2 × f150 / f328 / f599). Full-take side-by-side at `sxs_chibi_me_v2ratio_vs_prefix.mp4`.
