# LAM chibi recipe — topic index

Living interpretation of the "human → chibi 3DGS" thread on top of LAM-20K.

## Current belief

**Chibi-on-splats is concluded dead (2026-05-15).** Chibi is too large a deformation for a baked Gaussian-splat representation — appearance is glued to verts at fixed density, so chibi-magnitude stretch always rescatters it (blur/smear/leak), and the vertex-space fit loss cannot see that failure. The secant-basis and neck-pancake fixes shipped on `chibi-diff-leak-fix` but the representation limit is not a fixable bug. **Pivot: bake LAM avatar → FLAME-UV-textured mesh, chibi as ordinary mesh deformation.** Splats→mesh research at `2026-05-15-splats-to-mesh-conversion.md`.

Prior belief (splat path, retained for audit): LAM-20K plus a three-asset chibi injection (basis swap + xyz override + per-vertex splat-scale ratio) renders a coherent chibi avatar at FLAME-rate from any LAM-compatible anchor PNG.

## Asset surface

| Variable | Asset | Purpose | Emitted by |
|---|---|---|---|
| `LAM_CHIBI_ARKIT_BS` | `chibi_arkit_bs.npy` | Swap ARKit-52 basis at FLAME init | `chibi_make_assets.py` |
| `LAM_EDIT_XYZ_OBJ` | `chibi_textured_mesh.obj` | Override canonical xyz before LBS | `chibi_make_assets.py` |
| `LAM_CHIBI_SCALE_RATIO` | `chibi_scale_ratio.npy` | Per-vertex Gaussian splat-scale compensation | `chibi_make_assets.py` |
| `LAM_CHIBI_SCALE_BOOST` | — | Uniform splat-scale knob (sanity / fallback) | none |

## Load-bearing dated docs

- [`2026-05-12-flame-for-stylized-anchors.md`](../2026-05-12-flame-for-stylized-anchors.md) — LAM bake-off verdict: 4/4 human + 2/2 stylized humanoid (orc, demon) anchors pass at ~310 fps on RTX 5090. Anime + non-human out of FLAME morphology. Anchor for the chibi thread.
- [`2026-05-13-lam-arkit-spike-resolved.md`](../2026-05-13-lam-arkit-spike-resolved.md) — Released LAM-20K Python checkpoint accepts ARKit-52 with a two-line patch. Gaussian net is identity-only; expression never touches a trained weight. Live LAM avatar from iPhone = ~1 day plumbing.
- [`2026-05-13-arkit-flame-mapping-extracted.md`](../2026-05-13-arkit-flame-mapping-extracted.md) — `flame_arkit_bs.npy` (52, 5023, 3) lives at `model_zoo/human_parametric_models/flame_assets/flame_arkit_bs.npy`. Used by both Python runtime and offline GLB-bake path.
- [`2026-05-14-chibi-splat-scale-fix.md`](../2026-05-14-chibi-splat-scale-fix.md) — Falsification ladder + v1 boost + v2 per-vertex ratio. Closes the iris-through-lid artifact at chibi_strength=2.0.
- [`2026-05-14-neural-renderer-override-pattern.md`](../2026-05-14-neural-renderer-override-pattern.md) — Methodology note distilled from the splat-scale fix: the override-audit checklist, closed-form pull-backs over trained corrections, when the scalar edge-ratio model breaks (anisotropic / large-stretch cases).
- [`2026-05-15-splats-to-mesh-conversion.md`](../2026-05-15-splats-to-mesh-conversion.md) — Splats→mesh for the pivot. LAM already *is* a FLAME-topology UV-unwrapped rigged mesh → skip SuGaR/2DGS surface extraction. **Color claim superseded** by 2026-05-16 (per-splat SH-DC is NOT usable surface color). Teeth/eyelid/hair findings stand.
- [`2026-05-16-splat-appearance-baking.md`](../2026-05-16-splat-appearance-baking.md) — **Most recent.** Milestone-0 gate result: splats→mesh geometry is clean (LAM `shaped_mesh.obj` = `xyz − offset`), but per-vertex splat color (`f_dc`/`shs`) is blotchy — it's alpha-blend color, not albedo. Accepted fix = **render-and-bake**: render the splats from N views, fit a UV texture by differentiable rasterisation (nvdiffrast) so the textured mesh matches the splat renders. SuGaR's `extract_refined_mesh_with_texture.py` is the reference impl; we skip its mesh-recon half (we own the FLAME mesh). Gotcha: linear/sRGB mismatch → washed-out texture.
- [`2026-05-15-chibi-painter-proportion-rules.md`](../2026-05-15-chibi-painter-proportion-rules.md) — Digital-painter rule set for chibi faces (head=sphere/block, features clustered low, eyes huge/low/~1-eye-width apart, nose→button, mouth→strip, nose-mouth:mouth-chin≈1:2, flat skin). Each rule mapped to a mesh op + tagged for differentiability — rules 1–7/8a are vertex-displacement (differentiable), rule 8b (flat skin) is an SH-DC appearance edit and must be a separate optimization target. Anti-creepy = no realistic shading on chibi proportions, and no literal-infant proportions.

## Retired hypotheses

- "Pin eyeball verts to original FLAME positions" — caused ghost second pair of eyes.
- "Rescale blink/squint/eye-region basis rows to match chibi geometry" — falsified across four variants. The basis was never the problem.
- "Some lid vertices were forgotten during transformation" — architecturally no; verified at f599.
- "Lower chibi_strength is the production fix" — unnecessary now that splat scales follow mesh stretch. s=2.0 is shippable.

## Open knobs (not currently exercised)

- **Anime-style oversized eyes.** Multiply `chibi_scale_ratio` only on `eyeball ∪ eye_region` verts by an additional `eye_boost`. Current proportional sizing reads as chibi-correct enough.
- **Anisotropic splat-scale (v3).** Replace the scalar ratio with a 2×2 tangent-plane covariance ratio if directional streaks appear at higher chibi strengths. Not currently needed.
- **LBS-joint mismatch.** `eyes_pose` joint transforms in `flame_arkit.py:790-801` are computed from the canonical FLAME template, not the chibi mesh. Plausible secondary cause of subtle eye-pose artifacts at high chibi strength; not investigated since v2 closed the primary artifact.

## Scripts

- `scripts/chibi_anchor_render.sh` — end-to-end: baseline LAM bake → SH→display-RGB rebake → chibi asset gen → chibi LAM render → side-by-side.
- `scripts/chibi_render_with_assets.sh` — fast path when chibi assets already exist for an anchor.
- `scripts/chibi_eye_diag.sh` — single-frame f599 harness for rapid blink-artifact iteration.
- `scripts/chibi_make_assets.py` — emits the three asset files from a (template, arkit_bs, baked-display-mesh) triple.

## Sibling threads

- [`liveportrait-stylized.md`](liveportrait-stylized.md) — stylized rendering via LP+SPADE rather than LAM. Different product shape (2D warp vs 3DGS).
- [`arkit-bridge.md`](arkit-bridge.md) — the ARKit→PersonaLive bridge; supplies the motion side. LAM consumes the same ARKit-52 stream.
