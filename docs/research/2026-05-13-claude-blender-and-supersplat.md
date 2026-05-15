---
status: live
topic: liveportrait-stylized
---

# Claude × Blender × SuperSplat for editing LAM canonical splats

**Date:** 2026-05-13
**Question (one sentence):** What can Claude actually drive, remotely and programmatically, to edit a LAM-produced canonical 3DGS `.ply` — recolor splats, prune, project a 2D style image onto colors, retint hair — across Blender, SuperSplat, and the Python ecosystem?
**Depth:** standard, 13 sources.

---

## Executive Summary

There is a usable remote-control path: **Claude Desktop → `ahujasid/blender-mcp` → Blender → KIRI 3DGS Render addon**. That stack exposes arbitrary Python execution inside Blender, plus KIRI's purpose-built brush-painting and image-texture-painting on individual Gaussian splats, which is the closest off-the-shelf match for "project a 2D style image onto splat colors" [1][8][9]. A more formal Blender Foundation MCP server is now in Blender Labs alongside the community one [3][4]. SuperSplat is the easiest visual editor but has **no remote/MCP/scripting surface from the browser app itself** — its sibling project `splat-transform` is a CLI/JS API that does geometric transforms, filtering, merging, and SH-band stripping but **does not recolor splats or project images** [5][6][7]. Pure-Python recoloring requires either a custom ~30-line script over the `.ply` SH coefficients or use of `splatviz` for interactive runtime edits — the most-cited PyPI library (`3dgs-edit-tools`) explicitly does not support color or SH editing today [10][11]. Net: for an agent-driven recolor / restyle of a canonical LAM splat, the highest-leverage path is Blender-MCP + KIRI; the cheapest scripted path is direct `.ply` manipulation; SuperSplat is the visual fallback for a human.

---

## Key Findings

### Blender MCP: two servers, one capability surface

The community server `ahujasid/blender-mcp` is the de-facto integration today and the one all third-party guides target [1][2][8][12]. It connects Claude (Desktop, or via the same `uvx blender-mcp` invocation from Claude Code) to a Blender addon over a local socket, and exposes seven tools: scene inspection, object manipulation, material control, **arbitrary Python execution via `execute_blender_code`**, viewport screenshot capture back to the model, Poly Haven asset fetch, and Hyper3D Rodin generation [1]. The `execute_blender_code` tool is the load-bearing one for our use case — there is no sandbox, so any operation that Blender's `bpy` Python API supports is reachable from a Claude tool call, including driving third-party addons. The repository explicitly warns "this can be powerful but potentially dangerous… ALWAYS save your work before using it" [1]. Requirement: Blender 3.6+ [2].

In parallel, the Blender Foundation has shipped an official MCP server under Blender Labs (`blender.org/lab/mcp-server`) and opened a developer-forum thread on its security model as of 2025 [3][4]. The Foundation's framing — surfaced through the devtalk thread title "MCP security for Blender scripting — 3D-Agent notes" — implies the official server is more cautious about sandboxing than the community one, though I could not retrieve the page body (403 to WebFetch) to verify the exact tool surface. For our purposes, the community server's arbitrary-Python escape hatch is what unlocks KIRI; if the official server lands with a comparable escape hatch, it becomes a drop-in.

### KIRI 3DGS Render: the only Blender splat editor with per-splat color painting

KIRI Innovation's free addon `Kiri-Innovation/3dgs-render-blender-addon` (v4.1.0, Blender 4.3+) is the dominant 3DGS workflow addon for Blender [8][9][13]. It treats `.ply` either as a mesh-like object in Edit Mode (Box/Circle select → delete floaters) or as a real-time render layer, with `.obj → .ply` conversion in both directions [8][9]. The capability that matters here is **painting**: v3.0 introduced direct brush-painting and image-texture-based painting of splat colors, with edits persisted through export to other 3DGS viewers [13]. v4.0 adds proxy-based real-time viewport, one-click compositing with traditional meshes, HQ/LQ transparency modes, and a `Remove Higher SH` modifier; v4.1 piles on more crop-box modes and an auto-crop function [8][9]. The official write-up confirms colour edits visible in real-time Render Mode and persisted through export, but does not detail whether the brush writes to per-splat DC SH coefficients or to a texture overlay — a gap (see *Open Questions*).

Because the addon is just Blender Python, `bpy.context`-driven calls into its operators are reachable from `execute_blender_code` over the MCP. That means a Claude turn can: load a `.ply`, switch to Edit Mode, run box-select, run the painting operator, set a brush color or image-texture target, dab, and trigger export — entirely from chat. The viewport-screenshot tool lets Claude visually verify each step [1].

A second Blender addon, `Splats`, is published on the official Blender Extensions registry and is the lower-friction install path but ships fewer editing features than KIRI [14]. KIRI on Superhive (formerly Blender Market) is the same codebase with a marketplace listing [8].

### SuperSplat: best UI for splats, no remote control

PlayCanvas's `playcanvas/supersplat` is a free open-source browser-based 3DGS editor at `superspl.at/editor` [5][6][15]. Capabilities visible from the docs and release notes: import/export PLY, compressed PLY, SPLAT, SOG/SOGS, KSPLAT, CSV; three selection tools (Rect, Brush, Picker); two visualization modes (Centers, Rings); colour adjustment during editing; one-click web publishing; SuperSplat 2.0 timeline-based camera animation; SH bands loaded and rendered since v1.2.0, with export-time control over how many SH bands to keep [5][7][15][16]. What's not there: any documented Python/JS/CLI API exposed from inside the web app, nor any MCP server. Color editing exists in the UI, but the docs I retrieved do not specify whether it's per-splat brush-painted or a global grade — a gap.

The sibling project `playcanvas/splat-transform` is the actual scriptable surface for the PlayCanvas stack [6]. It's a CLI plus JS library with a `ProcessAction` API covering: `translate`, `rotate`, `scale`, `filterNaN`, `filterByValue` (filter by opacity/scale/colors), `filterBands` (drop SH bands above a threshold), `filterBox`, `filterSphere`, `filterFloaters`, `filterCluster`, `decimate`, `param`, `lod`, `mortonOrder`, `summary` [6]. Input formats: PLY, Compressed PLY, SOG, SPZ, SPLAT, KSPLAT, LCC, plus JS-generator inputs. Output: PLY, Compressed PLY, SOG, GLB, CSV, HTML viewer bundle, LOD bundles, voxel octrees. **It does not include a recolor, hue-shift, image-projection, or per-splat color paint action** — its color-related operator (`filterByValue`) only filters by, not changes, colors [6].

### Pure-Python paths: small library, easy custom script

The PyPI library `3dgs-edit-tools` advertises itself as the "editing" library but explicitly does not support color or SH editing, recoloring, or 2D image projection — its operations are PLY↔CSV conversion (round-tripped through a spreadsheet for human edits), merge, and geometric transform [10]. So if the goal is automated recolor, that library is not the answer.

The remaining Python options are `splatviz` (interactive PyQt-class real-time viewer with an "edit widget" that allows real-time mutation of the Gaussian Python object at runtime, useful for human iteration but not for headless batch) [11], or **direct manipulation of the `.ply` using `plyfile` / `numpy`**. The 3DGS PLY layout is well-documented: per-Gaussian properties include `xyz`, `f_dc_0..2` (DC term of the spherical harmonics = base color), `f_rest_0..44` (higher SH bands), `opacity`, `scale_0..2`, `rot_0..3` [17, original 3DGS reference impl]. A few-dozen-line numpy script can mask splats by spatial region or color-similarity, replace `f_dc_*`, zero `f_rest_*` (or rescale), and write back. This is the simplest agent-driven path: Claude writes the script and runs it with Bash, no Blender required.

### What "project a 2D style image onto splats" actually means

The closest off-the-shelf realization is **KIRI's image-based painting** in Blender [13]. The user (or `execute_blender_code` from Claude) loads a 2D image into the brush as a texture source and paints across the splat surface; the addon writes the sampled colors into the splats. This treats the splat cloud as a paintable surface and inherits Blender's standard projection-paint UX. For our LAM canonical, where the splat cloud is bound to FLAME mesh topology, you can additionally project a 2D anime portrait onto the FLAME mesh and then bake the projected colors onto the underlying splats. CGChannel's writeup of v3.0 confirms painting is real and end-to-end, not a viewport-only filter [13].

A scripted equivalent is feasible: rasterize the canonical splat positions through the same camera that produced the 2D anchor, project the 2D image's colors back to each splat by depth-test, and write the projected RGB into the DC SH coefficients. No off-the-shelf library does exactly this for 3DGS yet — it's a ~half-day script.

### Practical Claude-driven workflows for our LAM pipeline

Three paths matter for the LAM canonical-repaint experiment:

1. **Blender-MCP + KIRI, interactive-by-chat.** Spin up Blender, attach the addon, configure `ahujasid/blender-mcp` in Claude Desktop. Claude calls `execute_blender_code` to import `young_european_f_gs_offset.ply`, switch to Edit Mode, drive KIRI's painting operators with a 2D anime portrait as the brush texture, and export. The viewport-screenshot tool gives Claude visual feedback to course-correct. This is the highest-leverage path because it composes a tested off-the-shelf splat-painter with Claude's general competence.

2. **Direct `.ply` recolor via custom script.** Claude writes a numpy/plyfile script that edits `f_dc_*` and `f_rest_*` per-splat — masking by spatial region (hair, skin) using FLAME-bound positions, projecting a 2D image, or applying a learned color transform. No GUI, headless, agent-native. This is the fastest to iterate on if the goal is reproducible algorithmic edits rather than human-feel artistry.

3. **Human in SuperSplat, agent post-processes.** Open the canonical `.ply` in `superspl.at/editor`, hand-crop to head-only, export. Pipe through `splat-transform` for any geometric or filter pass, then run our own recolor script. SuperSplat does the things its UI is good at (visual cleanup, crop, opacity) while Claude scripts the color logic. Useful when the user wants tactile control over masking.

## Comparison

| Tool | Agent-controllable | Per-splat color edit | 2D image projection | SH band edits | Notes |
|---|---|---|---|---|---|
| ahujasid/blender-mcp + KIRI | ✅ via `execute_blender_code` | ✅ via KIRI brush | ✅ via KIRI image-texture paint | ⚠️ KIRI has `Remove Higher SH`, others via custom bpy | Most powerful; no sandbox [1][8][13] |
| Official Blender MCP (Labs) | ✅ (degree TBD) | same as above (via Blender) | same | same | Capabilities not fully verified [3][4] |
| SuperSplat web | ❌ | ✅ in UI only | ❌ | ⚠️ export-time band-count control | No scripting surface from the browser app [5][7] |
| splat-transform CLI/JS | ✅ via Bash | ❌ | ❌ | ✅ `filterBands` only drops | Geometric/filter-only [6] |
| `3dgs-edit-tools` (PyPI) | ✅ via Bash | ❌ | ❌ | ❌ | PLY↔CSV roundtrip; no color [10] |
| `splatviz` | ⚠️ interactive only | ✅ at runtime | ❌ | ✅ at runtime | Not headless batch [11] |
| Custom numpy + plyfile script | ✅ via Bash | ✅ | ✅ (with projection code) | ✅ | ~30-line floor; ~half-day for projection |

## Open Questions

- **KIRI brush internals.** The promo materials confirm "color edits persist through export" [9][13], but no source I retrieved spells out whether the brush writes per-splat DC SH coefficients or a separate texture layer that's baked at export. This matters for downstream compatibility — if it's a texture layer, exports to viewers without KIRI's pipeline may lose the colors. Worth confirming by importing a KIRI-painted `.ply` into SuperSplat and inspecting `f_dc_*` against the original.
- **Official Blender MCP scope.** The 403 from `blender.org/lab/mcp-server` blocked me from confirming the tool surface of the Foundation's official server [3]. The forum thread title strongly implies a tighter sandbox than ahujasid's, which could limit `execute_blender_code`-style escape hatches [4].
- **SuperSplat color editing UI specifics.** Multiple search snippets confirm "color editing" / "adjust colors" exists [5][6][7][15], but I couldn't retrieve a detailed description of whether it's a per-splat brush, a global grade, or both. Worth a live check in the editor.

## Sources

[1] Ahuja, Siddharth. "blender-mcp — Blender Model Context Protocol Integration." GitHub, https://github.com/ahujasid/blender-mcp (Retrieved 2026-05-13).
[2] Vagon Blog. "How to Use Blender MCP with Anthropic's Claude AI." https://vagon.io/blog/how-to-use-blender-mcp-with-anthropic-claude-ai (Retrieved 2026-05-13).
[3] Blender Foundation. "MCP Server — Blender Labs." https://www.blender.org/lab/mcp-server/ (Retrieved 2026-05-13; HTTP 403 on body; presence confirmed by search snippet).
[4] Blender Developer Forum. "Blender MCP Server after Claude: MCP security for Blender scripting — 3D-Agent notes." https://devtalk.blender.org/t/blender-mcp-server-after-claude-mcp-security-for-blender-scripting-3d-agent-notes/45131 (Retrieved 2026-05-13; 403 on body; presence confirmed by search snippet).
[5] PlayCanvas. "supersplat — 3D Gaussian Splat Editor." GitHub, https://github.com/playcanvas/supersplat (Retrieved 2026-05-13).
[6] PlayCanvas. "splat-transform — CLI tool and library for 3D Gaussian splat processing and conversion." GitHub, https://github.com/playcanvas/splat-transform (Retrieved 2026-05-13).
[7] PlayCanvas Developer Site. "SuperSplat | Gaussian Splatting editing." https://developer.playcanvas.com/user-manual/gaussian-splatting/editing/supersplat/ (Retrieved 2026-05-13).
[8] KIRI Innovation. "3dgs-render-blender-addon v4.1.0." GitHub, https://github.com/Kiri-Innovation/3dgs-render-blender-addon (Retrieved 2026-05-13).
[9] KIRI Engine. "KIRI Engine 3DGS Render v4.0 for Blender — A Faster Edit→Render Pipeline for Gaussian Splats." https://www.kiriengine.app/blog/kiri-engine-3DGS%20Render-Blender-v4.0 (Retrieved 2026-05-13).
[10] 404background. "3dgs-edit-tools." PyPI, https://pypi.org/project/3dgs-edit-tools/ — and GitHub repo https://github.com/404background/3dgs-edit-tools (Retrieved 2026-05-13).
[11] Barthel, Florian. "splatviz — Full python interactive 3D Gaussian Splatting viewer for real-time editing and analyzing." GitHub, https://github.com/Florian-Barthel/splatviz (Retrieved 2026-05-13).
[12] Eigent.ai Blog. "Claude for Creative Work: Blender MCP Connector Guide 2026." https://www.eigent.ai/blog/claude-blender-mcp (Retrieved 2026-05-13).
[13] CG Channel. "3DGS Render 3.0 lets you paint 3D Gaussian Splats in Blender." https://www.cgchannel.com/2025/03/3dgs-render-3-0-lets-you-paint-3d-gaussian-splats-in-blender/ (Retrieved 2026-05-13).
[14] Blender Extensions. "Splats." https://extensions.blender.org/add-ons/splats/ (Retrieved 2026-05-13).
[15] PlayCanvas Blog. "Publish Your Gaussian Splats with SuperSplat 2.0." https://blog.playcanvas.com/publish-your-gaussian-splats-with-supersplat/ (Retrieved 2026-05-13).
[16] Eastcott, Will. "SuperSplat 1.2.0 — Spherical harmonic bands are now loaded and rendered." X/Twitter, https://x.com/willeastcott/status/1825838435090718751 (Retrieved 2026-05-13).
[17] Kerbl, Kopanas, Leimkühler, Drettakis. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." Original reference implementation, https://github.com/graphdeco-inria/gaussian-splatting (Retrieved 2026-05-13).
